from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.validation.purged_cv import (
    PurgedWalkForwardConfig,
    PurgedWalkForwardSplitter,
    assert_time_sorted,
)


@dataclass(frozen=True)
class WalkForwardRunConfig:
    feature_cols: List[str]
    label_col: str

    # Purged walk-forward
    train_size: int
    test_size: int
    step_size: int
    purge_size: int
    embargo_size: int = 0
    min_train_rows: int = 200
    min_test_rows: int = 20

    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 100
    early_stop_val_frac: float = 0.15

    # Calibration
    calibrate: bool = True
    calibrator_method: str = "sigmoid"
    calibrator_val_frac: float = 0.15

    # Persistence
    save_models: bool = False
    model_dir: str = "models"
    save_latest_model_bundle: bool = False
    latest_model_path: str = "deployment/live_model_bundle.joblib"


def _is_lightgbm_model(model: BaseEstimator) -> bool:
    name = model.__class__.__name__.lower()
    mod = (model.__class__.__module__ or "").lower()
    return ("lgbm" in name) or ("lightgbm" in mod)


def _is_xgboost_model(model: BaseEstimator) -> bool:
    name = model.__class__.__name__.lower()
    mod = (model.__class__.__module__ or "").lower()
    return ("xgb" in name) or ("xgboost" in mod)


def _split_train_for_earlystop_and_calibration(
    train_idx: np.ndarray,
    early_stop_frac: float,
    calibrator_frac: float,
    enable_early_stopping: bool,
    enable_calibration: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    total_tail = 0.0

    if enable_early_stopping:
        if not (0.0 < early_stop_frac < 0.5):
            raise ValueError("early_stop_val_frac must be between (0, 0.5)")
        total_tail += early_stop_frac

    if enable_calibration:
        if not (0.0 < calibrator_frac < 0.5):
            raise ValueError("calibrator_val_frac must be between (0, 0.5)")
        total_tail += calibrator_frac

    if total_tail >= 0.5:
        raise ValueError("early_stop_val_frac + calibrator_val_frac must be < 0.5")

    n = len(train_idx)
    n_cal = max(1, int(n * calibrator_frac)) if enable_calibration else 0
    n_es = max(1, int(n * early_stop_frac)) if enable_early_stopping else 0

    end = n
    cal_idx = None
    es_idx = None

    if enable_calibration:
        cal_idx = train_idx[end - n_cal : end]
        end -= n_cal

    if enable_early_stopping:
        es_idx = train_idx[end - n_es : end]
        end -= n_es

    fit_idx = train_idx[:end]

    if len(fit_idx) < 50:
        raise ValueError("Train too small after splits; reduce fracs or increase train_size.")

    return fit_idx, es_idx, cal_idx


def _fit_with_optional_early_stopping(
    model: BaseEstimator,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_es: Optional[pd.DataFrame],
    y_es: Optional[pd.Series],
    cfg: WalkForwardRunConfig,
) -> BaseEstimator:
    use_es = (
        cfg.enable_early_stopping
        and X_es is not None
        and y_es is not None
        and len(X_es) >= 10
        and (_is_lightgbm_model(model) or _is_xgboost_model(model))
    )

    if not use_es:
        model.fit(X_fit, y_fit)
        return model

    try:
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_es, y_es)],
            early_stopping_rounds=cfg.early_stopping_rounds,
            verbose=False,
        )
    except TypeError:
        model.fit(X_fit, y_fit)

    return model


def walk_forward_train_predict(
    df: pd.DataFrame,
    model_factory: Callable[[], BaseEstimator],
    cfg: WalkForwardRunConfig,
    time_col: Optional[str] = None,
    split_group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      pred_df: out-of-sample predictions
      diag: diagnostics dict with OOS metrics and fold diagnostics
    """
    assert_time_sorted(df, time_col=split_group_col or time_col)

    X = df[cfg.feature_cols].astype(float)
    y = df[cfg.label_col].astype(int)

    if split_group_col is not None:
        split_groups = pd.Series(df[split_group_col]).reset_index(drop=True)
        group_codes, group_values = pd.factorize(split_groups, sort=False)
        n = len(group_values)
    else:
        split_groups = None
        group_codes = None
        n = len(df)
    splitter = PurgedWalkForwardSplitter(
        PurgedWalkForwardConfig(
            train_size=cfg.train_size,
            test_size=cfg.test_size,
            step_size=cfg.step_size,
            purge_size=cfg.purge_size,
            embargo_size=cfg.embargo_size,
        )
    )

    out_index = df[time_col] if time_col is not None else df.index
    preds_out: List[pd.DataFrame] = []
    fold_diags: List[Dict[str, Any]] = []
    latest_model_artifact: Optional[Dict[str, Any]] = None

    os.makedirs(cfg.model_dir, exist_ok=True)

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(n)):
        if split_group_col is not None:
            train_row_idx = np.flatnonzero(np.isin(group_codes, train_idx))
            test_row_idx = np.flatnonzero(np.isin(group_codes, test_idx))
        else:
            train_row_idx = train_idx
            test_row_idx = test_idx

        train_mask = np.isfinite(X.iloc[train_row_idx].to_numpy()).all(axis=1) & np.isfinite(y.iloc[train_row_idx].to_numpy())
        test_mask = np.isfinite(X.iloc[test_row_idx].to_numpy()).all(axis=1) & np.isfinite(y.iloc[test_row_idx].to_numpy())

        train_idx_eff = train_row_idx[train_mask]
        test_idx_eff = test_row_idx[test_mask]

        if len(train_idx_eff) < cfg.min_train_rows or len(test_idx_eff) < cfg.min_test_rows:
            continue

        X_test = X.iloc[test_idx_eff]
        y_test = y.iloc[test_idx_eff]

        try:
            fit_idx, es_idx, cal_idx = _split_train_for_earlystop_and_calibration(
                train_idx=train_idx_eff,
                early_stop_frac=cfg.early_stop_val_frac,
                calibrator_frac=cfg.calibrator_val_frac,
                enable_early_stopping=cfg.enable_early_stopping,
                enable_calibration=cfg.calibrate,
            )
        except ValueError:
            fit_idx = train_idx_eff
            es_idx = None
            cal_idx = None

        X_fit = X.iloc[fit_idx]
        y_fit = y.iloc[fit_idx]

        X_es = X.iloc[es_idx] if es_idx is not None else None
        y_es = y.iloc[es_idx] if es_idx is not None else None

        if pd.Series(y_fit).nunique() < 2:
            constant_proba = float(pd.Series(y_fit).iloc[0]) if len(y_fit) else 0.0
            fold_df = pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "proba_raw": np.full(len(y_test), constant_proba, dtype=float),
                    "y_true": y_test.to_numpy(),
                },
                index=out_index[test_idx_eff],
            )
            fold_diags.append({
                "fold_id": int(fold_id),
                "n": int(len(fold_df)),
                "pos_rate": float(np.mean(fold_df["y_true"])) if len(fold_df) else 0.0,
                "proba_col": "proba_raw",
                "auc": float("nan"),
                "auc_inverted": float("nan"),
                "logloss": float("nan"),
                "start": str(fold_df.index.min()),
                "end": str(fold_df.index.max()),
                "note": f"single_class_train={int(constant_proba)}",
            })
            preds_out.append(fold_df)
            print(f"Fold {fold_id} | single-class training slice -> constant proba={constant_proba:.3f}")
            continue

        model = model_factory()

        model = _fit_with_optional_early_stopping(model, X_fit, y_fit, X_es, y_es, cfg)

        proba_raw = model.predict_proba(X_test)[:, 1]

        if cfg.calibrate and cal_idx is not None and len(cal_idx) >= 20:
            X_cal = X.iloc[cal_idx]
            y_cal = y.iloc[cal_idx]

            calibrator = CalibratedClassifierCV(
                FrozenEstimator(model),
                method=cfg.calibrator_method,
            )
            calibrator.fit(X_cal, y_cal)
            proba_cal = calibrator.predict_proba(X_test)[:, 1]

            if cfg.save_models:
                joblib.dump(
                    {"model": model, "calibrator": calibrator, "cfg": cfg, "fold_id": fold_id},
                    os.path.join(cfg.model_dir, f"fold_{fold_id:03d}.joblib"),
                )

            latest_model_artifact = {
                "model": model,
                "calibrator": calibrator,
                "cfg": cfg,
                "fold_id": fold_id,
                "feature_cols": list(cfg.feature_cols),
                "label_col": cfg.label_col,
                "uses_calibrator": True,
            }

            fold_df = pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "proba_raw": proba_raw,
                    "proba_cal": proba_cal,
                    "y_true": y_test.to_numpy(),
                },
                index=out_index[test_idx_eff],
            )
        else:
            if cfg.save_models:
                joblib.dump(
                    {"model": model, "cfg": cfg, "fold_id": fold_id},
                    os.path.join(cfg.model_dir, f"fold_{fold_id:03d}.joblib"),
                )

            latest_model_artifact = {
                "model": model,
                "cfg": cfg,
                "fold_id": fold_id,
                "feature_cols": list(cfg.feature_cols),
                "label_col": cfg.label_col,
                "uses_calibrator": False,
            }

            fold_df = pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "proba_raw": proba_raw,
                    "y_true": y_test.to_numpy(),
                },
                index=out_index[test_idx_eff],
            )

        try:
            from sklearn.metrics import roc_auc_score, log_loss

            proba_col = "proba_cal" if "proba_cal" in fold_df.columns else "proba_raw"
            y_true_fold = fold_df["y_true"].astype(int).to_numpy()
            p_fold = fold_df[proba_col].astype(float).to_numpy()
            fold_pos_rate = float(np.mean(y_true_fold))
            if len(np.unique(y_true_fold)) < 2:
                fold_auc = float("nan")
                fold_auc_inv = float("nan")
                fold_ll = float(log_loss(y_true_fold, p_fold, labels=[0, 1]))
                fold_note = "single_class_test"
            else:
                fold_auc = float(roc_auc_score(y_true_fold, p_fold))
                fold_auc_inv = float(roc_auc_score(y_true_fold, 1.0 - p_fold))
                fold_ll = float(log_loss(y_true_fold, p_fold))
                fold_note = None

            fold_diags.append({
                "fold_id": int(fold_id),
                "n": int(len(fold_df)),
                "pos_rate": fold_pos_rate,
                "proba_col": proba_col,
                "auc": fold_auc,
                "auc_inverted": fold_auc_inv,
                "logloss": fold_ll,
                "start": str(fold_df.index.min()),
                "end": str(fold_df.index.max()),
                "note": fold_note,
            })

            if fold_note == "single_class_test":
                print(
                    f"Fold {fold_id} | n={len(fold_df)} | pos={fold_pos_rate:.3f} | "
                    f"AUC=nan | invAUC=nan | LL={fold_ll:.6f} | note={fold_note}"
                )
            else:
                print(
                    f"Fold {fold_id} | n={len(fold_df)} | pos={fold_pos_rate:.3f} | "
                    f"AUC={fold_auc:.4f} | invAUC={fold_auc_inv:.4f} | LL={fold_ll:.6f}"
                )
        except Exception as e:
            print(f"Fold {fold_id} diagnostics skipped: {e}")

        preds_out.append(fold_df)

    if not preds_out:
        raise RuntimeError("No folds produced predictions. Check window sizes / NaNs / label availability.")

    pred_df = pd.concat(preds_out).sort_index()

    oos_auc = float("nan")
    oos_auc_inverted = float("nan")
    oos_logloss = float("nan")
    proba_col = "proba_cal" if "proba_cal" in pred_df.columns else "proba_raw"

    try:
        from sklearn.metrics import roc_auc_score, log_loss

        y_true = pred_df["y_true"].astype(int).to_numpy()
        p = pred_df[proba_col].astype(float).to_numpy()
        oos_logloss = float(log_loss(y_true, p))
        if len(np.unique(y_true)) >= 2:
            oos_auc = float(roc_auc_score(y_true, p))
            oos_auc_inverted = float(roc_auc_score(y_true, 1.0 - p))

        print("\n=== OOS Classification Diagnostics ===")
        print(f"OOS AUC: {oos_auc:.4f}" if np.isfinite(oos_auc) else "OOS AUC: nan")
        print(f"OOS LogLoss: {oos_logloss:.6f}")
        print(
            f"OOS AUC (inverted probs): {oos_auc_inverted:.4f}"
            if np.isfinite(oos_auc_inverted)
            else "OOS AUC (inverted probs): nan"
        )
    except Exception as e:
        print(f"OOS diagnostics skipped: {e}")

    diag = {
        "proba_col": proba_col,
        "oos_auc": oos_auc,
        "oos_auc_inverted": oos_auc_inverted,
        "oos_logloss": oos_logloss,
        "fold_diags": fold_diags,
        "n_preds": int(len(pred_df)),
    }

    if cfg.save_latest_model_bundle and latest_model_artifact is not None:
        bundle_dir = os.path.dirname(cfg.latest_model_path)
        if bundle_dir:
            os.makedirs(bundle_dir, exist_ok=True)
        joblib.dump(latest_model_artifact, cfg.latest_model_path)
        diag["latest_model_path"] = cfg.latest_model_path
        diag["latest_model_fold_id"] = latest_model_artifact["fold_id"]

    return pred_df, diag
