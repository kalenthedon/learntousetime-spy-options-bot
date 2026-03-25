from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ml.walkforward_train import WalkForwardRunConfig, walk_forward_train_predict
from src.options.contracts import normalize_option_chain_frame
from src.options.features import make_option_features
from src.options.labels import make_option_labels


DEFAULT_OUTPUT_PATH = "experiments/options_research_summary.json"


def load_option_chain_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_option_chain_frame(df)


def prepare_option_research_frame(
    df: pd.DataFrame,
    horizon_bars: int,
    target_return_pct: float,
    max_adverse_return_pct: float,
) -> tuple[pd.DataFrame, list[str]]:
    feat = make_option_features(df)
    feat["y"] = make_option_labels(
        feat,
        horizon_bars=horizon_bars,
        target_return_pct=target_return_pct,
        max_adverse_return_pct=max_adverse_return_pct,
    )
    feat = feat.dropna().reset_index(drop=True)

    feature_cols = [
        "days_to_expiry",
        "moneyness",
        "spread_pct_mid",
        "extrinsic_pct_mid",
        "mark_iv",
        "delta",
        "delta_abs",
        "gamma",
        "theta_to_premium",
        "vega_to_premium",
        "dollar_gamma",
        "open_interest",
        "volume",
        "liquidity_score",
        "mid_ret_1",
        "mid_ret_3",
        "iv_chg_1",
        "iv_chg_3",
        "underlying_ret_1",
        "underlying_ret_3",
        "volume_z_10",
        "spread_rank_in_chain",
        "oi_rank_in_chain",
    ]
    return feat, feature_cols


def summarize_top_contracts(pred_df: pd.DataFrame, proba_col: str, top_k: int = 1) -> dict:
    if pred_df.empty:
        return {
            "timestamps": 0,
            "avg_top_probability": None,
            "avg_positive_rate_top": None,
            "top_k": top_k,
        }

    working = pred_df.copy()
    working["rank"] = working.groupby("timestamp")[proba_col].rank(method="first", ascending=False)
    top = working.loc[working["rank"] <= top_k].copy()
    return {
        "timestamps": int(working["timestamp"].nunique()),
        "avg_top_probability": float(top[proba_col].mean()),
        "avg_positive_rate_top": float(top["y_true"].mean()),
        "top_k": int(top_k),
        "rows_selected": int(len(top)),
    }


def run_options_research(
    csv_path: str,
    train_size: int,
    test_size: int,
    step_size: int,
    horizon_bars: int,
    target_return_pct: float,
    max_adverse_return_pct: float,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict:
    raw = load_option_chain_csv(csv_path)
    frame, feature_cols = prepare_option_research_frame(
        raw,
        horizon_bars=horizon_bars,
        target_return_pct=target_return_pct,
        max_adverse_return_pct=max_adverse_return_pct,
    )
    frame = frame.reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame))

    cfg = WalkForwardRunConfig(
        feature_cols=feature_cols,
        label_col="y",
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        purge_size=horizon_bars,
        calibrate=False,
        save_models=False,
        save_latest_model_bundle=False,
    )

    pred_df, diag = walk_forward_train_predict(
        frame,
        model_factory=lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs"),
        ),
        cfg=cfg,
        time_col="row_id",
    )
    pred_df = pred_df.join(
        frame.set_index("row_id")[["timestamp", "option_symbol", "underlying_symbol", "option_type", "days_to_expiry"]]
    )

    proba_col = "proba_cal" if "proba_cal" in pred_df.columns else "proba_raw"
    auc = None
    ll = None
    if not pred_df.empty and pred_df["y_true"].nunique() >= 2:
        auc = float(roc_auc_score(pred_df["y_true"], pred_df[proba_col]))
        ll = float(log_loss(pred_df["y_true"], pred_df[proba_col], labels=[0, 1]))

    summary = {
        "csv_path": csv_path,
        "rows_raw": int(len(raw)),
        "rows_model": int(len(frame)),
        "underlyings": sorted(frame["underlying_symbol"].dropna().unique().tolist()),
        "contracts": int(frame["option_symbol"].nunique()),
        "horizon_bars": int(horizon_bars),
        "target_return_pct": float(target_return_pct),
        "max_adverse_return_pct": float(max_adverse_return_pct),
        "oos_rows": int(len(pred_df)),
        "proba_col": proba_col,
        "oos_auc": auc,
        "oos_logloss": ll,
        "oos_positive_rate": float(pred_df["y_true"].mean()) if not pred_df.empty else None,
        "top_contract_summary": summarize_top_contracts(pred_df, proba_col=proba_col, top_k=1),
        "fold_count": int(len(diag.get("fold_diags", []))),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward ML research on option-chain snapshots.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--step-size", type=int, default=1000)
    parser.add_argument("--horizon-bars", type=int, default=8)
    parser.add_argument("--target-return-pct", type=float, default=0.25)
    parser.add_argument("--max-adverse-return-pct", type=float, default=-0.20)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    summary = run_options_research(
        csv_path=args.csv_path,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        horizon_bars=args.horizon_bars,
        target_return_pct=args.target_return_pct,
        max_adverse_return_pct=args.max_adverse_return_pct,
        output_path=args.output_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
