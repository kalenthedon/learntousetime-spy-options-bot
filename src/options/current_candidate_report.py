from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.backtests.run_options_research import (
    apply_selection_orientation,
    build_model_factory,
    build_option_feature_frame,
    load_option_chain_csv,
    prepare_option_research_frame,
    resolve_selection_orientation,
)
from src.ml.walkforward_train import WalkForwardRunConfig, walk_forward_train_predict
from src.options.contracts import normalize_option_chain_frame


DEFAULT_HISTORY_PATH = "centralized_data/options/SPY_put_chain_history.csv"
DEFAULT_LATEST_PATH = "centralized_data/options/SPY_put_chain_latest.csv"
DEFAULT_OUTPUT_PATH = "experiments/options_current_candidates.json"
DEFAULT_CSV_OUTPUT_PATH = "experiments/options_current_candidates.csv"


def _load_snapshot_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    return normalize_option_chain_frame(pd.read_csv(csv_path))


def _records_for_json(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].astype(str)
    return json.loads(safe.to_json(orient="records"))


def _fit_current_model(
    history_path: str,
    model_type: str,
    train_size: int,
    test_size: int,
    step_size: int,
    horizon_bars: int,
    target_return_pct: float,
    max_adverse_return_pct: float,
    selection_orientation: str,
) -> tuple[object, list[str], dict]:
    history = load_option_chain_csv(history_path)
    frame, feature_cols = prepare_option_research_frame(
        history,
        horizon_bars=horizon_bars,
        target_return_pct=target_return_pct,
        max_adverse_return_pct=max_adverse_return_pct,
    )
    frame = frame.reset_index(drop=True)
    if "collected_at" in frame.columns:
        frame["snapshot_time"] = pd.to_datetime(frame["collected_at"], utc=True, errors="coerce")
    else:
        frame["snapshot_time"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.floor("15min")
    frame = frame.dropna(subset=["snapshot_time"]).sort_values(
        ["snapshot_time", "timestamp", "expiration", "strike", "option_type", "option_symbol"]
    ).reset_index(drop=True)
    frame["row_id"] = range(len(frame))
    frame["snapshot_id"] = pd.factorize(frame["snapshot_time"], sort=False)[0]

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
        min_train_rows=150,
        min_test_rows=20,
    )
    pred_df, diag = walk_forward_train_predict(
        frame,
        model_factory=build_model_factory(model_type),
        cfg=cfg,
        time_col="row_id",
        split_group_col="snapshot_id",
    )
    proba_col = "proba_cal" if "proba_cal" in pred_df.columns else "proba_raw"
    resolved_orientation = resolve_selection_orientation(
        selection_orientation,
        float(diag.get("oos_auc")) if diag.get("oos_auc") is not None else None,
        float(diag.get("oos_auc_inverted")) if diag.get("oos_auc_inverted") is not None else None,
    )

    model = build_model_factory(model_type)()
    model.fit(frame[feature_cols].astype(float), frame["y"].astype(int))

    diagnostics = {
        "model_type": model_type,
        "proba_col": proba_col,
        "selection_orientation": resolved_orientation,
        "oos_auc": float(diag.get("oos_auc")) if diag.get("oos_auc") is not None else None,
        "oos_auc_inverted": float(diag.get("oos_auc_inverted")) if diag.get("oos_auc_inverted") is not None else None,
        "fold_count": int(len(diag.get("fold_diags", []))),
        "rows_model": int(len(frame)),
        "snapshot_count": int(frame["snapshot_id"].nunique()),
    }
    return model, feature_cols, diagnostics


def build_current_candidate_report(
    history_path: str = DEFAULT_HISTORY_PATH,
    latest_path: str = DEFAULT_LATEST_PATH,
    model_type: str = "logistic",
    train_size: int = 48,
    test_size: int = 12,
    step_size: int = 12,
    horizon_bars: int = 8,
    target_return_pct: float = 0.25,
    max_adverse_return_pct: float = -0.20,
    min_entry_score: float = 0.30,
    top_k: int = 3,
    selection_orientation: str = "auto",
    output_path: str = DEFAULT_OUTPUT_PATH,
    csv_output_path: str = DEFAULT_CSV_OUTPUT_PATH,
) -> dict:
    latest = _load_snapshot_csv(latest_path)
    if latest.empty:
        raise RuntimeError(f"No latest option snapshot rows found at {latest_path}")

    model, feature_cols, diagnostics = _fit_current_model(
        history_path=history_path,
        model_type=model_type,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        horizon_bars=horizon_bars,
        target_return_pct=target_return_pct,
        max_adverse_return_pct=max_adverse_return_pct,
        selection_orientation=selection_orientation,
    )

    history = load_option_chain_csv(history_path)
    history = history.copy()
    history["row_source"] = "history"
    latest = latest.copy()
    latest["row_source"] = "latest"
    combined = pd.concat([history, latest], ignore_index=True)
    combined = combined.sort_values(["timestamp", "expiration", "strike", "option_type", "option_symbol"]).reset_index(drop=True)
    feature_frame, _ = build_option_feature_frame(combined)

    scored = feature_frame.loc[feature_frame["row_source"] == "latest"].copy()
    if scored.empty:
        raise RuntimeError("Could not align latest snapshot rows into feature frame")

    scored["proba_raw"] = model.predict_proba(scored[feature_cols].astype(float))[:, 1]
    scored, score_col = apply_selection_orientation(
        scored,
        proba_col="proba_raw",
        orientation=diagnostics["selection_orientation"],
    )
    scored = scored.sort_values([score_col, "candidate_score", "open_interest", "volume"], ascending=[False, False, False, False]).reset_index(drop=True)
    scored["rank"] = range(1, len(scored) + 1)
    scored["selected"] = (scored["rank"] <= int(top_k)) & (scored[score_col] >= float(min_entry_score))

    cols = [
        "rank",
        "selected",
        "timestamp",
        "option_symbol",
        "underlying_symbol",
        "strike",
        "days_to_expiry",
        "mid",
        "delta",
        "open_interest",
        "volume",
        "candidate_score",
        "proba_raw",
        score_col,
    ]
    export = scored[cols].copy()
    export = export.rename(columns={score_col: "selection_score"})

    report = {
        "history_path": history_path,
        "latest_path": latest_path,
        "snapshot_timestamp": str(scored["timestamp"].max()),
        "model_type": diagnostics["model_type"],
        "selection_orientation": diagnostics["selection_orientation"],
        "selection_score_col": "selection_score",
        "min_entry_score": float(min_entry_score),
        "top_k": int(top_k),
        "oos_auc": diagnostics["oos_auc"],
        "oos_auc_inverted": diagnostics["oos_auc_inverted"],
        "fold_count": diagnostics["fold_count"],
        "rows_model": diagnostics["rows_model"],
        "snapshot_count": diagnostics["snapshot_count"],
        "candidate_count": int(len(export)),
        "selected_count": int(export["selected"].sum()),
        "selected_candidates": _records_for_json(export.loc[export["selected"]]),
        "top_candidates": _records_for_json(export.head(max(int(top_k), 5))),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    Path(csv_output_path).parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(csv_output_path, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the latest SPY options snapshot and write current candidate artifacts.")
    parser.add_argument("--history-path", default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--latest-path", default=DEFAULT_LATEST_PATH)
    parser.add_argument("--model-type", choices=["logistic", "random_forest", "hist_gbm"], default="logistic")
    parser.add_argument("--train-size", type=int, default=48)
    parser.add_argument("--test-size", type=int, default=12)
    parser.add_argument("--step-size", type=int, default=12)
    parser.add_argument("--horizon-bars", type=int, default=8)
    parser.add_argument("--target-return-pct", type=float, default=0.25)
    parser.add_argument("--max-adverse-return-pct", type=float, default=-0.20)
    parser.add_argument("--min-entry-score", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--selection-orientation", choices=["auto", "raw", "inverted"], default="auto")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv-output-path", default=DEFAULT_CSV_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_current_candidate_report(
        history_path=args.history_path,
        latest_path=args.latest_path,
        model_type=args.model_type,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        horizon_bars=args.horizon_bars,
        target_return_pct=args.target_return_pct,
        max_adverse_return_pct=args.max_adverse_return_pct,
        min_entry_score=args.min_entry_score,
        top_k=args.top_k,
        selection_orientation=args.selection_orientation,
        output_path=args.output_path,
        csv_output_path=args.csv_output_path,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
