from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.backtests.run_options_research import (
    apply_selection_orientation,
    apply_selection_contract_filters,
    build_model_factory,
    build_option_feature_frame,
    DEFAULT_SELECTOR_MAX_DTE,
    DEFAULT_SELECTOR_MAX_SPREAD_PCT,
    DEFAULT_SELECTOR_MIN_ABS_DELTA,
    DEFAULT_SELECTOR_MIN_ENTRY_SCORE,
    DEFAULT_SELECTOR_MIN_OPEN_INTEREST,
    DEFAULT_SELECTOR_TOP_K,
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
DEFAULT_JOURNAL_PATH = "experiments/options_candidate_journal.csv"
DEFAULT_THRESHOLD_SWEEP = "0.01,0.02,0.03,0.05,0.10,0.15,0.20,0.30"
DEFAULT_BACKFILL_SUMMARY_PATH = "experiments/options_backfill_candidate_summary.json"


def _load_snapshot_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    return normalize_option_chain_frame(pd.read_csv(csv_path))


def _latest_snapshot_from_history(history_path: str) -> pd.DataFrame:
    history = load_option_chain_csv(history_path)
    if history.empty:
        return pd.DataFrame()
    if "collected_at" in history.columns:
        snapshot_time = pd.to_datetime(history["collected_at"], utc=True, errors="coerce")
    else:
        snapshot_time = pd.to_datetime(history["timestamp"], utc=True, errors="coerce").dt.floor("15min")
    history = history.assign(_snapshot_time=snapshot_time).dropna(subset=["_snapshot_time"]).copy()
    if history.empty:
        return pd.DataFrame()
    latest_snapshot_time = history["_snapshot_time"].max()
    latest = history.loc[history["_snapshot_time"] == latest_snapshot_time].drop(columns=["_snapshot_time"]).copy()
    return latest.reset_index(drop=True)


def _records_for_json(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].astype(str)
    return json.loads(safe.to_json(orient="records"))


def _parse_threshold_sweep(value: str) -> list[float]:
    thresholds = []
    for raw in str(value).split(","):
        cleaned = raw.strip()
        if not cleaned:
            continue
        thresholds.append(float(cleaned))
    if not thresholds:
        raise ValueError("threshold_sweep must contain at least one numeric threshold")
    return sorted(set(thresholds))


def _load_backfill_summary(path: str) -> dict | None:
    summary_path = Path(path)
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_historical_threshold_context(summary: dict | None, threshold: float) -> dict | None:
    if not summary:
        return None
    threshold_rows = summary.get("threshold_trade_summary") or []
    if not threshold_rows:
        return None
    chosen = min(threshold_rows, key=lambda row: abs(float(row.get("threshold", 0.0)) - float(threshold)))
    return {
        "threshold": float(chosen.get("threshold", 0.0)),
        "decision_rate": chosen.get("decision_rate"),
        "trade_count": chosen.get("trade_count"),
        "win_rate": chosen.get("win_rate"),
        "avg_return_pct": chosen.get("avg_return_pct"),
        "median_return_pct": chosen.get("median_return_pct"),
        "total_return_pct": chosen.get("total_return_pct"),
        "target_hit_rate": chosen.get("target_hit_rate"),
        "stop_hit_rate": chosen.get("stop_hit_rate"),
    }


def _paper_trade_readiness(
    diagnostics: dict,
    historical_threshold_context: dict | None,
    min_historical_trades: int,
) -> tuple[bool, str]:
    if diagnostics.get("oos_auc") is None:
        return False, "missing_oos_auc"
    if float(diagnostics["oos_auc"]) < 0.55:
        return False, "oos_auc_below_threshold"
    if not historical_threshold_context:
        return False, "missing_backfill_threshold_context"
    if int(historical_threshold_context.get("trade_count") or 0) < int(min_historical_trades):
        return False, "insufficient_historical_trades"
    if historical_threshold_context.get("total_return_pct") is None:
        return False, "missing_historical_return_data"
    if float(historical_threshold_context["total_return_pct"]) <= 0.0:
        return False, "historical_threshold_not_profitable"
    return True, "historical_threshold_profitable"


def _append_decision_journal(row: dict, path: str) -> None:
    journal_path = Path(path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "snapshot_timestamp",
        "decision",
        "selected_count",
        "selected_option_symbol",
        "selected_selection_score",
        "selected_mid",
        "selected_delta",
        "model_type",
        "selection_orientation",
        "min_entry_score",
        "top_k",
        "oos_auc",
        "oos_auc_inverted",
    ]
    normalized_row = {k: row.get(k, "") for k in fieldnames}
    existing_rows: list[dict] = []
    if journal_path.exists() and journal_path.stat().st_size > 0:
        with journal_path.open("r", encoding="utf-8", newline="") as f:
            existing_rows = list(csv.DictReader(f))

    dedupe_keys = (
        str(normalized_row.get("snapshot_timestamp", "")),
        str(normalized_row.get("model_type", "")),
        str(normalized_row.get("selection_orientation", "")),
        str(normalized_row.get("min_entry_score", "")),
        str(normalized_row.get("top_k", "")),
    )
    filtered_rows = [
        existing
        for existing in existing_rows
        if (
            str(existing.get("snapshot_timestamp", "")),
            str(existing.get("model_type", "")),
            str(existing.get("selection_orientation", "")),
            str(existing.get("min_entry_score", "")),
            str(existing.get("top_k", "")),
        ) != dedupe_keys
    ]
    filtered_rows.append(normalized_row)

    with journal_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)


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
    min_entry_score: float = DEFAULT_SELECTOR_MIN_ENTRY_SCORE,
    top_k: int = DEFAULT_SELECTOR_TOP_K,
    min_open_interest: float = DEFAULT_SELECTOR_MIN_OPEN_INTEREST,
    max_spread_pct: float = DEFAULT_SELECTOR_MAX_SPREAD_PCT,
    min_abs_delta: float = DEFAULT_SELECTOR_MIN_ABS_DELTA,
    max_days_to_expiry: float = DEFAULT_SELECTOR_MAX_DTE,
    selection_orientation: str = "auto",
    threshold_sweep: str = DEFAULT_THRESHOLD_SWEEP,
    backfill_summary_path: str = DEFAULT_BACKFILL_SUMMARY_PATH,
    min_historical_trades_for_ready: int = 5,
    output_path: str = DEFAULT_OUTPUT_PATH,
    csv_output_path: str = DEFAULT_CSV_OUTPUT_PATH,
    journal_path: str = DEFAULT_JOURNAL_PATH,
) -> dict:
    latest = _load_snapshot_csv(latest_path)
    if latest.empty:
        latest = _latest_snapshot_from_history(history_path)
    if latest.empty:
        raise RuntimeError(
            f"No latest option snapshot rows found at {latest_path}, and no usable latest snapshot could be derived from {history_path}"
        )

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
    scored = apply_selection_contract_filters(
        scored,
        min_open_interest=min_open_interest,
        max_spread_pct=max_spread_pct,
        min_abs_delta=min_abs_delta,
        max_days_to_expiry=max_days_to_expiry,
    )
    scored = scored.sort_values([score_col, "candidate_score", "open_interest", "volume"], ascending=[False, False, False, False]).reset_index(drop=True)
    scored["rank"] = range(1, len(scored) + 1)
    scored["selected"] = (scored["rank"] <= int(top_k)) & (scored[score_col] >= float(min_entry_score))
    threshold_values = _parse_threshold_sweep(threshold_sweep)

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
    selected_export = export.loc[export["selected"]].copy()
    decision = "no_trade"
    selected_option_symbol = None
    selected_selection_score = None
    selected_mid = None
    selected_delta = None
    if not selected_export.empty:
        decision = "select_contract"
        top_selected = selected_export.iloc[0]
        selected_option_symbol = str(top_selected["option_symbol"])
        selected_selection_score = float(top_selected["selection_score"])
        selected_mid = float(top_selected["mid"])
        selected_delta = float(top_selected["delta"])

    threshold_summary = []
    for threshold in threshold_values:
        eligible = export.loc[export["selection_score"] >= float(threshold)].copy()
        threshold_summary.append(
            {
                "threshold": float(threshold),
                "eligible_count": int(len(eligible)),
                "top_option_symbol": str(eligible.iloc[0]["option_symbol"]) if not eligible.empty else None,
                "top_selection_score": float(eligible.iloc[0]["selection_score"]) if not eligible.empty else None,
            }
        )
    historical_threshold_context = _resolve_historical_threshold_context(
        _load_backfill_summary(backfill_summary_path),
        min_entry_score,
    )
    paper_trade_ready, paper_trade_readiness_reason = _paper_trade_readiness(
        diagnostics=diagnostics,
        historical_threshold_context=historical_threshold_context,
        min_historical_trades=min_historical_trades_for_ready,
    )

    report = {
        "history_path": history_path,
        "latest_path": latest_path,
        "snapshot_timestamp": str(scored["timestamp"].max()),
        "model_type": diagnostics["model_type"],
        "selection_orientation": diagnostics["selection_orientation"],
        "selection_score_col": "selection_score",
        "min_entry_score": float(min_entry_score),
        "top_k": int(top_k),
        "min_open_interest": float(min_open_interest),
        "max_spread_pct": float(max_spread_pct),
        "min_abs_delta": float(min_abs_delta),
        "max_days_to_expiry": float(max_days_to_expiry),
        "decision": decision,
        "selected_option_symbol": selected_option_symbol,
        "selected_selection_score": selected_selection_score,
        "selected_mid": selected_mid,
        "selected_delta": selected_delta,
        "paper_trade_ready": paper_trade_ready,
        "paper_trade_readiness_reason": paper_trade_readiness_reason,
        "historical_threshold_context": historical_threshold_context,
        "oos_auc": diagnostics["oos_auc"],
        "oos_auc_inverted": diagnostics["oos_auc_inverted"],
        "fold_count": diagnostics["fold_count"],
        "rows_model": diagnostics["rows_model"],
        "snapshot_count": diagnostics["snapshot_count"],
        "candidate_count": int(len(export)),
        "selected_count": int(selected_export["selected"].sum()) if not selected_export.empty else 0,
        "selected_candidates": _records_for_json(selected_export),
        "threshold_summary": threshold_summary,
        "top_candidates": _records_for_json(export.head(max(int(top_k), 5))),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    Path(csv_output_path).parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(csv_output_path, index=False)
    _append_decision_journal(
        {
            "snapshot_timestamp": report["snapshot_timestamp"],
            "decision": report["decision"],
            "selected_count": report["selected_count"],
            "selected_option_symbol": report["selected_option_symbol"],
            "selected_selection_score": report["selected_selection_score"],
            "selected_mid": report["selected_mid"],
            "selected_delta": report["selected_delta"],
            "model_type": report["model_type"],
            "selection_orientation": report["selection_orientation"],
            "min_entry_score": report["min_entry_score"],
            "top_k": report["top_k"],
            "oos_auc": report["oos_auc"],
            "oos_auc_inverted": report["oos_auc_inverted"],
        },
        path=journal_path,
    )
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
    parser.add_argument("--min-entry-score", type=float, default=DEFAULT_SELECTOR_MIN_ENTRY_SCORE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_SELECTOR_TOP_K)
    parser.add_argument("--min-open-interest", type=float, default=DEFAULT_SELECTOR_MIN_OPEN_INTEREST)
    parser.add_argument("--max-spread-pct", type=float, default=DEFAULT_SELECTOR_MAX_SPREAD_PCT)
    parser.add_argument("--min-abs-delta", type=float, default=DEFAULT_SELECTOR_MIN_ABS_DELTA)
    parser.add_argument("--max-days-to-expiry", type=float, default=DEFAULT_SELECTOR_MAX_DTE)
    parser.add_argument("--selection-orientation", choices=["auto", "raw", "inverted"], default="auto")
    parser.add_argument("--threshold-sweep", default=DEFAULT_THRESHOLD_SWEEP)
    parser.add_argument("--backfill-summary-path", default=DEFAULT_BACKFILL_SUMMARY_PATH)
    parser.add_argument("--min-historical-trades-for-ready", type=int, default=5)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--csv-output-path", default=DEFAULT_CSV_OUTPUT_PATH)
    parser.add_argument("--journal-path", default=DEFAULT_JOURNAL_PATH)
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
        min_open_interest=args.min_open_interest,
        max_spread_pct=args.max_spread_pct,
        min_abs_delta=args.min_abs_delta,
        max_days_to_expiry=args.max_days_to_expiry,
        selection_orientation=args.selection_orientation,
        threshold_sweep=args.threshold_sweep,
        backfill_summary_path=args.backfill_summary_path,
        min_historical_trades_for_ready=args.min_historical_trades_for_ready,
        output_path=args.output_path,
        csv_output_path=args.csv_output_path,
        journal_path=args.journal_path,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
