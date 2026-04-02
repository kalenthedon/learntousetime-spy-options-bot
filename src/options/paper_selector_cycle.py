from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.options.current_candidate_report import (
    DEFAULT_BACKFILL_SUMMARY_PATH,
    DEFAULT_CSV_OUTPUT_PATH,
    DEFAULT_HISTORY_PATH,
    DEFAULT_JOURNAL_PATH as DEFAULT_CANDIDATE_JOURNAL_PATH,
    DEFAULT_LATEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_CURRENT_REPORT_PATH,
    DEFAULT_THRESHOLD_SWEEP,
    build_current_candidate_report,
)
from src.options.market_hours_collect import run_market_hours_collect


DEFAULT_SELECTOR_LATEST_PATH = "experiments/options_paper_selector_latest.json"
DEFAULT_SELECTOR_JOURNAL_PATH = "experiments/options_paper_selector_journal.csv"


def _upsert_selector_journal(row: dict, path: str) -> None:
    journal_path = Path(path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cycle_timestamp",
        "snapshot_timestamp",
        "collect_status",
        "collect_reason",
        "captured_rows",
        "decision",
        "decision_reason",
        "paper_trade_ready",
        "paper_trade_readiness_reason",
        "selected_option_symbol",
        "selected_selection_score",
        "selected_mid",
        "selected_delta",
        "candidate_count",
        "selected_count",
        "model_type",
        "selection_orientation",
        "min_entry_score",
        "top_k",
        "min_open_interest",
        "max_spread_pct",
        "min_abs_delta",
        "max_days_to_expiry",
    ]
    normalized_row = {name: row.get(name, "") for name in fieldnames}
    existing_rows: list[dict] = []
    if journal_path.exists() and journal_path.stat().st_size > 0:
        with journal_path.open("r", encoding="utf-8", newline="") as f:
            existing_rows = list(csv.DictReader(f))

    dedupe_key = (
        str(normalized_row.get("snapshot_timestamp", "")),
        str(normalized_row.get("model_type", "")),
        str(normalized_row.get("selection_orientation", "")),
        str(normalized_row.get("min_entry_score", "")),
        str(normalized_row.get("top_k", "")),
        str(normalized_row.get("min_open_interest", "")),
        str(normalized_row.get("max_spread_pct", "")),
        str(normalized_row.get("min_abs_delta", "")),
        str(normalized_row.get("max_days_to_expiry", "")),
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
            str(existing.get("min_open_interest", "")),
            str(existing.get("max_spread_pct", "")),
            str(existing.get("min_abs_delta", "")),
            str(existing.get("max_days_to_expiry", "")),
        )
        != dedupe_key
    ]
    filtered_rows.append(normalized_row)
    with journal_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)


def run_paper_selector_cycle(
    underlying_symbol: str = "SPY",
    min_dte: int = 7,
    max_dte: int = 21,
    history_path: str = DEFAULT_HISTORY_PATH,
    latest_path: str = DEFAULT_LATEST_PATH,
    model_type: str = "logistic",
    train_size: int = 48,
    test_size: int = 12,
    step_size: int = 12,
    horizon_bars: int = 8,
    target_return_pct: float = 0.25,
    max_adverse_return_pct: float = -0.20,
    min_entry_score: float = 0.10,
    top_k: int = 3,
    min_open_interest: float = 100.0,
    max_spread_pct: float = 0.12,
    min_abs_delta: float = 0.25,
    max_days_to_expiry: float = 10.0,
    selection_orientation: str = "auto",
    threshold_sweep: str = DEFAULT_THRESHOLD_SWEEP,
    backfill_summary_path: str = DEFAULT_BACKFILL_SUMMARY_PATH,
    current_report_path: str = DEFAULT_CURRENT_REPORT_PATH,
    current_csv_output_path: str = DEFAULT_CSV_OUTPUT_PATH,
    current_journal_path: str = DEFAULT_CANDIDATE_JOURNAL_PATH,
    selector_latest_path: str = DEFAULT_SELECTOR_LATEST_PATH,
    selector_journal_path: str = DEFAULT_SELECTOR_JOURNAL_PATH,
    collect_snapshot: bool = True,
) -> dict:
    cycle_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    collect_result = {
        "status": "skipped",
        "reason": "collect_disabled",
        "underlying_symbol": underlying_symbol,
        "captured_rows": 0,
        "history_path": history_path,
    }
    if collect_snapshot:
        collect_result = run_market_hours_collect(
            underlying_symbol=underlying_symbol,
            min_dte=min_dte,
            max_dte=max_dte,
            history_path=history_path,
        )

    report = build_current_candidate_report(
        history_path=history_path,
        latest_path=latest_path,
        model_type=model_type,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        horizon_bars=horizon_bars,
        target_return_pct=target_return_pct,
        max_adverse_return_pct=max_adverse_return_pct,
        min_entry_score=min_entry_score,
        top_k=top_k,
        min_open_interest=min_open_interest,
        max_spread_pct=max_spread_pct,
        min_abs_delta=min_abs_delta,
        max_days_to_expiry=max_days_to_expiry,
        selection_orientation=selection_orientation,
        threshold_sweep=threshold_sweep,
        backfill_summary_path=backfill_summary_path,
        output_path=current_report_path,
        csv_output_path=current_csv_output_path,
        journal_path=current_journal_path,
    )

    selector_state = {
        "cycle_timestamp": cycle_timestamp,
        "collect_result": collect_result,
        "report": report,
    }
    Path(selector_latest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(selector_latest_path).write_text(json.dumps(selector_state, indent=2, default=str), encoding="utf-8")
    _upsert_selector_journal(
        {
            "cycle_timestamp": cycle_timestamp,
            "snapshot_timestamp": report["snapshot_timestamp"],
            "collect_status": collect_result.get("status"),
            "collect_reason": collect_result.get("reason"),
            "captured_rows": collect_result.get("captured_rows", 0),
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
            "paper_trade_ready": report["paper_trade_ready"],
            "paper_trade_readiness_reason": report["paper_trade_readiness_reason"],
            "selected_option_symbol": report["selected_option_symbol"],
            "selected_selection_score": report["selected_selection_score"],
            "selected_mid": report["selected_mid"],
            "selected_delta": report["selected_delta"],
            "candidate_count": report["candidate_count"],
            "selected_count": report["selected_count"],
            "model_type": report["model_type"],
            "selection_orientation": report["selection_orientation"],
            "min_entry_score": report["min_entry_score"],
            "top_k": report["top_k"],
            "min_open_interest": report["min_open_interest"],
            "max_spread_pct": report["max_spread_pct"],
            "min_abs_delta": report["min_abs_delta"],
            "max_days_to_expiry": report["max_days_to_expiry"],
        },
        path=selector_journal_path,
    )
    return selector_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one lightweight SPY options paper-selector cycle.")
    parser.add_argument("--underlying-symbol", default="SPY")
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=21)
    parser.add_argument("--history-path", default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--latest-path", default=DEFAULT_LATEST_PATH)
    parser.add_argument("--model-type", choices=["logistic", "random_forest", "hist_gbm"], default="logistic")
    parser.add_argument("--train-size", type=int, default=48)
    parser.add_argument("--test-size", type=int, default=12)
    parser.add_argument("--step-size", type=int, default=12)
    parser.add_argument("--horizon-bars", type=int, default=8)
    parser.add_argument("--target-return-pct", type=float, default=0.25)
    parser.add_argument("--max-adverse-return-pct", type=float, default=-0.20)
    parser.add_argument("--min-entry-score", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-open-interest", type=float, default=100.0)
    parser.add_argument("--max-spread-pct", type=float, default=0.12)
    parser.add_argument("--min-abs-delta", type=float, default=0.25)
    parser.add_argument("--max-days-to-expiry", type=float, default=10.0)
    parser.add_argument("--selection-orientation", choices=["auto", "raw", "inverted"], default="auto")
    parser.add_argument("--threshold-sweep", default=DEFAULT_THRESHOLD_SWEEP)
    parser.add_argument("--backfill-summary-path", default=DEFAULT_BACKFILL_SUMMARY_PATH)
    parser.add_argument("--current-report-path", default=DEFAULT_CURRENT_REPORT_PATH)
    parser.add_argument("--current-csv-output-path", default=DEFAULT_CSV_OUTPUT_PATH)
    parser.add_argument("--current-journal-path", default=DEFAULT_CANDIDATE_JOURNAL_PATH)
    parser.add_argument("--selector-latest-path", default=DEFAULT_SELECTOR_LATEST_PATH)
    parser.add_argument("--selector-journal-path", default=DEFAULT_SELECTOR_JOURNAL_PATH)
    parser.add_argument("--no-collect", action="store_true")
    args = parser.parse_args()

    result = run_paper_selector_cycle(
        underlying_symbol=args.underlying_symbol,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
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
        current_report_path=args.current_report_path,
        current_csv_output_path=args.current_csv_output_path,
        current_journal_path=args.current_journal_path,
        selector_latest_path=args.selector_latest_path,
        selector_journal_path=args.selector_journal_path,
        collect_snapshot=not args.no_collect,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
