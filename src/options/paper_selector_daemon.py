from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import time
from zoneinfo import ZoneInfo
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.options.paper_selector_cycle import (
    DEFAULT_PORTFOLIO_JOURNAL_PATH,
    DEFAULT_PORTFOLIO_STATE_PATH,
    DEFAULT_SELECTOR_JOURNAL_PATH,
    DEFAULT_SELECTOR_LATEST_PATH,
    run_paper_selector_cycle,
)
from src.options.current_candidate_report import (
    DEFAULT_BACKFILL_SUMMARY_PATH,
    DEFAULT_CSV_OUTPUT_PATH,
    DEFAULT_HISTORY_PATH,
    DEFAULT_JOURNAL_PATH as DEFAULT_CANDIDATE_JOURNAL_PATH,
    DEFAULT_LATEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_CURRENT_REPORT_PATH,
    DEFAULT_THRESHOLD_SWEEP,
)


US_EASTERN = ZoneInfo("America/New_York")


def seconds_until_next_interval(interval_minutes: int, delay_seconds: int) -> float:
    now = datetime.now(US_EASTERN)
    minute_bucket = (now.minute // interval_minutes) * interval_minutes
    current_bucket = now.replace(minute=minute_bucket, second=0, microsecond=0)
    next_bucket = current_bucket + timedelta(minutes=interval_minutes)
    target = next_bucket + timedelta(seconds=delay_seconds)
    return max(0.0, (target - now).total_seconds())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SPY options paper selector continuously on a fixed interval.")
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
    parser.add_argument("--portfolio-state-path", default=DEFAULT_PORTFOLIO_STATE_PATH)
    parser.add_argument("--portfolio-journal-path", default=DEFAULT_PORTFOLIO_JOURNAL_PATH)
    parser.add_argument("--position-size", type=int, default=1)
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--delay-seconds", type=int, default=30)
    parser.add_argument("--run-immediately", action="store_true")
    parser.add_argument("--no-collect", action="store_true")
    args = parser.parse_args()

    def _run_once() -> dict:
        return run_paper_selector_cycle(
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
            portfolio_state_path=args.portfolio_state_path,
            portfolio_journal_path=args.portfolio_journal_path,
            position_size=args.position_size,
            collect_snapshot=not args.no_collect,
        )

    if args.run_immediately:
        print(json.dumps(_run_once(), indent=2, default=str))

    while True:
        sleep_seconds = seconds_until_next_interval(args.interval_minutes, args.delay_seconds)
        print(f"[options-paper-daemon] sleeping {sleep_seconds:.0f}s until next cycle")
        time.sleep(sleep_seconds)
        try:
            print(json.dumps(_run_once(), indent=2, default=str))
        except Exception as exc:
            print(f"[options-paper-daemon] cycle failed: {exc}")
            time.sleep(60)


if __name__ == "__main__":
    main()
