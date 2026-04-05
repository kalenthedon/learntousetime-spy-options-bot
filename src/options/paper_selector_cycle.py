from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

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
    _latest_snapshot_from_history,
    _load_snapshot_csv,
    build_current_candidate_report,
)
from src.options.market_hours_collect import run_market_hours_collect


DEFAULT_SELECTOR_LATEST_PATH = "experiments/options_paper_selector_latest.json"
DEFAULT_SELECTOR_JOURNAL_PATH = "experiments/options_paper_selector_journal.csv"
DEFAULT_PORTFOLIO_STATE_PATH = "experiments/options_paper_portfolio_state.json"
DEFAULT_PORTFOLIO_JOURNAL_PATH = "experiments/options_paper_portfolio_journal.csv"
DEFAULT_POSITION_SIZE = 1
PORTFOLIO_JOURNAL_FIELDS = [
    "event_timestamp",
    "snapshot_timestamp",
    "event_type",
    "option_symbol",
    "qty",
    "entry_mid",
    "exit_mid",
    "current_mid",
    "score",
    "bars_held",
    "exit_reason",
    "realized_pnl",
    "realized_return_pct",
    "paper_trade_ready",
    "decision",
    "decision_reason",
]


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


def _load_portfolio_state(path: str) -> dict:
    state_path = Path(path)
    if not state_path.exists() or state_path.stat().st_size == 0:
        return {
            "open_position": None,
            "closed_positions": 0,
            "realized_pnl": 0.0,
            "realized_return_pct": 0.0,
            "last_cycle_timestamp": None,
        }
    return json.loads(state_path.read_text(encoding="utf-8"))


def _save_portfolio_state(state: dict, path: str) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _append_portfolio_journal(row: dict, path: str) -> None:
    journal_path = Path(path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    if journal_path.exists() and journal_path.stat().st_size > 0:
        with journal_path.open("r", encoding="utf-8", newline="") as f:
            existing_rows = list(csv.DictReader(f))
    normalized = {name: row.get(name, "") for name in PORTFOLIO_JOURNAL_FIELDS}
    existing_rows.append(normalized)
    with journal_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PORTFOLIO_JOURNAL_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)


def _ensure_portfolio_journal(path: str) -> None:
    journal_path = Path(path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    if journal_path.exists():
        return
    with journal_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PORTFOLIO_JOURNAL_FIELDS)
        writer.writeheader()


def _load_latest_snapshot(latest_path: str, history_path: str) -> pd.DataFrame:
    latest = _load_snapshot_csv(latest_path)
    if latest.empty:
        latest = _latest_snapshot_from_history(history_path)
    return latest


def _resolve_snapshot_timestamp(latest: pd.DataFrame) -> str | None:
    if latest.empty:
        return None
    if "collected_at" in latest.columns and latest["collected_at"].notna().any():
        return str(pd.to_datetime(latest["collected_at"], utc=True, errors="coerce").max())
    return str(pd.to_datetime(latest["timestamp"], utc=True, errors="coerce").max())


def _mark_open_position_to_market(
    state: dict,
    latest: pd.DataFrame,
    snapshot_timestamp: str | None,
    horizon_bars: int,
    cycle_timestamp: str,
    journal_path: str,
    report: dict,
) -> tuple[dict, list[dict]]:
    events: list[dict] = []
    open_position = state.get("open_position")
    if not open_position:
        return state, events

    symbol = str(open_position["option_symbol"])
    match = latest.loc[latest["option_symbol"].astype(str) == symbol].copy() if not latest.empty else pd.DataFrame()
    if match.empty:
        open_position["status"] = "stale_quote"
        state["open_position"] = open_position
        return state, events

    row = match.sort_values("timestamp").iloc[-1]
    current_mid = float(row["mid"])
    current_delta = float(row["delta"])
    open_position["last_snapshot_timestamp"] = snapshot_timestamp
    open_position["current_mid"] = current_mid
    open_position["current_delta"] = current_delta
    open_position["bars_held"] = int(open_position.get("bars_held", 0)) + 1
    open_position["unrealized_pnl"] = float((current_mid - float(open_position["entry_mid"])) * int(open_position["qty"]) * 100.0)
    open_position["unrealized_return_pct"] = float(current_mid / float(open_position["entry_mid"]) - 1.0)

    exit_reason = None
    if current_mid >= float(open_position["target_mid"]):
        exit_reason = "target"
    elif current_mid <= float(open_position["stop_mid"]):
        exit_reason = "stop"
    elif int(open_position["bars_held"]) >= int(horizon_bars):
        exit_reason = "timeout"

    if exit_reason:
        realized_pnl = float((current_mid - float(open_position["entry_mid"])) * int(open_position["qty"]) * 100.0)
        realized_return_pct = float(current_mid / float(open_position["entry_mid"]) - 1.0)
        state["closed_positions"] = int(state.get("closed_positions", 0)) + 1
        state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + realized_pnl
        state["realized_return_pct"] = float(state.get("realized_return_pct", 0.0)) + realized_return_pct
        event = {
            "event_timestamp": cycle_timestamp,
            "snapshot_timestamp": snapshot_timestamp,
            "event_type": "exit",
            "option_symbol": symbol,
            "qty": int(open_position["qty"]),
            "entry_mid": float(open_position["entry_mid"]),
            "exit_mid": current_mid,
            "current_mid": current_mid,
            "score": float(open_position["entry_score"]),
            "bars_held": int(open_position["bars_held"]),
            "exit_reason": exit_reason,
            "realized_pnl": realized_pnl,
            "realized_return_pct": realized_return_pct,
            "paper_trade_ready": report["paper_trade_ready"],
            "decision": report["decision"],
            "decision_reason": report["decision_reason"],
        }
        _append_portfolio_journal(event, journal_path)
        events.append(event)
        state["last_closed_position"] = {
            **open_position,
            "exit_mid": current_mid,
            "exit_reason": exit_reason,
            "realized_pnl": realized_pnl,
            "realized_return_pct": realized_return_pct,
            "exit_snapshot_timestamp": snapshot_timestamp,
            "exit_cycle_timestamp": cycle_timestamp,
        }
        state["open_position"] = None
    else:
        state["open_position"] = open_position
    return state, events


def _maybe_open_position(
    state: dict,
    latest: pd.DataFrame,
    report: dict,
    snapshot_timestamp: str | None,
    cycle_timestamp: str,
    qty: int,
    target_return_pct: float,
    max_adverse_return_pct: float,
    journal_path: str,
) -> tuple[dict, list[dict]]:
    events: list[dict] = []
    if state.get("open_position") is not None:
        return state, events
    if not report["paper_trade_ready"] or report["decision"] != "select_contract" or not report.get("selected_option_symbol"):
        return state, events

    symbol = str(report["selected_option_symbol"])
    match = latest.loc[latest["option_symbol"].astype(str) == symbol].copy() if not latest.empty else pd.DataFrame()
    if match.empty:
        return state, events

    row = match.sort_values("timestamp").iloc[-1]
    entry_mid = float(row["mid"])
    if entry_mid <= 0:
        return state, events
    open_position = {
        "option_symbol": symbol,
        "qty": int(qty),
        "entry_snapshot_timestamp": snapshot_timestamp,
        "entry_cycle_timestamp": cycle_timestamp,
        "entry_mid": entry_mid,
        "entry_score": float(report["selected_selection_score"]),
        "entry_delta": float(row["delta"]),
        "target_mid": float(entry_mid * (1.0 + float(target_return_pct))),
        "stop_mid": float(entry_mid * (1.0 + float(max_adverse_return_pct))),
        "bars_held": 0,
        "current_mid": entry_mid,
        "current_delta": float(row["delta"]),
        "unrealized_pnl": 0.0,
        "unrealized_return_pct": 0.0,
        "status": "open",
    }
    state["open_position"] = open_position
    event = {
        "event_timestamp": cycle_timestamp,
        "snapshot_timestamp": snapshot_timestamp,
        "event_type": "entry",
        "option_symbol": symbol,
        "qty": int(qty),
        "entry_mid": entry_mid,
        "exit_mid": "",
        "current_mid": entry_mid,
        "score": float(report["selected_selection_score"]),
        "bars_held": 0,
        "exit_reason": "",
        "realized_pnl": "",
        "realized_return_pct": "",
        "paper_trade_ready": report["paper_trade_ready"],
        "decision": report["decision"],
        "decision_reason": report["decision_reason"],
    }
    _append_portfolio_journal(event, journal_path)
    events.append(event)
    return state, events


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
    portfolio_state_path: str = DEFAULT_PORTFOLIO_STATE_PATH,
    portfolio_journal_path: str = DEFAULT_PORTFOLIO_JOURNAL_PATH,
    position_size: int = DEFAULT_POSITION_SIZE,
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
    latest_snapshot = _load_latest_snapshot(latest_path=latest_path, history_path=history_path)
    snapshot_timestamp = _resolve_snapshot_timestamp(latest_snapshot)
    portfolio_state = _load_portfolio_state(portfolio_state_path)
    _ensure_portfolio_journal(portfolio_journal_path)
    portfolio_state, exit_events = _mark_open_position_to_market(
        state=portfolio_state,
        latest=latest_snapshot,
        snapshot_timestamp=snapshot_timestamp,
        horizon_bars=horizon_bars,
        cycle_timestamp=cycle_timestamp,
        journal_path=portfolio_journal_path,
        report=report,
    )
    portfolio_state, entry_events = _maybe_open_position(
        state=portfolio_state,
        latest=latest_snapshot,
        report=report,
        snapshot_timestamp=snapshot_timestamp,
        cycle_timestamp=cycle_timestamp,
        qty=position_size,
        target_return_pct=target_return_pct,
        max_adverse_return_pct=max_adverse_return_pct,
        journal_path=portfolio_journal_path,
    )
    portfolio_state["last_cycle_timestamp"] = cycle_timestamp
    _save_portfolio_state(portfolio_state, portfolio_state_path)

    selector_state = {
        "cycle_timestamp": cycle_timestamp,
        "collect_result": collect_result,
        "report": report,
        "portfolio": portfolio_state,
        "portfolio_events": exit_events + entry_events,
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
    parser.add_argument("--portfolio-state-path", default=DEFAULT_PORTFOLIO_STATE_PATH)
    parser.add_argument("--portfolio-journal-path", default=DEFAULT_PORTFOLIO_JOURNAL_PATH)
    parser.add_argument("--position-size", type=int, default=DEFAULT_POSITION_SIZE)
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
        portfolio_state_path=args.portfolio_state_path,
        portfolio_journal_path=args.portfolio_journal_path,
        position_size=args.position_size,
        collect_snapshot=not args.no_collect,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
