from __future__ import annotations

import argparse
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.options.collect_alpaca_option_history import append_option_history


US_EASTERN = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 35)
MARKET_CLOSE = time(16, 5)


def is_regular_market_window(now: datetime | None = None) -> tuple[bool, str]:
    now = now or datetime.now(US_EASTERN)
    if now.tzinfo is None:
        now = now.replace(tzinfo=US_EASTERN)
    else:
        now = now.astimezone(US_EASTERN)

    if now.weekday() >= 5:
        return False, "weekend"
    current = now.time()
    if current < MARKET_OPEN:
        return False, "before_market_window"
    if current > MARKET_CLOSE:
        return False, "after_market_window"
    return True, "collect"


def run_market_hours_collect(
    underlying_symbol: str = "SPY",
    min_dte: int = 7,
    max_dte: int = 21,
    history_path: str = "centralized_data/options/SPY_put_chain_history.csv",
) -> dict:
    should_collect, reason = is_regular_market_window()
    if not should_collect:
        return {
            "status": "skipped",
            "reason": reason,
            "underlying_symbol": underlying_symbol,
        }

    latest = append_option_history(
        underlying_symbol=underlying_symbol,
        min_dte=min_dte,
        max_dte=max_dte,
        history_path=history_path,
    )
    return {
        "status": "collected",
        "reason": reason,
        "underlying_symbol": underlying_symbol,
        "captured_rows": int(len(latest)),
        "history_path": history_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect an options snapshot only during US market hours.")
    parser.add_argument("--underlying-symbol", default="SPY")
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=21)
    parser.add_argument("--history-path", default="centralized_data/options/SPY_put_chain_history.csv")
    args = parser.parse_args()

    result = run_market_hours_collect(
        underlying_symbol=args.underlying_symbol,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        history_path=args.history_path,
    )
    print(result)


if __name__ == "__main__":
    main()
