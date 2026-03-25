from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.options.capture_alpaca_option_chain import capture_spy_long_put_chain


DEFAULT_HISTORY_PATH = "centralized_data/options/SPY_put_chain_history.csv"


def append_option_history(
    underlying_symbol: str = "SPY",
    min_dte: int = 7,
    max_dte: int = 21,
    history_path: str = DEFAULT_HISTORY_PATH,
) -> pd.DataFrame:
    latest = capture_spy_long_put_chain(
        underlying_symbol=underlying_symbol,
        expiration_window_days=(min_dte, max_dte),
        output_path="centralized_data/options/SPY_put_chain_latest.csv",
    )

    history_file = Path(history_path)
    history_file.parent.mkdir(parents=True, exist_ok=True)

    if history_file.exists() and history_file.stat().st_size > 0:
        history = pd.read_csv(history_file)
        combined = pd.concat([history, latest], ignore_index=True)
    else:
        combined = latest.copy()

    if not combined.empty:
        dedupe_cols = ["timestamp", "option_symbol"]
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
        combined = combined.sort_values(["timestamp", "option_symbol"]).reset_index(drop=True)

    combined.to_csv(history_file, index=False)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one filtered Alpaca SPY put-chain snapshot to the local history CSV.")
    parser.add_argument("--underlying-symbol", default="SPY")
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=21)
    parser.add_argument("--history-path", default=DEFAULT_HISTORY_PATH)
    args = parser.parse_args()

    latest = append_option_history(
        underlying_symbol=args.underlying_symbol,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        history_path=args.history_path,
    )
    print(f"captured_rows={len(latest)}")
    print(f"history_path={args.history_path}")


if __name__ == "__main__":
    main()
