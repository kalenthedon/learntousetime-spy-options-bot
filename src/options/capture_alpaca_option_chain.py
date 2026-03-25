from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.alpaca_options import (
    fetch_option_contracts,
    fetch_option_snapshots,
    fetch_stock_bars,
    snapshots_to_frame,
)
from src.options.universe import filter_long_put_candidates, score_long_put_candidates


DEFAULT_OUTPUT_PATH = "centralized_data/options/SPY_put_chain_latest.csv"


def capture_spy_long_put_chain(
    underlying_symbol: str = "SPY",
    expiration_window_days: tuple[int, int] = (7, 21),
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    today = date.today()
    exp_gte = today + timedelta(days=expiration_window_days[0])
    exp_lte = today + timedelta(days=expiration_window_days[1])

    contracts = fetch_option_contracts(
        underlying_symbol=underlying_symbol,
        expiration_gte=exp_gte.isoformat(),
        expiration_lte=exp_lte.isoformat(),
        option_type="put",
    )
    if not contracts:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out_path, index=False)
        return pd.DataFrame()

    spot_bars = fetch_stock_bars(underlying_symbol, timeframe="1Hour", limit=1)
    if spot_bars.empty:
        raise RuntimeError(f"Could not fetch latest stock bar for {underlying_symbol}")
    spot_price = float(spot_bars["close"].iloc[-1])

    contracts_df = pd.DataFrame(contracts).copy()
    contracts_df["strike_price"] = pd.to_numeric(contracts_df["strike_price"], errors="coerce")
    contracts_df = contracts_df.loc[
        contracts_df["strike_price"].between(spot_price * 0.90, spot_price * 1.02)
    ].copy()
    contracts_df["strike_distance"] = (contracts_df["strike_price"] - spot_price).abs()
    contracts_df = contracts_df.sort_values(["expiration_date", "strike_distance"]).head(120)

    selected_contracts = contracts_df.to_dict(orient="records")
    snapshots = fetch_option_snapshots(contract.get("symbol") for contract in selected_contracts)
    frame = snapshots_to_frame(selected_contracts, snapshots)
    frame = filter_long_put_candidates(frame)
    frame = score_long_put_candidates(frame)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a filtered Alpaca option chain for SPY-style long put research.")
    parser.add_argument("--underlying-symbol", default="SPY")
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=21)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    frame = capture_spy_long_put_chain(
        underlying_symbol=args.underlying_symbol,
        expiration_window_days=(args.min_dte, args.max_dte),
        output_path=args.output_path,
    )
    if frame.empty:
        print("No filtered contracts matched current selection rules.")
        return
    cols = [
        "timestamp",
        "underlying_symbol",
        "option_symbol",
        "days_to_expiry",
        "strike",
        "bid",
        "ask",
        "mid",
        "delta",
        "open_interest",
        "volume",
        "candidate_score",
    ]
    print(frame[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
