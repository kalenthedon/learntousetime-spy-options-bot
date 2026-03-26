from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


LATEST_PATH = Path("centralized_data/options/SPY_put_chain_latest.csv")
HISTORY_PATH = Path("centralized_data/options/SPY_put_chain_history.csv")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def build_status() -> dict:
    latest = _load_csv(LATEST_PATH)
    history = _load_csv(HISTORY_PATH)

    latest_ts = None
    if not latest.empty and "timestamp" in latest.columns:
        latest_ts = str(latest["timestamp"].max())
    latest_collected_at = None
    if not latest.empty and "collected_at" in latest.columns:
        latest_collected_at = str(latest["collected_at"].max())
    elif not latest.empty and "timestamp" in latest.columns:
        latest_collected_at = str(pd.to_datetime(latest["timestamp"], utc=True, errors="coerce").dt.floor("15min").max())

    history_snapshot_count = 0
    if not history.empty:
        if "collected_at" in history.columns:
            history_snapshot_count = int(history["collected_at"].nunique())
        elif "timestamp" in history.columns:
            history_snapshot_count = int(pd.to_datetime(history["timestamp"], utc=True, errors="coerce").dt.floor("15min").nunique())
    history_contract_count = (
        int(history["option_symbol"].nunique()) if not history.empty and "option_symbol" in history.columns else 0
    )
    top_contracts = []
    if not latest.empty and "candidate_score" in latest.columns:
        cols = ["option_symbol", "strike", "days_to_expiry", "mid", "delta", "candidate_score"]
        view = latest[cols].sort_values("candidate_score", ascending=False).head(5)
        top_contracts = view.to_dict(orient="records")

    return {
        "latest_rows": int(len(latest)),
        "history_rows": int(len(history)),
        "latest_timestamp": latest_ts,
        "latest_collected_at": latest_collected_at,
        "history_snapshot_count": history_snapshot_count,
        "history_contract_count": history_contract_count,
        "ready_for_research": history_snapshot_count >= 50 and history_contract_count >= 20,
        "top_contracts": top_contracts,
    }


def render_report(status: dict) -> str:
    lines = [
        "SPY Options History Status",
        (
            f"Latest snapshot rows: {status['latest_rows']}"
            f" | history rows: {status['history_rows']}"
            f" | unique snapshots: {status['history_snapshot_count']}"
            f" | unique contracts: {status['history_contract_count']}"
        ),
        f"Latest timestamp: {status['latest_timestamp'] or 'n/a'}",
        f"Latest collected_at: {status['latest_collected_at'] or 'n/a'}",
        f"Ready for starter research: {status['ready_for_research']}",
    ]
    if status["top_contracts"]:
        lines.append("Top current contracts:")
        for row in status["top_contracts"]:
            lines.append(
                " - "
                f"{row['option_symbol']} | strike={row['strike']} | dte={row['days_to_expiry']:.2f}"
                f" | mid={row['mid']:.3f} | delta={row['delta']:.4f}"
                f" | score={row['candidate_score']:.4f}"
            )
    return "\n".join(lines)


def main() -> None:
    status = build_status()
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    print(render_report(status))


if __name__ == "__main__":
    main()
