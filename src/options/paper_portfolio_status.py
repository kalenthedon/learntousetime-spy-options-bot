from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.options.paper_selector_cycle import (
    DEFAULT_PORTFOLIO_JOURNAL_PATH,
    DEFAULT_PORTFOLIO_STATE_PATH,
)


def _load_state(path: str) -> dict:
    state_path = Path(path)
    if not state_path.exists() or state_path.stat().st_size == 0:
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _load_journal(path: str) -> pd.DataFrame:
    journal_path = Path(path)
    if not journal_path.exists() or journal_path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(journal_path)


def build_portfolio_status(
    state_path: str = DEFAULT_PORTFOLIO_STATE_PATH,
    journal_path: str = DEFAULT_PORTFOLIO_JOURNAL_PATH,
) -> dict:
    state = _load_state(state_path)
    journal = _load_journal(journal_path)
    entries = journal.loc[journal["event_type"] == "entry"].copy() if not journal.empty and "event_type" in journal.columns else pd.DataFrame()
    exits = journal.loc[journal["event_type"] == "exit"].copy() if not journal.empty and "event_type" in journal.columns else pd.DataFrame()

    realized_pnl = float(state.get("realized_pnl", 0.0) or 0.0)
    realized_return_pct = float(state.get("realized_return_pct", 0.0) or 0.0)
    exit_win_rate = None
    avg_exit_return_pct = None
    if not exits.empty and "realized_return_pct" in exits.columns:
        exit_returns = pd.to_numeric(exits["realized_return_pct"], errors="coerce").dropna()
        if not exit_returns.empty:
            exit_win_rate = float((exit_returns > 0).mean())
            avg_exit_return_pct = float(exit_returns.mean())

    open_position = state.get("open_position") or None
    return {
        "state_path": state_path,
        "journal_path": journal_path,
        "has_open_position": bool(open_position),
        "open_position": open_position,
        "closed_positions": int(state.get("closed_positions", 0) or 0),
        "realized_pnl": realized_pnl,
        "realized_return_pct": realized_return_pct,
        "entry_events": int(len(entries)),
        "exit_events": int(len(exits)),
        "exit_win_rate": exit_win_rate,
        "avg_exit_return_pct": avg_exit_return_pct,
        "last_cycle_timestamp": state.get("last_cycle_timestamp"),
    }


def render_report(status: dict) -> str:
    lines = [
        "SPY Options Paper Portfolio",
        (
            f"Open position: {status['has_open_position']}"
            f" | closed_positions: {status['closed_positions']}"
            f" | realized_pnl: {status['realized_pnl']}"
            f" | realized_return_pct: {status['realized_return_pct']}"
        ),
        (
            f"Entry events: {status['entry_events']}"
            f" | exit events: {status['exit_events']}"
            f" | exit_win_rate: {status['exit_win_rate'] if status['exit_win_rate'] is not None else 'n/a'}"
            f" | avg_exit_return_pct: {status['avg_exit_return_pct'] if status['avg_exit_return_pct'] is not None else 'n/a'}"
        ),
        f"Last cycle timestamp: {status['last_cycle_timestamp'] or 'n/a'}",
    ]
    open_position = status.get("open_position") or {}
    if open_position:
        lines.append(
            "Open position detail:"
        )
        lines.append(
            " - "
            f"{open_position.get('option_symbol')} | qty={open_position.get('qty')}"
            f" | entry_mid={open_position.get('entry_mid')}"
            f" | current_mid={open_position.get('current_mid')}"
            f" | unrealized_pnl={open_position.get('unrealized_pnl')}"
            f" | bars_held={open_position.get('bars_held')}"
        )
    return "\n".join(lines)


def main() -> None:
    status = build_portfolio_status()
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(status, indent=2, default=str))
        return
    print(render_report(status))


if __name__ == "__main__":
    main()
