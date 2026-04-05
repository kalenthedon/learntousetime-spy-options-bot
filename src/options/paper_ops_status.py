from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.options.history_status import build_status as build_history_status
from src.options.paper_portfolio_status import build_portfolio_status


PAPER_SELECTOR_LATEST_PATH = Path("experiments/options_paper_selector_latest.json")


def _load_selector_latest() -> dict:
    if not PAPER_SELECTOR_LATEST_PATH.exists() or PAPER_SELECTOR_LATEST_PATH.stat().st_size == 0:
        return {}
    return json.loads(PAPER_SELECTOR_LATEST_PATH.read_text(encoding="utf-8"))


def build_ops_status() -> dict:
    history_status = build_history_status()
    portfolio_status = build_portfolio_status()
    selector_latest = _load_selector_latest()
    selector_report = selector_latest.get("report", {}) or {}
    collect_result = selector_latest.get("collect_result", {}) or {}
    return {
        "history": history_status,
        "selector": {
            "cycle_timestamp": selector_latest.get("cycle_timestamp"),
            "snapshot_timestamp": selector_report.get("snapshot_timestamp"),
            "decision": selector_report.get("decision"),
            "decision_reason": selector_report.get("decision_reason"),
            "paper_trade_ready": selector_report.get("paper_trade_ready"),
            "paper_trade_readiness_reason": selector_report.get("paper_trade_readiness_reason"),
            "selected_option_symbol": selector_report.get("selected_option_symbol"),
            "selected_selection_score": selector_report.get("selected_selection_score"),
            "collect_status": collect_result.get("status"),
            "collect_reason": collect_result.get("reason"),
        },
        "portfolio": portfolio_status,
    }


def render_report(status: dict) -> str:
    history = status["history"]
    selector = status["selector"]
    portfolio = status["portfolio"]
    lines = [
        "SPY Options Paper Ops Status",
        (
            f"History: snapshots={history['history_snapshot_count']}"
            f" | contracts={history['history_contract_count']}"
            f" | latest_collected_at={history['latest_collected_at'] or 'n/a'}"
        ),
        (
            f"Selector: cycle={selector.get('cycle_timestamp') or 'n/a'}"
            f" | decision={selector.get('decision') or 'n/a'}"
            f" | ready={selector.get('paper_trade_ready')}"
            f" | collect={selector.get('collect_status') or 'n/a'}:{selector.get('collect_reason') or 'n/a'}"
        ),
        (
            f"Selector detail: selected={selector.get('selected_option_symbol') or 'none'}"
            f" | score={selector.get('selected_selection_score') if selector.get('selected_selection_score') is not None else 'n/a'}"
            f" | reason={selector.get('decision_reason') or 'n/a'}"
        ),
        (
            f"Portfolio: open={portfolio['has_open_position']}"
            f" | closed={portfolio['closed_positions']}"
            f" | realized_pnl={portfolio['realized_pnl']}"
            f" | exit_win_rate={portfolio['exit_win_rate'] if portfolio['exit_win_rate'] is not None else 'n/a'}"
        ),
    ]
    open_position = portfolio.get("open_position") or {}
    if open_position:
        lines.append(
            "Open position:"
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
    status = build_ops_status()
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(status, indent=2, default=str))
        return
    print(render_report(status))


if __name__ == "__main__":
    main()
