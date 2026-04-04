from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


LATEST_PATH = Path("centralized_data/options/SPY_put_chain_latest.csv")
HISTORY_PATH = Path("centralized_data/options/SPY_put_chain_history.csv")
PAPER_SELECTOR_LATEST_PATH = Path("experiments/options_paper_selector_latest.json")


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
    selector_status = {}
    if PAPER_SELECTOR_LATEST_PATH.exists() and PAPER_SELECTOR_LATEST_PATH.stat().st_size > 0:
        selector_payload = json.loads(PAPER_SELECTOR_LATEST_PATH.read_text(encoding="utf-8"))
        report = selector_payload.get("report", {}) or {}
        collect_result = selector_payload.get("collect_result", {}) or {}
        portfolio = selector_payload.get("portfolio", {}) or {}
        selector_status = {
            "cycle_timestamp": selector_payload.get("cycle_timestamp"),
            "collect_status": collect_result.get("status"),
            "collect_reason": collect_result.get("reason"),
            "decision": report.get("decision"),
            "decision_reason": report.get("decision_reason"),
            "paper_trade_ready": report.get("paper_trade_ready"),
            "paper_trade_readiness_reason": report.get("paper_trade_readiness_reason"),
            "selected_option_symbol": report.get("selected_option_symbol"),
            "selected_selection_score": report.get("selected_selection_score"),
            "snapshot_timestamp": report.get("snapshot_timestamp"),
            "portfolio_open_symbol": (portfolio.get("open_position") or {}).get("option_symbol"),
            "portfolio_unrealized_pnl": (portfolio.get("open_position") or {}).get("unrealized_pnl"),
            "portfolio_realized_pnl": portfolio.get("realized_pnl"),
            "portfolio_closed_positions": portfolio.get("closed_positions"),
        }

    return {
        "latest_rows": int(len(latest)),
        "history_rows": int(len(history)),
        "latest_timestamp": latest_ts,
        "latest_collected_at": latest_collected_at,
        "history_snapshot_count": history_snapshot_count,
        "history_contract_count": history_contract_count,
        "ready_for_research": history_snapshot_count >= 50 and history_contract_count >= 20,
        "top_contracts": top_contracts,
        "paper_selector_status": selector_status,
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
    selector = status.get("paper_selector_status") or {}
    if selector:
        lines.append("Latest paper selector cycle:")
        lines.append(
            " - "
            f"cycle={selector.get('cycle_timestamp') or 'n/a'}"
            f" | snapshot={selector.get('snapshot_timestamp') or 'n/a'}"
            f" | collect={selector.get('collect_status') or 'n/a'}:{selector.get('collect_reason') or 'n/a'}"
            f" | decision={selector.get('decision') or 'n/a'}"
            f" | ready={selector.get('paper_trade_ready')}"
        )
        if selector.get("selected_option_symbol"):
            lines.append(
                " - "
                f"selected={selector['selected_option_symbol']}"
                f" | score={float(selector['selected_selection_score']):.4f}"
            )
        else:
            lines.append(
                " - "
                f"decision_reason={selector.get('decision_reason') or 'n/a'}"
                f" | readiness_reason={selector.get('paper_trade_readiness_reason') or 'n/a'}"
            )
        lines.append(
            " - "
            f"portfolio_open={selector.get('portfolio_open_symbol') or 'none'}"
            f" | realized_pnl={selector.get('portfolio_realized_pnl') if selector.get('portfolio_realized_pnl') is not None else 'n/a'}"
            f" | closed_positions={selector.get('portfolio_closed_positions') if selector.get('portfolio_closed_positions') is not None else 'n/a'}"
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
