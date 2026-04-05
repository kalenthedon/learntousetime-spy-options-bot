from __future__ import annotations

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.options.paper_ops_status import build_ops_status


def _fmt(value) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_dashboard(status: dict) -> str:
    history = status.get("history") or {}
    selector = status.get("selector") or {}
    portfolio = status.get("portfolio") or {}
    top_contracts = history.get("top_contracts") or []
    open_position = portfolio.get("open_position") or {}

    def card(title: str, rows: list[tuple[str, object]]) -> str:
        body = "".join(
            f"<tr><th>{html.escape(label)}</th><td>{html.escape(_fmt(value))}</td></tr>"
            for label, value in rows
        )
        return f"<section class='card'><h2>{html.escape(title)}</h2><table>{body}</table></section>"

    top_rows = "".join(
        "<tr>"
        f"<td>{html.escape(_fmt(row.get('option_symbol')))}</td>"
        f"<td>{html.escape(_fmt(row.get('strike')))}</td>"
        f"<td>{html.escape(_fmt(row.get('days_to_expiry')))}</td>"
        f"<td>{html.escape(_fmt(row.get('mid')))}</td>"
        f"<td>{html.escape(_fmt(row.get('delta')))}</td>"
        f"<td>{html.escape(_fmt(row.get('candidate_score')))}</td>"
        "</tr>"
        for row in top_contracts
    )
    top_contracts_card = f"""
    <section class='card span-2'>
      <h2>Top Current Contracts</h2>
      <table>
        <thead>
          <tr>
            <th>Option Symbol</th>
            <th>Strike</th>
            <th>DTE</th>
            <th>Mid</th>
            <th>Delta</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>
          {top_rows or "<tr><td colspan='6'>n/a</td></tr>"}
        </tbody>
      </table>
    </section>
    """

    cards = [
        card(
            "History",
            [
                ("Snapshots", history.get("history_snapshot_count")),
                ("Contracts", history.get("history_contract_count")),
                ("History Rows", history.get("history_rows")),
                ("Latest Rows", history.get("latest_rows")),
                ("Latest Collected At", history.get("latest_collected_at")),
                ("Ready For Research", history.get("ready_for_research")),
            ],
        ),
        card(
            "Selector",
            [
                ("Cycle Timestamp", selector.get("cycle_timestamp")),
                ("Snapshot Timestamp", selector.get("snapshot_timestamp")),
                ("Decision", selector.get("decision")),
                ("Decision Reason", selector.get("decision_reason")),
                ("Paper Trade Ready", selector.get("paper_trade_ready")),
                ("Readiness Reason", selector.get("paper_trade_readiness_reason")),
                ("Selected Symbol", selector.get("selected_option_symbol")),
                ("Selected Score", selector.get("selected_selection_score")),
                ("Collect Status", selector.get("collect_status")),
                ("Collect Reason", selector.get("collect_reason")),
            ],
        ),
        card(
            "Portfolio",
            [
                ("Open Position", portfolio.get("has_open_position")),
                ("Closed Positions", portfolio.get("closed_positions")),
                ("Realized PnL", portfolio.get("realized_pnl")),
                ("Realized Return Pct", portfolio.get("realized_return_pct")),
                ("Entry Events", portfolio.get("entry_events")),
                ("Exit Events", portfolio.get("exit_events")),
                ("Exit Win Rate", portfolio.get("exit_win_rate")),
                ("Avg Exit Return Pct", portfolio.get("avg_exit_return_pct")),
                ("Last Cycle Timestamp", portfolio.get("last_cycle_timestamp")),
            ],
        ),
        card(
            "Open Position",
            [
                ("Option Symbol", open_position.get("option_symbol")),
                ("Qty", open_position.get("qty")),
                ("Entry Mid", open_position.get("entry_mid")),
                ("Current Mid", open_position.get("current_mid")),
                ("Entry Score", open_position.get("entry_score")),
                ("Entry Delta", open_position.get("entry_delta")),
                ("Bars Held", open_position.get("bars_held")),
                ("Unrealized PnL", open_position.get("unrealized_pnl")),
                ("Unrealized Return Pct", open_position.get("unrealized_return_pct")),
                ("Target Mid", open_position.get("target_mid")),
                ("Stop Mid", open_position.get("stop_mid")),
            ],
        ),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="30">
  <title>KTD SPY Options Dashboard</title>
  <style>
    :root {{
      --bg: #0e1116;
      --panel: #131922;
      --border: #243244;
      --text: #edf2f7;
      --muted: #9fb0c0;
      --accent: #7cc6fe;
      --accent-2: #8ce99a;
      --warn: #ffd27d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      background: radial-gradient(circle at top, #172233 0%, var(--bg) 50%);
      color: var(--text);
      padding: 24px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p.meta {{ margin: 0 0 24px; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: rgba(19, 25, 34, 0.94);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
    }}
    .span-2 {{ grid-column: span 2; }}
    @media (max-width: 900px) {{
      .span-2 {{ grid-column: span 1; }}
    }}
    h2 {{ margin: 0 0 12px; font-size: 16px; color: var(--accent); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      text-align: left;
      vertical-align: top;
      padding: 8px 0;
      border-bottom: 1px solid rgba(36, 50, 68, 0.7);
      font-size: 13px;
    }}
    thead th {{
      color: var(--accent-2);
      font-weight: 600;
      border-bottom: 1px solid rgba(124, 198, 254, 0.25);
      padding-right: 10px;
    }}
    tbody td {{ padding-right: 10px; }}
    th {{ color: var(--muted); width: 44%; font-weight: 500; }}
    tr:last-child th, tr:last-child td {{ border-bottom: 0; }}
    .footer {{ margin-top: 20px; color: var(--warn); font-size: 12px; }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <h1>KTD SPY Options Paper Dashboard</h1>
  <p class="meta">Auto-refreshes every 30s. JSON endpoint: <a href="/status.json">/status.json</a></p>
  <div class="grid">
    {''.join(cards)}
    {top_contracts_card}
  </div>
  <p class="footer">Paper dashboard reflects history depth, selector readiness, and paper portfolio state at a glance.</p>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def _write(self, body: bytes, content_type: str, status_code: int = 200) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        status = build_ops_status()
        if self.path in {"/status.json", "/api/status"}:
            body = json.dumps(status, indent=2, sort_keys=True, default=str).encode("utf-8")
            self._write(body, "application/json; charset=utf-8")
            return
        if self.path in {"/", "/index.html"}:
            body = render_dashboard(status).encode("utf-8")
            self._write(body, "text/html; charset=utf-8")
            return
        self._write(b"not found", "text/plain; charset=utf-8", status_code=404)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a live SPY options paper-trading dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8797)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"dashboard_listen=http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
