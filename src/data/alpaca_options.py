from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


ALPACA_API_KEY_ENV = "ALPACA_API_KEY"
ALPACA_API_SECRET_ENV = "ALPACA_API_SECRET"
ALPACA_DATA_FEED_ENV = "ALPACA_DATA_FEED"
ALPACA_ENV_PATH = Path("deployment/alpaca_options.env")
ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
ALPACA_TRADING_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_SNAPSHOT_BATCH_SIZE = 20


def load_local_env_file(path: Path = ALPACA_ENV_PATH) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def alpaca_headers() -> dict:
    load_local_env_file()
    return {
        "APCA-API-KEY-ID": require_env(ALPACA_API_KEY_ENV),
        "APCA-API-SECRET-KEY": require_env(ALPACA_API_SECRET_ENV),
    }


def alpaca_data_feed() -> str:
    load_local_env_file()
    return os.getenv(ALPACA_DATA_FEED_ENV, "indicative").strip() or "indicative"


def _request_json(base_url: str, path: str, params: dict | None = None) -> dict:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    response = requests.get(url, headers=alpaca_headers(), params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_option_contracts(
    underlying_symbol: str,
    expiration_gte: str | None = None,
    expiration_lte: str | None = None,
    option_type: str | None = None,
    status: str = "active",
    limit: int = 1000,
) -> list[dict]:
    params = {
        "underlying_symbols": underlying_symbol.upper(),
        "status": status,
        "limit": int(limit),
    }
    if expiration_gte:
        params["expiration_date_gte"] = expiration_gte
    if expiration_lte:
        params["expiration_date_lte"] = expiration_lte
    if option_type:
        params["type"] = option_type.lower()

    rows = []
    page_token = None
    while True:
        req_params = dict(params)
        if page_token:
            req_params["page_token"] = page_token
        payload = _request_json(ALPACA_TRADING_BASE_URL, "/v2/options/contracts", params=req_params)
        rows.extend(payload.get("option_contracts", []) or [])
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return rows


def fetch_option_snapshots(symbols: Iterable[str], feed: str | None = None) -> dict:
    symbols = [str(symbol).upper() for symbol in symbols]
    if not symbols:
        return {}
    merged = {"snapshots": {}}
    batch_size = ALPACA_SNAPSHOT_BATCH_SIZE
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        params = {
            "symbols": ",".join(batch),
            "feed": feed or alpaca_data_feed(),
        }
        payload = _request_json(ALPACA_DATA_BASE_URL, "/v1beta1/options/snapshots", params=params)
        merged["snapshots"].update(payload.get("snapshots", {}) or {})
    return merged


def fetch_stock_bars(
    symbol: str,
    timeframe: str = "1Hour",
    start: str | None = None,
    end: str | None = None,
    limit: int = 1000,
    feed: str = "iex",
) -> pd.DataFrame:
    params = {
        "timeframe": timeframe,
        "limit": int(limit),
        "feed": feed,
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    payload = _request_json(ALPACA_DATA_BASE_URL, f"/v2/stocks/{symbol.upper()}/bars", params=params)
    bars = payload.get("bars", []) or []
    if not bars:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(bars)
    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def snapshots_to_frame(contracts: list[dict], snapshots: dict) -> pd.DataFrame:
    if not contracts:
        return pd.DataFrame()

    rows = []
    by_symbol = snapshots.get("snapshots", {}) if isinstance(snapshots, dict) else {}

    for contract in contracts:
        symbol = str(contract.get("symbol", "")).upper()
        snap = by_symbol.get(symbol) or {}
        latest_quote = snap.get("latestQuote") or {}
        greeks = snap.get("greeks") or {}
        latest_trade = snap.get("latestTrade") or {}
        implied_vol = snap.get("impliedVolatility")
        quote_ts = latest_quote.get("t") or latest_trade.get("t")
        if not quote_ts:
            continue

        bid = latest_quote.get("bp")
        ask = latest_quote.get("ap")
        if bid is None or ask is None:
            continue
        bid = float(bid)
        ask = float(ask)
        mid = (bid + ask) / 2.0

        expiration = pd.to_datetime(contract.get("expiration_date"), utc=True)
        timestamp = pd.to_datetime(quote_ts, utc=True)
        dte = max((expiration - timestamp).total_seconds() / 86400.0, 0.0)

        rows.append(
            {
                "timestamp": timestamp,
                "underlying_symbol": str(contract.get("underlying_symbol", "")).upper(),
                "underlying_price": float(snap.get("underlyingPrice") or 0.0),
                "option_symbol": symbol,
                "option_type": str(contract.get("type", "")).lower(),
                "strike": float(contract.get("strike_price") or 0.0),
                "expiration": expiration,
                "days_to_expiry": dte,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "mark_iv": float(implied_vol) if implied_vol is not None else float("nan"),
                "delta": float(greeks.get("delta")) if greeks.get("delta") is not None else float("nan"),
                "gamma": float(greeks.get("gamma")) if greeks.get("gamma") is not None else float("nan"),
                "theta": float(greeks.get("theta")) if greeks.get("theta") is not None else float("nan"),
                "vega": float(greeks.get("vega")) if greeks.get("vega") is not None else float("nan"),
                "open_interest": float(contract.get("open_interest") or 0.0),
                "volume": float(latest_trade.get("s") or 0.0),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["timestamp", "expiration", "strike"]).reset_index(drop=True)
