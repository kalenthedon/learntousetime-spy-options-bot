from __future__ import annotations

import pandas as pd


REQUIRED_OPTION_COLUMNS = [
    "timestamp",
    "underlying_symbol",
    "underlying_price",
    "option_symbol",
    "option_type",
    "strike",
    "expiration",
    "days_to_expiry",
    "bid",
    "ask",
    "mid",
    "mark_iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "open_interest",
    "volume",
]


def validate_option_chain_frame(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_OPTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Option chain frame missing required columns: {missing}")


def normalize_option_chain_frame(df: pd.DataFrame) -> pd.DataFrame:
    validate_option_chain_frame(df)
    out = df.copy()

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["expiration"] = pd.to_datetime(out["expiration"], utc=True)
    out["option_type"] = out["option_type"].astype(str).str.lower()
    out["underlying_symbol"] = out["underlying_symbol"].astype(str).str.upper()
    out["option_symbol"] = out["option_symbol"].astype(str)

    numeric_cols = [
        "underlying_price",
        "strike",
        "days_to_expiry",
        "bid",
        "ask",
        "mid",
        "mark_iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "open_interest",
        "volume",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values(["timestamp", "expiration", "strike", "option_type", "option_symbol"]).reset_index(drop=True)
    return out
