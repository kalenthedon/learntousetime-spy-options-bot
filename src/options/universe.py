from __future__ import annotations

import pandas as pd


def filter_long_put_candidates(
    df: pd.DataFrame,
    min_dte: float = 7.0,
    max_dte: float = 21.0,
    min_abs_delta: float = 0.25,
    max_abs_delta: float = 0.45,
    max_spread_pct_mid: float = 0.08,
    min_open_interest: float = 500.0,
    min_volume: float = 1.0,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["delta_abs"] = out["delta"].abs()
    out["spread_pct_mid"] = (out["ask"] - out["bid"]) / out["mid"]
    mask = (
        out["option_type"].eq("put")
        & out["days_to_expiry"].between(min_dte, max_dte)
        & out["delta_abs"].between(min_abs_delta, max_abs_delta)
        & (out["spread_pct_mid"] <= max_spread_pct_mid)
        & (out["open_interest"] >= min_open_interest)
        & (out["volume"] >= min_volume)
        & (out["mid"] > 0.0)
    )
    return out.loc[mask].copy()


def score_long_put_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["delta_distance_score"] = 1.0 - (out["delta"].abs() - 0.35).abs()
    out["liquidity_score"] = out["open_interest"].rank(pct=True) + out["volume"].rank(pct=True)
    out["spread_score"] = 1.0 - out["spread_pct_mid"].rank(pct=True)
    out["dte_score"] = 1.0 - (out["days_to_expiry"] - 14.0).abs() / 14.0
    out["candidate_score"] = (
        0.35 * out["delta_distance_score"].fillna(0.0)
        + 0.30 * out["liquidity_score"].fillna(0.0)
        + 0.20 * out["spread_score"].fillna(0.0)
        + 0.15 * out["dte_score"].fillna(0.0)
    )
    return out.sort_values(
        ["timestamp", "candidate_score", "open_interest", "volume"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
