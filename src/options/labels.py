from __future__ import annotations

import numpy as np
import pandas as pd


def _forward_window_stat(s: pd.Series, horizon_bars: int, fn: str) -> pd.Series:
    shifted = s.shift(-1)
    reversed_shifted = shifted.iloc[::-1]
    if fn == "max":
        out = reversed_shifted.rolling(horizon_bars, min_periods=horizon_bars).max()
    elif fn == "min":
        out = reversed_shifted.rolling(horizon_bars, min_periods=horizon_bars).min()
    else:
        raise ValueError(f"Unsupported fn: {fn}")
    return out.iloc[::-1]


def make_option_labels(
    df: pd.DataFrame,
    horizon_bars: int = 8,
    target_return_pct: float = 0.25,
    max_adverse_return_pct: float = -0.20,
) -> pd.Series:
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")

    grouped = df.groupby("option_symbol", sort=False)
    future_mid = grouped["mid"].shift(-horizon_bars)
    future_peak = grouped["mid"].transform(lambda s: _forward_window_stat(s, horizon_bars, "max"))
    future_trough = grouped["mid"].transform(lambda s: _forward_window_stat(s, horizon_bars, "min"))

    base_mid = df["mid"].astype(float)
    terminal_return = future_mid / base_mid - 1.0
    peak_return = future_peak / base_mid - 1.0
    trough_return = future_trough / base_mid - 1.0

    y = (
        (peak_return >= float(target_return_pct))
        & (trough_return > float(max_adverse_return_pct))
        & np.isfinite(terminal_return)
    ).astype(int)
    return y
