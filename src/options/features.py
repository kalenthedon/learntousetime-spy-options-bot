from __future__ import annotations

import numpy as np
import pandas as pd


def make_option_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["option_symbol", "timestamp"]).reset_index(drop=True)

    out["spread"] = out["ask"] - out["bid"]
    out["spread_pct_mid"] = np.where(out["mid"] > 0, out["spread"] / out["mid"], np.nan)
    out["moneyness"] = np.where(
        out["option_type"].eq("call"),
        out["underlying_price"] / out["strike"] - 1.0,
        out["strike"] / out["underlying_price"] - 1.0,
    )
    out["intrinsic_value"] = np.where(
        out["option_type"].eq("call"),
        np.maximum(out["underlying_price"] - out["strike"], 0.0),
        np.maximum(out["strike"] - out["underlying_price"], 0.0),
    )
    out["extrinsic_value"] = out["mid"] - out["intrinsic_value"]
    out["extrinsic_pct_mid"] = np.where(out["mid"] > 0, out["extrinsic_value"] / out["mid"], np.nan)
    out["dollar_gamma"] = out["gamma"] * out["underlying_price"] ** 2
    out["theta_to_premium"] = np.where(out["mid"] > 0, out["theta"] / out["mid"], np.nan)
    out["vega_to_premium"] = np.where(out["mid"] > 0, out["vega"] / out["mid"], np.nan)
    out["delta_abs"] = out["delta"].abs()
    out["liquidity_score"] = np.log1p(out["open_interest"].clip(lower=0)) + np.log1p(out["volume"].clip(lower=0))

    grouped = out.groupby("option_symbol", sort=False)
    out["mid_ret_1"] = grouped["mid"].pct_change()
    out["mid_ret_3"] = grouped["mid"].pct_change(3)
    out["iv_chg_1"] = grouped["mark_iv"].diff()
    out["iv_chg_3"] = grouped["mark_iv"].diff(3)
    underlying_grouped = out.groupby("underlying_symbol", sort=False)
    out["underlying_ret_1"] = underlying_grouped["underlying_price"].pct_change()
    out["underlying_ret_3"] = underlying_grouped["underlying_price"].pct_change(3)
    out["volume_z_10"] = grouped["volume"].transform(
        lambda s: (s - s.rolling(10).mean()) / s.rolling(10).std()
    )
    out["spread_rank_in_chain"] = out.groupby(["timestamp", "underlying_symbol"])["spread_pct_mid"].rank(pct=True)
    out["oi_rank_in_chain"] = out.groupby(["timestamp", "underlying_symbol"])["open_interest"].rank(pct=True)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
