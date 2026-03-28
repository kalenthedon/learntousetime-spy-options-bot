from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ml.walkforward_train import WalkForwardRunConfig, walk_forward_train_predict
from src.data.alpaca_options import fetch_stock_bars
from src.options.contracts import normalize_option_chain_frame
from src.options.features import make_option_features
from src.options.labels import make_option_labels


DEFAULT_OUTPUT_PATH = "experiments/options_research_summary.json"
DEFAULT_TRADES_OUTPUT_PATH = "experiments/options_research_trades.csv"


def load_option_chain_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return repair_underlying_prices(normalize_option_chain_frame(df))


def repair_underlying_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "underlying_price" not in df.columns:
        return df

    out = df.copy()
    missing_mask = out["underlying_price"].isna() | (out["underlying_price"] <= 0)
    if not missing_mask.any():
        return out

    repaired_frames = []
    for symbol, group in out.groupby("underlying_symbol", sort=False):
        group = group.sort_values("timestamp").copy()
        group_missing = group["underlying_price"].isna() | (group["underlying_price"] <= 0)
        if not group_missing.any():
            repaired_frames.append(group)
            continue

        start = (group["timestamp"].min() - pd.Timedelta(hours=2)).isoformat()
        end = (group["timestamp"].max() + pd.Timedelta(hours=2)).isoformat()
        bars = fetch_stock_bars(str(symbol), timeframe="1Hour", start=start, end=end, limit=10_000)
        if bars.empty:
            repaired_frames.append(group)
            continue

        bars = bars.sort_values("timestamp")[["timestamp", "close"]].rename(columns={"close": "underlying_price_bar"})
        merged = pd.merge_asof(
            group.sort_values("timestamp"),
            bars,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(hours=2),
        )
        group["underlying_price"] = np.where(
            group_missing,
            merged["underlying_price_bar"].to_numpy(),
            group["underlying_price"].to_numpy(),
        )
        repaired_frames.append(group.drop(columns=[], errors="ignore"))

    repaired = pd.concat(repaired_frames, ignore_index=True)
    repaired["underlying_price"] = pd.to_numeric(repaired["underlying_price"], errors="coerce")
    return repaired.sort_values(["timestamp", "expiration", "strike", "option_type", "option_symbol"]).reset_index(drop=True)


def prepare_option_research_frame(
    df: pd.DataFrame,
    horizon_bars: int,
    target_return_pct: float,
    max_adverse_return_pct: float,
) -> tuple[pd.DataFrame, list[str]]:
    feat = make_option_features(df)
    feature_cols = [
        "days_to_expiry",
        "moneyness",
        "spread_pct_mid",
        "extrinsic_pct_mid",
        "mark_iv",
        "delta",
        "delta_abs",
        "gamma",
        "theta_to_premium",
        "vega_to_premium",
        "dollar_gamma",
        "open_interest",
        "volume",
        "liquidity_score",
        "mid_ret_1",
        "mid_ret_3",
        "iv_chg_1",
        "iv_chg_3",
        "underlying_ret_1",
        "underlying_ret_3",
        "volume_z_10",
        "spread_rank_in_chain",
        "oi_rank_in_chain",
    ]
    fill_zero_cols = [
        "mid_ret_1",
        "mid_ret_3",
        "iv_chg_1",
        "iv_chg_3",
        "underlying_ret_1",
        "underlying_ret_3",
        "volume_z_10",
    ]
    feat[fill_zero_cols] = feat[fill_zero_cols].fillna(0.0)
    feat["spread_rank_in_chain"] = feat["spread_rank_in_chain"].fillna(0.5)
    feat["oi_rank_in_chain"] = feat["oi_rank_in_chain"].fillna(0.5)
    feat["y"] = make_option_labels(
        feat,
        horizon_bars=horizon_bars,
        target_return_pct=target_return_pct,
        max_adverse_return_pct=max_adverse_return_pct,
    )
    feat = feat.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    return feat, feature_cols


def summarize_top_contracts(pred_df: pd.DataFrame, proba_col: str, top_k: int = 1) -> dict:
    if pred_df.empty:
        return {
            "timestamps": 0,
            "avg_top_probability": None,
            "avg_positive_rate_top": None,
            "top_k": top_k,
        }

    working = pred_df.copy()
    working["rank"] = working.groupby("timestamp")[proba_col].rank(method="first", ascending=False)
    top = working.loc[working["rank"] <= top_k].copy()
    return {
        "timestamps": int(working["timestamp"].nunique()),
        "avg_top_probability": float(top[proba_col].mean()),
        "avg_positive_rate_top": float(top["y_true"].mean()),
        "top_k": int(top_k),
        "rows_selected": int(len(top)),
    }


def build_model_factory(model_type: str):
    model_type = str(model_type).strip().lower()
    if model_type == "logistic":
        return lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced"),
        )
    if model_type == "random_forest":
        return lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced_subsample",
        )
    if model_type == "hist_gbm":
        return lambda: HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            random_state=42,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _simulate_option_trade(
    contract_frame: pd.DataFrame,
    entry_idx: int,
    horizon_bars: int,
    target_return_pct: float,
    max_adverse_return_pct: float,
) -> dict | None:
    if entry_idx >= len(contract_frame):
        return None
    entry_row = contract_frame.iloc[entry_idx]
    entry_mid = float(entry_row["mid"])
    if entry_mid <= 0:
        return None

    future = contract_frame.iloc[entry_idx + 1: entry_idx + 1 + horizon_bars].copy()
    if future.empty:
        return None

    target_mid = entry_mid * (1.0 + float(target_return_pct))
    stop_mid = entry_mid * (1.0 + float(max_adverse_return_pct))
    exit_row = future.iloc[-1]
    exit_reason = "timeout"

    for _, candidate in future.iterrows():
        candidate_mid = float(candidate["mid"])
        if candidate_mid >= target_mid:
            exit_row = candidate
            exit_reason = "target"
            break
        if candidate_mid <= stop_mid:
            exit_row = candidate
            exit_reason = "stop"
            break

    exit_mid = float(exit_row["mid"])
    net_return_pct = exit_mid / entry_mid - 1.0
    return {
        "entry_timestamp": str(entry_row["timestamp"]),
        "exit_timestamp": str(exit_row["timestamp"]),
        "option_symbol": str(entry_row["option_symbol"]),
        "underlying_symbol": str(entry_row["underlying_symbol"]),
        "days_to_expiry_entry": float(entry_row["days_to_expiry"]),
        "entry_mid": entry_mid,
        "exit_mid": exit_mid,
        "exit_reason": exit_reason,
        "net_return_pct": net_return_pct,
        "bars_held": int(len(contract_frame.loc[(contract_frame["timestamp"] > entry_row["timestamp"]) & (contract_frame["timestamp"] <= exit_row["timestamp"])])),
    }


def simulate_option_selection_trades(
    frame: pd.DataFrame,
    pred_df: pd.DataFrame,
    proba_col: str,
    horizon_bars: int,
    target_return_pct: float,
    max_adverse_return_pct: float,
    min_entry_proba: float,
    top_k: int,
    allow_overlapping_positions: bool,
) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()

    enriched = pred_df.reset_index().copy()
    enriched = enriched.sort_values(["timestamp", proba_col], ascending=[True, False]).copy()
    enriched["rank"] = enriched.groupby("timestamp")[proba_col].rank(method="first", ascending=False)
    selected = enriched.loc[(enriched["rank"] <= int(top_k)) & (enriched[proba_col] >= float(min_entry_proba))].copy()
    if selected.empty:
        return pd.DataFrame()

    trades: list[dict] = []
    contract_groups = {
        symbol: group.reset_index(drop=True)
        for symbol, group in frame.sort_values(["option_symbol", "timestamp"]).groupby("option_symbol", sort=False)
    }
    seen_entries: set[tuple[str, str]] = set()
    active_until_timestamp = None

    for row in selected.itertuples(index=False):
        key = (str(row.timestamp), str(row.option_symbol))
        if key in seen_entries:
            continue
        row_timestamp = pd.Timestamp(row.timestamp)
        if not allow_overlapping_positions and active_until_timestamp is not None and row_timestamp <= active_until_timestamp:
            continue
        seen_entries.add(key)
        contract_frame = contract_groups.get(str(row.option_symbol))
        if contract_frame is None or contract_frame.empty:
            continue
        entry_matches = contract_frame.index[contract_frame["timestamp"].astype(str) == str(row.timestamp)]
        if len(entry_matches) == 0:
            continue
        trade = _simulate_option_trade(
            contract_frame=contract_frame,
            entry_idx=int(entry_matches[0]),
            horizon_bars=horizon_bars,
            target_return_pct=target_return_pct,
            max_adverse_return_pct=max_adverse_return_pct,
        )
        if trade is None:
            continue
        trade["entry_probability"] = float(getattr(row, proba_col))
        trades.append(trade)
        active_until_timestamp = pd.Timestamp(trade["exit_timestamp"])

    return pd.DataFrame(trades)


def summarize_simulated_trades(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "trade_count": 0,
            "win_rate": None,
            "avg_return_pct": None,
            "median_return_pct": None,
            "total_return_pct": None,
            "target_hit_rate": None,
            "stop_hit_rate": None,
        }
    net_returns = trades_df["net_return_pct"].astype(float)
    return {
        "trade_count": int(len(trades_df)),
        "win_rate": float((net_returns > 0).mean()),
        "avg_return_pct": float(net_returns.mean()),
        "median_return_pct": float(net_returns.median()),
        "total_return_pct": float(net_returns.sum()),
        "target_hit_rate": float((trades_df["exit_reason"] == "target").mean()),
        "stop_hit_rate": float((trades_df["exit_reason"] == "stop").mean()),
    }


def run_options_research(
    csv_path: str,
    model_type: str,
    train_size: int,
    test_size: int,
    step_size: int,
    horizon_bars: int,
    target_return_pct: float,
    max_adverse_return_pct: float,
    min_entry_proba: float,
    top_k: int,
    allow_overlapping_positions: bool,
    output_path: str = DEFAULT_OUTPUT_PATH,
    trades_output_path: str = DEFAULT_TRADES_OUTPUT_PATH,
) -> dict:
    raw = load_option_chain_csv(csv_path)
    frame, feature_cols = prepare_option_research_frame(
        raw,
        horizon_bars=horizon_bars,
        target_return_pct=target_return_pct,
        max_adverse_return_pct=max_adverse_return_pct,
    )
    frame = frame.reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame))

    cfg = WalkForwardRunConfig(
        feature_cols=feature_cols,
        label_col="y",
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        purge_size=horizon_bars,
        calibrate=False,
        save_models=False,
        save_latest_model_bundle=False,
    )

    pred_df, diag = walk_forward_train_predict(
        frame,
        model_factory=build_model_factory(model_type),
        cfg=cfg,
        time_col="row_id",
    )
    pred_df = pred_df.join(
        frame.set_index("row_id")[["timestamp", "option_symbol", "underlying_symbol", "option_type", "days_to_expiry"]]
    )

    proba_col = "proba_cal" if "proba_cal" in pred_df.columns else "proba_raw"
    auc = None
    ll = None
    if not pred_df.empty and pred_df["y_true"].nunique() >= 2:
        auc = float(roc_auc_score(pred_df["y_true"], pred_df[proba_col]))
        ll = float(log_loss(pred_df["y_true"], pred_df[proba_col], labels=[0, 1]))

    summary = {
        "csv_path": csv_path,
        "rows_raw": int(len(raw)),
        "rows_model": int(len(frame)),
        "underlyings": sorted(frame["underlying_symbol"].dropna().unique().tolist()),
        "contracts": int(frame["option_symbol"].nunique()),
        "model_type": model_type,
        "horizon_bars": int(horizon_bars),
        "target_return_pct": float(target_return_pct),
        "max_adverse_return_pct": float(max_adverse_return_pct),
        "oos_rows": int(len(pred_df)),
        "proba_col": proba_col,
        "oos_auc": auc,
        "oos_logloss": ll,
        "oos_positive_rate": float(pred_df["y_true"].mean()) if not pred_df.empty else None,
        "top_contract_summary": summarize_top_contracts(pred_df, proba_col=proba_col, top_k=top_k),
        "fold_count": int(len(diag.get("fold_diags", []))),
    }

    trades_df = simulate_option_selection_trades(
        frame=frame,
        pred_df=pred_df,
        proba_col=proba_col,
        horizon_bars=horizon_bars,
        target_return_pct=target_return_pct,
        max_adverse_return_pct=max_adverse_return_pct,
        min_entry_proba=min_entry_proba,
        top_k=top_k,
        allow_overlapping_positions=allow_overlapping_positions,
    )
    summary["selection_rules"] = {
        "min_entry_proba": float(min_entry_proba),
        "top_k": int(top_k),
        "allow_overlapping_positions": bool(allow_overlapping_positions),
        "horizon_bars": int(horizon_bars),
        "target_return_pct": float(target_return_pct),
        "max_adverse_return_pct": float(max_adverse_return_pct),
    }
    summary["simulated_trade_summary"] = summarize_simulated_trades(trades_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(trades_output_path).parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(trades_output_path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward ML research on option-chain snapshots.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--model-type", choices=["logistic", "random_forest", "hist_gbm"], default="random_forest")
    parser.add_argument("--train-size", type=int, default=320)
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--step-size", type=int, default=60)
    parser.add_argument("--horizon-bars", type=int, default=8)
    parser.add_argument("--target-return-pct", type=float, default=0.25)
    parser.add_argument("--max-adverse-return-pct", type=float, default=-0.20)
    parser.add_argument("--min-entry-proba", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--allow-overlapping-positions", action="store_true")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--trades-output-path", default=DEFAULT_TRADES_OUTPUT_PATH)
    args = parser.parse_args()

    summary = run_options_research(
        csv_path=args.csv_path,
        model_type=args.model_type,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        horizon_bars=args.horizon_bars,
        target_return_pct=args.target_return_pct,
        max_adverse_return_pct=args.max_adverse_return_pct,
        min_entry_proba=args.min_entry_proba,
        top_k=args.top_k,
        allow_overlapping_positions=args.allow_overlapping_positions,
        output_path=args.output_path,
        trades_output_path=args.trades_output_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
