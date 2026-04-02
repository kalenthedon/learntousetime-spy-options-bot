from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.backtests.run_options_research import (
    apply_selection_orientation,
    build_model_factory,
    prepare_option_research_frame,
    load_option_chain_csv,
    resolve_selection_orientation,
)
from src.ml.walkforward_train import WalkForwardRunConfig, walk_forward_train_predict


DEFAULT_HISTORY_PATH = "centralized_data/options/SPY_put_chain_history.csv"
DEFAULT_OUTPUT_PATH = "experiments/options_backfill_candidate_summary.json"
DEFAULT_JOURNAL_PATH = "experiments/options_backfill_candidate_journal.csv"


def _prepare_research_frame(
    history_path: str,
    horizon_bars: int,
    target_return_pct: float,
    max_adverse_return_pct: float,
) -> tuple[pd.DataFrame, list[str]]:
    raw = load_option_chain_csv(history_path)
    frame, feature_cols = prepare_option_research_frame(
        raw,
        horizon_bars=horizon_bars,
        target_return_pct=target_return_pct,
        max_adverse_return_pct=max_adverse_return_pct,
    )
    frame = frame.reset_index(drop=True)
    if "collected_at" in frame.columns:
        frame["snapshot_time"] = pd.to_datetime(frame["collected_at"], utc=True, errors="coerce")
    else:
        frame["snapshot_time"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.floor("15min")
    frame = frame.dropna(subset=["snapshot_time"]).sort_values(
        ["snapshot_time", "timestamp", "expiration", "strike", "option_type", "option_symbol"]
    ).reset_index(drop=True)
    frame["row_id"] = range(len(frame))
    frame["snapshot_id"] = pd.factorize(frame["snapshot_time"], sort=False)[0]
    frame["selection_time"] = frame["snapshot_time"].astype(str)
    return frame, feature_cols


def _records_for_json(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].astype(str)
    return json.loads(safe.to_json(orient="records"))


def _longest_streak(values: list[bool], target: bool) -> int:
    best = 0
    current = 0
    for value in values:
        if bool(value) is bool(target):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def build_backfill_candidate_journal(
    history_path: str = DEFAULT_HISTORY_PATH,
    model_type: str = "logistic",
    train_size: int = 48,
    test_size: int = 12,
    step_size: int = 12,
    horizon_bars: int = 8,
    target_return_pct: float = 0.25,
    max_adverse_return_pct: float = -0.20,
    min_entry_score: float = 0.30,
    top_k: int = 3,
    selection_orientation: str = "auto",
    output_path: str = DEFAULT_OUTPUT_PATH,
    journal_path: str = DEFAULT_JOURNAL_PATH,
) -> dict:
    frame, feature_cols = _prepare_research_frame(
        history_path=history_path,
        horizon_bars=horizon_bars,
        target_return_pct=target_return_pct,
        max_adverse_return_pct=max_adverse_return_pct,
    )

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
        min_train_rows=150,
        min_test_rows=20,
    )
    pred_df, diag = walk_forward_train_predict(
        frame,
        model_factory=build_model_factory(model_type),
        cfg=cfg,
        time_col="row_id",
        split_group_col="snapshot_id",
    )
    pred_df = pred_df.join(
        frame.set_index("row_id")[
            [
                "timestamp",
                "snapshot_time",
                "selection_time",
                "option_symbol",
                "underlying_symbol",
                "strike",
                "days_to_expiry",
                "mid",
                "delta",
                "open_interest",
                "volume",
                "candidate_score",
            ]
        ]
    )
    proba_col = "proba_cal" if "proba_cal" in pred_df.columns else "proba_raw"
    resolved_orientation = resolve_selection_orientation(
        selection_orientation,
        float(diag.get("oos_auc")) if diag.get("oos_auc") is not None else None,
        float(diag.get("oos_auc_inverted")) if diag.get("oos_auc_inverted") is not None else None,
    )
    pred_df, score_col = apply_selection_orientation(pred_df, proba_col=proba_col, orientation=resolved_orientation)
    pred_df = pred_df.sort_values(["selection_time", score_col, "candidate_score"], ascending=[True, False, False]).copy()
    pred_df["rank"] = pred_df.groupby("selection_time")[score_col].rank(method="first", ascending=False)
    pred_df["selected"] = (pred_df["rank"] <= int(top_k)) & (pred_df[score_col] >= float(min_entry_score))

    rows = []
    for selection_time, group in pred_df.groupby("selection_time", sort=True):
        group = group.sort_values(["rank", score_col], ascending=[True, False]).copy()
        selected = group.loc[group["selected"]].copy()
        decision = "select_contract" if not selected.empty else "no_trade"
        top_row = selected.iloc[0] if not selected.empty else group.iloc[0]
        rows.append(
            {
                "selection_time": str(selection_time),
                "decision": decision,
                "selected_count": int(len(selected)),
                "selected_option_symbol": str(top_row["option_symbol"]) if decision == "select_contract" else "",
                "selected_selection_score": float(top_row[score_col]) if decision == "select_contract" else "",
                "selected_mid": float(top_row["mid"]) if decision == "select_contract" else "",
                "selected_delta": float(top_row["delta"]) if decision == "select_contract" else "",
                "top_option_symbol": str(group.iloc[0]["option_symbol"]),
                "top_selection_score": float(group.iloc[0][score_col]),
                "top_rank_mid": float(group.iloc[0]["mid"]),
                "top_rank_delta": float(group.iloc[0]["delta"]),
                "positive_rate_top_rank": float(group.iloc[0]["y_true"]),
            }
        )

    journal_df = pd.DataFrame(rows)
    if journal_df.empty:
        raise RuntimeError("No backfill candidate journal rows were produced")

    select_df = journal_df.loc[journal_df["decision"] == "select_contract"].copy()
    decisions = journal_df["decision"].eq("select_contract").tolist()
    summary = {
        "history_path": history_path,
        "model_type": model_type,
        "selection_orientation": resolved_orientation,
        "proba_col": proba_col,
        "selection_score_col": "selection_score",
        "min_entry_score": float(min_entry_score),
        "top_k": int(top_k),
        "oos_auc": float(diag.get("oos_auc")) if diag.get("oos_auc") is not None else None,
        "oos_auc_inverted": float(diag.get("oos_auc_inverted")) if diag.get("oos_auc_inverted") is not None else None,
        "fold_count": int(len(diag.get("fold_diags", []))),
        "snapshot_count": int(frame["snapshot_id"].nunique()),
        "journal_rows": int(len(journal_df)),
        "select_count": int(len(select_df)),
        "no_trade_count": int((journal_df["decision"] == "no_trade").sum()),
        "select_rate": float((journal_df["decision"] == "select_contract").mean()),
        "avg_selected_count_when_trading": float(select_df["selected_count"].astype(float).mean()) if not select_df.empty else None,
        "avg_top_selection_score": float(journal_df["top_selection_score"].mean()),
        "avg_selected_score": float(select_df["selected_selection_score"].astype(float).mean()) if not select_df.empty else None,
        "longest_select_streak": _longest_streak(decisions, True),
        "longest_no_trade_streak": _longest_streak(decisions, False),
        "top_selected_symbols": (
            select_df["selected_option_symbol"].value_counts().rename_axis("option_symbol").reset_index(name="count").head(10).to_dict(orient="records")
            if not select_df.empty
            else []
        ),
        "journal_preview": _records_for_json(journal_df.head(10)),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    Path(journal_path).parent.mkdir(parents=True, exist_ok=True)
    journal_df.to_csv(journal_path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill the SPY options selector across historical snapshots.")
    parser.add_argument("--history-path", default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--model-type", choices=["logistic", "random_forest", "hist_gbm"], default="logistic")
    parser.add_argument("--train-size", type=int, default=48)
    parser.add_argument("--test-size", type=int, default=12)
    parser.add_argument("--step-size", type=int, default=12)
    parser.add_argument("--horizon-bars", type=int, default=8)
    parser.add_argument("--target-return-pct", type=float, default=0.25)
    parser.add_argument("--max-adverse-return-pct", type=float, default=-0.20)
    parser.add_argument("--min-entry-score", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--selection-orientation", choices=["auto", "raw", "inverted"], default="auto")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--journal-path", default=DEFAULT_JOURNAL_PATH)
    args = parser.parse_args()

    summary = build_backfill_candidate_journal(
        history_path=args.history_path,
        model_type=args.model_type,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        horizon_bars=args.horizon_bars,
        target_return_pct=args.target_return_pct,
        max_adverse_return_pct=args.max_adverse_return_pct,
        min_entry_score=args.min_entry_score,
        top_k=args.top_k,
        selection_orientation=args.selection_orientation,
        output_path=args.output_path,
        journal_path=args.journal_path,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
