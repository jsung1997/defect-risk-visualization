# drv/cli.py
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import os, sys

# --- dual import header ---
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# --------------------------

from drv.visualize import risk_heatmap_binned, topk_bar, risk_trend
from drv.config import DRVConfig
from drv.models import train_baselines, score_dataframe, save_models, load_models
from drv.ranking import evaluate_ranking
from drv.data import save_df  # keep your existing writer

# We will NOT use load_commits() here; it keeps collapsing module/time.

REQUIRED_COLS = {"commit_id", "module", "label", "commit_time"}

def _read_and_adapt(commits_path: str, module_col: str, time_col: str) -> pd.DataFrame:
    """Read raw CSV and adapt to expected schema without hidden normalization."""
    df = pd.read_csv(commits_path)

    # Map module/time explicitly
    if module_col not in df.columns:
        raise KeyError(f"Missing '{module_col}' in CSV columns: {list(df.columns)}")
    if time_col not in df.columns:
        raise KeyError(f"Missing '{time_col}' in CSV columns: {list(df.columns)}")

    df = df.copy()
    df["module"] = df[module_col]

    # commit_time: accept seconds epoch or already-datetime
    if np.issubdtype(df[time_col].dtype, np.datetime64):
        df["commit_time"] = df[time_col]
    else:
        # force seconds → datetime; if it looks like ms, divide by 1000
        s = pd.to_numeric(df[time_col], errors="coerce")
        # heuristic: big values → ms
        if s.dropna().median() > 1e12:
            s = (s / 1000.0).round()
        df["commit_time"] = pd.to_datetime(s, unit="s", errors="coerce")

    # commit_id
    if "commit_id" not in df.columns:
        # synthesize a stable id if missing
        df["commit_id"] = (
            df["module"].astype(str) + "_" +
            df["commit_time"].astype(str) + "_" +
            df.reset_index().index.astype(str)
        )

    # label from buggy TRUE/FALSE if missing
    if "label" not in df.columns:
        if "buggy" in df.columns:
            df["label"] = (
                df["buggy"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(0).astype(int)
            )
        else:
            raise KeyError("Neither 'label' nor 'buggy' present; need one for training.")

    # sanity checks
    if df["module"].nunique() <= 1:
        print("[WARN] Only one module detected. Check your 'project' column has variety.")
    if df["commit_time"].isna().all():
        raise ValueError("All commit_time values are NaT. Check your time column and epoch units.")

    # order
    df = df.sort_values(["module", "commit_time", "commit_id"]).reset_index(drop=True)
    return df

def _validate(df: pd.DataFrame):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

def main():
    ap = argparse.ArgumentParser(description="DRV: Defect-Risk Visualization CLI (explicit mapping)")
    ap.add_argument("--commits", type=str, default="data/apachejit_total.csv",
                    help="Path to commit-level CSV")
    ap.add_argument("--mode", type=str, choices=["train", "score", "visualize"], default="train")
    ap.add_argument("--output", type=str, default="outputs", help="Output directory")

    # mapping/visual controls
    ap.add_argument("--module_col", default="project", help="Column that holds module/project")
    ap.add_argument("--time_col", default="author_date", help="Column with commit timestamp (epoch seconds)")
    ap.add_argument("--heatmap_value", choices=["risk_score", "buggy"], default="risk_score")
    ap.add_argument("--heatmap_freq", choices=["W", "M", "Q", "Y"], default="M")
    ap.add_argument("--heatmap_topn", type=int, default=25)
    ap.add_argument("--trend_window", default="30D")

    args = ap.parse_args()

    cfg = DRVConfig()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read + adapt
    df = _read_and_adapt(args.commits, args.module_col, args.time_col)
    _validate(df)

    if args.mode == "train":
        # Train baseline, score, evaluate
        models, metrics, _ = train_baselines(df, seed=cfg.seed)
        (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("Training metrics:", metrics)

        save_models(models, out_dir=str(output_dir))

        _, proba_f = score_dataframe(df, models)
        rank_eval = evaluate_ranking(df["label"].values, proba_f, ks=(5, 10, 20))
        (output_dir / "ranking_eval.json").write_text(json.dumps(rank_eval, indent=2), encoding="utf-8")
        print("Ranking evaluation:", rank_eval)

        out = df.copy()
        out["risk_score"] = proba_f
        save_df(out, str(output_dir / "scored_commits.csv"))

        mod = (
            out.groupby("module", as_index=False)["risk_score"]
              .mean()
              .sort_values("risk_score", ascending=False)
        )
        save_df(mod, str(output_dir / "module_risk.csv"))

        # Visuals
        risk_heatmap_binned(
            out,
            value_col=args.heatmap_value,
            time_col="commit_time",
            module_col="module",
            topn=args.heatmap_topn,
            freq=args.heatmap_freq,
            out_path=str(output_dir / "heatmap.png"),
        )
        topk_bar(out, out["risk_score"].values, k=10, out_path=str(output_dir / "topk.png"))
        risk_trend(out, value_col="risk_score", window=args.trend_window, out_path=str(output_dir / "trend.png"))

    elif args.mode == "score":
        models = load_models(in_dir=str(output_dir))
        _, proba_f = score_dataframe(df, models)
        out = df.copy()
        out["risk_score"] = proba_f
        save_df(out, str(output_dir / "scored_commits.csv"))
        print(f"Scored {len(out)} rows → {output_dir / 'scored_commits.csv'}")

    elif args.mode == "visualize":
        scored_path = output_dir / "scored_commits.csv"
        if scored_path.exists():
            out = pd.read_csv(scored_path)
            # re-enforce schema if old file
            if "module" not in out or "commit_time" not in out:
                out = _read_and_adapt(str(scored_path), args.module_col, args.time_col)
        else:
            # score on the fly with saved models
            models = load_models(in_dir=str(output_dir))
            _, proba_f = score_dataframe(df, models)
            out = df.copy()
            out["risk_score"] = proba_f

        _validate(out)

        risk_heatmap_binned(
            out,
            value_col=args.heatmap_value,
            time_col="commit_time",
            module_col="module",
            topn=args.heatmap_topn,
            freq=args.heatmap_freq,
            out_path=str(output_dir / "heatmap.png"),
        )
        topk_bar(out, out["risk_score"].values, k=10, out_path=str(output_dir / "topk.png"))
        risk_trend(out, value_col="risk_score", window=args.trend_window, out_path=str(output_dir / "trend.png"))
        print(f"Saved plots to {output_dir}")

if __name__ == "__main__":
    main()
