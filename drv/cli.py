import argparse, json
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

# --- dual import header: makes script runnable directly ---
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# ----------------------------------------------------------

from drv.visualize import risk_heatmap_binned, topk_bar, risk_trend
from drv.config import DRVConfig
from drv.data import load_commits, save_df
from drv.models import train_baselines, score_dataframe, save_models, load_models
from drv.ranking import evaluate_ranking
from drv.eval import threshold_scores, f1_auc, report

REQUIRED_COLS = {"commit_id", "module", "label"}

def _validate_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    # ensure required columns exist (after normalization in load_commits)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {sorted(missing)}. "
            f"Got: {sorted(df.columns.tolist())}"
        )

    df = df.copy()
    df["commit_id"] = df["commit_id"].astype(str)

    # stable order for visuals
    if "commit_time" in df.columns:
        # commit_time can be int (unix seconds) or pandas datetime; sorting works for both
        df = df.sort_values(["module", "commit_time", "commit_id"])
    else:
        df = df.sort_values(["module", "commit_id"])
    return df


def main():
    ap = argparse.ArgumentParser(description="DRV: Defect-Risk Visualization CLI")
    ap.add_argument("--commits", type=str, default="data/commits.csv",
                    help="Path to commit-level CSV")
    ap.add_argument("--mode", type=str, choices=["train", "score", "visualize"], default="train")
    ap.add_argument("--output", type=str, default="outputs",
                    help="Directory to store outputs")

    # ---- NEW: Heatmap controls ----
    ap.add_argument("--heatmap_value", choices=["risk_score", "buggy"], default="risk_score",
                    help="What to visualize as color: model risk_score (default) or ground-truth buggy rate")
    ap.add_argument("--heatmap_freq", choices=["W", "M", "Q", "Y"], default="M",
                    help="Time bucket for X-axis (Week/Month/Quarter/Year)")
    ap.add_argument("--heatmap_topn", type=int, default=25,
                    help="Show only the top-N hottest modules by mean value")
    # --------------------------------

    args = ap.parse_args()

    cfg = DRVConfig()
    commits_path = Path(args.commits).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_commits(str(commits_path))
    df = _validate_and_sort(df)

    if args.mode == "train":
        # Train baselines and evaluate
        models, metrics, _ = train_baselines(df, seed=cfg.seed)
        (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("Training metrics:", metrics)

        save_models(models, out_dir=str(output_dir))

        # Score and ranking eval
        _, proba_f = score_dataframe(df, models)
        rank_eval = evaluate_ranking(df["label"].values, proba_f, ks=(5, 10, 20))
        (output_dir / "ranking_eval.json").write_text(json.dumps(rank_eval, indent=2), encoding="utf-8")
        print("Ranking evaluation:", rank_eval)

        # Persist scored commits
        out = df.copy()
        out["risk_score"] = proba_f
        save_df(out, str(output_dir / "scored_commits.csv"))

        # Module-level aggregation (useful for dashboards)
        mod = (
            out.groupby("module", as_index=False)["risk_score"]
              .mean()
              .sort_values("risk_score", ascending=False)
        )
        save_df(mod, str(output_dir / "module_risk.csv"))

        # --- Visuals ---
        # New, readable heatmap (time-binned, top-N modules)
        risk_heatmap_binned(
            out,
            value_col=args.heatmap_value,        # "risk_score" or "buggy"
            time_col="commit_time",
            module_col="module",
            topn=args.heatmap_topn,
            freq=args.heatmap_freq,
            out_path=str(output_dir / "heatmap.png"),
        )

        # Keep the other plots (they assume risk_score is present)
        topk_bar(out, out["risk_score"].values, k=10, out_path=str(output_dir / "topk.png"))
        risk_trend(out, value_col="risk_score", window="7D", out_path=str(output_dir / "trend.png"))

    elif args.mode == "visualize":
        # Visualize from previously scored data
        scored_path = output_dir / "scored_commits.csv"
        if not scored_path.exists():
            raise FileNotFoundError(f"{scored_path} not found. Run with --mode train first.")
        out = pd.read_csv(scored_path)
        out = _validate_and_sort(out)

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
        risk_trend(out, value_col="risk_score", window="7D", out_path=str(output_dir / "trend.png"))
        print(f"Saved plots to {output_dir}")

    elif args.mode == "score":
        models = load_models(in_dir=str(output_dir))
        _, proba_f = score_dataframe(df, models)
        out = df.copy()
        out["risk_score"] = proba_f
        save_df(out, str(output_dir / "scored_commits.csv"))
        print(f"Scored {len(out)} rows → {output_dir / 'scored_commits.csv'}")

if __name__ == "__main__":
    main()
