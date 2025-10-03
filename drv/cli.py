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

from drv.visualize import risk_heatmap, topk_bar, risk_trend
from drv.config import DRVConfig
from drv.data import load_commits, save_df
from drv.models import train_baselines, score_dataframe, save_models, load_models
from drv.ranking import evaluate_ranking
from drv.eval import threshold_scores, f1_auc, report

REQUIRED_COLS = {"commit_id", "module", "message", "label"}

def _validate_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["commit_id"] = df["commit_id"].astype(str)
    if "commit_time" in df.columns:
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
    args = ap.parse_args()

    cfg = DRVConfig()
    commits_path = Path(args.commits).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_commits(str(commits_path))
    df = _validate_and_sort(df)

    if args.mode == "train":
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

        # module-level aggregation
        mod = out.groupby("module", as_index=False)["risk_score"].mean().sort_values("risk_score", ascending=False)
        save_df(mod, str(output_dir / "module_risk.csv"))

        # plots
        risk_heatmap(out, value_col="risk_score", out_path=str(output_dir / "heatmap.png"))
        topk_bar(out, out["risk_score"].values, k=10, out_path=str(output_dir / "topk.png"))
        risk_trend(out, value_col="risk_score", window="7D", out_path=str(output_dir / "trend.png"))

    elif args.mode == "visualize":
        scored_path = output_dir / "scored_commits.csv"
        if not scored_path.exists():
            raise FileNotFoundError(f"{scored_path} not found. Run with --mode train first.")
        out = pd.read_csv(scored_path)
        out = _validate_and_sort(out)
        risk_heatmap(out, value_col="risk_score", out_path=str(output_dir / "heatmap.png"))
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
