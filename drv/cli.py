import argparse, json
import pandas as pd
import numpy as np
from pathlib import Path

from .config import DRVConfig
from .data import load_commits, save_df
from .models import train_baselines, score_dataframe, save_models, load_models
from .ranking import evaluate_ranking
from .eval import threshold_scores, f1_auc, report
from .visualize import risk_heatmap, topk_bar

REQUIRED_COLS = {"commit_id", "module", "message", "label"}

def _validate_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # enforce string for IDs & stable deterministic order
    df = df.copy()
    df["commit_id"] = df["commit_id"].astype(str)
    if "commit_time" in df.columns:
        df = df.sort_values(["module", "commit_time", "commit_id"])
    else:
        df = df.sort_values(["module", "commit_id"])
    return df

def main():
    ap = argparse.ArgumentParser(description="DRV: Defect-Risk Visualization CLI")
    ap.add_argument("--commits", type=str, default="data/commits.csv", help="Path to commit-level CSV")
    ap.add_argument("--mode", type=str, choices=["train", "score", "visualize"], default="train")
    ap.add_argument("--output", type=str, default="outputs")
    args = ap.parse_args()

    cfg = DRVConfig()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    df = load_commits(args.commits)
    df = _validate_and_sort(df)

    if args.mode == "train":
        models, metrics, feats = train_baselines(df, seed=cfg.seed)
        with open(f"{args.output}/train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("Training metrics:", metrics)

        # Persist models for later scoring
        save_models(models, out_dir=args.output)

        # Score whole df with fused model for a quick ranking eval example
        proba_m, proba_f = score_dataframe(df, models)
        rank_eval = evaluate_ranking(df["label"].values, proba_f, ks=(5, 10, 20))
        with open(f"{args.output}/ranking_eval.json", "w") as f:
            json.dump(rank_eval, f, indent=2)
        print("Ranking evaluation:", rank_eval)

        # Save scores
        out = df.copy()
        out["risk_score"] = proba_f
        save_df(out, f"{args.output}/scored_commits.csv")

        # Module-level aggregation (optional but useful)
        mod = out.groupby("module", as_index=False)["risk_score"].mean().sort_values("risk_score", ascending=False)
        save_df(mod, f"{args.output}/module_risk.csv")

        # Plots
        try:
            risk_heatmap(out, value_col="risk_score", out_path=f"{args.output}/heatmap.png")
            topk_bar(out, out["risk_score"].values, k=10, out_path=f"{args.output}/topk.png")
        except Exception as e:
            print("Plotting failed:", e)

    elif args.mode == "visualize":
        # assumes scored_commits.csv exists
        out = pd.read_csv(f"{args.output}/scored_commits.csv")
        out = _validate_and_sort(out)
        risk_heatmap(out, value_col="risk_score", out_path=f"{args.output}/heatmap.png")
        topk_bar(out, out["risk_score"].values, k=10, out_path=f"{args.output}/topk.png")
        print(f"Saved plots to {args.output}")

    elif args.mode == "score":
        # score a new CSV using previously saved models
        models = load_models(in_dir=args.output)
        proba_m, proba_f = score_dataframe(df, models)
        out = df.copy()
        out["risk_score"] = proba_f
        save_df(out, f"{args.output}/scored_commits.csv")
        print(f"Scored {len(out)} rows → {args.output}/scored_commits.csv")

if __name__ == "__main__":
    main()
