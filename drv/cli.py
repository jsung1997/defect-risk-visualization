
import argparse, json
import pandas as pd
import numpy as np
from pathlib import Path

from .config import DRVConfig
from .data import load_commits, save_df
from .models import train_baselines, score_dataframe
from .ranking import evaluate_ranking
from .eval import threshold_scores, f1_auc, report
from .visualize import risk_heatmap, topk_bar

def main():
    ap = argparse.ArgumentParser(description="DRV: Defect-Risk Visualization CLI")
    ap.add_argument("--commits", type=str, default="data/commits.csv", help="Path to commit-level CSV")
    ap.add_argument("--mode", type=str, choices=["train","score","visualize"], default="train")
    ap.add_argument("--output", type=str, default="outputs")
    args = ap.parse_args()

    cfg = DRVConfig()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    df = load_commits(args.commits)

    if args.mode == "train":
        models, metrics, feats = train_baselines(df, seed=cfg.seed)
        with open(f"{args.output}/train_metrics.json","w") as f:
            json.dump(metrics, f, indent=2)
        print("Training metrics:", metrics)

        # Score whole df with fused model for a quick ranking eval example
        proba_m, proba_f = score_dataframe(df, models)
        rank_eval = evaluate_ranking(df["label"].values, proba_f, ks=(5,10,20))
        with open(f"{args.output}/ranking_eval.json","w") as f:
            json.dump(rank_eval, f, indent=2)
        print("Ranking evaluation:", rank_eval)

        # Save scores
        out = df.copy()
        out["risk_score"] = proba_f
        save_df(out, f"{args.output}/scored_commits.csv")

        # Plots
        try:
            risk_heatmap(out, out["risk_score"].values, out_path=f"{args.output}/heatmap.png")
            topk_bar(out, out["risk_score"].values, k=10, out_path=f"{args.output}/topk.png")
        except Exception as e:
            print("Plotting failed:", e)

    elif args.mode == "visualize":
        # assumes scored_commits.csv exists
        out = pd.read_csv(f"{args.output}/scored_commits.csv")
        risk_heatmap(out, out["risk_score"].values, out_path=f"{args.output}/heatmap.png")
        topk_bar(out, out["risk_score"].values, k=10, out_path=f"{args.output}/topk.png")

    elif args.mode == "score":
        print("Use 'train' mode to fit models and generate scores.")

if __name__ == "__main__":
    main()
