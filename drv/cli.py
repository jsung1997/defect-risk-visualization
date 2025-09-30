import argparse, json
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Dual-import header: allows both `python -m drv.cli` and `python drv/cli.py` ---
if __package__ in (None, ""):
    # repo_root = folder that contains the "drv" package (this file lives in drv/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from drv.config import DRVConfig
    from drv.data import load_commits, save_df
    from drv.models import train_baselines, score_dataframe, save_models, load_models
    from drv.ranking import evaluate_ranking
    from drv.eval import threshold_scores, f1_auc, report
    from drv.visualize import risk_heatmap, topk_bar
else:
    from .config import DRVConfig
    from .data import load_commits, save_df
    from .models import train_baselines, score_dataframe, save_models, load_models
    from .ranking import evaluate_ranking
    from .eval import threshold_scores, f1_auc, report
    from .visualize import risk_heatmap, topk_bar

REQUIRED_COLS = {"commit_id", "module", "message", "label"}

def _repo_root() -> Path:
    """Return repository root (parent of this cli.py's folder)."""
    return Path(__file__).resolve().parent.parent

def _abs_from_root(p: str | Path) -> Path:
    """If path is relative, interpret it relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else (_repo_root() / p)

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
                    help="Path to commit-level CSV (relative to repo root or absolute)")
    ap.add_argument("--mode", type=str, choices=["train", "score", "visualize"], default="train")
    ap.add_argument("--output", type=str, default="outputs",
                    help="Output directory (relative to repo root or absolute)")
    args = ap.parse_args()

    # Resolve paths relative to repo root so 'Run' works from anywhere
    commits_path = _abs_from_root(args.commits)
    output_dir = _abs_from_root(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DRVConfig()

    # Load & prep data
    df = load_commits(str(commits_path))
    df = _validate_and_sort(df)

    if args.mode == "train":
        models, metrics, _ = train_baselines(df, seed=cfg.seed)
        (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("Training metrics:", metrics)

        # Persist models
        save_models(models, out_dir=str(output_dir))

        # Score whole df with fused model for ranking eval
        _, proba_f = score_dataframe(df, models)
        rank_eval = evaluate_ranking(df["label"].values, proba_f, ks=(5, 10, 20))
        (output_dir / "ranking_eval.json").write_text(json.dumps(rank_eval, indent=2), encoding="utf-8")
        print("Ranking evaluation:", rank_eval)

        # Save scored commits
        out = df.copy()
        out["risk_score"] = proba_f
        save_df(out, str(output_dir / "scored_commits.csv"))

        # Module-level aggregation
        mod = out.groupby("module", as_index=False)["risk_score"].mean().sort_values("risk_score", ascending=False)
        save_df(mod, str(output_dir / "module_risk.csv"))

        # Plots
        try:
            risk_heatmap(out, value_col="risk_score", out_path=str(output_dir / "heatmap.png"))
            topk_bar(out, out["risk_score"].values, k=10, out_path=str(output_dir / "topk.png"))
        except Exception as e:
            print("Plotting failed:", e)

    elif args.mode == "visualize":
        # assumes scored_commits.csv exists
        scored_path = output_dir / "scored_commits.csv"
        if not scored_path.exists():
            raise FileNotFoundError(f"{scored_path} not found. Run with --mode train or --mode score first.")
        out = pd.read_csv(scored_path)
        out = _validate_and_sort(out)
        risk_heatmap(out, value_col="risk_score", out_path=str(output_dir / "heatmap.png"))
        topk_bar(out, out["risk_score"].values, k=10, out_path=str(output_dir / "topk.png"))
        print(f"Saved plots to {output_dir}")

    elif args.mode == "score":
        # Load saved models and score the provided CSV
        models = load_models(in_dir=str(output_dir))
        _, proba_f = score_dataframe(df, models)
        out = df.copy()
        out["risk_score"] = proba_f
        save_df(out, str(output_dir / "scored_commits.csv"))
        print(f"Scored {len(out)} rows → {output_dir / 'scored_commits.csv'}")

if __name__ == "__main__":
    main()
