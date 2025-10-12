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
from drv.data import save_df  # reuse your existing writer

REQUIRED_COLS = {"commit_id", "module", "label", "commit_time"}


def _read_and_adapt(commits_path: str, module_col: str, time_col: str) -> pd.DataFrame:
    """Read raw CSV and adapt to expected schema (no hidden normalization)."""
    df = pd.read_csv(commits_path)
    if module_col not in df.columns:
        raise KeyError(f"Missing '{module_col}' in CSV columns: {list(df.columns)}")
    if time_col not in df.columns:
        raise KeyError(f"Missing '{time_col}' in CSV columns: {list(df.columns)}")

    x = df.copy()
    x["module"] = x[module_col]

    # Parse time: seconds / ms / datetime
    tc = x[time_col]
    if np.issubdtype(tc.dtype, np.datetime64):
        x["commit_time"] = tc
    else:
        sn = pd.to_numeric(tc, errors="coerce")
        if sn.dropna().median() > 1e12:  # ms
            sn = (sn / 1000.0).round()
        x["commit_time"] = pd.to_datetime(sn, unit="s", errors="coerce")
        # fallback parse if still NaT
        nat_mask = x["commit_time"].isna()
        if nat_mask.any():
            x.loc[nat_mask, "commit_time"] = pd.to_datetime(tc[nat_mask], errors="coerce")

    if "commit_id" not in x.columns:
        x["commit_id"] = (
            x["module"].astype(str) + "_" +
            x["commit_time"].astype(str) + "_" +
            x.reset_index().index.astype(str)
        )

    if "label" not in x.columns:
        if "buggy" in x.columns:
            x["label"] = x["buggy"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(0).astype(int)
        else:
            raise KeyError("Neither 'label' nor 'buggy' present; need one for training/eval.")

    x = x.sort_values(["module", "commit_time", "commit_id"]).reset_index(drop=True)
    return x


def _validate(df: pd.DataFrame):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def main():
    ap = argparse.ArgumentParser(description="DRV: Defect-Risk Visualization CLI (explicit mapping)")
    ap.add_argument("--commits", type=str, default="data/apachejit_total.csv", help="Path to commit-level CSV")
    ap.add_argument("--mode", type=str, choices=["train", "score", "visualize"], default="train")
    ap.add_argument("--output", type=str, default="outputs", help="Output directory")

    # mapping/visual controls
    ap.add_argument("--module_col", default="project", help="Column holding module/project")
    ap.add_argument("--time_col", default="author_date", help="Column with commit timestamp (epoch secs or datetime)")
    ap.add_argument("--heatmap_value", choices=["risk_score", "buggy"], default="risk_score")
    ap.add_argument("--heatmap_freq", choices=["W", "M", "Q", "Y"], default="M")
    ap.add_argument("--heatmap_topn", type=int, default=25)
    ap.add_argument("--trend_window", default="30D")

    args = ap.parse_args()

    cfg = DRVConfig()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read raw + adapt to schema
    df = _read_and_adapt(args.commits, args.module_col, args.time_col)
    _validate(df)

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

        mod = (
            out.groupby("module", as_index=False)["risk_score"]
              .mean().sort_values("risk_score", ascending=False)
        )
        save_df(mod, str(output_dir / "module_risk.csv"))

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
        out = df.copy(); out["risk_score"] = proba_f
        save_df(out, str(output_dir / "scored_commits.csv"))
        print(f"Scored {len(out)} rows → {output_dir / 'scored_commits.csv'}")

    elif args.mode == "visualize":
        scored_path = output_dir / "scored_commits.csv"
        if not scored_path.exists():
            # score on the fly using saved models
            models = load_models(in_dir=str(output_dir))
            _, proba_f = score_dataframe(df, models)
            out = df.copy(); out["risk_score"] = proba_f
        else:
            # merge risks back onto raw to ensure commit_time/module exist
            scored = pd.read_csv(scored_path)
            if "risk_score" not in scored.columns or "commit_id" not in scored.columns:
                raise ValueError(f"{scored_path} must contain 'commit_id' and 'risk_score'.")
            base = _read_and_adapt(args.commits, args.module_col, args.time_col)
            out = base.merge(scored[["commit_id", "risk_score"]], on="commit_id", how="inner")
            if out.empty:
                raise ValueError("Merge produced no rows. Mismatched commit_id between raw and scored.")

        _validate(out)
        out = out.sort_values(["module", "commit_time", "commit_id"])

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
