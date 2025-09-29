
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def risk_heatmap(df_commits: pd.DataFrame, risk_scores: np.ndarray, out_path: str=None):
    """Plot a heatmap: rows=modules, cols=commits (chronological), values=risk score (0-1)."""
    # pivot to modules x commits (fill missing with nan, then 0)
    df = df_commits.copy()
    df = df.sort_values(by=["module","commit_time"])
    pivot = df.pivot_table(index="module", columns="commit_id", values=risk_scores, aggfunc="mean")
    pivot = pivot.fillna(0.0)
    plt.figure(figsize=(10,6))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=90)
    plt.yticks(range(pivot.shape[0]), pivot.index)
    plt.title("DRV: Module x Commit Risk Heatmap")
    plt.xlabel("Commits")
    plt.ylabel("Modules")
    plt.colorbar(label="Risk")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()

def topk_bar(df_commits: pd.DataFrame, risk_scores: np.ndarray, k:int=10, out_path:str=None):
    order = np.argsort(-risk_scores)[:k]
    subset = df_commits.iloc[order][["commit_id","module"]].copy()
    values = risk_scores[order]
    plt.figure(figsize=(8,5))
    plt.bar(range(k), values)
    plt.xticks(range(k), [f"{r.commit_id}\n{r.module}" for _, r in subset.iterrows()], rotation=45, ha="right")
    plt.ylabel("Risk")
    plt.title(f"Top-{k} Risky Commits/Modules")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
