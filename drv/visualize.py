import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def risk_heatmap(df_commits: pd.DataFrame, value_col: str = "risk_score", out_path: str | None = None):
    """
    Plot a heatmap: rows=modules, cols=commits (chronological),
    cell value = df[value_col] (e.g., 'risk_score').
    """
    df = df_commits.copy()

    # Ensure chronological ordering per module if commit_time exists
    if "commit_time" in df.columns:
        df = df.sort_values(by=["module", "commit_time", "commit_id"])
    else:
        df = df.sort_values(by=["module", "commit_id"])

    # Pivot expects a *column name* for values
    pivot = df.pivot_table(index="module", columns="commit_id", values=value_col, aggfunc="mean")
    pivot = pivot.fillna(0.0)

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=90)
    plt.yticks(range(pivot.shape[0]), pivot.index)
    plt.title("DRV: Module × Commit Risk Heatmap")
    plt.xlabel("Commits")
    plt.ylabel("Modules")
    plt.colorbar(label="Risk")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()

def topk_bar(df_commits: pd.DataFrame, risk_scores: np.ndarray, k: int = 10, out_path: str | None = None):
    order = np.argsort(-risk_scores)[:k]
    subset = df_commits.iloc[order][["commit_id", "module"]].copy()
    values = risk_scores[order]

    plt.figure(figsize=(8, 5))
    plt.bar(range(k), values)

    labels = [f"{r.commit_id}\n{r.module[:18]}" for _, r in subset.iterrows()]
    plt.xticks(range(k), labels, rotation=45, ha="right")
    plt.ylabel("Risk")
    plt.title(f"Top-{k} Risky Commits/Modules")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
