import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# --- Heatmap ---
def risk_heatmap(
    df_commits: pd.DataFrame,
    value_col: str = "risk_score",
    out_path: Optional[str] = None,
    *,
    max_xticks: int = 20,
    window: Optional[str] = None,  # e.g., "7D" for weekly
):
    df = df_commits.copy()

    # Ensure ordering
    if "commit_time" in df.columns:
        if np.issubdtype(df["commit_time"].dtype, np.number):
            df["commit_time"] = pd.to_datetime(df["commit_time"], unit="s", errors="coerce")
        else:
            df["commit_time"] = pd.to_datetime(df["commit_time"], errors="coerce")
        df = df.sort_values(["module", "commit_time", "commit_id"])
    else:
        df = df.sort_values(["module", "commit_id"])

    # Aggregate by time window if requested
    if window is not None:
        df["_bucket"] = df["commit_time"].dt.to_period(window).dt.start_time.dt.strftime("%Y-%m-%d")
        col_key = "_bucket"
    else:
        col_key = "commit_id"

    pivot = df.pivot_table(index="module", columns=col_key, values=value_col, aggfunc="mean").fillna(0.0)

    plt.figure(figsize=(14, 7))
    im = plt.imshow(pivot.values, aspect="auto")
    # Limit x-ticks
    cols = list(pivot.columns.astype(str))
    step = max(1, len(cols) // max_xticks)
    idx = list(range(0, len(cols), step))
    if idx[-1] != len(cols) - 1:
        idx.append(len(cols) - 1)
    plt.xticks(idx, [cols[i] for i in idx], rotation=45, ha="right", fontsize=8)

    plt.yticks(range(pivot.shape[0]), pivot.index, fontsize=9)
    plt.title("DRV: Module × Risk Heatmap")
    plt.xlabel("Commits" if window is None else f"Buckets ({window})")
    plt.ylabel("Modules")
    plt.colorbar(im, label="Risk")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# --- Top-K Bar Chart ---
def topk_bar(
    df_commits: pd.DataFrame,
    risk_scores: np.ndarray,
    k: int = 10,
    out_path: Optional[str] = None
):
    order = np.argsort(-risk_scores)[:k]
    subset = df_commits.iloc[order][["commit_id", "module"]].copy()
    values = risk_scores[order]

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(values)), values)

    labels = [f"{r.commit_id}\n{r.module[:18]}" for _, r in subset.iterrows()]
    plt.xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Risk")
    plt.title(f"Top-{k} Risky Commits/Modules")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# --- Trend Line ---
def risk_trend(
    df_commits: pd.DataFrame,
    value_col: str = "risk_score",
    window: str = "7D",
    out_path: Optional[str] = None
):
    if "commit_time" not in df_commits.columns:
        raise ValueError("risk_trend requires 'commit_time' column.")

    df = df_commits.copy()
    if np.issubdtype(df["commit_time"].dtype, np.number):
        df["commit_time"] = pd.to_datetime(df["commit_time"], unit="s", errors="coerce")
    else:
        df["commit_time"] = pd.to_datetime(df["commit_time"], errors="coerce")

    df = df.set_index("commit_time").sort_index()
    trend = df[value_col].resample(window).mean().dropna()

    plt.figure(figsize=(9, 4))
    plt.plot(trend.index, trend.values, marker="o", color="tab:blue")
    plt.title(f"Average {value_col} Trend ({window})")
    plt.xlabel("Time")
    plt.ylabel(f"Average {value_col}")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
