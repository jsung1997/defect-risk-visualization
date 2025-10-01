import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def _downsample_xticks(num_cols: int, labels: list[str], max_xticks: int = 20):
    """
    Return (indices, labels) for ~max_xticks evenly spaced x-ticks.
    """
    if num_cols == 0:
        return [], []
    max_xticks = max(1, int(max_xticks))
    step = max(1, num_cols // max_xticks)
    idx = list(range(0, num_cols, step))
    # Ensure the last label is included
    if idx[-1] != num_cols - 1:
        idx.append(num_cols - 1)
    return idx, [labels[i] for i in idx]

def risk_heatmap(
    df_commits: pd.DataFrame,
    value_col: str = "risk_score",
    out_path: Optional[str] = None,
    *,
    max_xticks: int = 20,
    window: Optional[str] = None,  # e.g., "7D" to aggregate weekly by commit_time
):
    """
    Plot a heatmap: rows = modules, cols = commits or time-buckets, values = df[value_col].

    Args:
        df_commits: DataFrame with at least ['module','commit_id', value_col].
                    If 'commit_time' exists and `window` is provided, it will be used for resampling.
        value_col:  Column name to visualize (default: 'risk_score').
        out_path:   Optional path to save PNG.
        max_xticks: Cap the number of x-axis labels by showing ~evenly spaced ones.
        window:     Optional Pandas offset alias to aggregate by time window (e.g., "7D", "14D").
                    Requires a 'commit_time' column (unix seconds or datetime-like).
    """
    df = df_commits.copy()

    # Ensure types/ordering
    if "commit_time" in df.columns:
        # accept epoch seconds or datetime
        if np.issubdtype(df["commit_time"].dtype, np.number):
            df["commit_time"] = pd.to_datetime(df["commit_time"], unit="s", errors="coerce")
        else:
            df["commit_time"] = pd.to_datetime(df["commit_time"], errors="coerce")
        df = df.sort_values(["module", "commit_time", "commit_id"])
    else:
        df = df.sort_values(["module", "commit_id"])

    # Optional: aggregate over time windows (reduces #columns, clearer x-axis)
    if window is not None:
        if "commit_time" not in df.columns or df["commit_time"].isna().all():
            raise ValueError("window aggregation requires a valid 'commit_time' column.")
        # bucket label by period start
        df["_bucket"] = df["commit_time"].dt.to_period(window).dt.start_time.dt.strftime("%Y-%m-%d")
        col_key = "_bucket"
    else:
        col_key = "commit_id"

    # Build pivot (modules x columns)
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame.")
    pivot = df.pivot_table(index="module", columns=col_key, values=value_col, aggfunc="mean").fillna(0.0)

    # Plot
    plt.figure(figsize=(14, 7))
    im = plt.imshow(pivot.values, aspect="auto")
    # X ticks: downsample to keep labels readable
    cols = list(pivot.columns.astype(str))
    xt_idx, xt_labels = _downsample_xticks(pivot.shape[1], cols, max_xticks=max_xticks)
    plt.xticks(xt_idx, xt_labels, rotation=45, ha="right", fontsize=8)

    # Y ticks: show all modules (usually manageable); truncate long names a bit
    ylabels = [str(y) if len(str(y)) <= 30 else str(y)[:27] + "..." for y in pivot.index]
    plt.yticks(range(pivot.shape[0]), ylabels, fontsize=9)

    title_suffix = f" (window={window})" if window else ""
    plt.title(f"DRV: Module × {'Bucket' if window else 'Commit'} Risk Heatmap{title_suffix}")
    plt.xlabel("Buckets" if window else "Commits")
    plt.ylabel("Modules")
    cbar = plt.colorbar(im)
    cbar.set_label("Risk")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

def topk_bar(
    df_commits: pd.DataFrame,
    risk_scores: np.ndarray,
    k: int = 10,
    out_path: Optional[str] = None
):
    """
    Simple Top-K bar chart of highest-risk rows (commit_id + module).
    """
    order = np.argsort(-risk_scores)[:k]
    subset = df_commits.iloc[order][["commit_id", "module"]].copy()
    values = risk_scores[order]

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(values)), values)

    # Compact x labels
    def _short(s: str, n=18):
        s = str(s)
        return s if len(s) <= n else s[:n-3] + "..."
    labels = [f"{_short(r.commit_id)}\n{_short(r.module)}" for _, r in subset.iterrows()]
    plt.xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Risk")
    plt.title(f"Top-{k} Risky Commits/Modules")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
