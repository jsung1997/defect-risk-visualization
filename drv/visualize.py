import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional

# --- Existing functions (keep your heatmap and topk_bar) ---
# risk_heatmap(...)
# topk_bar(...)

def risk_trend(
    df_commits: pd.DataFrame,
    value_col: str = "risk_score",
    window: str = "7D",
    out_path: Optional[str] = None
):
    """
    Plot a trend line of average risk over time.

    Args:
        df_commits: DataFrame with at least [value_col, commit_time].
        value_col:  Column to average (default: 'risk_score').
        window:     Pandas offset alias for resampling window (e.g., '7D' for weekly, '14D', 'M' for monthly).
        out_path:   Optional PNG file path to save.
    """
    df = df_commits.copy()

    if "commit_time" not in df.columns:
        raise ValueError("risk_trend requires a 'commit_time' column in the DataFrame.")

    # Ensure commit_time is datetime
    if np.issubdtype(df["commit_time"].dtype, np.number):
        df["commit_time"] = pd.to_datetime(df["commit_time"], unit="s", errors="coerce")
    else:
        df["commit_time"] = pd.to_datetime(df["commit_time"], errors="coerce")

    # Resample into windows (e.g., weekly averages)
    df = df.set_index("commit_time").sort_index()
    trend = df[value_col].resample(window).mean().dropna()

    # Plot
    plt.figure(figsize=(9, 4))
    plt.plot(trend.index, trend.values, marker="o", linestyle="-", color="tab:blue")
    plt.title(f"Average {value_col} Trend ({window} window)")
    plt.xlabel("Time")
    plt.ylabel(f"Average {value_col}")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
