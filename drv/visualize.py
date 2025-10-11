# drv/visualize.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional

# Use a consistent style without forcing colors explicitly beyond palettes
sns.set_context("talk")


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Coerce a pandas Series to datetime64[ns] (assumes unix seconds if numeric)."""
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    # If object/str or numeric, try unix-seconds first, then general parse
    try:
        return pd.to_datetime(series, unit="s", errors="coerce")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def risk_heatmap_binned(
    df: pd.DataFrame,
    value_col: str = "risk_score",          # "risk_score" for model risk, or "buggy" for defect rate
    time_col: str = "commit_time",          # must exist in df; unix seconds or datetime
    module_col: str = "module",
    topn: int = 25,                         # show top-N hottest modules
    freq: str = "M",                        # "W", "M", "Q", "Y"
    clip_quantiles: Tuple[float, float] = (0.05, 0.95),
    out_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Developer-readable heatmap:
      Y = modules (top-N by mean value)
      X = time buckets (freq)
      Color = mean value_col (risk or bug rate)

    Returns the pivot table (modules × buckets) for inspection/testing.
    """
    x = df.copy()

    # Validate columns
    for col in (value_col, time_col, module_col):
        if col not in x.columns:
            raise KeyError(f"Column '{col}' not found in dataframe. Available: {list(x.columns)}")

    # Coerce time
    x[time_col] = _ensure_datetime(x[time_col])
    if x[time_col].isna().all():
        raise ValueError(f"Could not parse '{time_col}' to datetime. Provide unix seconds or ISO timestamps.")

    # Map buggy → {TRUE/FALSE}→{1/0} if requested
    if value_col == "buggy":
        x[value_col] = (
            x[value_col]
            .astype(str)
            .str.upper()
            .map({"TRUE": 1, "FALSE": 0})
            .fillna(0)
            .astype(float)
        )

    # Build time buckets
    x["bucket"] = x[time_col].dt.to_period(freq).dt.to_timestamp()

    # Select top-N hottest modules by overall mean value
    mod_mean = x.groupby(module_col, as_index=False)[value_col].mean()
    hot_mods = (
        mod_mean.sort_values(value_col, ascending=False)
        .head(max(1, topn))[module_col]
        .tolist()
    )
    x = x[x[module_col].isin(hot_mods)]

    # Pivot to module × bucket
    pivot = (
        x.groupby([module_col, "bucket"], as_index=False)[value_col]
         .mean()
         .pivot(index=module_col, columns="bucket", values=value_col)
    )

    # Sort rows by their global mean (descending)
    row_mean = pivot.mean(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[row_mean.index]

    # Robust color clipping to avoid washed-out maps
    if pivot.notna().values.any():
        q_low = float(np.nanquantile(pivot.values, clip_quantiles[0]))
        q_hi  = float(np.nanquantile(pivot.values, clip_quantiles[1]))
        vmin, vmax = q_low, q_hi
        # If all values identical (vmin==vmax), widen the range slightly
        if np.isclose(vmin, vmax, equal_nan=True):
            vmin, vmax = vmin - 1e-6, vmax + 1e-6
    else:
        # Fallback if everything is NaN
        vmin, vmax = 0.0, 1.0

    # Plot
    plt.figure(figsize=(18, max(6, 0.35 * len(pivot))))
    sns.heatmap(
        pivot,
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.3,
        linecolor="#efefef",
        cbar_kws={"label": "Avg Risk" if value_col != "buggy" else "Bug Rate"},
    )
    plt.title(
        f"DRV: Top-{max(1, topn)} Modules × {freq}-binned "
        f"{'risk_score' if value_col!='buggy' else 'buggy'} Heatmap",
        fontsize=14
    )
    plt.xlabel("Time")
    plt.ylabel("Modules")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return pivot


def topk_bar(
    df: pd.DataFrame,
    values,                                   # kept for backward compat with your CLI signature
    k: int = 10,
    out_path: Optional[str] = None
) -> None:
    """
    Bar chart of top-k modules by average risk_score.
    Uses a hue to avoid Seaborn 0.14 warning when palette is provided without hue.
    """
    if "module" not in df.columns or "risk_score" not in df.columns:
        raise KeyError("Expected columns 'module' and 'risk_score' in dataframe.")

    df_bar = (
        df.groupby("module", as_index=False)["risk_score"]
          .mean()
          .sort_values("risk_score", ascending=False)
          .head(max(1, k))
    )

    plt.figure(figsize=(10, max(4, 0.4 * len(df_bar))))
    # Use hue=module (dodge=False, legend=False) to keep stable behavior and silence FutureWarning
    sns.barplot(
        data=df_bar,
        x="risk_score",
        y="module",
        hue="module",
        dodge=False,
        palette="Reds_r",
        legend=False
    )
    plt.title(f"Top-{max(1, k)} Modules by Average Risk", fontsize=13)
    plt.xlabel("Average Risk Score")
    plt.ylabel("Module")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def risk_trend(
    df: pd.DataFrame,
    value_col: str = "risk_score",
    window: str = "7D",
    out_path: Optional[str] = None
) -> None:
    """
    Rolling trend of risk over time (global).
      - Parses commit_time → datetime
      - Plots rolling mean with window like '7D' or '30D'
    """
    if "commit_time" not in df.columns:
        raise KeyError("Expected column 'commit_time' in dataframe.")
    if value_col not in df.columns:
        raise KeyError(f"Expected column '{value_col}' in dataframe.")

    x = df.copy()
    x["commit_time"] = _ensure_datetime(x["commit_time"])
    x = x.sort_values("commit_time")

    # Rolling mean (time-based window if DatetimeIndex)
    ts = (
        x.set_index("commit_time")[value_col]
         .astype(float)
         .rolling(window=window)
         .mean()
         .reset_index()
    )

    plt.figure(figsize=(12, 4))
    plt.plot(ts["commit_time"], ts[value_col], linewidth=2)
    plt.title(f"Risk Trend (Rolling {window})", fontsize=13)
    plt.xlabel("Time")
    plt.ylabel(f"Mean {value_col}")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
