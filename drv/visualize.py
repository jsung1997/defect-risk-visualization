# drv/visualize.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from typing import Tuple, Optional

sns.set_context("talk")


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Robustly coerce to datetime from seconds, ms, ISO strings, or mixed."""
    if np.issubdtype(series.dtype, np.datetime64):
        return series

    s = series.copy()

    # Try numeric first (epoch seconds or ms)
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().any():
        med = sn.dropna().median()
        if med > 1e12:  # looks like milliseconds
            sn = (sn / 1000.0).round()
        return pd.to_datetime(sn, unit="s", errors="coerce")

    # Fallback: generic parse
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)


def risk_heatmap_binned(
    df: pd.DataFrame,
    value_col: str = "risk_score",          # "risk_score" or "buggy"
    time_col: str = "commit_time",
    module_col: str = "module",
    topn: int = 25,
    freq: str = "M",                        # "W","M","Q","Y"
    clip_quantiles: Tuple[float, float] = (0.05, 0.95),
    out_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Heatmap: Y=top-N modules, X=time buckets, Color=mean value_col.
    Returns the pivot for inspection.
    """
    x = df.copy()
    for col in (value_col, time_col, module_col):
        if col not in x.columns:
            raise KeyError(f"Column '{col}' not found. Available: {list(x.columns)}")

    x[time_col] = _ensure_datetime(x[time_col])
    if x[time_col].isna().all():
        raise ValueError(f"Could not parse '{time_col}' to datetime. Check epoch units or format.")

    if value_col == "buggy":
        x[value_col] = (
            x[value_col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(0).astype(float)
        )

    x["bucket"] = x[time_col].dt.to_period(freq).dt.to_timestamp()

    mod_mean = x.groupby(module_col, as_index=False)[value_col].mean()
    hot_mods = (
        mod_mean.sort_values(value_col, ascending=False)
        .head(max(1, topn))[module_col]
        .tolist()
    )
    x = x[x[module_col].isin(hot_mods)]

    pivot = (
        x.groupby([module_col, "bucket"], as_index=False)[value_col]
         .mean()
         .pivot(index=module_col, columns="bucket", values=value_col)
    )
    row_mean = pivot.mean(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[row_mean.index]

    if pivot.notna().values.any():
        q_low = float(np.nanquantile(pivot.values, clip_quantiles[0]))
        q_hi  = float(np.nanquantile(pivot.values, clip_quantiles[1]))
        if np.isclose(q_low, q_hi):
            vmin, vmax = q_low - 1e-6, q_hi + 1e-6
        else:
            vmin, vmax = q_low, q_hi
    else:
        vmin, vmax = 0.0, 1.0

    fig_h = max(6, 0.35 * len(pivot))
    fig = plt.figure(figsize=(18, fig_h))
    ax = sns.heatmap(
        pivot,
        cmap="Reds", vmin=vmin, vmax=vmax,
        linewidths=0.3, linecolor="#efefef",
        cbar_kws={"label": "Avg Risk" if value_col != "buggy" else "Bug Rate"},
    )

    # Title & axis labels
    ax.set_title(f"DRV: Top-{max(1, topn)} Modules × {freq}-binned {value_col} Heatmap", fontsize=14, pad=20)
    ax.set_xlabel("Time", fontsize=12, labelpad=10)
    ax.set_ylabel("Modules", fontsize=12, labelpad=10)

    # X tick formatting
    ax.xaxis.set_major_locator(MaxNLocator(nbins=16))  # limit number of ticks
    plt.xticks(rotation=30, ha="right", fontsize=9)
    xlabels = [lbl.get_text().split("T")[0].split(" ")[0] for lbl in ax.get_xticklabels()]
    if freq.upper() == "Y":
        xlabels = [s[:4] for s in xlabels]
    ax.set_xticklabels(xlabels)

    # Y ticks smaller
    plt.yticks(fontsize=9)

    # Ensure labels are not clipped (no tight_layout here)
    plt.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.22)

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return pivot


def topk_bar(
    df: pd.DataFrame,
    values,                                  # kept for CLI signature compat
    k: int = 10,
    out_path: Optional[str] = None
) -> None:
    """Top-k modules by mean risk_score (Seaborn 0.14-safe)."""
    if "module" not in df.columns or "risk_score" not in df.columns:
        raise KeyError("Expected columns 'module' and 'risk_score'.")

    df_bar = (
        df.groupby("module", as_index=False)["risk_score"]
          .mean()
          .sort_values("risk_score", ascending=False)
          .head(max(1, k))
    )

    fig = plt.figure(figsize=(10, max(4, 0.4 * len(df_bar))))
    sns.barplot(
        data=df_bar,
        x="risk_score", y="module",
        hue="module", dodge=False, legend=False,
        palette="Reds_r"
    )
    plt.title(f"Top-{max(1, k)} Modules by Average Risk", fontsize=13)
    plt.xlabel("Average Risk Score")
    plt.ylabel("Module")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def risk_trend(
    df: pd.DataFrame,
    value_col: str = "risk_score",
    window: str = "7D",
    out_path: Optional[str] = None
) -> None:
    """Rolling mean of risk over time."""
    if "commit_time" not in df.columns:
        raise KeyError("Expected column 'commit_time'.")
    if value_col not in df.columns:
        raise KeyError(f"Expected column '{value_col}'.")

    x = df.copy()
    x["commit_time"] = _ensure_datetime(x["commit_time"])
    x = x.sort_values("commit_time")

    ts = (
        x.set_index("commit_time")[value_col]
         .astype(float)
         .rolling(window=window)
         .mean()
         .reset_index()
    )

    fig = plt.figure(figsize=(12, 4))
    plt.plot(ts["commit_time"], ts[value_col], linewidth=2)
    plt.title(f"Risk Trend (Rolling {window})", fontsize=13)
    plt.xlabel("Time"); plt.ylabel(f"Mean {value_col}")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
