import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def risk_heatmap_binned(
    df: pd.DataFrame,
    value_col: str = "risk_score",           # or "buggy"
    time_col: str = "commit_time",           # unix seconds or pandas datetime
    module_col: str = "module",
    topn: int = 25,                          # show top-N hottest modules
    freq: str = "M",                         # "W", "M", "Q", "Y"
    clip_quantiles=(0.05, 0.95),
    out_path: str = None
):
    x = df.copy()

    # ensure datetime
    if not np.issubdtype(x[time_col].dtype, np.datetime64):
        x[time_col] = pd.to_datetime(x[time_col], unit="s", errors="coerce")

    # if using ground truth instead of scores, use mean of 0/1 -> defect rate
    if value_col == "buggy":
        if x[value_col].dtype != np.float64 and x[value_col].dtype != np.int64:
            x[value_col] = x[value_col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(0)
    # aggregate by module × time bucket
    x["bucket"] = x[time_col].dt.to_period(freq).dt.to_timestamp()

    # pick top-N hot modules by overall mean
    mod_mean = x.groupby(module_col, as_index=False)[value_col].mean()
    hot_mods = (mod_mean.sort_values(value_col, ascending=False)
                        .head(topn)[module_col].tolist())
    x = x[x[module_col].isin(hot_mods)]

    # pivot
    pivot = (x.groupby([module_col, "bucket"], as_index=False)[value_col]
               .mean()
               .pivot(index=module_col, columns="bucket", values=value_col)
               .sort_values(x[value_col].mean(level=0).sort_values(ascending=False).index))

    # robust color clipping to avoid all-pink maps
    vmin, vmax = pivot.quantile(clip_quantiles[0]).min(), pivot.quantile(clip_quantiles[1]).max()

    plt.figure(figsize=(18, max(6, 0.35*len(pivot))))
    sns.heatmap(pivot, cmap="Reds", vmin=vmin, vmax=vmax,
                cbar_kws={"label": ("Avg Risk" if value_col!="buggy" else "Bug Rate")},
                linewidths=0.3, linecolor="#efefef")
    plt.title(f"DRV: Top-{topn} Modules × {freq} Heatmap ({'risk' if value_col!='buggy' else 'bug rate'})", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Modules")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    return pivot  # handy for inspection/tests
