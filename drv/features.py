import pandas as pd
import numpy as np
from typing import List

# Preferred JIT/CK-style features, but some datasets omit a few.
PREFERRED_FEATURES: List[str] = [
    "ns","nd","nf","entropy","la","ld","lt","fix","ndev","age","nuc","exp","rexp","sexp"
]

def available_features(df: pd.DataFrame) -> List[str]:
    """Return intersection of preferred features and df columns (order preserved)."""
    cols = [c for c in PREFERRED_FEATURES if c in df.columns]
    if not cols:
        # Extremely minimal fallback: use whatever numeric JIT-ish columns we can find
        candidates = ["ns","nd","nf","la","ld","ndev","age","nuc"]
        cols = [c for c in candidates if c in df.columns]
    return cols

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build numeric feature matrix robustly:
      - pick only available columns
      - coerce to numeric
      - fill NaNs with 0
      - z-score normalize (safe if var==0)
    """
    cols = available_features(df)
    if not cols:
        # last-resort: return a single zero feature to keep sklearn happy
        X = pd.DataFrame({"_bias": np.zeros(len(df), dtype=float)})
        return X

    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)

    # z-score with epsilon to avoid div-by-zero
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0) + 1e-9
    X = (X - mean) / std
    return X

def fuse_semantic_and_metrics(semantic: np.ndarray, metrics: pd.DataFrame) -> np.ndarray:
    """Concatenate semantic embeddings (may be zeros) with normalized metrics (>=1 col)."""
    return np.hstack([semantic, metrics.values])
