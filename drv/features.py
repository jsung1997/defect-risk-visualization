
import pandas as pd
import numpy as np

CK_FEATURES = ["ns","nd","nf","entropy","la","ld","lt","fix","ndev","age","nuc","exp","rexp","sexp"]

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric features for classical models (XGB baseline)."""
    X = df[CK_FEATURES].copy()
    # Simple normalization (z-score) for stability
    X = (X - X.mean()) / (X.std(ddof=0) + 1e-9)
    return X

def fuse_semantic_and_metrics(semantic: np.ndarray, metrics: pd.DataFrame) -> np.ndarray:
    """Concatenate semantic embeddings with normalized metrics."""
    return np.hstack([semantic, metrics.values])
