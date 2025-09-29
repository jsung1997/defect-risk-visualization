
import pandas as pd
from pathlib import Path

def load_commits(path: str) -> pd.DataFrame:
    """Load commit-level data.
    Expected columns:
      commit_id, module, message, la, ld, ns, nd, nf, entropy, lt, fix, ndev, age, nuc, exp, rexp, sexp, label
    label: 1 = defective, 0 = clean
    """
    return pd.read_csv(path)

def load_modules(path: str) -> pd.DataFrame:
    """Load module metadata (module -> subsystem/dependencies if available)."""
    return pd.read_csv(path)

def save_df(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
