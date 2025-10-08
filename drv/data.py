import pandas as pd
from pathlib import Path

# Common aliases seen in JIT datasets (e.g., apachejit_total.csv)
_ALIAS_MAP = {
    "commit_id": ["commit_id", "commit", "hash", "revision", "rev", "sha"],
    "bug":       ["bug", "label", "is_bug", "isbug", "is_buggy", "buggy", "defect"],
    "project":   ["project", "repo", "repository", "project_name", "proj"],
    "message":   ["message", "msg", "commit_message", "subject", "title"],
    "commit_time": ["commit_time", "time", "timestamp", "ts", "date", "authored_date"],
}

def _lower_strip(cols):
    return [str(c).strip().lower() for c in cols]

def _pick(df, logical_name):
    want = _ALIAS_MAP.get(logical_name, [])
    cols = list(df.columns)
    for w in want:
        if w in cols:
            return w
    return None

def load_commits(path: str) -> pd.DataFrame:
    """Load a commit-level CSV and normalize to expected schema:
       - commit_id (str)
       - label (0/1)
       - module (project)
       - message (string; can be empty)
       - commit_time (optional; datetime-convertible)
       - CK/JIT metrics if present (ns, nd, nf, entropy, la, ld, lt, ndev, age, nuc, exp, rexp, sexp)
    """
    df = pd.read_csv(path)
    # Normalize header casing/whitespace
    df.columns = _lower_strip(df.columns)

    # Map essential fields
    cid_col = _pick(df, "commit_id")
    bug_col = _pick(df, "bug")
    proj_col = _pick(df, "project")
    msg_col = _pick(df, "message")
    time_col = _pick(df, "commit_time")

    if cid_col is None:
        raise ValueError(f"Could not find commit id column. Available columns: {list(df.columns)}")

    # Create normalized columns
    df["commit_id"] = df[cid_col].astype(str)

    if bug_col is None:
        raise ValueError(
            "Could not find label/bug column (e.g., 'bug', 'label', 'is_bug'). "
            f"Available columns: {list(df.columns)}"
        )
    # Coerce to 0/1 integers
    lbl = df[bug_col]
    if lbl.dtype.kind in "bif":
        df["label"] = (lbl > 0).astype(int)
    else:
        df["label"] = lbl.astype(str).str.lower().isin(["1","true","yes","y","bug","buggy"]).astype(int)

    # Module: prefer 'project'; else single global module
    if proj_col is not None:
        df["module"] = df[proj_col].astype(str)
    else:
        df["module"] = "global"

    # Message: optional
    df["message"] = df[msg_col].astype(str) if msg_col is not None else ""

    # Commit time: optional; keep raw if present, let viz convert
    if time_col is not None:
        df["commit_time"] = df[time_col]

    return df

def save_df(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
