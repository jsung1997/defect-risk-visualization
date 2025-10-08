from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from .features import build_feature_matrix, fuse_semantic_and_metrics

@dataclass
class DRVModels:
    tfidf: TfidfVectorizer | None
    clf_metrics: GradientBoostingClassifier | None
    clf_fused: GradientBoostingClassifier | None

def _safe_tfidf_fit(text_series: pd.Series) -> tuple[TfidfVectorizer | None, np.ndarray]:
    """
    Fit TF-IDF on commit messages.
    If messages are empty or vocabulary can't be built, return (None, zeros).
    """
    msgs = text_series.fillna("").astype(str)
    if (msgs.str.len() > 0).sum() == 0:
        return None, np.zeros((len(msgs), 1), dtype=float)
    tfidf = TfidfVectorizer(max_features=3000)
    try:
        X_text = tfidf.fit_transform(msgs).toarray()
        return tfidf, X_text
    except ValueError:
        # empty vocabulary -> use zeros
        return None, np.zeros((len(msgs), 1), dtype=float)

def _safe_tfidf_transform(text_series: pd.Series, tfidf: TfidfVectorizer | None) -> np.ndarray:
    if tfidf is None:
        return np.zeros((len(text_series), 1), dtype=float)
    return tfidf.transform(text_series.fillna("").astype(str)).toarray()

def train_baselines(df: pd.DataFrame, seed: int = 42):
    # Metrics
    X_metrics = build_feature_matrix(df)
    y = df["label"].astype(int).values

    # Text (commit messages). May be unavailable/empty in apachejit_total -> handle safely.
    tfidf, X_text = _safe_tfidf_fit(df["message"] if "message" in df.columns else pd.Series([""]*len(df)))
    X_fused = fuse_semantic_and_metrics(X_text, X_metrics)

    # Train metrics-only
    Xtr_m, Xte_m, ytr, yte = train_test_split(X_metrics, y, test_size=0.2, random_state=seed, stratify=y)
    clf_m = GradientBoostingClassifier(random_state=seed)
    clf_m.fit(Xtr_m, ytr)
    pred_m = clf_m.predict(Xte_m)
    proba_m = clf_m.predict_proba(Xte_m)[:, 1]
    f1_m = f1_score(yte, pred_m)
    auc_m = roc_auc_score(yte, proba_m)

    # Train fused (text+metrics). If text is zeros, this still works.
    Xtr_f, Xte_f, ytr_f, yte_f = train_test_split(X_fused, y, test_size=0.2, random_state=seed, stratify=y)
    clf_f = GradientBoostingClassifier(random_state=seed)
    clf_f.fit(Xtr_f, ytr_f)
    pred_f = clf_f.predict(Xte_f)
    proba_f = clf_f.predict_proba(Xte_f)[:, 1]
    f1_f = f1_score(yte_f, pred_f)
    auc_f = roc_auc_score(yte_f, proba_f)

    models = DRVModels(tfidf=tfidf, clf_metrics=clf_m, clf_fused=clf_f)
    return models, {"metrics_f1": f1_m, "metrics_auc": auc_m, "fused_f1": f1_f, "fused_auc": auc_f}, (X_text, X_metrics, X_fused)

def score_dataframe(df: pd.DataFrame, models: DRVModels):
    X_metrics = build_feature_matrix(df)
    X_text = _safe_tfidf_transform(df["message"] if "message" in df.columns else pd.Series([""]*len(df)), models.tfidf)
    X_fused = fuse_semantic_and_metrics(X_text, X_metrics)

    proba_m = models.clf_metrics.predict_proba(X_metrics)[:, 1] if models.clf_metrics else np.zeros(len(df))
    proba_f = models.clf_fused.predict_proba(X_fused)[:, 1] if models.clf_fused else proba_m
    return proba_m, proba_f

# ---------- Persistence ----------
def save_models(models: DRVModels, out_dir: str = "outputs"):
    joblib.dump(models.tfidf, f"{out_dir}/tfidf.joblib")
    joblib.dump(models.clf_metrics, f"{out_dir}/clf_metrics.joblib")
    joblib.dump(models.clf_fused, f"{out_dir}/clf_fused.joblib")

def load_models(in_dir: str = "outputs") -> DRVModels:
    tfidf = joblib.load(f"{in_dir}/tfidf.joblib")
    clf_m = joblib.load(f"{in_dir}/clf_metrics.joblib")
    clf_f = joblib.load(f"{in_dir}/clf_fused.joblib")
    return DRVModels(tfidf=tfidf, clf_metrics=clf_m, clf_fused=clf_f)
