from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib

# Local baseline models (no internet)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from .features import build_feature_matrix, fuse_semantic_and_metrics

@dataclass
class DRVModels:
    tfidf: TfidfVectorizer
    clf_metrics: GradientBoostingClassifier
    clf_fused: GradientBoostingClassifier

def train_baselines(df: pd.DataFrame, seed: int = 42):
    # Prepare features
    X_metrics = build_feature_matrix(df)
    y = df["label"].astype(int).values

    # Semantic proxy (stand-in for BiCC embeddings): TF-IDF on commit messages
    tfidf = TfidfVectorizer(max_features=3000)
    X_text = tfidf.fit_transform(df["message"].fillna("")).toarray()

    X_fused = fuse_semantic_and_metrics(X_text, X_metrics)

    Xtr_m, Xte_m, ytr, yte = train_test_split(X_metrics, y, test_size=0.2, random_state=seed, stratify=y)
    Xtr_f, Xte_f, ytr_f, yte_f = train_test_split(X_fused, y, test_size=0.2, random_state=seed, stratify=y)

    clf_m = GradientBoostingClassifier(random_state=seed)
    clf_m.fit(Xtr_m, ytr)
    pred_m = clf_m.predict(Xte_m)
    proba_m = clf_m.predict_proba(Xte_m)[:, 1]
    f1_m = f1_score(yte, pred_m)
    auc_m = roc_auc_score(yte, proba_m)

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
    X_text = models.tfidf.transform(df["message"].fillna("")).toarray()
    X_fused = fuse_semantic_and_metrics(X_text, X_metrics)
    proba_m = models.clf_metrics.predict_proba(X_metrics)[:, 1]
    proba_f = models.clf_fused.predict_proba(X_fused)[:, 1]
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
