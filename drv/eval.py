
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import numpy as np

def f1_auc(y_true, y_pred, y_proba):
    return {
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba)
    }

def threshold_scores(y_scores, threshold=0.5):
    return (y_scores >= threshold).astype(int)

def report(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=3)
