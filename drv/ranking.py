
import numpy as np
import pandas as pd

def recall_at_k(y_true, y_scores, k):
    order = np.argsort(-y_scores)
    topk = order[:k]
    return y_true[topk].sum() / max(1, y_true.sum())

def ndcg_at_k(y_true, y_scores, k):
    order = np.argsort(-y_scores)[:k]
    gains = (2**y_true[order] - 1) / np.log2(np.arange(2, k+2))
    ideal_order = np.argsort(-y_true)[:k]
    ideal_gains = (2**y_true[ideal_order] - 1) / np.log2(np.arange(2, k+2))
    denom = ideal_gains.sum() if ideal_gains.sum() > 0 else 1.0
    return gains.sum() / denom

def first_priority_accuracy(y_true, y_scores):
    """FPA: 1 if the very top-ranked item is actually defective, else 0."""
    return float(y_true[np.argmax(y_scores)] == 1)

def evaluate_ranking(y_true, y_scores, ks=(5,10,20)):
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    out = {"FPA": first_priority_accuracy(y_true, y_scores)}
    for k in ks:
        out[f"Recall@{k}"] = recall_at_k(y_true, y_scores, k)
        out[f"NDCG@{k}"] = ndcg_at_k(y_true, y_scores, k)
    return out
