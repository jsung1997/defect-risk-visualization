
# DRV Project (Defect-Risk Visualization)

OVERVIEW

DRV (Defect-Risk Visualization) is a framework that helps developers see and prioritize defect-prone code modules.
It combines defect prediction models with interactive visualizations so teams can quickly identify which parts of the codebase are most risky.

Objective

-Integrate software defect prediction (metrics + semantic models) into a unified workflow.

-Provide visual analytics (heatmaps, rankings, dashboards) to make predictions transparent and interpretable.

-Support developers in prioritizing high-risk modules, rather than treating all code equally.

-Deliver a practical tool that complements existing defect prediction research with developer-friendly insights.







This is a **self-contained Python scaffold** for DRV thesis project. It includes:
- ApacheJIT_Total Dataset
- Baseline models (metrics-only and fused semantic+metrics)
- Ranking metrics (FPA, Recall@K, NDCG@K)
- Visualization (risk heatmap, Top-K bar chart)
- CLI entrypoint

> Note: Semantic representation uses **TF‑IDF of commit messages** as a *stand-in* for BiCC-BERT embeddings (offline-friendly). Replace with your BiCC module if available.

## Quickstart

```bash
# From this folder:
python -m drv.cli --commits data/commits.csv --mode train --output outputs
```

Artifacts:
- `outputs/train_metrics.json` — F1/AUC for metrics-only and fused models
- `outputs/ranking_eval.json` — FPA/Recall@K/NDCG@K
- `outputs/scored_commits.csv` — risk scores for each commit
- `outputs/heatmap.png`, `outputs/topk.png` — example visualizations

## Structure

- `drv/config.py` — config dataclass
- `drv/data.py` — CSV loaders/savers
- `drv/features.py` — CK feature handling + fusion
- `drv/models.py` — training and scoring (GradientBoosting as a local stand-in)
- `drv/ranking.py` — FPA, Recall@K, NDCG@K
- `drv/eval.py` — F1, AUC helpers
- `drv/visualize.py` — heatmap & top-k plot
- `drv/cli.py` — command-line workflow

## Replace with Real Data/Models
- Replace `data/commits.csv` with your JIT‑Defects4J export (same columns).
- Swap TF‑IDF with your **BiCC‑BERT embeddings** in `models.py`.
- Add dependency graph visualization to `visualize.py` if you have module graphs.
