# Typed Heterogeneous Claim-Evidence-Entity Graphs for Hallucination Detection in LLM Outputs

Code and reproducibility materials for the IEEE Big Data 2026 paper: Padmavathi Kadium, Shafiya Mubeen Umme (California State University, Long Beach), and Benyamin Haghi (California Institute of Technology).

## Overview

This repository contains the notebook and cached results behind the paper's experiments: a typed heterogeneous claim-evidence-entity graph for hallucination detection, instantiated with R-GCN and GAT, evaluated against a flat cosine baseline, a trained MLP, and a zero-shot instruction-tuned LLM judge (FLAN-T5-large) on three corpora - FEVER, HaluEval, and TruthfulQA.

## Repository contents

- `MLBD_2026.ipynb` — the notebook that produced the paper's results: data loading and preprocessing for all three corpora, graph construction (5 relation types), model training (R-GCN/GAT), score fusion, ablations, and the statistical tests (McNemar, cluster-bootstrap, sign-flip permutation) reported in the paper.
- `curves_cache.pkl` — cached per-instance predictions (true label and score) for each (corpus, method) pair at seed 42, used by the statistical-testing cells so they don't require retraining to reproduce Tables VIII-IX.
- `judge_scores_FEVER.pkl`, `judge_scores_HaluEval.pkl`, `judge_scores_TruthfulQA.pkl` — cached validation and test scores from the zero-shot FLAN-T5-large judge for each corpus, so the judge doesn't need to be re-run to reproduce fusion and comparison results.
- `results_main.csv` — aggregate metrics (accuracy, precision, recall, macro-F1, AUROC, etc.) for every (corpus, method) pair in the main comparison (Table IV).

## Requirements

- Python 3.x
- PyTorch
- PyTorch Geometric
- transformers (for `google/flan-t5-large`)
- sentence-transformers (`all-mpnet-base-v2`)
- scikit-learn
- SciPy
- spaCy (`en_core_web_sm`)

## Reproducing the results

1. Open `MLBD_2026.ipynb` in Colab or Jupyter.
2. Run the setup and data-loading cells — these will reuse the cached files in this repo (`curves_cache.pkl`, `judge_scores_*.pkl`) rather than recomputing them where applicable.
3. Run the main comparison, ablation, fusion, and statistical-testing cells in order; each checkpoints its own results to CSV/pickle so the notebook can resume after a restart.

**Note on exact reproducibility:** GPU scatter operations in PyTorch Geometric are non-deterministic, so independently retrained runs of the same configuration can vary by up to ~0.006 macro-F1 at a fixed seed (discussed in Section IV-B of the paper). This does not affect the paper's qualitative conclusions.

## Data

FEVER, HaluEval, and TruthfulQA are loaded from their public releases (see citations in the paper). No proprietary data is included in this repository.
