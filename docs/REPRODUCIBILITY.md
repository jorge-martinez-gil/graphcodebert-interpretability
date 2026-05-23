# Reproducibility Guide

This document provides a step-by-step workflow to reproduce the figures and outputs in the repository.

## 1. Hardware and Software Requirements

### Hardware
- CPU-only execution is supported.
- GPU is optional (helps with faster embedding extraction for repeated runs).
- Recommended RAM: 8 GB minimum (16 GB preferred for smoother notebook/script execution).

### Software
- Python **3.7 to 3.11** (tested range).
- `pip` and a virtual environment tool (`venv` recommended).
- Dependencies listed in [`requirements.txt`](../requirements.txt).

## 2. Environment Setup

```bash
git clone https://github.com/jorge-martinez-gil/graphcodebert-interpretability.git
cd graphcodebert-interpretability
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Reproduce All Figures and Outputs

Run the following commands from the repository root.

### 3.1 Token-level similarity HTML
```bash
python comparison.py
```
Expected output:
- `code_similarity.html`

Estimated runtime:
- ~1-3 minutes (first run may be longer due to model download).

### 3.2 Global similarity heatmap
```bash
python heatmap.py
```
Expected output:
- `sorting_algorithms_similarity.png`

Estimated runtime:
- ~2-5 minutes.

### 3.3 PCA projections
```bash
python pca.py
```
Expected output directory:
- `pca_pairwise_comparisons/` (pairwise `*_tokens_2d_pca.png` files)

Estimated runtime:
- ~3-8 minutes.

### 3.4 t-SNE projections
```bash
python tsne.py
```
Expected output directory:
- `tsne_pairwise_comparisons/` (pairwise `*_tokens_2d_tsne.png` files)

Estimated runtime:
- ~5-15 minutes.

### 3.5 UMAP projections
```bash
python pumap.py
```
Expected output directory:
- `umap_pairwise_comparisons/` (pairwise `*_tokens_2d_umap.png` files)

Estimated runtime:
- ~4-12 minutes.

### 3.6 Saliency maps
```bash
python saliency_maps.py
```
Expected output directory:
- `saliency_maps/` (pairwise `*_saliency_map.png` files)

Estimated runtime:
- ~6-20 minutes.

### 3.7 Ablation study outputs
```bash
python ablation.py
```
Expected output directory:
- `ablation_study_results/`

Estimated runtime:
- ~4-12 minutes.

### 3.8 AST-based structural similarity baseline
```bash
python ast-s.py
```
Expected output directory:
- `ast_tree_kernel_visualizations/`

Estimated runtime:
- <1 minute.

### 3.9 TF-IDF textual similarity baseline
```bash
python tf-s.py
```
Expected output directory:
- `cosine_similarity_visualizations/`

Estimated runtime:
- <1 minute.

## 4. Notes on Determinism
- `tsne.py` sets `random_state=42` in `TSNE(...)`, which improves reproducibility of t-SNE projections.
- `pumap.py` sets `random_state=42` in `UMAP(...)`, which improves reproducibility of UMAP projections.
- For additional consistency across hardware, set deterministic seeds before running scripts:

```bash
export PYTHONHASHSEED=0
```

Within Python sessions/notebooks, also set:

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

Even with fixed seeds, minor numerical differences may occur across library versions and hardware backends.
