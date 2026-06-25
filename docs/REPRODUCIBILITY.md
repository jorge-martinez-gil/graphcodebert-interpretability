# Reproducibility Guide

This document provides a step-by-step workflow to reproduce the figures and outputs in the repository.

## TL;DR - reproduce everything with one command

```bash
pip install ".[full]"
gcbi reproduce            # regenerates the heatmap, PCA/t-SNE/UMAP projections,
                          # and the benchmark report (Markdown + LaTeX + figure)
```

The per-script instructions below remain valid: each script (`pca.py`, `heatmap.py`, ...)
is now a thin wrapper over the `graphcodebert_interpretability` library and produces
the same output files. Use whichever entry point you prefer.

## 1. Hardware and Software Requirements

### Hardware
- CPU-only execution is supported.
- GPU is optional (helps with faster embedding extraction for repeated runs).
- Recommended RAM: 8 GB minimum (16 GB preferred for smoother notebook/script execution).

### Software
- Python **3.8 to 3.12** (tested range).
- `pip` and a virtual environment tool (`venv` recommended).
- Dependencies listed in [`requirements.txt`](../requirements.txt), or install the package with `pip install ".[full]"`.

## 2. Environment Setup

```bash
git clone https://github.com/jorge-martinez-gil/graphcodebert-interpretability.git
cd graphcodebert-interpretability
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install ".[full]"
```

## 3. Reproduce All Figures and Outputs

Run the following commands from the repository root. Each script is a thin wrapper
over the library and writes the same files as before.

### 3.1 Token-level similarity HTML
```bash
python comparison.py
```
Expected output: `code_similarity.html`. Estimated runtime: ~1-3 minutes (first run longer due to model download).

### 3.2 Global similarity heatmap
```bash
python heatmap.py
```
Expected output: `sorting_algorithms_similarity.png`. Estimated runtime: ~2-5 minutes.

### 3.3 PCA projections
```bash
python pca.py
```
Expected output directory: `pca_pairwise_comparisons/`. Estimated runtime: ~3-8 minutes.

### 3.4 t-SNE projections
```bash
python tsne.py
```
Expected output directory: `tsne_pairwise_comparisons/`. Estimated runtime: ~5-15 minutes.

### 3.5 UMAP projections
```bash
python pumap.py
```
Expected output directory: `umap_pairwise_comparisons/`. Estimated runtime: ~4-12 minutes.

### 3.6 Saliency maps
```bash
python saliency_maps.py
```
Expected output directory: `saliency_maps/`. Estimated runtime: ~6-20 minutes.

### 3.7 Ablation study outputs
```bash
python ablation.py
```
Expected output directory: `ablation_study_results/`. Estimated runtime: ~4-12 minutes.

### 3.8 AST-based structural similarity baseline
```bash
python ast-s.py
```
Expected output directory: `ast_tree_kernel_visualizations/`. Estimated runtime: <1 minute (no model needed).

### 3.9 TF-IDF textual similarity baseline
```bash
python tf-s.py
```
Expected output directory: `cosine_similarity_visualizations/`. Estimated runtime: <1 minute (no model needed).

### 3.10 Automated benchmark report
```bash
gcbi report --dataset sorting --outdir benchmark_report
```
Expected outputs: `benchmark_report/report.md`, `benchmark_report/similarity_table.tex`, `benchmark_report/similarity_comparison.png`.

## 4. Notes on Determinism

- Call `graphcodebert_interpretability.set_seed(42)` (or pass `--seed` on the CLI)
  to seed Python, NumPy and PyTorch in one step. The reproduction scripts already do this.
- t-SNE and UMAP projections use `random_state=42`.
- For additional consistency across hardware, set `PYTHONHASHSEED=0` before running.

Even with fixed seeds, minor numerical differences may occur across library versions and hardware backends.
