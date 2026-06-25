# Changelog

All notable changes to this project are documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-06-25
### Added
- **Installable Python library** `graphcodebert_interpretability` with a clean,
  documented public API (`compare`, `token_alignment`, `saliency`, `project`,
  `similarity_matrix`, `embed`, baselines, datasets, `set_seed`, ...) that works on
  **any** code, not just the bundled examples.
- **`gcbi` command-line interface**: `compare`, `saliency`, `project`, `heatmap`,
  `report`, and `reproduce`.
- **Model-agnostic** model loading (`load_model`) with caching - supports
  GraphCodeBERT, CodeBERT, UniXcoder and other compatible checkpoints.
- **Automated benchmark reports**: `generate_report` emits a Markdown table, a
  `booktabs` LaTeX table and a 300-DPI comparison figure across GraphCodeBERT,
  AST and TF-IDF lenses.
- **Dataset loaders**: bundled `sorting` corpus plus `load_snippets_from_dir`
  and `load_jsonl` for user-provided code.
- **`pytest` test suite** (40+ fast tests, no model download required) and
  **GitHub Actions CI** (lint + tests on Python 3.9-3.12, plus a build job).
- `pyproject.toml` packaging with optional extras (`[model]`, `[umap]`,
  `[full]`, `[dev]`) and a `.gitignore`.

### Changed
- The nine analysis scripts (`comparison.py`, `heatmap.py`, `pca.py`, `tsne.py`,
  `pumap.py`, `saliency_maps.py`, `ablation.py`, `ast-s.py`, `tf-s.py`) are now
  **thin wrappers** over the library, eliminating the per-script duplication of
  the sorting-algorithm corpus and the model-loading boilerplate.
- README rewritten around the library and CLI, with installation, API quickstart,
  capability matrix and SEO-friendly headings.
- Stronger, centralised determinism via `set_seed` (Python, NumPy, PyTorch).

## [1.1.0] - 2026-05-23
### Added
- Comprehensive README overhaul with expanded methodology, usage, results, and citation guidance.
- `requirements.txt` for reproducible environment setup.
- `CONTRIBUTING.md` with contribution and coding-style guidance.
- `docs/REPRODUCIBILITY.md` with end-to-end reproduction steps, runtimes, and determinism notes.
- `examples/demo_similarity.ipynb` notebook with an Open in Colab badge.
- GitHub issue templates for bug reports and feature requests.

## [1.0.0] - 2024-09-01
### Initial release
- Initial publication of GraphCodeBERT interpretability scripts and generated visualizations.
- MIT licensing and citation metadata.
