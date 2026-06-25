# Contributing

Thank you for your interest in improving this project. Contributions that improve
**reproducibility**, **clarity**, and **research impact** are especially welcome.

## Ways to contribute

- Add a new **dataset** loader or bundled corpus (e.g. clone-detection benchmarks).
- Add support for another **code model** checkpoint and compare interpretability lenses.
- Add a new **explanation method** (e.g. attention rollout, integrated gradients).
- Improve **visualizations**, documentation, or tutorials.
- Report bugs in the library, scripts, notebooks, or generated outputs.

Please open a GitHub Issue first to discuss larger changes:
<https://github.com/jorge-martinez-gil/graphcodebert-interpretability/issues>.

## Project layout

```
graphcodebert_interpretability/   # the installable library
  datasets.py      # single source of truth for bundled code corpora
  model.py         # lazy, cached model/tokenizer loading
  embeddings.py    # token- and sequence-level embeddings
  similarity.py    # cosine math, compare(), HTML highlighting
  saliency.py      # gradient saliency maps
  projection.py    # PCA / t-SNE / UMAP
  baselines.py     # AST + TF-IDF baselines (no model needed)
  visualize.py     # publication-quality plotting helpers
  report.py        # benchmark report: Markdown + LaTeX + figure
  cli.py           # the `gcbi` command
tests/             # pytest suite (fast tests need no model)
*.py               # thin reproduction scripts (pca.py, heatmap.py, ...)
```

The heavy dependencies (`torch`, `transformers`) are imported **lazily inside
functions**, so the library imports and the baseline/projection/report code run
without the deep-learning stack. Please preserve this property: do not add
top-level `import torch` to modules other than where it is already isolated.

## Development setup

```bash
pip install -e ".[dev]"     # editable install + ruff + pytest + build
```

## Running checks before a pull request

```bash
ruff check .                # lint + import sorting (must pass)
pytest -m "not model" -q    # fast suite, no model download (must pass)
pytest -m model             # optional: model-backed integration tests
```

CI runs the lint + fast suite on Python 3.9-3.12 and builds the distribution.

## Coding style

- Follow **PEP 8**; `ruff` enforces style and import order (line length 100).
- Add a clear **docstring** to every public function and module.
- Every new feature should ship with **tests** and **documentation**.
- Keep numerical claims **reproducible**: never hard-code or fabricate results.

## Reproducing results locally

```bash
pip install ".[full]"
gcbi reproduce              # regenerates the full paper figure set
```

For full reproducibility details (runtimes, deterministic settings) see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
