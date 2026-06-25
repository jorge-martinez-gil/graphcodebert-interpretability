# GraphCodeBERT Interpretability — Explaining Code Similarity Models

[![DOI](https://img.shields.io/badge/DOI-10.1142%2FS0218194025500160-blue)](https://doi.org/10.1142/S0218194025500160)
[![arXiv](https://img.shields.io/badge/arXiv-2410.05275-b31b1b.svg)](https://arxiv.org/abs/2410.05275)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/jorge-martinez-gil/graphcodebert-interpretability/actions/workflows/ci.yml/badge.svg)](https://github.com/jorge-martinez-gil/graphcodebert-interpretability/actions/workflows/ci.yml)
[![GitHub stars](https://img.shields.io/github/stars/jorge-martinez-gil/graphcodebert-interpretability?style=social)](https://github.com/jorge-martinez-gil/graphcodebert-interpretability/stargazers)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorge-martinez-gil/graphcodebert-interpretability/blob/main/examples/demo_similarity.ipynb)

**A reusable, tested Python toolkit for interpreting [GraphCodeBERT](https://huggingface.co/microsoft/graphcodebert-base) and related code language models on code-similarity tasks.** It turns the black-box embeddings of a code model into inspectable evidence: token-level similarity alignments, gradient saliency maps, PCA/t-SNE/UMAP embedding projections, structural (AST) and lexical (TF-IDF) baselines, and automated, publication-ready benchmark reports.

> **Official implementation** of:
> Martinez-Gil, J. (2025). *Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks.* *International Journal of Software Engineering and Knowledge Engineering*, 35(05), 657-678. https://doi.org/10.1142/S0218194025500160

```python
import graphcodebert_interpretability as gcbi

result = gcbi.compare(open("bubble_sort.py").read(), open("insertion_sort.py").read())
print(result.score)                       # one symmetric similarity score in [0, 1]
open("report.html", "w").write(gcbi.highlight_html(result))   # token-by-token evidence
```

---

## Contents

- [What problem does this solve?](#what-problem-does-this-solve)
- [Why is it useful and different?](#why-is-it-useful-and-different)
- [Installation](#installation)
- [60-second quickstart (Python API)](#60-second-quickstart-python-api)
- [Command-line interface](#command-line-interface)
- [What you can do](#what-you-can-do)
- [Benchmark reports (Markdown + LaTeX + figures)](#benchmark-reports-markdown--latex--figures)
- [Reproduce the paper](#reproduce-the-paper)
- [Use your own code, models and datasets](#use-your-own-code-models-and-datasets)
- [Visualization gallery](#visualization-gallery)
- [How to cite](#how-to-cite)
- [Related work](#related-work)
- [Contributing](#contributing)
- [License](#license)

## What problem does this solve?

Code language models such as GraphCodeBERT, CodeBERT, CodeT5 and UniXcoder are increasingly used for **code clone detection, code search, and code-similarity** scoring in software-engineering pipelines. But their predictions are opaque: when a model says two snippets are "similar," it does not say *why*. This makes the results hard to trust, debug, or teach.

This repository provides a **practical interpretability layer** that answers *why* by combining several complementary lenses and presenting the evidence visually and quantitatively.

## Why is it useful and different?

Most code-model repositories ship one-off scripts that reproduce a single paper's figures. This one is built as **research infrastructure you can import and extend**:

- **An installable, documented library** (`pip install`) with a small, stable API - not a folder of copy-paste scripts.
- **Works on any code**, not just the bundled examples: pass your own snippets, files, directories or JSONL corpora.
- **Model-agnostic**: point it at `microsoft/graphcodebert-base`, `microsoft/codebert-base`, `microsoft/unixcoder-base` or any compatible checkpoint to compare interpretability across models.
- **Five interpretability lenses in one place**: token-similarity, saliency, PCA, t-SNE, UMAP - plus AST and TF-IDF baselines for honest comparison.
- **Automated, publication-ready output**: one command emits a Markdown report, a `booktabs` LaTeX table and a 300-DPI comparison figure.
- **Reproducible and tested**: fixed seeds, a `pytest` suite, and continuous integration. The lightweight analyses run **without** the heavy deep-learning stack, so the math is fast and verifiable.

## Installation

```bash
pip install git+https://github.com/jorge-martinez-gil/graphcodebert-interpretability.git
```

Or from a clone:

```bash
git clone https://github.com/jorge-martinez-gil/graphcodebert-interpretability.git
cd graphcodebert-interpretability
pip install .            # core: baselines, projections, reports (no torch needed)
pip install ".[model]"   # add torch + transformers to run GraphCodeBERT
pip install ".[full]"    # everything, incl. UMAP - to reproduce the paper end to end
```

The core install is intentionally light. The deep-learning dependencies (`torch`, `transformers`) are optional extras and are only needed for the neural lenses (`compare`, `saliency`, embeddings, projections of real code).

## 60-second quickstart (Python API)

```python
import graphcodebert_interpretability as gcbi

gcbi.set_seed(42)  # reproducible runs

code_a = "def bubble(a):\n    for i in range(len(a)):\n        ...\n"
code_b = "def insertion(a):\n    for i in range(1, len(a)):\n        ...\n"

# 1. Token-level similarity with a single interpretable score
result = gcbi.compare(code_a, code_b)
print(result.score)            # symmetric max-alignment score in [0, 1]
print(result.matrix.shape)     # full token-by-token cosine alignment matrix

# 2. Save a colour-coded HTML report showing which tokens drove the score
open("similarity.html", "w").write(gcbi.highlight_html(result))

# 3. Gradient saliency: which tokens most influence the representation?
sal = gcbi.saliency(code_a)
for word, score in sorted(zip(sal.words, sal.normalized()), key=lambda x: -x[1])[:5]:
    print(f"{score:5.2f}  {word}")

# 4. Project the bundled corpus into 2D (PCA / t-SNE / UMAP)
data = gcbi.load_dataset("sorting")
labels, sim = gcbi.similarity_matrix(data)     # global pairwise heatmap data

# 5. Baselines for comparison - no model required
gcbi.ast_similarity(code_a, code_b)            # structural (AST node bag)
```

Every public function works on arbitrary strings of code, so you can drop your own snippets straight in.

## Command-line interface

Installing the package also installs the `gcbi` command:

```bash
gcbi compare a.py b.py --html report.html      # token similarity + HTML evidence
gcbi saliency a.py --out saliency.png          # gradient saliency map
gcbi heatmap --dataset sorting --out heat.png  # global similarity heatmap
gcbi project --dir ./snippets --method umap    # 2D projections of your own files
gcbi report --dataset sorting                  # MD + LaTeX + figure benchmark
gcbi report --no-neural                        # baselines only (no model download)
gcbi reproduce                                 # regenerate the full paper figure set
```

## What you can do

| Capability | API | CLI | Needs model |
|---|---|---|---|
| Token-level similarity + score | `compare`, `token_alignment` | `gcbi compare` | yes |
| HTML token-highlight report | `highlight_html` | `gcbi compare --html` | yes |
| Gradient saliency maps | `saliency` | `gcbi saliency` | yes |
| Global similarity heatmap | `similarity_matrix` | `gcbi heatmap` | yes |
| PCA / t-SNE / UMAP projection | `project`, `project_pair` | `gcbi project` | yes (1) |
| AST structural baseline | `ast_similarity`, `ast_similarity_matrix` | - | no |
| TF-IDF lexical baseline | `tfidf_similarity_matrix` | - | no |
| Benchmark report (MD/LaTeX/figure) | `generate_report` | `gcbi report` | optional |
| Reproducible seeding | `set_seed` | `--seed` | no |

(1) `project()` operates on any embedding array with no model; projecting *code* requires the model to produce embeddings first.

## Benchmark reports (Markdown + LaTeX + figures)

A single call computes pairwise similarity under three lenses (GraphCodeBERT, AST, TF-IDF) and writes a complete, citable report - useful for papers and for comparing how a neural model differs from structural/lexical baselines:

```python
gcbi.generate_report(dataset="sorting", outdir="benchmark_report")
# -> benchmark_report/report.md            (GitHub-rendered comparison table)
# -> benchmark_report/similarity_table.tex (booktabs LaTeX table for your paper)
# -> benchmark_report/similarity_comparison.png (300-DPI grouped bar figure)
```

All numbers are computed on the fly from the snippets you provide; nothing is hard-coded or fabricated.

## Reproduce the paper

Every figure in the paper is reproducible with one command:

```bash
pip install ".[full]"
gcbi reproduce          # heatmap + PCA/t-SNE/UMAP projections + benchmark report
```

The original per-figure scripts still work too (`python pca.py`, `python heatmap.py`, ...); they are now thin wrappers over the library. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for runtimes, expected outputs and determinism notes.

## Use your own code, models and datasets

```python
# Your own files or directory
data = gcbi.load_snippets_from_dir("my_project/snippets")     # {filename: code}
data = gcbi.load_jsonl("clones.jsonl")                        # {name: code}

# A different code model
emb = gcbi.sequence_embedding(code, model_name="microsoft/unixcoder-base")

# Compare two models on the same pair
for m in ["microsoft/graphcodebert-base", "microsoft/codebert-base"]:
    print(m, gcbi.compare(code_a, code_b, model_name=m).score)
```

## Visualization gallery

![PCA token projection for Bubble Sort vs Insertion Sort](Bubble_Sort_vs_Insertion_Sort_tokens_2d_pca.png)
*Figure 1. PCA token-level projection: Bubble Sort and Insertion Sort tokens occupy nearby embedding regions, reflecting their procedural similarity.*

![GraphCodeBERT similarity heatmap across sorting algorithms](sorting_algorithms_similarity.png)
*Figure 2. Pairwise GraphCodeBERT similarity heatmap across classical sorting implementations.*

## How to cite

If you use this software or build on it, please cite the paper:

```bibtex
@article{martinezgil2025augmenting,
  author  = {Martinez-Gil, Jorge},
  title   = {Augmenting the Interpretability of GraphCodeBERT for Code Similarity Tasks},
  journal = {International Journal of Software Engineering and Knowledge Engineering},
  volume  = {35},
  number  = {05},
  pages   = {657--678},
  year    = {2025},
  doi     = {10.1142/S0218194025500160},
  url     = {https://doi.org/10.1142/S0218194025500160}
}
```

A machine-readable citation is in [`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button uses it.

## Related work

- Guo et al. (2021). [GraphCodeBERT: Pre-training Code Representations with Data Flow](https://arxiv.org/abs/2009.08366).
- Feng et al. (2020). [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155).
- Vig (2019). [A Multiscale Visualization of Attention in the Transformer Model](https://arxiv.org/abs/1906.05714).
- Jain & Wallace (2019). [Attention is not Explanation](https://arxiv.org/abs/1902.10186).

## Contributing

Contributions that improve reproducibility, clarity and research impact are very welcome - new datasets, additional models, new explanation methods, and documentation. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md). Development setup:

```bash
pip install -e ".[dev]"
ruff check .
pytest -m "not model"   # fast suite; drop the filter to run model-backed tests
```

## License

Released under the [MIT License](LICENSE).
