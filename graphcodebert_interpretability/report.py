# -*- coding: utf-8 -*-
"""Automated, reproducible benchmark reports.

Computes pairwise code similarity under three lenses -- GraphCodeBERT (neural),
AST (structural) and TF-IDF (lexical) -- and emits a Markdown report, a
publication-ready LaTeX table and a comparison figure. All numbers are computed
on the fly from the provided snippets; nothing is hard-coded or fabricated.

The table-formatting helpers (:func:`pairwise_rows`, :func:`to_markdown`,
:func:`to_latex`) are pure and unit testable without a model.
"""
from __future__ import annotations

import itertools
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .baselines import ast_similarity_matrix, tfidf_similarity_matrix
from .datasets import load_dataset

__all__ = [
    "pairwise_rows",
    "to_markdown",
    "to_latex",
    "compute_matrices",
    "generate_report",
    "METHODS",
]

METHODS = ("GraphCodeBERT", "AST", "TF-IDF")


def pairwise_rows(
    labels: List[str],
    matrices: Dict[str, np.ndarray],
) -> List[dict]:
    """Flatten per-method similarity matrices into one row per snippet pair.

    Parameters
    ----------
    labels:
        Snippet names (matrix index order).
    matrices:
        Mapping of method name -> symmetric ``(n, n)`` similarity matrix.

    Returns
    -------
    list of dict
        Each row has ``pair`` plus one key per method.
    """
    rows: List[dict] = []
    for i, j in itertools.combinations(range(len(labels)), 2):
        row = {"pair": f"{labels[i]} vs {labels[j]}"}
        for method, matrix in matrices.items():
            row[method] = float(np.asarray(matrix)[i, j])
        rows.append(row)
    return rows


def _method_keys(rows: List[dict]) -> List[str]:
    return [k for k in rows[0] if k != "pair"] if rows else []


def to_markdown(rows: List[dict]) -> str:
    """Render pairwise rows as a GitHub-flavoured Markdown table."""
    if not rows:
        return ""
    methods = _method_keys(rows)
    header = "| Pair | " + " | ".join(methods) + " |"
    sep = "|" + "---|" * (len(methods) + 1)
    lines = [header, sep]
    for row in rows:
        cells = " | ".join(f"{row[m]:.4f}" for m in methods)
        lines.append(f"| {row['pair']} | {cells} |")
    return "\n".join(lines)


def to_latex(
    rows: List[dict],
    caption: str = "Pairwise code similarity across interpretability lenses.",
    label: str = "tab:similarity",
) -> str:
    """Render pairwise rows as a ``booktabs`` LaTeX table."""
    if not rows:
        return ""
    methods = _method_keys(rows)
    col_spec = "l" + "r" * len(methods)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Pair & " + " & ".join(methods) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = " & ".join(f"{row[m]:.4f}" for m in methods)
        pair = row["pair"].replace("_", r"\_")
        lines.append(f"{pair} & {cells} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def compute_matrices(
    snippets: Dict[str, str],
    model_name: Optional[str] = None,
    include_neural: bool = True,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Compute similarity matrices for every available method.

    When ``include_neural`` is ``False`` the GraphCodeBERT lens is skipped, which
    keeps the report fully runnable without the deep-learning stack.
    """
    labels = list(snippets)
    matrices: Dict[str, np.ndarray] = {}

    if include_neural:
        from .model import DEFAULT_MODEL
        from .similarity import similarity_matrix

        _, gcb = similarity_matrix(
            snippets, model_name=model_name or DEFAULT_MODEL
        )
        matrices["GraphCodeBERT"] = gcb

    _, ast_mat = ast_similarity_matrix(snippets)
    matrices["AST"] = ast_mat
    _, tfidf_mat = tfidf_similarity_matrix(snippets)
    matrices["TF-IDF"] = tfidf_mat

    return labels, matrices


def _comparison_figure(rows: List[dict], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    methods = _method_keys(rows)
    pairs = [r["pair"] for r in rows]
    x = np.arange(len(pairs))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 1.1), 6), dpi=300)
    for k, method in enumerate(methods):
        values = [r[method] for r in rows]
        ax.bar(x + k * width, values, width, label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0, 1)
    ax.set_title("Pairwise code similarity by interpretability lens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_report(
    snippets: Optional[Dict[str, str]] = None,
    dataset: str = "sorting",
    outdir: str = "benchmark_report",
    model_name: Optional[str] = None,
    include_neural: bool = True,
) -> Dict[str, str]:
    """Generate a full benchmark report into ``outdir``.

    Returns a mapping of artefact name -> file path for the Markdown report,
    LaTeX table and comparison figure.
    """
    if snippets is None:
        snippets = load_dataset(dataset)

    os.makedirs(outdir, exist_ok=True)
    labels, matrices = compute_matrices(
        snippets, model_name=model_name, include_neural=include_neural
    )
    rows = pairwise_rows(labels, matrices)

    md_path = os.path.join(outdir, "report.md")
    tex_path = os.path.join(outdir, "similarity_table.tex")
    fig_path = os.path.join(outdir, "similarity_comparison.png")

    markdown = to_markdown(rows)
    latex = to_latex(rows)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Code Similarity Benchmark Report\n\n")
        handle.write(
            f"Dataset: **{dataset}** ({len(labels)} snippets, "
            f"{len(rows)} pairs).\n\n"
        )
        handle.write("Methods compared: " + ", ".join(matrices.keys()) + ".\n\n")
        handle.write("## Pairwise similarity\n\n")
        handle.write(markdown + "\n\n")
        handle.write("![Comparison](similarity_comparison.png)\n")
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(latex + "\n")

    _comparison_figure(rows, fig_path)

    return {"markdown": md_path, "latex": tex_path, "figure": fig_path}
