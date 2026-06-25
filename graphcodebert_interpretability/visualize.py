# -*- coding: utf-8 -*-
"""Publication-quality plotting helpers.

``matplotlib`` is imported lazily so importing the package never requires it.
When ``out_path`` is given the figure is saved at 300 DPI and closed; otherwise
the figure is returned for further customisation or interactive display.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "scatter_projection",
    "plot_similarity_heatmap",
    "plot_saliency_pair",
    "plot_saliency",
]

_COLOR_A = "#d62728"  # red
_COLOR_B = "#1f77b4"  # blue


def _save_or_return(fig, out_path: Optional[str]):
    if out_path is not None:
        fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        return out_path
    return fig


def scatter_projection(
    coords: np.ndarray,
    split_index: int,
    label_a: str,
    label_b: str,
    title: Optional[str] = None,
    out_path: Optional[str] = None,
):
    """Scatter a 2D projection, colouring the two snippets' tokens differently."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    coords = np.asarray(coords)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.scatter(
        coords[:split_index, 0], coords[:split_index, 1],
        color=_COLOR_A, s=50, alpha=0.8, label=label_a,
    )
    ax.scatter(
        coords[split_index:, 0], coords[split_index:, 1],
        color=_COLOR_B, s=50, alpha=0.8, label=label_b,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title)
    ax.legend()
    return _save_or_return(fig, out_path)


def plot_similarity_heatmap(
    labels: Sequence[str],
    matrix: np.ndarray,
    title: str = "Code similarity",
    out_path: Optional[str] = None,
    cmap: str = "coolwarm",
):
    """Render a labelled, annotated similarity heatmap."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    im = ax.imshow(matrix, cmap=cmap, vmin=matrix.min(), vmax=matrix.max())
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center",
                color="black" if 0.25 < value < 0.85 else "white", fontsize=9,
            )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save_or_return(fig, out_path)


def plot_saliency_pair(
    words_a: Sequence[str],
    scores_a: Sequence[float],
    words_b: Sequence[str],
    scores_b: Sequence[float],
    name_a: str,
    name_b: str,
    title: Optional[str] = None,
    out_path: Optional[str] = None,
):
    """Overlay two normalised saliency profiles for visual comparison."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    ax.bar(range(len(scores_a)), scores_a, color=_COLOR_A, alpha=0.7, label=name_a)
    ax.bar(range(len(scores_b)), scores_b, color=_COLOR_B, alpha=0.7, label=name_b)
    ax.set_xticks([])
    ax.set_xlabel("Token position")
    ax.set_ylabel("Normalised saliency")
    ax.set_title(title or f"Saliency: {name_a} vs {name_b}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, out_path)


def plot_saliency(
    words: Sequence[str],
    scores: Sequence[float],
    name: str = "Snippet",
    top_k: Optional[int] = None,
    out_path: Optional[str] = None,
):
    """Horizontal bar chart of per-word saliency, optionally top-k only."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    words = list(words)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)
    if top_k is not None:
        order = order[-top_k:]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(order))), dpi=300)
    ax.barh([words[i] for i in order], scores[order], color=_COLOR_B)
    ax.set_xlabel("Saliency")
    ax.set_title(f"Token saliency: {name}")
    fig.tight_layout()
    return _save_or_return(fig, out_path)
