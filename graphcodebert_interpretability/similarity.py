# -*- coding: utf-8 -*-
"""Token-level and sequence-level code similarity.

The numerical core (:func:`cosine_similarity_matrix`, :func:`aggregate_similarity`)
is pure NumPy and has no model dependency, which makes it fast and fully unit
testable. The :func:`compare`/:func:`similarity_matrix` helpers add a thin
GraphCodeBERT layer on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .embeddings import sequence_embedding, token_embeddings
from .model import DEFAULT_MODEL, load_model
from .utils import NEWLINE_MARKER, SPACE_MARKER

__all__ = [
    "cosine_similarity_matrix",
    "aggregate_similarity",
    "SimilarityResult",
    "compare",
    "token_alignment",
    "similarity_matrix",
    "highlight_html",
]

_EPS = 1e-12


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the pairwise cosine-similarity matrix between two sets of vectors.

    Parameters
    ----------
    a:
        Array of shape ``(n, d)``.
    b:
        Array of shape ``(m, d)``.

    Returns
    -------
    np.ndarray
        A ``(n, m)`` matrix where entry ``[i, j]`` is the cosine similarity
        between ``a[i]`` and ``b[j]``.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + _EPS)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + _EPS)
    return a_norm @ b_norm.T


def aggregate_similarity(matrix: np.ndarray) -> float:
    """Aggregate a token alignment matrix into a single similarity score.

    Following the paper, the score is the mean of the best match for every token
    in snippet A and every token in snippet B, then averaged. This symmetric
    "max-align" score rewards snippets whose tokens each find a close counterpart
    in the other snippet.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.size == 0:
        return 0.0
    best_for_a = matrix.max(axis=1).mean()
    best_for_b = matrix.max(axis=0).mean()
    return float((best_for_a + best_for_b) / 2.0)


@dataclass
class SimilarityResult:
    """Container for a pairwise token-similarity comparison."""

    score: float
    tokens_a: List[str]
    tokens_b: List[str]
    matrix: np.ndarray = field(repr=False)
    name_a: str = "Snippet A"
    name_b: str = "Snippet B"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SimilarityResult(score={self.score:.4f}, "
            f"name_a={self.name_a!r}, name_b={self.name_b!r}, "
            f"tokens_a={len(self.tokens_a)}, tokens_b={len(self.tokens_b)})"
        )


def token_alignment(
    code_a: str,
    code_b: str,
    model=None,
    tokenizer=None,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Return the token-by-token cosine-similarity matrix for two snippets."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_name)
    _, emb_a = token_embeddings(code_a, model=model, tokenizer=tokenizer)
    _, emb_b = token_embeddings(code_b, model=model, tokenizer=tokenizer)
    return cosine_similarity_matrix(emb_a, emb_b)


def compare(
    code_a: str,
    code_b: str,
    name_a: str = "Snippet A",
    name_b: str = "Snippet B",
    model=None,
    tokenizer=None,
    model_name: str = DEFAULT_MODEL,
) -> SimilarityResult:
    """Compare two code snippets at the token level with GraphCodeBERT.

    Works on *any* code, not just the bundled sorting algorithms.

    Returns
    -------
    SimilarityResult
        Holds the aggregate score, both token lists and the alignment matrix.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_name)
    tokens_a, emb_a = token_embeddings(code_a, model=model, tokenizer=tokenizer)
    tokens_b, emb_b = token_embeddings(code_b, model=model, tokenizer=tokenizer)
    matrix = cosine_similarity_matrix(emb_a, emb_b)
    score = aggregate_similarity(matrix)
    return SimilarityResult(
        score=score,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        matrix=matrix,
        name_a=name_a,
        name_b=name_b,
    )


def similarity_matrix(
    snippets: Dict[str, str],
    pooling: str = "mean",
    model=None,
    tokenizer=None,
    model_name: str = DEFAULT_MODEL,
) -> Tuple[List[str], np.ndarray]:
    """Return a full pairwise sequence-similarity matrix across many snippets.

    This powers the global similarity heatmap.

    Returns
    -------
    (labels, matrix)
        ``labels`` preserves the input order; ``matrix`` is symmetric.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_name)
    labels = list(snippets)
    vectors = np.vstack(
        [
            sequence_embedding(
                snippets[name], pooling=pooling, model=model, tokenizer=tokenizer
            )
            for name in labels
        ]
    )
    matrix = cosine_similarity_matrix(vectors, vectors)
    return labels, matrix


# ---------------------------------------------------------------------------
# HTML highlighting of token-level similarity
# ---------------------------------------------------------------------------
def _intensity_color(intensity: int) -> str:
    """Map a 0-255 intensity to a blue -> green -> yellow -> red colour."""
    if intensity < 64:
        return "rgb(173, 216, 230)"  # light blue
    if intensity < 128:
        return "rgb(144, 238, 144)"  # light green
    if intensity < 192:
        return "rgb(255, 255, 102)"  # yellow
    return "rgb(255, 69, 0)"  # red


def _render_tokens(tokens: Sequence[str], intensities: Sequence[int]) -> str:
    parts: List[str] = []
    previous = ""
    for token, intensity in zip(tokens, intensities):
        if token == NEWLINE_MARKER:
            parts.append("<br>")
        elif token == SPACE_MARKER:
            parts.append(" ")
        else:
            if (
                previous
                and not previous.endswith((NEWLINE_MARKER, SPACE_MARKER))
                and not previous.isspace()
            ):
                parts.append(" ")
            clean = token.lstrip(SPACE_MARKER)
            clean = (
                clean.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            bg = _intensity_color(intensity)
            fg = "#000" if intensity < 192 else "#fff"
            parts.append(
                f"<span class='token' style='background-color:{bg}; "
                f"color:{fg};'>{clean}</span>"
            )
        previous = token
    return "".join(parts)


def highlight_html(
    result: SimilarityResult, similarity_threshold: float = 0.8
) -> str:
    """Render a :class:`SimilarityResult` as a self-contained HTML report.

    Tokens are coloured by their strongest cross-snippet alignment, making the
    model's notion of similarity directly inspectable.
    """
    matrix = result.matrix
    best_a = matrix.max(axis=1)
    best_b = matrix.max(axis=0)
    intensity_a = [
        int(255 * v) if v > similarity_threshold else 0 for v in best_a
    ]
    intensity_b = [
        int(255 * v) if v > similarity_threshold else 0 for v in best_b
    ]

    body_a = _render_tokens(result.tokens_a, intensity_a)
    body_b = _render_tokens(result.tokens_b, intensity_b)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GraphCodeBERT token similarity: {result.name_a} vs {result.name_b}</title>
<style>
  body {{ font-family: Arial, sans-serif; background:#f4f4f4; color:#333; padding:20px; }}
  h2 {{ color:#444; border-bottom:2px solid #ddd; padding-bottom:10px; }}
  pre {{ background:#272822; color:#f8f8f2; padding:20px; border-radius:5px;
         overflow-x:auto; font-size:16px; line-height:1.5; }}
  .token {{ display:inline-block; padding:2px 5px; border-radius:3px; }}
  .legend span {{ padding:2px 8px; border-radius:3px; margin-right:6px; }}
</style>
</head>
<body>
<p class="legend">Token alignment strength:
  <span style="background:rgb(173,216,230)">low</span>
  <span style="background:rgb(144,238,144)">medium</span>
  <span style="background:rgb(255,255,102)">high</span>
  <span style="background:rgb(255,69,0);color:#fff">very high</span>
</p>
<h2>{result.name_a}</h2><pre style="font-family:monospace;">{body_a}</pre>
<h2>{result.name_b}</h2><pre style="font-family:monospace;">{body_b}</pre>
<h2>Final Similarity Score: {result.score:.2f}</h2>
</body>
</html>
"""
