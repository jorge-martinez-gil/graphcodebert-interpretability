# -*- coding: utf-8 -*-
"""Dimensionality reduction of embeddings (PCA, t-SNE, UMAP).

:func:`project` is pure NumPy/scikit-learn and operates on an embedding array,
so it is fully unit testable without a model. Convenience helpers build the
embeddings from code first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .embeddings import token_embeddings
from .model import DEFAULT_MODEL, load_model

__all__ = ["ProjectionResult", "project", "project_pair", "VALID_METHODS"]

VALID_METHODS = ("pca", "tsne", "umap")


@dataclass
class ProjectionResult:
    """2D coordinates produced by a dimensionality-reduction method."""

    coords: np.ndarray = field(repr=False)
    method: str
    labels: Optional[List[str]] = None
    split_index: Optional[int] = None


def project(
    embeddings: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Reduce an ``(n, d)`` embedding array to ``(n, n_components)``.

    Parameters
    ----------
    method:
        One of ``"pca"``, ``"tsne"`` or ``"umap"``.
    random_state:
        Fixed seed for reproducible projections (t-SNE/UMAP and randomized PCA).
    kwargs:
        Forwarded to the underlying estimator (e.g. ``perplexity`` for t-SNE,
        ``n_neighbors``/``min_dist`` for UMAP).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        # Perplexity must stay below the sample count.
        perplexity = kwargs.pop("perplexity", min(30, max(2, len(embeddings) - 1)))
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
            init="pca",
            **kwargs,
        )
    elif method == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "UMAP projection requires 'umap-learn'. Install it with: "
                "pip install graphcodebert-interpretability[umap]"
            ) from exc

        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=kwargs.pop("n_neighbors", min(5, max(2, len(embeddings) - 1))),
            min_dist=kwargs.pop("min_dist", 0.3),
            **kwargs,
        )
    else:  # pragma: no cover - simple guard
        raise ValueError(
            f"Unknown method {method!r}; choose from {VALID_METHODS}."
        )

    return reducer.fit_transform(embeddings)


def project_pair(
    code_a: str,
    code_b: str,
    method: str = "pca",
    name_a: str = "Snippet A",
    name_b: str = "Snippet B",
    model=None,
    tokenizer=None,
    model_name: str = DEFAULT_MODEL,
    **kwargs,
) -> ProjectionResult:
    """Project the token embeddings of two snippets into a shared 2D space."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_name)
    tokens_a, emb_a = token_embeddings(code_a, model=model, tokenizer=tokenizer)
    tokens_b, emb_b = token_embeddings(code_b, model=model, tokenizer=tokenizer)
    combined = np.concatenate([emb_a, emb_b], axis=0)
    coords = project(combined, method=method, **kwargs)
    return ProjectionResult(
        coords=coords,
        method=method,
        labels=[name_a] * len(tokens_a) + [name_b] * len(tokens_b),
        split_index=len(tokens_a),
    )
