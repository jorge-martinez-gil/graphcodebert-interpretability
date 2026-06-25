# -*- coding: utf-8 -*-
"""GraphCodeBERT Interpretability.

A reusable toolkit for interpreting GraphCodeBERT (and related code language
models) on code-similarity tasks: token-level similarity, gradient saliency,
PCA/t-SNE/UMAP projections, structural/lexical baselines and automated,
publication-ready benchmark reports.

Reference
---------
Martinez-Gil, J. (2025). *Augmenting the Interpretability of GraphCodeBERT for
Code Similarity Tasks*. International Journal of Software Engineering and
Knowledge Engineering, 35(05), 657-678. https://doi.org/10.1142/S0218194025500160

Quick start
-----------
>>> import graphcodebert_interpretability as gcbi
>>> result = gcbi.compare("def f(x): return x+1", "def g(y): return y+1")
>>> round(result.score, 2)  # doctest: +SKIP
0.97
"""
from __future__ import annotations

from ._version import __version__

# --- Lightweight, model-free API (safe to import without torch) -------------
from .baselines import (
    ast_features,
    ast_similarity,
    ast_similarity_matrix,
    tfidf_similarity_matrix,
)
from .datasets import (
    SORTING_ALGORITHMS,
    list_datasets,
    load_dataset,
    load_jsonl,
    load_snippets_from_dir,
    slug,
)

# --- Model-backed API (imports torch lazily *inside* the functions) ---------
from .embeddings import embed, sequence_embedding, token_embeddings
from .model import DEFAULT_MODEL, clear_cache, load_model
from .projection import ProjectionResult, project, project_pair
from .report import generate_report, pairwise_rows, to_latex, to_markdown
from .saliency import SaliencyResult, saliency
from .similarity import (
    SimilarityResult,
    aggregate_similarity,
    compare,
    cosine_similarity_matrix,
    highlight_html,
    similarity_matrix,
    token_alignment,
)
from .utils import merge_subwords, resolve_device, set_seed

__all__ = [
    "__version__",
    # similarity
    "compare",
    "token_alignment",
    "similarity_matrix",
    "cosine_similarity_matrix",
    "aggregate_similarity",
    "highlight_html",
    "SimilarityResult",
    # embeddings / model
    "embed",
    "token_embeddings",
    "sequence_embedding",
    "load_model",
    "clear_cache",
    "DEFAULT_MODEL",
    # saliency
    "saliency",
    "SaliencyResult",
    # projection
    "project",
    "project_pair",
    "ProjectionResult",
    # baselines
    "ast_features",
    "ast_similarity",
    "ast_similarity_matrix",
    "tfidf_similarity_matrix",
    # datasets
    "SORTING_ALGORITHMS",
    "load_dataset",
    "list_datasets",
    "load_snippets_from_dir",
    "load_jsonl",
    "slug",
    # report
    "generate_report",
    "pairwise_rows",
    "to_markdown",
    "to_latex",
    # utils
    "set_seed",
    "resolve_device",
    "merge_subwords",
]
