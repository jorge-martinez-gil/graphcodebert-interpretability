# -*- coding: utf-8 -*-
"""Non-neural similarity baselines: AST structural and TF-IDF lexical.

These baselines are pure ``ast``/scikit-learn and require no model, providing
fast, deterministic points of comparison for the GraphCodeBERT embeddings.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "ast_features",
    "ast_similarity",
    "ast_similarity_matrix",
    "tfidf_similarity_matrix",
]


def ast_features(code: str) -> str:
    """Return a space-joined sequence of AST node-type names for ``code``.

    Returns an empty string if the code does not parse, so callers can handle
    syntactically invalid snippets gracefully.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    return " ".join(type(node).__name__ for node in ast.walk(tree))


def _cosine(matrix) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity

    return cosine_similarity(matrix)


def ast_similarity_matrix(
    snippets: Dict[str, str],
) -> Tuple[List[str], np.ndarray]:
    """Pairwise AST structural similarity across snippets.

    Node-type bags are vectorised with a count vectoriser and compared by cosine
    similarity (a bag-of-AST-nodes kernel).
    """
    from sklearn.feature_extraction.text import CountVectorizer

    labels = list(snippets)
    features = [ast_features(snippets[name]) for name in labels]
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(features)
    return labels, _cosine(matrix)


def ast_similarity(code_a: str, code_b: str) -> float:
    """AST structural similarity between exactly two snippets, in ``[0, 1]``."""
    _, matrix = ast_similarity_matrix({"a": code_a, "b": code_b})
    return float(matrix[0, 1])


def tfidf_similarity_matrix(
    snippets: Dict[str, str],
) -> Tuple[List[str], np.ndarray]:
    """Pairwise TF-IDF lexical similarity across snippets."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    labels = list(snippets)
    codes = [snippets[name] for name in labels]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(codes)
    return labels, _cosine(matrix)
