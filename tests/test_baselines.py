"""Tests for the model-free AST and TF-IDF baselines."""
import numpy as np
import pytest

from graphcodebert_interpretability import (
    ast_features,
    ast_similarity,
    ast_similarity_matrix,
    tfidf_similarity_matrix,
)


def test_ast_features_extracts_node_types():
    features = ast_features("def f(x):\n    return x + 1\n")
    assert "FunctionDef" in features
    assert "Return" in features


def test_ast_features_invalid_code_returns_empty():
    assert ast_features("def (((") == ""


def test_identical_code_has_ast_similarity_one():
    code = "def f(x):\n    return x + 1\n"
    assert ast_similarity(code, code) == pytest.approx(1.0, abs=1e-6)


def test_ast_similarity_matrix_is_symmetric(sorting_snippets):
    labels, matrix = ast_similarity_matrix(sorting_snippets)
    assert matrix.shape == (len(labels), len(labels))
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-6)


def test_tfidf_similarity_matrix_is_symmetric(sorting_snippets):
    labels, matrix = tfidf_similarity_matrix(sorting_snippets)
    assert matrix.shape == (len(labels), len(labels))
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-6)


def test_similarity_values_in_unit_interval(sorting_snippets):
    _, ast_m = ast_similarity_matrix(sorting_snippets)
    _, tf_m = tfidf_similarity_matrix(sorting_snippets)
    for m in (ast_m, tf_m):
        assert m.min() >= -1e-9
        assert m.max() <= 1.0 + 1e-9
