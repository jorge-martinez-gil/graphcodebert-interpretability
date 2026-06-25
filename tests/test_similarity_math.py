"""Unit tests for the pure-NumPy similarity core (no model required)."""
import numpy as np
import pytest

from graphcodebert_interpretability import (
    aggregate_similarity,
    cosine_similarity_matrix,
)


def test_identical_vectors_have_similarity_one():
    a = np.array([[1.0, 2.0, 3.0]])
    m = cosine_similarity_matrix(a, a)
    assert m.shape == (1, 1)
    assert m[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_vectors_have_zero_similarity():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert cosine_similarity_matrix(a, b)[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_opposite_vectors_have_negative_similarity():
    a = np.array([[1.0, 0.0]])
    b = np.array([[-1.0, 0.0]])
    assert cosine_similarity_matrix(a, b)[0, 0] == pytest.approx(-1.0, abs=1e-6)


def test_matrix_shape():
    a = np.random.RandomState(0).randn(5, 8)
    b = np.random.RandomState(1).randn(3, 8)
    assert cosine_similarity_matrix(a, b).shape == (5, 3)


def test_cosine_is_scale_invariant():
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[10.0, 20.0, 30.0]])
    assert cosine_similarity_matrix(a, b)[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_aggregate_similarity_bounds_and_perfect_match():
    a = np.eye(3)
    m = cosine_similarity_matrix(a, a)
    assert aggregate_similarity(m) == pytest.approx(1.0, abs=1e-6)


def test_aggregate_similarity_empty_matrix():
    assert aggregate_similarity(np.empty((0, 0))) == 0.0


def test_zero_vector_does_not_raise():
    a = np.zeros((2, 4))
    b = np.ones((2, 4))
    # Should not divide by zero thanks to the epsilon guard.
    out = cosine_similarity_matrix(a, b)
    assert np.all(np.isfinite(out))
