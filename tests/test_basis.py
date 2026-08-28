"""Basis functions: recurrence correctness and orthonormality."""

import numpy as np
import pytest
from scipy.special import eval_legendre

from khaos.basis import (
    legendre_poly,
    make_basis,
    make_basis_matrix,
    ss_legendre_poly,
)


@pytest.mark.parametrize("j", range(0, 13))
def test_legendre_matches_scipy(j):
    x = np.linspace(-1, 1, 101)
    np.testing.assert_allclose(legendre_poly(x, j), eval_legendre(j, x), atol=1e-11)


def test_shifted_definition():
    """psi_a(x) = sqrt(2a+1) P_a(2x-1)."""
    x = np.linspace(0, 1, 51)
    for j in range(8):
        expected = np.sqrt(2 * j + 1) * eval_legendre(j, 2 * x - 1)
        np.testing.assert_allclose(ss_legendre_poly(x, j), expected, atol=1e-11)


def test_orthonormal_on_unit_interval():
    """<psi_a, psi_b> = delta_ab under the uniform measure on [0, 1]."""
    nodes, weights = np.polynomial.legendre.leggauss(40)
    x = 0.5 * (nodes + 1)  # map to [0, 1]
    w = 0.5 * weights
    P = np.column_stack([ss_legendre_poly(x, j) for j in range(9)])
    gram = P.T @ (w[:, None] * P)
    np.testing.assert_allclose(gram, np.eye(9), atol=1e-10)


def test_make_basis_is_a_tensor_product():
    rng = np.random.default_rng(0)
    X = rng.random((20, 4))
    variables = np.array([0, 3])
    degrees = np.array([2, 3])
    expected = ss_legendre_poly(X[:, 0], 2) * ss_legendre_poly(X[:, 3], 3)
    np.testing.assert_allclose(make_basis(variables, degrees, X), expected)


def test_make_basis_degree_zero_is_constant():
    rng = np.random.default_rng(1)
    X = rng.random((10, 2))
    np.testing.assert_allclose(make_basis([1], [0], X), np.ones(10))


def test_basis_columns_are_nearly_uncorrelated_under_uniform_inputs():
    """Design columns are orthogonal in expectation for i.i.d. uniform X."""
    rng = np.random.default_rng(2)
    X = rng.random((200_000, 3))
    B = make_basis_matrix([[0], [1], [0, 1]], [[1], [2], [1, 1]], X)
    gram = B.T @ B / X.shape[0]
    np.testing.assert_allclose(gram, np.eye(4), atol=0.02)


def test_make_basis_length_mismatch():
    with pytest.raises(ValueError):
        make_basis([0, 1], [1], np.zeros((5, 2)))
