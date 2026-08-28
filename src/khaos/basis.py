"""Orthonormal polynomial chaos basis functions.

Port of ``R/chaos_helpers.R`` (``legendre_poly``, ``ss_legendre_poly``,
``make_basis``) from the ``khaos`` R package.

Inputs are assumed to be scaled to the unit hypercube :math:`[0, 1]^p`, so the
univariate basis is the *standardised shifted* Legendre polynomial

.. math::

    \\psi_\\alpha(x) = \\sqrt{2\\alpha + 1}\\, P_\\alpha(2x - 1),

which is orthonormal with respect to the uniform measure on :math:`[0, 1]`.
Multivariate basis functions are tensor products over the active variables,

.. math::

    \\Psi_m(x \\mid \\boldsymbol{\\alpha}_m)
        = \\prod_{j=1}^{p} \\psi_{\\alpha_{mj}}(x_j).
"""

from __future__ import annotations

import numpy as np

from ._compat import njit

__all__ = [
    "legendre_poly",
    "ss_legendre_poly",
    "make_basis",
    "make_basis_matrix",
]


@njit
def _legendre_kernel(z: np.ndarray, j: int) -> np.ndarray:
    """Legendre polynomial ``P_j`` evaluated at ``z`` by the standard recurrence.

    ``(k + 1) P_{k+1}(z) = (2k + 1) z P_k(z) - k P_{k-1}(z)``
    """
    n = z.shape[0]
    if j == 0:
        return np.ones(n)
    if j == 1:
        return z.copy()

    p_prev = np.ones(n)
    p_curr = z.copy()
    for k in range(1, j):
        p_next = ((2.0 * k + 1.0) * z * p_curr - k * p_prev) / (k + 1.0)
        p_prev = p_curr
        p_curr = p_next
    return p_curr


@njit
def _ss_legendre_kernel(x: np.ndarray, j: int) -> np.ndarray:
    """Standardised shifted Legendre polynomial ``sqrt(2j+1) P_j(2x-1)``."""
    z = 2.0 * x - 1.0
    return np.sqrt(2.0 * j + 1.0) * _legendre_kernel(z, j)


@njit
def _make_basis_kernel(
    variables: np.ndarray, degrees: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Tensor-product basis column for one multi-index.

    ``variables`` holds 0-based column indices of the active inputs and
    ``degrees`` the matching (strictly positive) univariate degrees.
    """
    n = X.shape[0]
    out = np.ones(n)
    for k in range(variables.shape[0]):
        out = out * _ss_legendre_kernel(
            np.ascontiguousarray(X[:, variables[k]]), degrees[k]
        )
    return out


def legendre_poly(x, j: int) -> np.ndarray:
    """Legendre polynomial :math:`P_j(x)` (R: ``legendre_poly``)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return _legendre_kernel(np.ascontiguousarray(x), int(j))


def ss_legendre_poly(x, j: int) -> np.ndarray:
    """Standardised shifted Legendre polynomial (R: ``ss_legendre_poly``).

    Orthonormal on :math:`[0, 1]`: :math:`\\int_0^1 \\psi_a \\psi_b = \\delta_{ab}`.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return _ss_legendre_kernel(np.ascontiguousarray(x), int(j))


def make_basis(variables, degrees, X) -> np.ndarray:
    """One basis-function column (R: ``make_basis``).

    Parameters
    ----------
    variables : array_like of int
        0-based indices of the active inputs.
    degrees : array_like of int
        Univariate degrees, same length as ``variables``.
    X : ndarray of shape (n, p)
        Inputs scaled to :math:`[0, 1]`.
    """
    v = np.ascontiguousarray(np.atleast_1d(np.asarray(variables, dtype=np.int64)))
    d = np.ascontiguousarray(np.atleast_1d(np.asarray(degrees, dtype=np.int64)))
    if v.shape[0] != d.shape[0]:
        raise ValueError("variables and degrees must have the same length")
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    return _make_basis_kernel(v, d, X)


def make_basis_matrix(variables_list, degrees_list, X) -> np.ndarray:
    """Design matrix ``[1, Psi_1, ..., Psi_M]`` for a list of multi-indices."""
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    n = X.shape[0]
    M = len(variables_list)
    B = np.ones((n, M + 1))
    for m in range(M):
        B[:, m + 1] = make_basis(variables_list[m], degrees_list[m], X)
    return B
