"""Small linear-algebra utilities matching the R reference implementation."""

from __future__ import annotations

import numpy as np
from scipy.linalg import lapack

__all__ = ["rcond1", "safe_inverse", "logdet_spd", "rmvnorm_eigen"]


def rcond1(A: np.ndarray) -> float:
    """Reciprocal 1-norm condition estimate, as R's ``rcond(A)`` computes it.

    Uses LAPACK ``dgetrf`` + ``dgecon``, the same routines R calls.
    """
    A = np.asarray(A, dtype=float, order="F")
    if A.size == 0:
        return np.inf
    anorm = float(np.abs(A).sum(axis=0).max())
    if not np.isfinite(anorm):
        return np.nan
    lu, _piv, info = lapack.dgetrf(A)
    if info != 0:
        return 0.0
    rc, info = lapack.dgecon(lu, anorm, norm="1")
    if info != 0:
        return 0.0
    return float(rc)


def safe_inverse(A: np.ndarray, tol: float = 1e-9):
    """Inverse of a symmetric positive-definite matrix, or ``None``.

    Port of the ``safe_solve`` closure defined inside ``adaptive_khaos_*``:
    bail out when the reciprocal condition number falls below ``tol`` or the
    Cholesky factorisation fails.  Returning ``None`` means "reject this move
    immediately", exactly as ``FALSE`` does in R.

    Under i.i.d. uniform inputs the columns of the design matrix are
    uncorrelated, so the default tolerance is safe unless ``X`` is extremely
    correlated.
    """
    A = np.asarray(A, dtype=float)
    rc = rcond1(A)
    if not np.isfinite(rc) or rc < tol:
        return None
    c, info = lapack.dpotrf(np.asfortranarray(A), lower=0, clean=1)
    if info != 0:
        return None
    inv, info = lapack.dpotri(c, lower=0)
    if info != 0:
        return None
    # dpotri fills only one triangle.
    inv = np.triu(inv) + np.triu(inv, 1).T
    return inv


def logdet_spd(A: np.ndarray):
    """``log|A|`` for symmetric positive-definite ``A``; ``None`` if not SPD."""
    c, info = lapack.dpotrf(np.asfortranarray(np.asarray(A, dtype=float)),
                            lower=0, clean=1)
    if info != 0:
        return None
    return float(2.0 * np.log(np.diag(c)).sum())


def rmvnorm_eigen(
    mean: np.ndarray, sigma: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Single draw from ``N(mean, sigma)`` using ``mvtnorm::rmvnorm``'s default.

    ``mvtnorm`` uses the symmetric eigendecomposition with negative eigenvalues
    clipped at zero, which tolerates the numerically semi-definite covariance
    matrices that show up for near-collinear bases.
    """
    mean = np.asarray(mean, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    vals, vecs = np.linalg.eigh(sigma)
    # eigh returns ascending; mvtnorm's eigen() is descending. Order is
    # irrelevant for the resulting distribution.
    root = vecs @ (vecs.T * np.sqrt(np.maximum(vals, 0.0))[:, None])
    z = rng.standard_normal(mean.shape[0])
    return mean + root @ z
