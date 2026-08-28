"""Linear-algebra helpers and the modified g-prior's Laplace machinery."""

import warnings

import numpy as np
import pytest

from khaos.gprior import (
    build_G,
    dgsq_full,
    dgsq_orth,
    g_weight,
    laplace_full,
    laplace_orth,
    log_dgsq_full,
    log_dgsq_orth,
)
from khaos.linalg import (
    logdet_spd,
    rcond1,
    rmvnorm_eigen,
    safe_inverse,
    safe_logdet,
)


def _spd(k, seed=0, cond=None):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((k + 5, k))
    return A.T @ A + np.eye(k)


# --------------------------------------------------------------------------
def test_rcond_matches_exact_reciprocal_condition_number():
    A = _spd(6)
    assert rcond1(A) == pytest.approx(1.0 / np.linalg.cond(A, 1), rel=0.3)


def test_safe_inverse_agrees_with_numpy():
    A = _spd(8)
    np.testing.assert_allclose(safe_inverse(A), np.linalg.inv(A), atol=1e-9)


def test_safe_inverse_rejects_singular():
    A = np.ones((4, 4))
    assert safe_inverse(A) is None


def test_safe_inverse_is_symmetric():
    inv = safe_inverse(_spd(7, seed=3))
    np.testing.assert_allclose(inv, inv.T, atol=1e-12)


def test_logdet_spd():
    A = _spd(5, seed=2)
    assert logdet_spd(A) == pytest.approx(np.linalg.slogdet(A)[1])
    assert logdet_spd(-A) is None


def test_rmvnorm_eigen_moments():
    rng = np.random.default_rng(0)
    mean = np.array([1.0, -2.0, 0.5])
    sigma = _spd(3, seed=1)
    draws = np.array([rmvnorm_eigen(mean, sigma, rng) for _ in range(200_000)])
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.02)
    np.testing.assert_allclose(np.cov(draws.T), sigma, rtol=0.05, atol=0.05)


def test_rmvnorm_eigen_handles_semidefinite():
    rng = np.random.default_rng(0)
    sigma = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
    draws = np.array([rmvnorm_eigen(np.zeros(2), sigma, rng) for _ in range(500)])
    np.testing.assert_allclose(draws[:, 0], draws[:, 1], atol=1e-8)


# --------------------------------------------------------------------------
def test_g_weight_formula():
    assert g_weight(q=2, d=5, zeta=1.0) == pytest.approx((1 + 2 * (2 + 5 - 2)) ** -0.5)
    # zeta = 0 switches off the complexity penalty (standard g-prior).
    assert g_weight(q=4, d=9, zeta=0.0) == 1.0


def test_build_G_formula():
    g = np.array([1.0, 0.5, 0.25])
    G = build_G(g, g0sq=3.0)
    for m in range(3):
        for l in range(3):
            assert G[m, l] == pytest.approx(1.0 + 1.0 / (3.0 * g[m] * g[l]))


def test_gprior_precision_equals_BtB_plus_S0inv():
    """G o (B'B) is exactly B'B + S0^{-1} for S0 = g0^2 D(g)(B'B)^-1 D(g)."""
    rng = np.random.default_rng(0)
    B = rng.standard_normal((40, 5))
    BtB = B.T @ B
    g = np.array([1.0, 0.8, 0.5, 0.3, 0.2])
    g0sq = 7.0
    S0 = g0sq * np.diag(g) @ np.linalg.inv(BtB) @ np.diag(g)
    np.testing.assert_allclose(
        build_G(g, g0sq) * BtB, BtB + np.linalg.inv(S0), rtol=1e-8
    )


# --------------------------------------------------------------------------
def test_log_and_natural_scale_densities_agree():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((30, 4))
    BtB = B.T @ B
    g = np.array([1.0, 0.7, 0.4, 0.3])
    for theta in (0.5, 2.0, 50.0):
        assert np.log(dgsq_full(theta, 1.0, 1.0, g, BtB)[0]) == pytest.approx(
            log_dgsq_full(theta, 1.0, 1.0, g, BtB)
        )
        assert np.log(dgsq_orth(theta, 1.0, 1.0, g)[0]) == pytest.approx(
            log_dgsq_orth(theta, 1.0, 1.0, g)
        )


def test_densities_reject_nonpositive_theta():
    g = np.ones(3)
    assert log_dgsq_orth(-1.0, 1.0, 1.0, g) == -np.inf
    assert log_dgsq_orth(0.0, 1.0, 1.0, g) == -np.inf


def test_laplace_orth_is_centred_on_the_mode():
    """The inverse-gamma fit's mode should maximise the orthogonal posterior."""
    a, b = 2.0, 4.0
    g = np.array([1.0, 0.6, 0.35, 0.2, 0.15])
    a_t, b_t = laplace_orth(a, b, g)
    # With the reference's c = 0 the fit is centred at m_theta = b/a (neither
    # the inverse-gamma mode nor its mean -- see gprior._inv_gamma_from_mode).
    mode = b_t / a_t

    grid = mode * np.exp(np.linspace(-1.5, 1.5, 601))
    vals = np.array([log_dgsq_orth(t, a, b, g**2) for t in grid])
    assert grid[np.argmax(vals)] == pytest.approx(mode, rel=0.05)


def test_laplace_full_finds_a_stationary_point():
    rng = np.random.default_rng(3)
    B = rng.standard_normal((60, 6))
    BtB = B.T @ B
    g = np.array([1.0, 0.7, 0.5, 0.4, 0.3, 0.25])
    a, b = 1.5, 3.0
    pars = laplace_full(a, b, g, BtB)
    assert pars is not None
    a_t, b_t = pars
    mode = b_t / a_t

    grid = mode * np.exp(np.linspace(-1.0, 1.0, 401))
    vals = np.array([log_dgsq_full(t, a, b, g, BtB) for t in grid])
    assert grid[np.argmax(vals)] == pytest.approx(mode, rel=0.1)


def test_laplace_returns_none_on_degenerate_input():
    # A wildly mis-specified curvature should be reported, not silently used.
    assert laplace_orth(0.0, 0.0, np.array([1.0])) is None


# --------------------------------------------------------------------------
def test_safe_logdet_agrees_with_cholesky():
    A = _spd(6, seed=4)
    assert safe_logdet(A) == pytest.approx(np.linalg.slogdet(A)[1])


def test_safe_logdet_rejects_singular_without_warning():
    """A singular matrix is a handled outcome, not a source of RuntimeWarnings."""
    A = np.ones((4, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert safe_logdet(A) is None


def test_safe_logdet_rejects_negative_determinant():
    A = np.diag([1.0, -2.0, 3.0])
    assert safe_logdet(A) is None
