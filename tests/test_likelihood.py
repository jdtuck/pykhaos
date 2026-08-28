"""Collapsed marginal likelihoods and the incremental sufficient statistics.

The key verification here compares the (M+1) x (M+1) algebra the sampler
actually uses against an independent n x n computation of the
normal-inverse-gamma marginal likelihood

.. math::

    p(y \\mid B) \\propto |I + B S_0 B'|^{-1/2}
        \\bigl(b_\\sigma + \\tfrac12 y'(I + B S_0 B')^{-1} y\\bigr)^{-(a_\\sigma + n/2)},

which shares no code with the implementation.
"""

import numpy as np
import pytest

from khaos.likelihood import (
    GPriorState,
    RidgeState,
    _birth_stats,
    _death_stats,
    _replace_stats,
)


def _reference_log_marginal(B, S0, y, a_sigma, b_sigma):
    n = y.shape[0]
    V = np.eye(n) + B @ S0 @ B.T
    sign, logdet = np.linalg.slogdet(V)
    quad = float(y @ np.linalg.solve(V, y))
    return -0.5 * logdet - (a_sigma + n / 2.0) * np.log(b_sigma + 0.5 * quad)


def _data(n=40, k=3, seed=0):
    rng = np.random.default_rng(seed)
    B = np.column_stack([np.ones(n), rng.standard_normal((n, k))])
    y = B @ rng.standard_normal(k + 1) + rng.normal(0, 0.5, n)
    return B, y, rng


# --------------------------------------------------------------------------
# Sufficient-statistic updates
# --------------------------------------------------------------------------
def test_birth_stats_match_full_recomputation():
    B, y, rng = _data()
    b_new = rng.standard_normal(B.shape[0])
    Bn, BtBn, vn = _birth_stats(B, B.T @ B, B.T @ y, b_new, y)
    np.testing.assert_allclose(BtBn, Bn.T @ Bn, atol=1e-10)
    np.testing.assert_allclose(vn, Bn.T @ y, atol=1e-10)


def test_death_stats_match_full_recomputation():
    B, y, _ = _data()
    Bn, BtBn, vn = _death_stats(B, B.T @ B, B.T @ y, 2)
    np.testing.assert_allclose(BtBn, Bn.T @ Bn, atol=1e-10)
    np.testing.assert_allclose(vn, Bn.T @ y, atol=1e-10)


def test_replace_stats_match_full_recomputation():
    B, y, rng = _data()
    b_new = rng.standard_normal(B.shape[0])
    Bn, BtBn, vn = _replace_stats(B, B.T @ B, B.T @ y, 2, b_new, y)
    np.testing.assert_allclose(Bn[:, 2], b_new)
    np.testing.assert_allclose(BtBn, Bn.T @ Bn, atol=1e-10)
    np.testing.assert_allclose(vn, Bn.T @ y, atol=1e-10)


# --------------------------------------------------------------------------
# Ridge prior
# --------------------------------------------------------------------------
@pytest.mark.parametrize("b_sigma", [0.0, 1.5])
def test_ridge_ratio_matches_reference_marginal(b_sigma):
    B, y, rng = _data(k=3, seed=1)
    tau2, a_sigma = 3.0, 1.0
    st = RidgeState(B, y, tau2, a_sigma, b_sigma, exact_marginal=True)

    b_new = rng.standard_normal(B.shape[0])
    cand = st.propose_birth(b_new, q=1, dtot=2)
    B_cand = np.hstack([B, b_new[:, None]])

    expected = _reference_log_marginal(
        B_cand, tau2 * np.eye(B_cand.shape[1]), y, a_sigma, b_sigma
    ) - _reference_log_marginal(B, tau2 * np.eye(B.shape[1]), y, a_sigma, b_sigma)
    assert cand.loglik_ratio == pytest.approx(expected, rel=1e-8)


def test_ridge_death_is_the_inverse_of_birth():
    B, y, rng = _data(k=4, seed=2)
    st = RidgeState(B, y, tau2=10.0, a_sigma=0.0, b_sigma=0.0)
    b_new = rng.standard_normal(B.shape[0])

    birth = st.propose_birth(b_new, q=1, dtot=1)
    st.accept(birth)
    death = st.propose_death(B.shape[1] - 1)  # kill the column just added
    assert death.loglik_ratio == pytest.approx(-birth.loglik_ratio, rel=1e-8)


def test_ridge_reference_default_equals_exact_when_b_sigma_zero():
    """R drops a factor of 1/2 on the quadratic form; harmless at b_sigma = 0."""
    B, y, rng = _data(k=3, seed=4)
    b_new = rng.standard_normal(B.shape[0])
    ref = RidgeState(B, y, 5.0, 0.0, 0.0, exact_marginal=False)
    exact = RidgeState(B, y, 5.0, 0.0, 0.0, exact_marginal=True)
    assert ref.propose_birth(b_new, 1, 1).loglik_ratio == pytest.approx(
        exact.propose_birth(b_new, 1, 1).loglik_ratio, rel=1e-9
    )


# --------------------------------------------------------------------------
# Modified g-prior
# --------------------------------------------------------------------------
def test_gprior_exact_ratio_matches_reference_marginal():
    B, y, rng = _data(k=3, seed=5)
    g0sq, zeta, a_sigma, b_sigma = 6.0, 1.0, 1.0, 0.7
    st = GPriorState(B, y, g0sq, zeta, a_sigma, b_sigma, exact_marginal=True)

    b_new = rng.standard_normal(B.shape[0])
    q, dtot = 2, 4
    cand = st.propose_birth(b_new, q, dtot, g0sq)

    from khaos.gprior import g_weight

    B_cand = np.hstack([B, b_new[:, None]])
    g_curr = np.ones(B.shape[1])
    g_cand = np.append(g_curr, g_weight(q, dtot, zeta))

    def S0(Bm, g):
        D = np.diag(g)
        return g0sq * D @ np.linalg.inv(Bm.T @ Bm) @ D

    expected = _reference_log_marginal(
        B_cand, S0(B_cand, g_cand), y, a_sigma, b_sigma
    ) - _reference_log_marginal(B, S0(B, g_curr), y, a_sigma, b_sigma)
    assert cand.loglik_ratio == pytest.approx(expected, rel=1e-7)


def test_gprior_default_omits_the_S0_normaliser():
    """Reference behaviour: |S0|^(-1/2) is left out of the birth ratio."""
    B, y, rng = _data(k=3, seed=6)
    b_new = rng.standard_normal(B.shape[0])
    ref = GPriorState(B, y, 6.0, 1.0, 0.0, 0.0, exact_marginal=False)
    exact = GPriorState(B, y, 6.0, 1.0, 0.0, 0.0, exact_marginal=True)
    assert ref.propose_birth(b_new, 2, 4, 6.0).loglik_ratio != pytest.approx(
        exact.propose_birth(b_new, 2, 4, 6.0).loglik_ratio
    )


def test_gprior_zeta_zero_is_the_standard_g_prior():
    B, y, rng = _data(k=3, seed=7)
    g0sq = 4.0
    st = GPriorState(B, y, g0sq, zeta=0.0, a_sigma=0.0, b_sigma=0.0,
                     exact_marginal=True)
    b_new = rng.standard_normal(B.shape[0])
    cand = st.propose_birth(b_new, q=3, dtot=9, g0sq=g0sq)

    B_cand = np.hstack([B, b_new[:, None]])

    def S0(Bm):
        return g0sq * np.linalg.inv(Bm.T @ Bm)

    expected = _reference_log_marginal(
        B_cand, S0(B_cand), y, 0.0, 0.0
    ) - _reference_log_marginal(B, S0(B), y, 0.0, 0.0)
    assert cand.loglik_ratio == pytest.approx(expected, rel=1e-7)


def test_gprior_posterior_mean_is_the_shrunk_least_squares_estimate():
    B, y, _ = _data(k=4, seed=8)
    g0sq = 9.0
    st = GPriorState(B, y, g0sq, zeta=0.0, a_sigma=0.0, b_sigma=0.0)
    mu, Sigma = st.posterior_moments()
    ols = np.linalg.solve(B.T @ B, B.T @ y)
    np.testing.assert_allclose(mu, ols * g0sq / (1.0 + g0sq), rtol=1e-8)


def test_refresh_updates_state_for_a_new_g0sq():
    B, y, _ = _data(k=3, seed=9)
    st = GPriorState(B, y, 2.0, 1.0, 0.0, 0.0)
    old = st.ldet
    assert st.refresh(50.0)
    assert st.ldet != old
    fresh = GPriorState(B, y, 50.0, 1.0, 0.0, 0.0)
    assert st.ldet == pytest.approx(fresh.ldet)
    assert st.Q == pytest.approx(fresh.Q)


def test_ill_conditioned_candidate_is_rejected():
    """A duplicated column must be refused rather than silently inverted."""
    B, y, _ = _data(k=3, seed=10)
    st = GPriorState(B, y, 5.0, 1.0, 0.0, 0.0)
    assert st.propose_birth(B[:, 1].copy(), 1, 1, 5.0) is None

    st_r = RidgeState(B, y, 10.0, 0.0, 0.0)
    assert st_r.propose_birth(B[:, 1].copy(), 1, 1) is None
