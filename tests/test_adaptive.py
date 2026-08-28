"""End-to-end behaviour of the RJMCMC sampler."""

import numpy as np
import pytest

from khaos import CoinPars, adaptive_khaos
from khaos.basis import ss_legendre_poly


def _linear_data(n=200, seed=0, noise=0.05):
    """y = psi_1(x1) + 2 psi_1(x2), i.e. exactly two basis functions."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 4))
    y = (
        ss_legendre_poly(X[:, 0], 1)
        + 2.0 * ss_legendre_poly(X[:, 1], 1)
        + rng.normal(0, noise, n)
    )
    return X, y


@pytest.fixture(scope="module")
def linear_fit():
    X, y = _linear_data()
    return adaptive_khaos(
        X, y, degree=6, order=2, nmcmc=4000, nburn=3000, seed=11, verbose=False
    )


# --------------------------------------------------------------------------
def test_output_shapes_are_consistent(linear_fit):
    fit = linear_fit
    k = fit.n_samples
    assert fit.nbasis.shape == (k,)
    assert fit.beta.shape[0] == k
    assert fit.s2.shape == (k,) and fit.lam.shape == (k,)
    assert fit.vars.shape[:2] == fit.nint.shape
    assert fit.degs.shape == fit.vars.shape
    for i in range(k):
        M = fit.nbasis[i]
        assert np.all(np.isfinite(fit.beta[i, : M + 1]))
        assert np.all(fit.nint[i, :M] >= 1)
        assert np.all(fit.dtot[i, :M] >= fit.nint[i, :M])


def test_thinning_and_burn_in_are_respected():
    X, y = _linear_data(n=80)
    fit = adaptive_khaos(
        X, y, nmcmc=1000, nburn=500, thin=5, seed=1, verbose=False
    )
    assert fit.n_samples == len(range(500, 1000, 5))


def test_recovers_a_two_term_expansion(linear_fit):
    fit = linear_fit
    # The truth needs exactly two basis functions; allow a little slack.
    assert 2 <= np.median(fit.nbasis) <= 5
    yhat = fit.predict().mean(axis=0)
    assert np.sqrt(np.mean((fit.y - yhat) ** 2)) < 0.15
    # Residual variance should land near the simulated noise level.
    assert np.median(fit.s2) < 0.02


def test_identifies_the_active_variables(linear_fit):
    fit = linear_fit
    used = np.zeros(fit.X.shape[1])
    for i in range(fit.n_samples):
        for m in range(fit.nbasis[i]):
            k = fit.nint[i, m]
            used[fit.vars[i, m, :k]] += 1
    used /= fit.n_samples
    assert used[0] > 0.5 and used[1] > 0.5
    assert used[2] < 0.2 and used[3] < 0.2


def test_predictions_extrapolate_to_new_points(linear_fit):
    rng = np.random.default_rng(99)
    Xt = rng.random((300, 4))
    yt = ss_legendre_poly(Xt[:, 0], 1) + 2.0 * ss_legendre_poly(Xt[:, 1], 1)
    yhat = linear_fit.predict(Xt).mean(axis=0)
    assert np.sqrt(np.mean((yt - yhat) ** 2)) < 0.15


def test_predict_shapes_and_nugget(linear_fit):
    fit = linear_fit
    assert fit.predict().shape == (fit.n_samples, fit.X.shape[0])
    assert fit.predict(mcmc_use=[0, 1, 2]).shape == (3, fit.X.shape[0])
    assert fit.predict(nugget=True, nreps=4, seed=0).shape == (
        4 * fit.n_samples,
        fit.X.shape[0],
    )
    # Adding the nugget must widen the predictive spread.
    clean = fit.predict(mcmc_use=[0]).std()
    noisy = fit.predict(mcmc_use=[0], nugget=True, nreps=200, seed=0).std()
    assert noisy > clean


def test_seeding_is_reproducible():
    X, y = _linear_data(n=100, seed=3)
    kw = dict(nmcmc=600, nburn=300, seed=7, verbose=False)
    a = adaptive_khaos(X, y, **kw)
    b = adaptive_khaos(X, y, **kw)
    np.testing.assert_array_equal(a.nbasis, b.nbasis)
    np.testing.assert_allclose(a.beta, b.beta, equal_nan=True)
    c = adaptive_khaos(X, y, **{**kw, "seed": 8})
    assert not np.array_equal(a.nbasis, c.nbasis) or not np.allclose(
        a.s2, c.s2
    )


@pytest.mark.parametrize("prior_type", ["ridge", "gprior"])
def test_both_priors_fit(prior_type):
    X, y = _linear_data(n=120, seed=5)
    fit = adaptive_khaos(
        X, y, prior_type=prior_type, degree=6, order=2, nmcmc=2500,
        nburn=1500, seed=4, verbose=False,
    )
    assert fit.prior_type == prior_type
    assert (fit.g2 is not None) == (prior_type == "gprior")
    yhat = fit.predict().mean(axis=0)
    assert np.sqrt(np.mean((y - yhat) ** 2)) < 0.3


def test_gprior_g2_samplers_all_run():
    X, y = _linear_data(n=60, seed=6)
    for scheme in ("f", "lf", "lo", "mh", "mho", "mhoo"):
        fit = adaptive_khaos(
            X, y, prior_type="gprior", degree=4, order=2, nmcmc=400,
            nburn=200, g2_sample=scheme, seed=2, verbose=False,
        )
        assert np.all(fit.g2 > 0) and np.all(np.isfinite(fit.g2))
    with pytest.warns(UserWarning):
        adaptive_khaos(
            X, y, prior_type="gprior", nmcmc=60, nburn=30,
            g2_sample="nonsense", seed=2, verbose=False,
        )


def test_max_basis_is_enforced():
    X, y = _linear_data(n=100, seed=8)
    fit = adaptive_khaos(
        X, y, nmcmc=1500, nburn=500, max_basis=3, seed=3, verbose=False
    )
    assert fit.nbasis.max() <= 3


def test_degree_and_order_limits_are_respected():
    X, y = _linear_data(n=120, seed=9)
    fit = adaptive_khaos(
        X, y, degree=5, order=2, nmcmc=1500, nburn=500, seed=3, verbose=False
    )
    assert fit.nint.max() <= 2
    assert fit.dtot.max() <= 5


def test_custom_coin_pars_run():
    X, y = _linear_data(n=80, seed=10)
    cp = CoinPars(q0_weights=lambda j: j ** -2.0, base_weight=0.5,
                  exponent=1.0, num_passes=2)
    fit = adaptive_khaos(
        X, y, order=3, nmcmc=800, nburn=400, coin_pars=cp, seed=1,
        verbose=False,
    )
    assert fit.n_samples == 400


def test_acceptance_bookkeeping(linear_fit):
    fit = linear_fit
    assert set(fit.count_accept) == set(fit.count_propose)
    for k in fit.count_propose:
        assert fit.count_accept[k] <= fit.count_propose[k]
    rates = fit.acceptance_rates()
    assert all(np.isnan(v) or 0 <= v <= 1 for v in rates.values())


# --------------------------------------------------------------------------
def test_input_validation():
    X, y = _linear_data(n=30)
    with pytest.raises(ValueError):
        adaptive_khaos(X, y[:10], nmcmc=10, nburn=1, verbose=False)
    with pytest.raises(ValueError):
        adaptive_khaos(X, y, degree=2, order=3, nmcmc=10, nburn=1, verbose=False)
    with pytest.raises(ValueError):
        adaptive_khaos(X, y, nmcmc=10, nburn=10, verbose=False)
    with pytest.raises(ValueError):
        adaptive_khaos(X, y, prior_type="banana", nmcmc=10, nburn=1,
                       verbose=False)


def test_unscaled_inputs_are_rescaled_by_default():
    """Out-of-range inputs are handled, not merely complained about."""
    rng = np.random.default_rng(0)
    X = rng.random((40, 2)) * 10
    y = rng.normal(size=40)
    fit = adaptive_khaos(X, y, nmcmc=50, nburn=10, seed=0, verbose=False)
    assert not fit.scaler.identity
    assert fit.X_scaled.max() <= 1.0 and fit.X_scaled.min() >= 0.0


def test_warns_when_rescaling_is_switched_off():
    rng = np.random.default_rng(0)
    X = rng.random((40, 2)) * 10
    y = rng.normal(size=40)
    with pytest.warns(UserWarning, match="scaled"):
        adaptive_khaos(X, y, scale_inputs=False, nmcmc=50, nburn=10, seed=0,
                       verbose=False)


def test_order_is_capped_at_p():
    """order > p is silently reduced, as in the R implementation."""
    rng = np.random.default_rng(0)
    X = rng.random((60, 2))
    y = ss_legendre_poly(X[:, 0], 2) + rng.normal(0, 0.05, 60)
    fit = adaptive_khaos(X, y, order=5, degree=6, nmcmc=600, nburn=300,
                         seed=0, verbose=False)
    assert fit.nint.max() <= 2


def test_accepts_one_dimensional_x():
    rng = np.random.default_rng(0)
    x = rng.random(80)
    y = ss_legendre_poly(x, 2) + rng.normal(0, 0.05, 80)
    fit = adaptive_khaos(x, y, degree=4, order=1, nmcmc=800, nburn=400,
                         seed=0, verbose=False)
    assert fit.predict().shape == (400, 80)


# --------------------------------------------------------------------------
# Reference-compatibility switches
# --------------------------------------------------------------------------
def _has_duplicate_variables(fit):
    for i in range(fit.n_samples):
        for m in range(fit.nbasis[i]):
            k = fit.nint[i, m]
            v = fit.vars[i, m, :k]
            if len(np.unique(v)) != k:
                return True
    return False


def test_default_swap_never_duplicates_a_variable():
    X, y = _linear_data(n=150, seed=20)
    fit = adaptive_khaos(
        X, y, degree=8, order=3, nmcmc=4000, nburn=1000, seed=5, verbose=False
    )
    assert not _has_duplicate_variables(fit)


def test_legacy_swap_reproduces_the_reference_quirk():
    """R's variable swap can put the same input in a term twice."""
    X, y = _linear_data(n=150, seed=20)
    fit = adaptive_khaos(
        X, y, degree=8, order=3, nmcmc=4000, nburn=1000, seed=5,
        legacy_swap=True, verbose=False,
    )
    assert _has_duplicate_variables(fit)


def test_sync_g2_improves_transdimensional_mixing():
    """Comparing models at a common g0^2 is what unsticks birth/death."""
    rng = np.random.default_rng(21)
    X = rng.random((250, 5))
    y = (
        3 * ss_legendre_poly(X[:, 0], 1)
        + 2 * ss_legendre_poly(X[:, 1], 2)
        + ss_legendre_poly(X[:, 2], 1) * ss_legendre_poly(X[:, 3], 1)
        + rng.normal(0, 0.3, 250)
    )
    kw = dict(prior_type="gprior", degree=8, order=3, nmcmc=3000, nburn=1500,
              seed=6, verbose=False)
    synced = adaptive_khaos(X, y, sync_g2=True, **kw)
    stale = adaptive_khaos(X, y, sync_g2=False, **kw)
    assert (
        synced.acceptance_rates()["Birth"] > stale.acceptance_rates()["Birth"]
    )
