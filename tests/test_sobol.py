"""Sobol index extraction from the posterior."""

import numpy as np
import pytest

from khaos import adaptive_khaos, sobol_khaos
from khaos.basis import ss_legendre_poly


@pytest.fixture(scope="module")
def additive_fit():
    """y = psi_1(x1) + 2 psi_1(x2): true S1 = 1/5, S2 = 4/5, no interaction."""
    rng = np.random.default_rng(0)
    X = rng.random((300, 3))
    y = (
        ss_legendre_poly(X[:, 0], 1)
        + 2.0 * ss_legendre_poly(X[:, 1], 1)
        + rng.normal(0, 0.02, 300)
    )
    return adaptive_khaos(
        X, y, degree=6, order=2, nmcmc=4000, nburn=3000, seed=12, verbose=False
    )


def test_recovers_analytic_variance_shares(additive_fit):
    res = sobol_khaos(additive_fit)
    T = np.nanmean(res.T, axis=0)
    assert T[0] == pytest.approx(0.2, abs=0.06)
    assert T[1] == pytest.approx(0.8, abs=0.06)
    assert T[2] < 0.05


def test_no_spurious_interaction_terms(additive_fit):
    res = sobol_khaos(additive_fit)
    inter = {
        lab: np.nanmean(res.S[:, i])
        for i, lab in enumerate(res.labels)
        if ":" in lab
    }
    assert all(v < 0.05 for v in inter.values())


def test_indices_are_a_partition_of_unity(additive_fit):
    res = sobol_khaos(additive_fit)
    total = res.S.sum(axis=1) + res.leftover
    np.testing.assert_allclose(total, 1.0, atol=1e-8)


def test_total_effects_dominate_first_order(additive_fit):
    res = sobol_khaos(additive_fit)
    assert np.all(res.T >= res.first_order - 1e-12)


def test_shapes_and_labels(additive_fit):
    res = sobol_khaos(additive_fit)
    k = additive_fit.n_samples
    assert res.S.shape[0] == k
    assert res.T.shape == (k, additive_fit.X.shape[1])
    assert res.first_order.shape == res.T.shape
    assert len(res.labels) == res.S.shape[1]
    assert all(lab.startswith("x") for lab in res.labels)
    # Labels are ordered by interaction order, then lexicographically.
    orders = [lab.count(":") for lab in res.labels]
    assert orders == sorted(orders)


def test_interaction_is_detected():
    """A pure interaction term should carry essentially all the variance."""
    rng = np.random.default_rng(1)
    X = rng.random((300, 3))
    y = 3.0 * ss_legendre_poly(X[:, 0], 1) * ss_legendre_poly(
        X[:, 1], 1
    ) + rng.normal(0, 0.02, 300)
    fit = adaptive_khaos(X, y, degree=6, order=2, nmcmc=4000, nburn=3000,
                         seed=13, verbose=False)
    res = sobol_khaos(fit)
    means = dict(zip(res.labels, np.nanmean(res.S, axis=0)))
    assert means.get("x1:x2", 0.0) > 0.85
    assert np.nanmean(res.first_order[:, 0]) < 0.1


def test_method_on_the_fit_object(additive_fit):
    res = additive_fit.sobol()
    assert res.S.shape[0] == additive_fit.n_samples


def test_summary_quantiles(additive_fit):
    s = additive_fit.sobol().summary(q=(0.1, 0.9))
    assert s["T"].shape == (2, additive_fit.X.shape[1])
    assert np.all(s["T"][0] <= s["T"][1] + 1e-12)
