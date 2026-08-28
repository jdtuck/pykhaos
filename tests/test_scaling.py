"""Input rescaling onto the unit hypercube."""

import numpy as np
import pytest

from khaos import InputScaler, adaptive_khaos, sobol_khaos
from khaos.basis import ss_legendre_poly
from khaos.scaling import resolve_scaler


# --------------------------------------------------------------------------
# InputScaler
# --------------------------------------------------------------------------
def test_from_data_maps_training_range_onto_the_unit_cube():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3.0, 17.0, (200, 4))
    sc = InputScaler.from_data(X)
    Z = sc.transform(X)
    np.testing.assert_allclose(Z.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.max(axis=0), 1.0, atol=1e-12)


def test_round_trip_is_exact():
    rng = np.random.default_rng(1)
    X = rng.uniform(-100.0, 5.0, (50, 3))
    sc = InputScaler.from_data(X)
    np.testing.assert_allclose(sc.inverse_transform(sc.transform(X)), X, atol=1e-10)


def test_the_map_is_affine_per_column():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 10, (40, 2))
    sc = InputScaler.from_range((0.0, 10.0), 2)
    np.testing.assert_allclose(sc.transform(X), X / 10.0)


def test_from_range_accepts_scalars_and_per_column_bounds():
    X = np.array([[0.0, 100.0], [10.0, 200.0]])
    per_col = InputScaler.from_range([[0.0, 100.0], [10.0, 200.0]], 2)
    np.testing.assert_allclose(per_col.transform(X), [[0.0, 0.0], [1.0, 1.0]])

    shared = InputScaler.from_range((0.0, 10.0), 2)
    np.testing.assert_allclose(shared.transform(X)[:, 0], [0.0, 1.0])


def test_from_range_rejects_bad_shapes_and_inverted_bounds():
    with pytest.raises(ValueError, match="x_range"):
        InputScaler.from_range(np.zeros((3, 4)), 4)
    with pytest.raises(ValueError, match="upper bounds"):
        InputScaler.from_range([[5.0, 0.0], [1.0, 1.0]], 2)


def test_constant_column_is_mapped_to_one_half_with_a_warning():
    X = np.column_stack([np.linspace(0, 5, 20), np.full(20, 7.0)])
    with pytest.warns(UserWarning, match="constant"):
        sc = InputScaler.from_data(X)
    Z = sc.transform(X)
    assert np.all(Z[:, 1] == 0.5)
    assert np.isfinite(Z).all()


def test_identity_map_is_a_no_op():
    rng = np.random.default_rng(3)
    X = rng.random((10, 3))
    sc = InputScaler.identity_map(3)
    assert sc.identity
    np.testing.assert_array_equal(sc.transform(X), X)


def test_transform_checks_the_column_count():
    sc = InputScaler.from_range((0.0, 1.0), 3)
    with pytest.raises(ValueError, match="columns"):
        sc.transform(np.zeros((5, 2)))


def test_out_of_range_and_clipping():
    sc = InputScaler.from_range((0.0, 10.0), 1)
    Z = sc.transform(np.array([[-1.0], [5.0], [11.0]]))
    np.testing.assert_array_equal(sc.out_of_range(Z), [True, False, True])
    Zc = sc.transform(np.array([[-1.0], [11.0]]), clip=True)
    np.testing.assert_allclose(Zc.ravel(), [0.0, 1.0])


# --------------------------------------------------------------------------
# resolve_scaler policy
# --------------------------------------------------------------------------
def test_auto_leaves_already_scaled_data_untouched():
    rng = np.random.default_rng(4)
    X = rng.random((50, 3))
    assert resolve_scaler(X, "auto", None).identity


def test_auto_rescales_out_of_range_data():
    rng = np.random.default_rng(5)
    X = rng.uniform(2.0, 9.0, (50, 3))
    sc = resolve_scaler(X, "auto", None)
    assert not sc.identity
    assert sc.transform(X).max() == pytest.approx(1.0)


def test_true_always_rescales_even_when_already_in_range():
    X = np.linspace(0.2, 0.8, 20)[:, None]
    sc = resolve_scaler(X, True, None)
    assert not sc.identity
    np.testing.assert_allclose(sc.transform(X).ravel()[[0, -1]], [0.0, 1.0])


def test_false_disables_rescaling_and_warns():
    rng = np.random.default_rng(6)
    X = rng.uniform(0, 10, (30, 2))
    with pytest.warns(UserWarning, match=r"scaled to \[0, 1\]"):
        sc = resolve_scaler(X, False, None)
    assert sc.identity


def test_x_range_overrides_the_data_range():
    X = np.array([[2.0], [8.0]])
    sc = resolve_scaler(X, "auto", (0.0, 10.0))
    np.testing.assert_allclose(sc.transform(X).ravel(), [0.2, 0.8])


def test_x_range_warns_when_training_data_escapes_it():
    X = np.array([[-1.0], [5.0]])
    with pytest.warns(UserWarning, match="outside the supplied x_range"):
        resolve_scaler(X, "auto", (0.0, 10.0))


def test_x_range_with_scaling_disabled_is_an_error():
    with pytest.raises(ValueError, match="x_range was given"):
        resolve_scaler(np.zeros((3, 1)), False, (0.0, 1.0))


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="scale_inputs"):
        resolve_scaler(np.zeros((3, 1)), "sometimes", None)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------
def _shifted_problem(n=250, lo=-40.0, hi=90.0, seed=0):
    """The same two-term truth as elsewhere, on an arbitrary input box."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(lo, hi, (n, 3))
    Z = (X - lo) / (hi - lo)
    y = (
        ss_legendre_poly(Z[:, 0], 1)
        + 2.0 * ss_legendre_poly(Z[:, 1], 1)
        + rng.normal(0, 0.05, n)
    )
    return X, y, (lo, hi)


def test_fit_works_on_arbitrarily_scaled_inputs():
    X, y, rng_box = _shifted_problem()
    fit = adaptive_khaos(
        X, y, degree=6, order=2, nmcmc=3000, nburn=2000, x_range=rng_box,
        seed=3, verbose=False,
    )
    yhat = fit.predict().mean(axis=0)
    assert np.sqrt(np.mean((y - yhat) ** 2)) < 0.15
    assert 2 <= np.median(fit.nbasis) <= 5


def test_predict_takes_and_returns_user_units():
    X, y, box = _shifted_problem()
    fit = adaptive_khaos(
        X, y, degree=6, order=2, nmcmc=3000, nburn=2000, x_range=box,
        seed=3, verbose=False,
    )
    rng = np.random.default_rng(99)
    Xt = rng.uniform(box[0], box[1], (300, 3))
    Zt = (Xt - box[0]) / (box[1] - box[0])
    yt = ss_legendre_poly(Zt[:, 0], 1) + 2.0 * ss_legendre_poly(Zt[:, 1], 1)
    # Xt is passed in raw units; no manual scaling anywhere.
    assert np.sqrt(np.mean((yt - fit.predict(Xt).mean(axis=0)) ** 2)) < 0.15


def test_fit_stores_raw_and_scaled_inputs():
    X, y, box = _shifted_problem(n=80)
    fit = adaptive_khaos(X, y, nmcmc=400, nburn=200, x_range=box, seed=0,
                         verbose=False)
    np.testing.assert_allclose(fit.X, X)
    assert fit.X_scaled.min() >= 0.0 and fit.X_scaled.max() <= 1.0
    np.testing.assert_allclose(fit.scaler.inverse_transform(fit.X_scaled), X,
                               atol=1e-9)


def test_rescaling_is_equivalent_to_scaling_by_hand():
    """A shifted/stretched problem must give exactly the pre-scaled answer."""
    X, y, box = _shifted_problem(n=120, seed=7)
    Z = (X - box[0]) / (box[1] - box[0])
    kw = dict(degree=6, order=2, nmcmc=1200, nburn=600, seed=5, verbose=False)
    auto = adaptive_khaos(X, y, x_range=box, **kw)
    manual = adaptive_khaos(Z, y, **kw)
    np.testing.assert_array_equal(auto.nbasis, manual.nbasis)
    np.testing.assert_allclose(auto.beta, manual.beta, equal_nan=True)
    np.testing.assert_allclose(auto.predict(X), manual.predict(Z))


def test_already_scaled_fit_is_unchanged_by_the_default():
    """'auto' must not perturb data that is already on the cube."""
    rng = np.random.default_rng(8)
    X = rng.random((120, 3))
    y = ss_legendre_poly(X[:, 0], 2) + rng.normal(0, 0.05, 120)
    kw = dict(degree=6, order=2, nmcmc=1200, nburn=600, seed=2, verbose=False)
    a = adaptive_khaos(X, y, **kw)
    b = adaptive_khaos(X, y, scale_inputs=False, **kw)
    np.testing.assert_array_equal(a.nbasis, b.nbasis)
    np.testing.assert_allclose(a.beta, b.beta, equal_nan=True)


def test_predicting_outside_the_scaled_range_warns():
    X, y, box = _shifted_problem(n=80)
    fit = adaptive_khaos(X, y, nmcmc=400, nburn=200, x_range=box, seed=0,
                         verbose=False)
    far = np.full((3, 3), box[1] * 10.0)
    with pytest.warns(UserWarning, match="outside the range"):
        fit.predict(far)


def test_predict_rejects_the_wrong_number_of_columns():
    X, y, box = _shifted_problem(n=60)
    fit = adaptive_khaos(X, y, nmcmc=300, nburn=150, x_range=box, seed=0,
                         verbose=False)
    with pytest.raises(ValueError, match="columns"):
        fit.predict(np.zeros((5, 2)))


def test_sobol_is_invariant_to_the_input_units():
    """Rescaling is affine, so the variance decomposition is unchanged."""
    X, y, box = _shifted_problem(n=200, seed=11)
    Z = (X - box[0]) / (box[1] - box[0])
    kw = dict(degree=6, order=2, nmcmc=2500, nburn=1500, seed=4, verbose=False)
    raw = sobol_khaos(adaptive_khaos(X, y, x_range=box, **kw))
    pre = sobol_khaos(adaptive_khaos(Z, y, **kw))
    np.testing.assert_allclose(np.nanmean(raw.T, axis=0),
                               np.nanmean(pre.T, axis=0))


def test_non_finite_inputs_are_rejected():
    rng = np.random.default_rng(0)
    X = rng.random((20, 2))
    y = rng.normal(size=20)
    X[3, 1] = np.nan
    with pytest.raises(ValueError, match="X contains non-finite"):
        adaptive_khaos(X, y, nmcmc=50, nburn=10, verbose=False)
    X[3, 1] = 0.5
    y[7] = np.inf
    with pytest.raises(ValueError, match="y contains non-finite"):
        adaptive_khaos(X, y, nmcmc=50, nburn=10, verbose=False)
