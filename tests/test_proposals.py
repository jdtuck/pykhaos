"""Proposal machinery: admissible-set size, coin flips, degree partitions."""

import itertools
import math

import numpy as np
import pytest

from khaos.proposals import (
    A_size,
    WeightCache,
    log_A_size,
    log_coinflip_prob,
    make_weights,
    random_partition,
)


def _brute_force_A(p, d, q):
    """Enumerate A_{p,d,q} directly: all nonzero multi-indices within budget."""
    count = 0
    for alpha in itertools.product(range(d + 1), repeat=p):
        deg = sum(alpha)
        order = sum(a > 0 for a in alpha)
        if 0 < order <= q and deg <= d:
            count += 1
    return count


@pytest.mark.parametrize(
    "p,d,q", [(2, 3, 2), (3, 4, 2), (3, 5, 3), (4, 4, 2), (1, 6, 1), (5, 3, 3)]
)
def test_A_size_matches_enumeration(p, d, q):
    assert A_size(p, d, q) == _brute_force_A(p, d, q)


@pytest.mark.parametrize("p,d,q", [(5, 10, 3), (20, 15, 5), (100, 15, 5)])
def test_log_A_size_agrees(p, d, q):
    assert log_A_size(p, d, q) == pytest.approx(math.log(A_size(p, d, q)))


# --------------------------------------------------------------------------
def test_make_weights_are_probabilities_summing_to_q0():
    rng = np.random.default_rng(0)
    eta = rng.integers(0, 30, size=12).astype(float)
    for q0 in (1, 2, 3, 5):
        w = make_weights(eta, q0, epsilon=2.0, alpha=1.0, num_passes=3)
        assert np.all(w > 0) and np.all(w <= 1)
        # The fixed-point calibration targets E[sum(chi)] = q0.
        assert w.sum() == pytest.approx(q0, rel=0.02)


def test_make_weights_uniform_when_no_history():
    w = make_weights(np.zeros(8), 2, epsilon=2.0, alpha=1.0, num_passes=3)
    np.testing.assert_allclose(w, w[0])


def test_make_weights_favour_used_variables():
    eta = np.array([0.0, 0.0, 50.0, 0.0])
    w = make_weights(eta, 2, epsilon=2.0, alpha=1.0, num_passes=3)
    assert w[2] > w[0]


def test_weight_cache_matches_direct_call():
    eta = np.array([3.0, 0.0, 7.0, 1.0, 0.0])
    cache = WeightCache(order=3, epsilon=2.0, alpha=1.0, num_passes=3)
    W = cache.weights(eta)
    for j in range(1, 4):
        np.testing.assert_allclose(
            W[j - 1], make_weights(eta, j, 2.0, 1.0, 3)
        )
    # A second call must return the cached array, unchanged.
    np.testing.assert_array_equal(W, cache.weights(eta))


def test_coinflip_probabilities_normalise():
    """Marginalising over q0, the subset probabilities form a distribution."""
    p, order = 5, 3
    eta = np.array([2.0, 0.0, 5.0, 1.0, 0.0])
    cache = WeightCache(order, 2.0, 1.0, 3)
    W = cache.weights(eta)
    J = 1.0 / np.arange(1, order + 1)
    log_J, log_H = np.log(J), math.log(J.sum())

    total = 0.0
    for k in range(1, p + 1):
        for subset in itertools.combinations(range(p), k):
            total += math.exp(
                log_coinflip_prob(W, log_J, log_H, np.array(subset))
            )
    # Plus the empty set, which the helper special-cases.
    empty = sum(
        (J[j] / J.sum()) * np.prod(1.0 - W[j]) for j in range(order)
    )
    assert total + empty == pytest.approx(1.0, abs=1e-10)


def test_coinflip_empty_set_returns_zero():
    W = np.full((2, 3), 0.4)
    J = np.array([1.0, 0.5])
    assert log_coinflip_prob(W, np.log(J), math.log(J.sum()),
                             np.array([], dtype=int)) == 0.0


# --------------------------------------------------------------------------
@pytest.mark.parametrize("d,q", [(1, 1), (5, 1), (5, 5), (12, 3), (15, 5)])
def test_random_partition_is_valid(d, q):
    rng = np.random.default_rng(0)
    for _ in range(200):
        part = random_partition(d, q, rng)
        assert part.shape == (q,)
        assert part.sum() == d
        assert np.all(part >= 1)


def test_random_partition_covers_every_composition():
    rng = np.random.default_rng(1)
    seen = {tuple(random_partition(5, 3, rng)) for _ in range(5000)}
    expected = {
        c for c in itertools.product(range(1, 4), repeat=3) if sum(c) == 5
    }
    assert seen == expected


def test_random_partition_rejects_bad_input():
    with pytest.raises(ValueError):
        random_partition(2, 3, np.random.default_rng(0))
