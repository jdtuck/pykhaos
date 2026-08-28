"""Proposal machinery for the adaptive (RJMCMC) KHAOS sampler.

Port of the proposal helpers in ``R/chaos_helpers.R`` (``A_size``,
``make_weights``, ``random_partition``) together with the coin-flipping
proposal probabilities that appear inline in the birth/death steps of
``adaptive_khaos_ridge`` / ``adaptive_khaos_gprior``.

The coin-flipping proposal (Rumsey et al., 2026, Sec. 3.2) replaces the NKD
scheme of Nott et al. (2005):

1. draw an expected interaction order :math:`q_0 \\in \\{1, \\dots, q_{\\max}\\}`
   with weights proportional to ``q0_weights(q0)`` (default :math:`1/q_0`);
2. build per-variable inclusion probabilities :math:`\\eta_j` that sum to
   :math:`q_0` and are inflated for variables already active in the model;
3. flip an independent coin per variable, :math:`\\chi_j \\sim \\mathrm{Bern}(\\eta_j)`;
4. reject and repeat if :math:`\\sum_j \\chi_j` is 0 or exceeds :math:`q_{\\max}`
   (the "delayed rejection" component, whose density is accumulated into the
   acceptance ratio);
5. draw the total degree, then a random composition of that degree over the
   active variables.
"""

from __future__ import annotations

import math

import numpy as np

from ._compat import njit

__all__ = [
    "A_size",
    "log_A_size",
    "make_weights",
    "random_partition",
    "WeightCache",
    "log_coinflip_prob",
]


# --------------------------------------------------------------------------
# Size of the admissible multi-index set
# --------------------------------------------------------------------------
def A_size(p: int, d: int, q: int) -> float:
    """Cardinality of :math:`\\mathcal{A}_{p,d,q}` (R: ``A_size``).

    .. math::

        |\\mathcal{A}_{p,d,q}|
            = \\sum_{i=1}^{q} \\sum_{j=1}^{d} \\binom{p}{i}\\binom{j-1}{i-1}
    """
    total = 0
    for qq in range(1, int(q) + 1):
        cpq = math.comb(int(p), qq) if qq <= p else 0
        if cpq == 0:
            continue
        inner = 0
        for dd in range(1, int(d) + 1):
            if dd - 1 >= qq - 1:
                inner += math.comb(dd - 1, qq - 1)
        total += cpq * inner
    return float(total)


def log_A_size(p: int, d: int, q: int) -> float:
    """``log(A_size(p, d, q))``, computed in log-space to avoid overflow."""
    terms = []
    for qq in range(1, int(q) + 1):
        if qq > p:
            continue
        lcpq = math.lgamma(p + 1) - math.lgamma(qq + 1) - math.lgamma(p - qq + 1)
        inner = 0
        for dd in range(1, int(d) + 1):
            if dd - 1 >= qq - 1:
                inner += math.comb(dd - 1, qq - 1)
        if inner > 0:
            terms.append(lcpq + math.log(inner))
    if not terms:
        return -math.inf
    mx = max(terms)
    return mx + math.log(sum(math.exp(t - mx) for t in terms))


# --------------------------------------------------------------------------
# Coin-flip inclusion probabilities
# --------------------------------------------------------------------------
@njit
def _make_weights_kernel(
    eta: np.ndarray, p0: float, epsilon: float, alpha: float, num_passes: int
) -> np.ndarray:
    p = eta.shape[0]
    base = np.power(eta, alpha) + epsilon
    v = base / np.sum(base) * p0 / p

    logv = np.log(v)
    # v_j == 1 (which happens when p0 == p, e.g. a single input) makes the
    # exponent solve degenerate: every weight is already 1 and any exponent
    # works.  The R implementation divides by log(1) = 0 and returns NaN here.
    n_free = 0
    for j in range(p):
        if logv[j] != 0.0:
            n_free += 1
    if n_free == 0:
        return np.ones(p)

    delta = 0.0
    beta = 1.0
    for _ in range(num_passes):
        acc = 0.0
        for j in range(p):
            if logv[j] != 0.0:
                acc += (math.log(p0 + delta) - math.log(p)) / logv[j]
        beta = acc / n_free
        s = 0.0
        for j in range(p):
            s += v[j] ** beta
        delta = delta + p0 - s

    out = np.power(v, beta)
    for j in range(p):
        if out[j] < 0.0:
            out[j] = 0.0
        elif out[j] > 1.0:
            out[j] = 1.0
    return out


def make_weights(eta, p0, epsilon, alpha, num_passes) -> np.ndarray:
    """Per-variable inclusion probabilities (R: ``make_weights``).

    Starting from base weights :math:`(\\eta_j^{\\alpha} + \\epsilon)`, the
    exponent ``beta`` is tuned over ``num_passes`` fixed-point passes so that
    the resulting probabilities sum (approximately) to ``p0``, the target
    expected interaction order.

    Notes
    -----
    ``eta`` is the running count of how many times each variable has been
    *added* to the model, so variables that have proved useful get proposed
    more often.  With ``eta = 0`` for every variable the weights are uniform.

    Mirrors the R argument order exactly, including the fact that the R
    implementation calls this as ``make_weights(eta, q0, coin_pars[[3]],
    coin_pars[[2]], coin_pars[[4]])`` -- i.e. ``epsilon`` receives
    ``coin_pars[[3]]`` and ``alpha`` receives ``coin_pars[[2]]``, which is the
    transpose of what the R documentation says.  :func:`khaos.adaptive.CoinPars`
    preserves that behaviour so results match the reference implementation.

    One deliberate departure: when ``p0 == len(eta)`` (a single input, or
    ``q_max == p``) every weight is already 1 and R's exponent solve divides by
    ``log(1) = 0``, returning ``NaN``.  This implementation returns the
    all-ones vector, which is the limit R is aiming at.
    """
    eta = np.ascontiguousarray(np.asarray(eta, dtype=float))
    return _make_weights_kernel(
        eta, float(p0), float(epsilon), float(alpha), int(num_passes)
    )


class WeightCache:
    """Caches ``make_weights(eta, q0, ...)`` for ``q0 = 1..order``.

    The reference R code recomputes these weights inside every birth/death
    acceptance-ratio loop.  They depend only on ``eta`` (which changes at most
    once per iteration) so caching them is a pure speed-up, not a change of
    algorithm.
    """

    __slots__ = ("order", "epsilon", "alpha", "num_passes", "_cache", "_maxsize")

    def __init__(self, order: int, epsilon: float, alpha: float, num_passes: int,
                 maxsize: int = 4):
        self.order = int(order)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.num_passes = int(num_passes)
        self._cache: dict[bytes, np.ndarray] = {}
        self._maxsize = int(maxsize)

    def weights(self, eta: np.ndarray) -> np.ndarray:
        """``(order, p)`` matrix whose row ``j-1`` is ``make_weights(eta, j)``."""
        eta = np.ascontiguousarray(np.asarray(eta, dtype=float))
        key = eta.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        p = eta.shape[0]
        W = np.empty((self.order, p))
        for j in range(1, self.order + 1):
            W[j - 1] = make_weights(eta, j, self.epsilon, self.alpha, self.num_passes)
        if len(self._cache) >= self._maxsize:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = W
        return W


def log_coinflip_prob(
    W: np.ndarray,
    log_J_probs: np.ndarray,
    log_Hjs: float,
    active: np.ndarray,
) -> float:
    """Log marginal probability of proposing exactly the variable set ``active``.

    Marginalises over the latent expected-order draw :math:`q_0`:

    .. math::

        \\pi(\\chi) = \\sum_{q_0} \\frac{w_{q_0}}{\\sum_k w_k}
                     \\prod_{j} \\eta_j(q_0)^{\\chi_j}(1 - \\eta_j(q_0))^{1-\\chi_j}

    Parameters
    ----------
    W : ndarray of shape (order, p)
        Inclusion probabilities from :class:`WeightCache`.
    active : ndarray of int
        0-based indices of the selected variables.

    Notes
    -----
    When ``active`` is empty this returns ``0.0`` (probability 1).  That is not
    a mathematical statement -- it reproduces an R indexing quirk in the
    reference code, where ``wts[-integer(0)]`` yields an *empty* vector rather
    than the whole vector, so both product terms collapse to 1.  The value is
    only ever used inside the delayed-rejection accumulator, where the
    contribution of a rejected all-zero draw is meant to be neutral.
    """
    order, p = W.shape
    if active.shape[0] == 0:
        return 0.0

    mask = np.zeros(p, dtype=bool)
    mask[active] = True

    # Weights can hit exactly 1 (when q_max == p), giving log1p(-1) = -inf.
    # That is the correct value -- such a variable is proposed with certainty --
    # so the -inf is intentional rather than an error.
    with np.errstate(divide="ignore"):
        terms = (
            log_J_probs
            - log_Hjs
            + np.log(W[:, mask]).sum(axis=1)
            + np.log1p(-W[:, ~mask]).sum(axis=1)
        )
    mx = terms.max()
    if not np.isfinite(mx):
        return -np.inf
    return float(mx + np.log(np.exp(terms - mx).sum()))


# --------------------------------------------------------------------------
# Degree partitions
# --------------------------------------------------------------------------
def random_partition(d: int, q: int, rng: np.random.Generator) -> np.ndarray:
    """Split total degree ``d`` into ``q`` parts, each at least 1.

    Port of R ``random_partition``: place ``q - 1`` cut points uniformly (with
    replacement) in ``0..d-q``, sort them, and take successive differences.
    """
    d = int(d)
    q = int(q)
    if d < q:
        raise ValueError("d must be greater than or equal to q")
    remaining = d - q
    if q == 1:
        return np.array([d], dtype=np.int64)
    cuts = np.sort(rng.integers(0, remaining + 1, size=q - 1))
    padded = np.concatenate(([0], cuts, [remaining]))
    additions = np.diff(padded)
    return (1 + additions).astype(np.int64)
