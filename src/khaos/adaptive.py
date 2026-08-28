"""Bayesian adaptive polynomial chaos expansions via reversible-jump MCMC.

Python port of ``adaptive_khaos_ridge`` and ``adaptive_khaos_gprior`` from the
``khaos`` R package, implementing the method of

    Rumsey, K. N., Francom, D., Gibson, G., Tucker, J. D. and Huerta, G. (2026).
    "Bayesian Adaptive Polynomial Chaos Expansions." *Stat*, 15(1), e70151.

which adapts the BMARS RJMCMC sampler of Francom & Sanso (2020) to a
polynomial chaos basis, with a coin-flipping proposal for data-driven
interaction selection (in place of the NKD scheme of Nott et al., 2005).

Model
-----
.. math::

    y_i = f(x_i) + \\epsilon_i, \\qquad \\epsilon_i \\sim N(0, \\sigma^2),

    f(x) = \\beta_0 + \\sum_{m=1}^{M} \\beta_m \\Psi_m(x \\mid \\boldsymbol\\alpha_m),
    \\qquad
    \\Psi_m(x \\mid \\boldsymbol\\alpha_m) = \\prod_{j=1}^{p} \\psi_{\\alpha_{mj}}(x_j)

with :math:`\\boldsymbol\\alpha_m \\sim \\mathrm{Unif}(\\mathcal{A}_{p,d_{\\max},q_{\\max}})`,
:math:`M \\mid \\lambda \\sim \\mathrm{Poisson}(\\lambda)` and
:math:`\\lambda \\sim \\mathrm{Gamma}(a_M, b_M)`.  Two coefficient priors are
available: an independent ridge prior and the modified g-prior of the paper
(see :mod:`khaos.gprior`).

Sampler
-------
Each sweep performs one reversible-jump move -- **birth**, **death**, or
**mutate** (either a re-draw of the degree partition or a single-variable swap,
chosen adaptively from their running acceptance rates) -- followed by Gibbs
updates of :math:`\\beta`, :math:`\\sigma^2`, :math:`\\lambda` and, for the
g-prior, a Metropolis-Hastings update of :math:`g_0^2`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import norm

from .basis import make_basis
from .gprior import laplace_full, laplace_orth, log_dgsq_full, log_dgsq_orth
from .likelihood import GPriorState, RidgeState
from .linalg import rmvnorm_eigen
from .model import AdaptiveKhaos
from .proposals import (
    WeightCache,
    log_A_size,
    log_coinflip_prob,
    random_partition,
)

__all__ = ["CoinPars", "adaptive_khaos", "adaptive_khaos_ridge",
           "adaptive_khaos_gprior"]


# --------------------------------------------------------------------------
# Proposal tuning
# --------------------------------------------------------------------------
@dataclass
class CoinPars:
    """Control parameters for the coin-flipping variable proposal.

    Mirrors R's ``coin_pars = list(function(j) 1/j, 1, 2, 3)``.

    Attributes
    ----------
    q0_weights : callable
        Unnormalised weights for the expected interaction order
        :math:`q_0 \\in \\{1, \\dots, q_{\\max}\\}`.  Default :math:`1/q_0`
        (``s_q = 1`` in the paper).
    base_weight : float
        ``coin_pars[[2]]``.  Also the additive floor in the variable-swap
        mutation, ``probs = 1e-9 + base_weight + eta``.
    exponent : float
        ``coin_pars[[3]]``.
    num_passes : int
        ``coin_pars[[4]]``; fixed-point passes used to calibrate the weights.

    Notes
    -----
    The R code calls ``make_weights(eta, q0, coin_pars[[3]], coin_pars[[2]],
    coin_pars[[4]])`` against the signature ``make_weights(eta, p0, epsilon,
    alpha, num_passes)`` -- so ``exponent`` lands in ``epsilon`` and
    ``base_weight`` lands in ``alpha``, the reverse of what the R docs say.
    That wiring is preserved here so the port reproduces the reference
    implementation; with the defaults the base weights are
    ``eta**1 + 2``.
    """

    q0_weights: Callable[[np.ndarray], np.ndarray] = lambda j: 1.0 / j
    base_weight: float = 1.0
    exponent: float = 2.0
    num_passes: int = 3

    def make_weights_args(self):
        """``(epsilon, alpha, num_passes)`` in :func:`make_weights` order."""
        return self.exponent, self.base_weight, self.num_passes


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _as_matrix(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be a 2-d array, DataFrame, or 1-d vector")
    return np.ascontiguousarray(X)


def _degree_probs(q: int, degree: int, degree_penalty: float) -> np.ndarray:
    """Weights over total degrees ``q, q+1, ..., degree``.

    R: ``seq_along(q:degree)^(-degree_penalty)``, normalised.  Larger
    ``degree_penalty`` pushes mass toward low-degree terms.
    """
    w = np.arange(1, degree - q + 2, dtype=float) ** (-degree_penalty)
    return w / w.sum()


def _logsumexp(terms: np.ndarray) -> float:
    terms = np.asarray(terms, dtype=float)
    mx = terms.max() if terms.size else -np.inf
    if not np.isfinite(mx):
        return -np.inf
    return float(mx + np.log(np.exp(terms - mx).sum()))


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------
def _fit(
    X,
    y,
    prior_type: str,
    degree: int,
    order: int,
    nmcmc: int,
    nburn: int,
    thin: int,
    max_basis: int,
    move_probs: Sequence[float],
    coin_pars: CoinPars,
    degree_penalty: float,
    s2_lower: float,
    a_sigma: float,
    b_sigma: float,
    a_M: float,
    b_M: float,
    # ridge only
    tau2: float,
    # g-prior only
    a_g: float,
    b_g: float,
    zeta: float,
    g2_sample: str,
    g2_init: Optional[float],
    sync_g2: bool,
    exact_marginal: bool,
    legacy_swap: bool,
    rcond_tol: float,
    rng: np.random.Generator,
    verbose: bool,
) -> AdaptiveKhaos:
    X = _as_matrix(X)
    y = np.asarray(y, dtype=float).ravel()

    if X.max() > 1 or X.min() < 0:
        import warnings

        warnings.warn(
            "Inputs are expected to be scaled to [0, 1]. Is this intentional?",
            stacklevel=3,
        )
    if y.shape[0] != X.shape[0]:
        raise ValueError("len(y) must equal X.shape[0]")

    n, p = X.shape
    order = min(int(order), p)
    degree = int(degree)
    if degree < order:
        raise ValueError("degree must be at least as large as order")
    if nburn >= nmcmc:
        raise ValueError("nburn must be smaller than nmcmc")

    move_probs = np.asarray(move_probs, dtype=float)
    if move_probs.shape != (3,):
        raise ValueError("move_probs must have length 3 (birth, death, mutate)")

    # Proposal constants ---------------------------------------------------
    q0_grid = np.arange(1, order + 1)
    J_probs = np.asarray(coin_pars.q0_weights(q0_grid), dtype=float)
    if J_probs.shape != q0_grid.shape:
        raise ValueError("q0_weights must be vectorised over 1..order")
    Hjs = float(J_probs.sum())
    log_J = np.log(J_probs)
    log_H = math.log(Hjs)
    logA = log_A_size(p, degree, order)

    eps_w, alpha_w, passes_w = coin_pars.make_weights_args()
    wcache = WeightCache(order, eps_w, alpha_w, passes_w)
    dprob_cache = {q: _degree_probs(q, degree, degree_penalty)
                   for q in range(1, order + 1)}

    # State ----------------------------------------------------------------
    vars_cur: list[np.ndarray] = []
    degs_cur: list[np.ndarray] = []
    eta = np.zeros(p)

    s2_prev = float(np.var(y, ddof=1) + 1e-9)
    lam_prev = 1.0
    g0sq = (b_g / (a_g + 1.0)) if g2_init is None else float(g2_init)

    B0 = np.ones((n, 1))
    if prior_type == "ridge":
        lik = RidgeState(B0, y, tau2, a_sigma, b_sigma, rcond_tol,
                         exact_marginal)
        beta_prev = np.linalg.solve(lik.Vinv, lik.v)
    else:
        lik = GPriorState(B0, y, g0sq, zeta, a_sigma, b_sigma, rcond_tol,
                          exact_marginal)
        beta_prev = np.array([y.mean()])

    count_accept = np.zeros(4, dtype=np.int64)
    count_propose = np.zeros(4, dtype=np.int64)

    # Storage --------------------------------------------------------------
    keep = set(range(nburn, nmcmc, thin))
    st_nbasis, st_s2, st_lam, st_g2, st_ss = [], [], [], [], []
    st_beta, st_vars, st_degs = [], [], []
    global_max_basis = 0

    def _store(nb, beta, s2v, lamv, g2v, ssv):
        st_nbasis.append(nb)
        st_beta.append(np.asarray(beta, dtype=float).copy())
        st_s2.append(s2v)
        st_lam.append(lamv)
        st_g2.append(g2v)
        st_ss.append(ssv)
        st_vars.append([v.copy() for v in vars_cur])
        st_degs.append([d.copy() for d in degs_cur])

    if 0 in keep:
        _store(0, beta_prev, s2_prev, lam_prev, g0sq, np.nan)

    if verbose:
        print(f"MCMC iteration 0 {time.strftime('#-- %b %d %X --#')} nbasis: 0")

    mutate_eps = 0.1  # floor on either mutation type's probability

    # ---------------------------------------------------------------- loop
    for i in range(1, nmcmc):
        M = len(vars_cur)

        if M == 0:
            move = "birth"
        elif M == max_basis:
            pr = move_probs[1:] / move_probs[1:].sum()
            move = ("death", "change")[rng.choice(2, p=pr)]
        else:
            move = ("birth", "death", "change")[
                rng.choice(3, p=move_probs / move_probs.sum())
            ]

        # ---------------------------------------------------------- BIRTH
        if move == "birth":
            q0 = int(rng.choice(q0_grid, p=J_probs / Hjs))
            W = wcache.weights(eta)
            wts = W[q0 - 1]

            # Coin flips, rejecting empty / over-order draws.  The density of
            # every draw (including rejected ones) accumulates into the
            # acceptance ratio -- the "delayed rejection" term.
            delayed_reject_term = 0.0
            while True:
                chi = rng.binomial(1, wts)
                active = np.flatnonzero(chi)
                delayed_reject_term += log_coinflip_prob(W, log_J, log_H, active)
                if 1 <= active.shape[0] <= order:
                    break

            q = int(active.shape[0])
            d_probs = dprob_cache[q]
            dtot = int(rng.choice(np.arange(q, degree + 1), p=d_probs))
            degs_new = random_partition(dtot, q, rng)

            b_new = make_basis(active, degs_new, X)
            cand = lik.propose_birth(b_new, q, dtot, g0sq)

            if cand is not None:
                # Prior ratio: Poisson(lambda) on M, uniform on the admissible
                # multi-index set.  The two log(M+1) terms account for basis
                # ordering and cancel (kept explicit to mirror the reference).
                lprior = (
                    math.log(lam_prev) - math.log(M + 1) - logA + math.log(M + 1)
                )
                lprop = (
                    math.log(move_probs[1]) + math.log(1.0 / (M + 1))
                ) - (
                    math.log(move_probs[0])
                    + math.log(d_probs[dtot - q])
                    - _lchoose(dtot, q)
                    + delayed_reject_term
                )
                log_alpha = cand.loglik_ratio + lprior + lprop
                if math.log(rng.random()) < log_alpha:
                    lik.accept(cand)
                    vars_cur.append(active.astype(np.int64))
                    degs_cur.append(degs_new.astype(np.int64))
                    eta[active] += 1
                    count_accept[0] += 1
            count_propose[0] += 1

        # ---------------------------------------------------------- DEATH
        elif move == "death":
            tokill = int(rng.integers(M))
            cand = lik.propose_death(tokill, g0sq)

            if cand is not None:
                active = vars_cur[tokill]
                degs_k = degs_cur[tokill]
                q = int(active.shape[0])
                dtot = int(degs_k.sum())
                d_probs = dprob_cache[q]

                eta_cand = eta.copy()
                eta_cand[active] -= 1
                W2 = wcache.weights(eta_cand)

                # Probability the reverse birth proposes exactly this term ...
                log_pn = log_coinflip_prob(W2, log_J, log_H, active)
                # ... conditional on not having been rejected for being empty
                # or exceeding q_max (normal approximation for the upper tail,
                # exactly as in the reference implementation).
                with np.errstate(divide="ignore"):
                    log_p0 = _logsumexp(log_J - log_H + np.log1p(-W2).sum(axis=1))
                if order < p:
                    mu_chi = W2.sum(axis=1)
                    sig_chi = np.sqrt((W2 * (1.0 - W2)).sum(axis=1))
                    log_pq = _logsumexp(
                        log_J - log_H
                        + norm.logsf(order + 0.5, loc=mu_chi, scale=sig_chi)
                    )
                else:
                    log_pq = -np.inf
                reject = min(1.0, math.exp(log_p0) + math.exp(log_pq))

                if reject < 1.0 and np.isfinite(log_pn):
                    lprior = (
                        -math.log(lam_prev) + math.log(M) + logA - math.log(M + 1)
                    )
                    lprop = (
                        math.log(move_probs[0])
                        + log_pn
                        - math.log1p(-reject)
                        + math.log(d_probs[dtot - q])
                        - _lchoose(dtot, q)
                    ) - (math.log(move_probs[1]) + math.log(1.0 / M))
                    log_alpha = cand.loglik_ratio + lprior + lprop
                    if math.log(rng.random()) < log_alpha:
                        lik.accept(cand)
                        vars_cur.pop(tokill)
                        degs_cur.pop(tokill)
                        eta = eta_cand
                        count_accept[1] += 1
            count_propose[1] += 1

        # --------------------------------------------------------- MUTATE
        else:
            if p <= 3:
                # A variable swap buys little when there are few inputs.
                mutate_type = 1
            else:
                rates = count_accept[2:4] / (0.01 + count_propose[2:4])
                z = np.clip(norm.ppf(rates), -5.0, 5.0)
                mutate_prob = mutate_eps + (1.0 - 2.0 * mutate_eps) * norm.cdf(
                    -(z[1] - z[0])
                )
                mutate_type = int(rng.binomial(1, mutate_prob))

            tochange = int(rng.integers(M))
            vars_k = vars_cur[tochange]
            degs_k = degs_cur[tochange]
            q = int(vars_k.shape[0])
            dtot_curr = int(degs_k.sum())

            if mutate_type == 1:
                # ---- Type 1: re-draw the degree partition, same variables.
                d_probs = dprob_cache[q]
                dtot_cand = int(rng.choice(np.arange(q, degree + 1), p=d_probs))
                degs_cand = random_partition(dtot_cand, q, rng)
                b_new = make_basis(vars_k, degs_cand, X)
                cand = lik.propose_replace(
                    tochange, b_new, q, dtot_cand, g0sq, update_g=True
                )
                if cand is not None:
                    # Uniform priors and no dimension change => no prior term.
                    lprop = (
                        math.log(d_probs[dtot_curr - q])
                        - math.log(d_probs[dtot_cand - q])
                        - _lchoose(dtot_curr, q)
                        + _lchoose(dtot_cand, q)
                    )
                    if math.log(rng.random()) < cand.loglik_ratio + lprop:
                        lik.accept(cand)
                        degs_cur[tochange] = degs_cand.astype(np.int64)
                        count_accept[2] += 1
                count_propose[2] += 1

            else:
                # ---- Type 2: swap one active variable for an inactive one.
                ind = int(rng.integers(q))
                var_old = int(vars_k[ind])

                probs_curr = 1e-9 + coin_pars.base_weight + eta
                if legacy_swap:
                    # R excludes only the variable being removed, so the swap
                    # can duplicate a variable already in the term.
                    blocked_fwd = np.array([var_old])
                else:
                    blocked_fwd = vars_k
                probs_curr = probs_curr.copy()
                probs_curr[blocked_fwd] = 0.0
                total = probs_curr.sum()

                cand = None
                if total > 0:
                    # (total == 0 means every input is already in this term,
                    #  so there is nothing to swap in and the move is a no-op.)
                    probs_curr = probs_curr / total
                    var_new = int(rng.choice(p, p=probs_curr))

                    vars_cand = vars_k.copy()
                    vars_cand[ind] = var_new

                    eta_cand = eta.copy()
                    eta_cand[var_old] -= 1
                    eta_cand[var_new] += 1
                    probs_cand = 1e-9 + coin_pars.base_weight + eta_cand
                    blocked_rev = (
                        np.array([var_new]) if legacy_swap else vars_cand
                    )
                    probs_cand[blocked_rev] = 0.0
                    probs_cand = probs_cand / probs_cand.sum()

                    b_new = make_basis(vars_cand, degs_k, X)
                    cand = lik.propose_replace(
                        tochange, b_new, q, dtot_curr, g0sq, update_g=False
                    )
                if cand is not None and probs_cand[var_old] > 0:
                    lprop = math.log(probs_cand[var_old]) - math.log(
                        probs_curr[var_new]
                    )
                    if math.log(rng.random()) < cand.loglik_ratio + lprop:
                        lik.accept(cand)
                        vars_cur[tochange] = vars_cand.astype(np.int64)
                        eta = eta_cand
                        count_accept[3] += 1
                count_propose[3] += 1

        # ------------------------------------------------------ GIBBS STEPS
        M = len(vars_cur)
        mu_n, Sigma_unit = lik.posterior_moments()
        beta_prev = rmvnorm_eigen(mu_n, s2_prev * Sigma_unit, rng)

        resid = y - lik.B @ beta_prev
        sum_sq = float(resid @ resid)

        shape = n / 2.0 + a_sigma
        rate = b_sigma + 0.5 * sum_sq
        s2_prev = max(s2_lower, 1.0 / rng.gamma(shape, 1.0 / rate))
        lam_prev = float(rng.gamma(a_M + M, 1.0 / (b_M + 1.0)))

        if prior_type == "gprior":
            g0sq = _update_g2(
                g0sq, g2_sample, a_g, b_g, lik.g_vec, lik.BtB, rng
            )
            if sync_g2:
                lik.refresh(g0sq)

        global_max_basis = max(global_max_basis, M)
        if i in keep:
            _store(M, beta_prev, s2_prev, lam_prev, g0sq, sum_sq)

        if verbose and (i + 1) % 1000 == 0:
            print(
                f"MCMC iteration {i + 1} {time.strftime('#-- %b %d %X --#')} "
                f"nbasis: {M}"
            )

    return _assemble(
        X, y, prior_type, st_nbasis, st_beta, st_s2, st_lam, st_g2, st_ss,
        st_vars, st_degs, eta, count_accept, count_propose, lik.B,
        global_max_basis, order,
    )


def _lchoose(n: int, k: int) -> float:
    """``log C(n, k)`` (R's ``lchoose``)."""
    if k < 0 or k > n:
        return -math.inf
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    )


def _update_g2(g0sq, g2_sample, a_g, b_g, g_vec, BtB, rng):
    """One update of the global regulariser ``g0^2``.

    ``g2_sample`` selects the strategy (R's ``g2_sample`` argument):

    ``"f"``
        fixed -- carry the current value forward;
    ``"lf"`` / ``"lo"``
        draw directly from the Laplace approximation (full design /
        orthogonal-design assumption);
    ``"mh"`` / ``"mho"``
        Metropolis-Hastings against the exact conditional posterior, using the
        full / orthogonal Laplace fit as the proposal;
    ``"mhoo"``
        Metropolis-Hastings against the *orthogonal* posterior with the
        orthogonal Laplace proposal.
    """
    if g2_sample == "f":
        return g0sq

    if g2_sample in ("lf", "mh"):
        pars = laplace_full(a_g, b_g, g_vec, BtB)
    elif g2_sample in ("lo", "mho", "mhoo"):
        pars = laplace_orth(a_g, b_g, g_vec)
    else:
        import warnings

        warnings.warn("g2_sample not recognized. Holding g2 fixed.", stacklevel=2)
        return g0sq

    if pars is None:
        return g0sq
    a_t, b_t = pars

    if g2_sample in ("lf", "lo"):
        return float(1.0 / rng.gamma(a_t, 1.0 / b_t))

    cand = float(1.0 / rng.gamma(a_t, 1.0 / b_t))
    if not np.isfinite(cand) or cand <= 0:
        return g0sq

    if g2_sample == "mhoo":
        lp_cand = log_dgsq_orth(cand, a_g, b_g, g_vec)
        lp_curr = log_dgsq_orth(g0sq, a_g, b_g, g_vec)
    else:
        lp_cand = log_dgsq_full(cand, a_g, b_g, g_vec, BtB)
        lp_curr = log_dgsq_full(g0sq, a_g, b_g, g_vec, BtB)

    from scipy.stats import gamma as _gamma

    lprop_cand = _gamma.logpdf(1.0 / cand, a_t, scale=1.0 / b_t) - 2.0 * math.log(cand)
    lprop_curr = _gamma.logpdf(1.0 / g0sq, a_t, scale=1.0 / b_t) - 2.0 * math.log(g0sq)

    log_alpha = (lp_cand - lp_curr) + (lprop_curr - lprop_cand)
    if np.isfinite(log_alpha) and math.log(rng.random()) < log_alpha:
        return cand
    return g0sq


def _assemble(X, y, prior_type, st_nbasis, st_beta, st_s2, st_lam, st_g2,
              st_ss, st_vars, st_degs, eta, count_accept, count_propose,
              B_last, global_max_basis, order):
    n_keep = len(st_nbasis)
    max_basis_used = max(global_max_basis, 1)
    max_order_used = 1
    for vv in st_vars:
        for v in vv:
            max_order_used = max(max_order_used, int(v.shape[0]))

    nbasis = np.asarray(st_nbasis, dtype=np.int64)
    beta = np.full((n_keep, max_basis_used + 1), np.nan)
    for i, b in enumerate(st_beta):
        beta[i, : b.shape[0]] = b

    vars_arr = np.full((n_keep, max_basis_used, max_order_used), -1, dtype=np.int64)
    degs_arr = np.zeros((n_keep, max_basis_used, max_order_used), dtype=np.int64)
    nint = np.zeros((n_keep, max_basis_used), dtype=np.int64)
    dtot = np.zeros((n_keep, max_basis_used), dtype=np.int64)
    for i in range(n_keep):
        for m, (v, d) in enumerate(zip(st_vars[i], st_degs[i])):
            k = v.shape[0]
            vars_arr[i, m, :k] = v
            degs_arr[i, m, :k] = d
            nint[i, m] = k
            dtot[i, m] = int(d.sum())

    names = ["Birth", "Death", "Mutate (degree)", "Mutate (vars)"]
    return AdaptiveKhaos(
        X=X,
        y=y,
        prior_type=prior_type,
        nbasis=nbasis,
        beta=beta,
        vars=vars_arr,
        degs=degs_arr,
        nint=nint,
        dtot=dtot,
        s2=np.asarray(st_s2, dtype=float),
        lam=np.asarray(st_lam, dtype=float),
        g2=np.asarray(st_g2, dtype=float) if prior_type == "gprior" else None,
        sum_sq=np.asarray(st_ss, dtype=float),
        eta=eta,
        count_accept=dict(zip(names, count_accept.tolist())),
        count_propose=dict(zip(names, count_propose.tolist())),
        B=B_last,
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def adaptive_khaos_ridge(
    X,
    y,
    degree: int = 15,
    order: int = 5,
    nmcmc: int = 10000,
    nburn: int = 9000,
    thin: int = 1,
    max_basis: int = 1000,
    tau2: float = 1e5,
    s2_lower: float = 0.0,
    a_sigma: float = 0.0,
    b_sigma: float = 0.0,
    a_M: float = 4.0,
    b_M: Optional[float] = None,
    move_probs: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
    coin_pars: Optional[CoinPars] = None,
    degree_penalty: float = 0.0,
    exact_marginal: bool = False,
    legacy_swap: bool = False,
    rcond_tol: float = 1e-9,
    seed=None,
    verbose: bool = True,
) -> AdaptiveKhaos:
    """Adaptive PCE with an independent ridge prior on the coefficients.

    Port of R ``adaptive_khaos_ridge`` (whose ``g1``/``g2`` arguments are named
    ``a_sigma``/``b_sigma`` here, and ``h1``/``h2`` are ``a_M``/``b_M``).

    Parameters
    ----------
    X : array_like of shape (n, p)
        Predictors, scaled to :math:`[0, 1]`.
    y : array_like of shape (n,)
        Response.
    degree, order : int
        Maximum total degree :math:`d_{\\max}` and interaction order
        :math:`q_{\\max}` of any single basis function.
    nmcmc, nburn, thin : int
        Chain length, burn-in, and thinning.
    tau2 : float
        Prior variance scale for the coefficients.
    a_sigma, b_sigma : float
        Shape/rate of the inverse-gamma prior on :math:`\\sigma^2`
        (default is Jeffreys').
    a_M, b_M : float
        Shape/rate of the gamma prior on :math:`\\lambda`, the expected number
        of basis functions.  ``b_M`` defaults to ``40 / n``.
    move_probs : sequence of 3 floats
        Probabilities of (birth, death, mutate).
    degree_penalty : float
        Larger values push the proposal toward lower-degree terms.
    exact_marginal : bool
        Use the textbook normal-inverse-gamma marginal
        (``b_sigma + quad/2``) instead of the reference implementation's
        ``b_sigma + quad``.  The two are equivalent whenever ``b_sigma == 0``,
        which is the default; ``False`` reproduces R exactly.
    legacy_swap : bool
        Reproduce the reference implementation's variable-swap mutation, which
        excludes only the variable being removed from the candidate pool and so
        can put the *same* input into a basis function twice.  The paper
        describes the move as swapping in an *inactive* variable, which is what
        the default (``False``) does.
    seed : int or Generator, optional
        Seed for reproducibility.

    Returns
    -------
    AdaptiveKhaos
    """
    y = np.asarray(y, dtype=float).ravel()
    if b_M is None:
        b_M = 40.0 / y.shape[0]
    return _fit(
        X, y, "ridge", degree, order, nmcmc, nburn, thin, max_basis,
        move_probs, coin_pars or CoinPars(), degree_penalty, s2_lower,
        a_sigma, b_sigma, a_M, b_M, tau2,
        a_g=1e-3, b_g=1e3, zeta=1.0, g2_sample="f", g2_init=None,
        sync_g2=False, exact_marginal=exact_marginal,
        legacy_swap=legacy_swap, rcond_tol=rcond_tol,
        rng=np.random.default_rng(seed), verbose=verbose,
    )


def adaptive_khaos_gprior(
    X,
    y,
    degree: int = 15,
    order: int = 5,
    nmcmc: int = 10000,
    nburn: int = 9000,
    thin: int = 1,
    max_basis: int = 1000,
    a_g: float = 1e-3,
    b_g: float = 1e3,
    zeta: float = 1.0,
    g2_sample: str = "mh",
    g2_init: Optional[float] = None,
    s2_lower: float = 0.0,
    a_sigma: float = 0.0,
    b_sigma: float = 0.0,
    a_M: float = 4.0,
    b_M: Optional[float] = None,
    move_probs: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
    coin_pars: Optional[CoinPars] = None,
    degree_penalty: float = 0.0,
    sync_g2: bool = True,
    exact_marginal: bool = False,
    legacy_swap: bool = False,
    rcond_tol: float = 1e-9,
    seed=None,
    verbose: bool = True,
) -> AdaptiveKhaos:
    """Adaptive PCE with the modified g-prior of Rumsey et al. (2026).

    Port of R ``adaptive_khaos_gprior``.

    Parameters
    ----------
    a_g, b_g : float
        Shape/rate of the inverse-gamma prior on the global regulariser
        :math:`g_0^2`.
    zeta : float
        Complexity-penalty strength in the modified g-prior; ``zeta = 0``
        recovers the standard g-prior.
    g2_sample : {'mh', 'mho', 'mhoo', 'lf', 'lo', 'f'}
        How :math:`g_0^2` is updated; see :func:`_update_g2`.
    g2_init : float, optional
        Starting value; defaults to the prior mode ``b_g / (a_g + 1)``.
    sync_g2 : bool
        Recompute the current state's marginal likelihood after each
        :math:`g_0^2` update, so that every acceptance ratio compares the two
        models at the *same* :math:`g_0^2`.  The R reference does not do this:
        its stored ``Sigma``/``Q`` still reflect whatever :math:`g_0^2` was
        current when the last move was accepted, which stalls the
        trans-dimensional moves badly (birth/death acceptance around 1%
        instead of 25% on the Friedman benchmark).  Default ``True``; set
        ``False`` to reproduce the reference exactly.
    exact_marginal : bool
        Experimental.  Include the :math:`-\\tfrac12\\log|S_0|` normalising
        term that the reference implementation drops from the birth/death
        acceptance ratio (see :class:`khaos.likelihood.GPriorState`).  It
        yields a genuine Bayes factor -- with ``zeta = 0`` the classical
        :math:`(1 + g_0^2)^{-1/2}` Occam factor per term -- but that penalty is
        *weaker* than what the reference effectively applies, so combined with
        the permissive default ``b_M = 4/n`` the model can grow very large and
        the sampler slow.  Tighten ``b_M`` if you use it.  ``False``
        (the default) reproduces R.
    b_M : float
        Rate of the gamma prior on :math:`\\lambda`; defaults to ``4 / n``.

    Other parameters are as in :func:`adaptive_khaos_ridge`.

    Returns
    -------
    AdaptiveKhaos
    """
    y = np.asarray(y, dtype=float).ravel()
    if b_M is None:
        b_M = 4.0 / y.shape[0]
    return _fit(
        X, y, "gprior", degree, order, nmcmc, nburn, thin, max_basis,
        move_probs, coin_pars or CoinPars(), degree_penalty, s2_lower,
        a_sigma, b_sigma, a_M, b_M, tau2=1e5,
        a_g=a_g, b_g=b_g, zeta=zeta, g2_sample=g2_sample, g2_init=g2_init,
        sync_g2=sync_g2, exact_marginal=exact_marginal,
        legacy_swap=legacy_swap, rcond_tol=rcond_tol,
        rng=np.random.default_rng(seed), verbose=verbose,
    )


def adaptive_khaos(X, y, prior_type: str = "ridge", **kwargs) -> AdaptiveKhaos:
    """Fit an adaptive Bayesian PCE (R: ``adaptive_khaos``).

    Parameters
    ----------
    prior_type : {'ridge', 'gprior'}
        Coefficient prior.  ``'gprior'`` is the modified g-prior introduced in
        the paper; ``'ridge'`` is the simpler independent-normal prior and is
        the R package's default.
    **kwargs
        Passed to :func:`adaptive_khaos_ridge` or
        :func:`adaptive_khaos_gprior`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((100, 2))
    >>> f = lambda x: 10.391 * ((x[:, 0] - 0.4) * (x[:, 1] - 0.6) + 0.36)
    >>> y = f(X) + rng.normal(0, 0.1, 100)
    >>> fit = adaptive_khaos(X, y, nmcmc=2000, nburn=1000, seed=1, verbose=False)
    >>> fit.predict().shape
    (1000, 100)
    """
    if prior_type == "ridge":
        return adaptive_khaos_ridge(X, y, **kwargs)
    if prior_type == "gprior":
        return adaptive_khaos_gprior(X, y, **kwargs)
    raise ValueError("prior_type must be one of {'ridge', 'gprior'}")
