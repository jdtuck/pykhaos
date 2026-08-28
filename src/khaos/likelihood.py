"""Marginal-likelihood state for the two coefficient priors.

Both the ridge prior (:class:`RidgeState`) and the modified g-prior
(:class:`GPriorState`) integrate :math:`\\boldsymbol\\beta` and
:math:`\\sigma^2` out of the acceptance ratio analytically, leaving a
collapsed log marginal likelihood that depends only on the current design
matrix.  The RJMCMC driver in :mod:`khaos.adaptive` is shared; these classes
supply the prior-specific algebra and are the only place the two fits differ.

Each class maintains the design matrix ``B = [1, Psi_1, ..., Psi_M]`` and the
sufficient statistics ``B'B`` and ``B'y``, and offers three candidate
constructors -- birth, death and column replacement -- that return a candidate
object carrying the log marginal-likelihood ratio ``loglik_ratio`` relative to
the current state.  ``None`` means the move must be rejected outright (the
design matrix is too ill-conditioned), matching the ``FALSE`` short-circuit of
the R ``safe_solve``.

Unlike the R reference, which recomputes ``crossprod(B)`` from scratch every
iteration, the updates here are rank-one / single-column and therefore exact
but :math:`O(nM)` rather than :math:`O(nM^2)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .gprior import build_G, g_weight
from .linalg import rcond1, safe_inverse, safe_logdet

__all__ = ["RidgeState", "GPriorState", "Candidate"]


@dataclass
class Candidate:
    """A proposed design matrix plus the collapsed log-likelihood ratio."""

    loglik_ratio: float
    B: np.ndarray
    BtB: np.ndarray
    v: np.ndarray
    extra: dict


# --------------------------------------------------------------------------
# Shared sufficient-statistic updates
# --------------------------------------------------------------------------
def _birth_stats(B, BtB, v, b_new, y):
    cross = B.T @ b_new
    k = BtB.shape[0]
    BtB_new = np.empty((k + 1, k + 1))
    BtB_new[:k, :k] = BtB
    BtB_new[:k, k] = cross
    BtB_new[k, :k] = cross
    BtB_new[k, k] = b_new @ b_new
    B_new = np.hstack([B, b_new[:, None]])
    v_new = np.append(v, b_new @ y)
    return B_new, BtB_new, v_new


def _death_stats(B, BtB, v, col):
    keep = np.ones(BtB.shape[0], dtype=bool)
    keep[col] = False
    return B[:, keep], BtB[np.ix_(keep, keep)], v[keep]


def _replace_stats(B, BtB, v, col, b_new, y):
    cross = B.T @ b_new
    cross[col] = b_new @ b_new
    B_new = B.copy()
    B_new[:, col] = b_new
    BtB_new = BtB.copy()
    BtB_new[:, col] = cross
    BtB_new[col, :] = cross
    v_new = v.copy()
    v_new[col] = b_new @ y
    return B_new, BtB_new, v_new


# --------------------------------------------------------------------------
# Ridge (independent normal) prior
# --------------------------------------------------------------------------
class RidgeState:
    """Collapsed likelihood under ``beta | sigma^2 ~ N(0, sigma^2 tau^2 I)``.

    Port of the linear algebra in ``adaptive_khaos_ridge``.  With
    :math:`V^{-1} = B'B + I/\\tau^2` and
    :math:`d = b_\\sigma + y'y - \\hat\\beta' V^{-1} \\hat\\beta`, the collapsed
    log marginal likelihood is (up to constants)

    .. math::

        -\\tfrac{1}{2}\\log|V^{-1}| - \\tfrac{k}{2}\\log\\tau^2
        - (a_\\sigma + n/2)\\log d,

    where ``k`` is the number of columns.  The ``0.5 * log(1/tau2)`` term shows
    up explicitly in the birth/death ratios and cancels for mutations.

    Note
    ----
    The R reference sets ``d = b_sigma + y'y - bhat' V^{-1} bhat``, without the
    factor of one half that the normal-inverse-gamma marginal calls for.  With
    the default Jeffreys prior (``b_sigma = 0``) the factor cancels out of every
    acceptance ratio, so the two agree; they differ only when ``b_sigma > 0``.
    Pass ``exact_marginal=True`` for the textbook form.
    """

    prior_type = "ridge"
    uses_g2 = False

    def __init__(self, B, y, tau2, a_sigma, b_sigma, rcond_tol=1e-9,
                 exact_marginal=False):
        self.y = np.asarray(y, dtype=float)
        self.n = self.y.shape[0]
        self.ssy = float(self.y @ self.y)
        self.tau2 = float(tau2)
        self.a_sigma = float(a_sigma)
        self.b_sigma = float(b_sigma)
        self.rcond_tol = float(rcond_tol)
        self.quad_scale = 0.5 if exact_marginal else 1.0

        self.B = np.asarray(B, dtype=float)
        self.BtB = self.B.T @ self.B
        self.v = self.B.T @ self.y
        stats = self._stats(self.BtB, self.v)
        if stats is None:
            raise np.linalg.LinAlgError(
                "initial design matrix is numerically singular"
            )
        self.Vinv, self.d, self.ldet_Vinv = stats

    # -- internals ---------------------------------------------------------
    def _stats(self, BtB, v):
        k = BtB.shape[0]
        Vinv = BtB + np.eye(k) / self.tau2
        ld = safe_logdet(Vinv)
        if ld is None:
            return None
        try:
            bhat = np.linalg.solve(Vinv, v)
        except np.linalg.LinAlgError:
            return None
        quad = self.ssy - float(bhat @ (Vinv @ bhat))
        d = self.b_sigma + self.quad_scale * quad
        return Vinv, d, ld

    def _ratio(self, ldet_cand, d_cand, tau_term):
        if not np.isfinite(d_cand) or d_cand <= 0:
            return None
        return float(
            tau_term
            + 0.5 * (self.ldet_Vinv - ldet_cand)
            + (self.a_sigma + self.n / 2.0) * (np.log(self.d) - np.log(d_cand))
        )

    def _finish(self, B, BtB, v, tau_term):
        # R guards every move with safe_solve(crossprod(B)); reproduce that.
        if rcond1(BtB) < self.rcond_tol:
            return None
        stats = self._stats(BtB, v)
        if stats is None:
            return None
        Vinv, d, ld = stats
        ratio = self._ratio(ld, d, tau_term)
        if ratio is None:
            return None
        return Candidate(ratio, B, BtB, v, {"Vinv": Vinv, "d": d, "ldet": ld})

    # -- public API --------------------------------------------------------
    def propose_birth(self, b_new, q, dtot, g0sq=None) -> Optional[Candidate]:
        B, BtB, v = _birth_stats(self.B, self.BtB, self.v, b_new, self.y)
        return self._finish(B, BtB, v, 0.5 * np.log(1.0 / self.tau2))

    def propose_death(self, index, g0sq=None) -> Optional[Candidate]:
        B, BtB, v = _death_stats(self.B, self.BtB, self.v, index + 1)
        return self._finish(B, BtB, v, -0.5 * np.log(1.0 / self.tau2))

    def propose_replace(self, index, b_new, q, dtot, g0sq=None,
                        update_g=True) -> Optional[Candidate]:
        B, BtB, v = _replace_stats(self.B, self.BtB, self.v, index + 1, b_new, self.y)
        return self._finish(B, BtB, v, 0.0)

    def accept(self, cand: Candidate) -> None:
        self.B, self.BtB, self.v = cand.B, cand.BtB, cand.v
        self.Vinv = cand.extra["Vinv"]
        self.d = cand.extra["d"]
        self.ldet_Vinv = cand.extra["ldet"]

    def posterior_moments(self):
        """``(mu_n, Sigma_n_over_sigma2)`` for the Gibbs update of ``beta``."""
        Lambda_inv = np.linalg.inv(self.Vinv)
        return Lambda_inv @ self.v, Lambda_inv


# --------------------------------------------------------------------------
# Modified g-prior
# --------------------------------------------------------------------------
class GPriorState:
    """Collapsed likelihood under the modified g-prior of Rumsey et al. (2026).

    With :math:`\\Sigma_n = (G \\circ B'B)^{-1}`,
    :math:`Q = b_\\sigma + \\tfrac{1}{2}(y'y - v'\\Sigma_n v)` and ``v = B'y``,
    the collapsed log marginal likelihood is (up to constants)

    .. math::

        \\tfrac{1}{2}\\log|\\Sigma_n| - (a_\\sigma + n/2)\\log Q .

    Note
    ----
    A normal-inverse-gamma marginal also carries :math:`-\\tfrac12\\log|S_0|`,
    which for this prior is not constant across models:

    .. math::

        \\log|S_0| = k \\log g_0^2 + 2\\sum_m \\log g_m - \\log|B'B|,
        \\qquad k = M + 1 .

    The R reference omits it from the birth/death ratio (it *does* include the
    analogous ``0.5 * log(1/tau2)`` in the ridge branch), so the default here
    omits it too and reproduces the reference exactly.  Set
    ``exact_marginal=True`` to include it.
    """

    prior_type = "gprior"
    uses_g2 = True

    def __init__(self, B, y, g0sq, zeta, a_sigma, b_sigma, rcond_tol=1e-9,
                 exact_marginal=False):
        self.y = np.asarray(y, dtype=float)
        self.n = self.y.shape[0]
        self.ssy = float(self.y @ self.y)
        self.zeta = float(zeta)
        self.a_sigma = float(a_sigma)
        self.b_sigma = float(b_sigma)
        self.rcond_tol = float(rcond_tol)
        self.exact_marginal = bool(exact_marginal)

        self.B = np.asarray(B, dtype=float)
        self.BtB = self.B.T @ self.B
        self.v = self.B.T @ self.y
        # Intercept carries weight 1.
        self.g_vec = np.ones(self.B.shape[1])
        self.G = build_G(self.g_vec, g0sq)
        self.Sigma = np.linalg.inv(self.G * self.BtB)
        self.quad = self.ssy - float(self.v @ (self.Sigma @ self.v))
        self.Q = self.b_sigma + 0.5 * self.quad
        self.ldet = safe_logdet(self.Sigma)
        if self.ldet is None:
            raise np.linalg.LinAlgError(
                "initial design matrix is numerically singular"
            )
        self.ldet_S0 = self._logdet_S0(self.g_vec, g0sq, self.BtB)

    def _logdet_S0(self, g_vec, g0sq, BtB) -> float:
        """``log|S_0|`` for the modified g-prior; 0 when the term is disabled."""
        if not self.exact_marginal:
            return 0.0
        ld = safe_logdet(BtB)
        if ld is None:
            return np.nan
        k = len(g_vec)
        return float(k * np.log(g0sq) + 2.0 * np.log(g_vec).sum() - ld)

    def _finish(self, B, BtB, v, g_vec, G, g0sq) -> Optional[Candidate]:
        Sigma = safe_inverse(G * BtB, self.rcond_tol)
        if Sigma is None:
            return None
        quad = self.ssy - float(v @ (Sigma @ v))
        Q = self.b_sigma + 0.5 * quad
        if not np.isfinite(Q) or Q <= 0:
            return None
        ldet = safe_logdet(Sigma)
        if ldet is None:
            return None
        ldet_S0 = self._logdet_S0(g_vec, g0sq, BtB)
        ratio = float(
            0.5 * (ldet - self.ldet)
            - 0.5 * (ldet_S0 - self.ldet_S0)
            + (self.a_sigma + self.n / 2.0) * (np.log(self.Q) - np.log(Q))
        )
        if not np.isfinite(ratio):
            return None
        return Candidate(
            ratio, B, BtB, v,
            {"g_vec": g_vec, "G": G, "Sigma": Sigma, "Q": Q, "ldet": ldet,
             "quad": quad, "ldet_S0": ldet_S0},
        )

    def propose_birth(self, b_new, q, dtot, g0sq) -> Optional[Candidate]:
        B, BtB, v = _birth_stats(self.B, self.BtB, self.v, b_new, self.y)
        g_vec = np.append(self.g_vec, g_weight(q, dtot, self.zeta))
        return self._finish(B, BtB, v, g_vec, build_G(g_vec, g0sq), g0sq)

    def propose_death(self, index, g0sq) -> Optional[Candidate]:
        B, BtB, v = _death_stats(self.B, self.BtB, self.v, index + 1)
        g_vec = np.delete(self.g_vec, index + 1)
        return self._finish(B, BtB, v, g_vec, build_G(g_vec, g0sq), g0sq)

    def propose_replace(self, index, b_new, q, dtot, g0sq,
                        update_g=True) -> Optional[Candidate]:
        B, BtB, v = _replace_stats(self.B, self.BtB, self.v, index + 1, b_new, self.y)
        if update_g:
            # Degree mutation changes the term's complexity weight.
            g_vec = self.g_vec.copy()
            g_vec[index + 1] = g_weight(q, dtot, self.zeta)
            G = build_G(g_vec, g0sq)
        else:
            # Variable swap leaves (q, d) -- hence g and G -- untouched.
            g_vec, G = self.g_vec, self.G
        return self._finish(B, BtB, v, g_vec, G, g0sq)

    def accept(self, cand: Candidate) -> None:
        self.B, self.BtB, self.v = cand.B, cand.BtB, cand.v
        e = cand.extra
        self.g_vec, self.G = e["g_vec"], e["G"]
        self.Sigma, self.Q, self.ldet = e["Sigma"], e["Q"], e["ldet"]
        self.quad = e["quad"]
        self.ldet_S0 = e["ldet_S0"]

    def refresh(self, g0sq: float) -> bool:
        """Recompute the current-state algebra for a new ``g0^2``.

        The R reference does *not* do this: after the Gibbs update of ``g0^2``
        the stored ``Sigma``/``Q``/``ldet`` still reflect whatever value was
        current when the last move was accepted, so the next acceptance ratio
        mixes two values of ``g0^2``.  Enable via ``sync_g2=True``.
        """
        G = build_G(self.g_vec, g0sq)
        Sigma = safe_inverse(G * self.BtB, self.rcond_tol)
        if Sigma is None:
            return False
        ldet = safe_logdet(Sigma)
        if ldet is None:
            return False
        self.G = G
        self.Sigma = Sigma
        self.quad = self.ssy - float(self.v @ (Sigma @ self.v))
        self.Q = self.b_sigma + 0.5 * self.quad
        self.ldet = ldet
        self.ldet_S0 = self._logdet_S0(self.g_vec, g0sq, self.BtB)
        return True

    def posterior_moments(self):
        return self.Sigma @ self.v, self.Sigma
