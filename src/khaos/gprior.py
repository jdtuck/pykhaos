"""Modified g-prior: densities and Laplace approximations for ``g0^2``.

Port of the ``rg0sq_laplace_*`` / ``dgsq_*`` helpers in ``R/chaos_helpers.R``.

The modified g-prior of Rumsey et al. (2026) is

.. math::

    \\boldsymbol{\\beta} \\mid M, \\sigma^2, g_0^2 \\sim
        \\mathcal{N}_{M+1}\\!\\left(0,\\;
        \\sigma^2 g_0^2 D(g)(\\Psi^{T}\\Psi)^{-1} D(g)\\right),
    \\qquad g_0^2 \\sim \\mathrm{Inv\\text{-}Gamma}(a_g, b_g),

with complexity weights (as implemented in ``khaos``)

.. math::

    g_m = \\bigl(1 + q(\\boldsymbol{\\alpha}_m)
          [\\,q(\\boldsymbol{\\alpha}_m) + d(\\boldsymbol{\\alpha}_m) - 2\\,]
          \\bigr)^{-\\zeta/2},

so higher-degree, higher-order terms are shrunk harder.  Setting
:math:`\\zeta = 0` recovers the usual g-prior.  Writing
:math:`G_{m\\ell} = 1 + 1/(g_0^2 g_m g_\\ell)`, the posterior precision is
:math:`\\Sigma_n^{-1} = G \\circ \\Psi^{T}\\Psi` (Hadamard product).
"""

from __future__ import annotations

import numpy as np

from .linalg import logdet_spd

__all__ = [
    "g_weight",
    "build_G",
    "log_dgsq_full",
    "log_dgsq_orth",
    "dgsq_full",
    "dgsq_orth",
    "laplace_orth",
    "laplace_full",
]


def g_weight(q: int, d: int, zeta: float) -> float:
    """Complexity weight ``g_m = (1 + q (q + d - 2))^(-zeta/2)``."""
    return float((1.0 + q * (q + d - 2.0)) ** (-zeta / 2.0))


def build_G(g_vec: np.ndarray, g0sq: float) -> np.ndarray:
    """``G = 1 + outer(1/g, 1/g) / g0^2`` (R: ``1 + tcrossprod(1/g_vec)/g2``)."""
    inv = 1.0 / np.asarray(g_vec, dtype=float)
    return 1.0 + np.outer(inv, inv) / g0sq


# --------------------------------------------------------------------------
# Conditional posterior of g0^2 (up to a constant)
# --------------------------------------------------------------------------
def log_dgsq_full(theta: float, a: float, b: float, gm: np.ndarray,
                  BtB: np.ndarray) -> float:
    """``log pi(g0^2 | .)`` under the exact design (R: ``log(dgsq_full(...))``).

    .. math::

        \\pi(g_0^2 \\mid y) \\propto
            (g_0^2)^{-(a_g + M/2)} e^{-b_g/g_0^2} |\\Sigma_n|^{1/2}
    """
    if not np.isfinite(theta) or theta <= 0:
        return -np.inf
    gm = np.asarray(gm, dtype=float)
    M = gm.shape[0]
    Sigma_inv = build_G(gm, theta) * BtB
    ld = logdet_spd(Sigma_inv)
    if ld is None:
        sign, ld = np.linalg.slogdet(Sigma_inv)
        if sign <= 0:
            return -np.inf
    return float(-(a + M / 2.0) * np.log(theta) - b / theta - 0.5 * ld)


def log_dgsq_orth(theta: float, a: float, b: float, gm_sq: np.ndarray) -> float:
    """``log pi(g0^2 | .)`` under the orthogonal-design approximation.

    .. math::

        \\pi(g_0^2 \\mid y, \\text{orth}) \\propto
            (g_0^2)^{-(a_g + M/2)} e^{-b_g/g_0^2}
            \\prod_m \\left(\\frac{g_0^2 g_m^2}{1 + g_0^2 g_m^2}\\right)^{1/2}

    Note
    ----
    The R implementation passes the *unsquared* weight vector ``g_vec`` into an
    argument named ``gm_sq``.  That behaviour is preserved here so results match
    the reference; pass ``g**2`` yourself if you want the formula as written.
    """
    if not np.isfinite(theta) or theta <= 0:
        return -np.inf
    gm_sq = np.atleast_1d(np.asarray(gm_sq, dtype=float))
    M = gm_sq.shape[0]
    ratio = theta * gm_sq / (1.0 + theta * gm_sq)
    if np.any(ratio <= 0):
        return -np.inf
    return float(
        -(a + M / 2.0) * np.log(theta) - b / theta + 0.5 * np.log(ratio).sum()
    )


def dgsq_full(theta, a=1.0, b=1.0, gm=None, BtB=None) -> np.ndarray:
    """Natural-scale version of :func:`log_dgsq_full` (R: ``dgsq_full``)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    if gm is None:
        gm = np.ones(BtB.shape[0])
    return np.array([np.exp(log_dgsq_full(t, a, b, gm, BtB)) for t in theta])


def dgsq_orth(theta, a=1.0, b=1.0, gm_sq=None) -> np.ndarray:
    """Natural-scale version of :func:`log_dgsq_orth` (R: ``dgsq_orth``)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    if gm_sq is None:
        gm_sq = np.ones(1)
    return np.array([np.exp(log_dgsq_orth(t, a, b, gm_sq)) for t in theta])


# --------------------------------------------------------------------------
# Laplace approximations
# --------------------------------------------------------------------------
def _inv_gamma_from_mode(m_theta: float, s2_theta: float, c: float = 0.0):
    """Match an inverse-gamma to a mode ``m_theta`` and curvature ``s2_theta``.

    ``c = -1`` is moment matching; ``c = 1`` matches modes.  The R default
    (``c = 0``) sits between the two and is what the sampler uses -- so the
    returned distribution is centred at ``b_theta / a_theta == m_theta``,
    which is neither its own mode nor its mean.
    """
    if not np.isfinite(m_theta) or m_theta <= 0:
        return None
    if not np.isfinite(s2_theta) or s2_theta <= 0:
        return None
    a_theta = 2.0 + m_theta**2 / s2_theta
    b_theta = m_theta * (a_theta + c)
    if not (np.isfinite(a_theta) and np.isfinite(b_theta)) or b_theta <= 0:
        return None
    return float(a_theta), float(b_theta)


def _map_orth(a: float, b: float, g: np.ndarray, n_iter: int = 5) -> float:
    """Fixed-point iteration for the mode under the orthogonality assumption.

    .. math::

        \\theta_k = \\frac{-a_g + \\sqrt{a_g^2 + 4 b_g G_k}}{2 G_k},
        \\qquad G_k = \\tfrac{1}{2}\\sum_m \\frac{g_m^2}{1 + \\theta_{k-1} g_m^2}
    """
    g2 = np.asarray(g, dtype=float) ** 2
    theta = b / a if a > 0 else 1.0
    for _ in range(n_iter):
        Gl = 0.5 * np.sum(g2 / (1.0 + theta * g2))
        if Gl <= 0 or not np.isfinite(Gl):
            break
        theta = max(0.0, (-a + np.sqrt(a * a + 4.0 * Gl * b)) / (2.0 * Gl))
    return float(theta)


def laplace_orth(a: float, b: float, g: np.ndarray, n_iter: int = 5,
                 c: float = 0.0):
    """Inverse-gamma Laplace approximation to ``pi(g0^2 | .)`` (orthogonal).

    Returns ``(a_theta, b_theta)`` or ``None`` if the approximation degenerates.
    Port of ``rg0sq_laplace_orth`` with ``n <= 0`` (parameters only).
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    g2 = g**2
    m_theta = _map_orth(a, b, g, n_iter)
    if m_theta <= 0:
        return None
    curvature = (
        2.0 * b / m_theta**3
        - a / (2.0 * m_theta**2)
        - 0.5 * np.sum(g2**2 / (1.0 + m_theta * g2) ** 2)
    )
    if curvature == 0:
        return None
    s2_theta = 1.0 / curvature
    return _inv_gamma_from_mode(m_theta, s2_theta, c)


def laplace_full(a: float, b: float, g: np.ndarray, BtB: np.ndarray,
                 n_iter: int = 5, n_newtonsteps: int = 3, c: float = 0.0):
    """Inverse-gamma Laplace approximation using the exact design matrix.

    Starts from the orthogonal-design mode, then runs Newton-Raphson on the
    log scale for the exact log posterior.  Port of ``rg0sq_laplace_full``
    with ``n <= 0``.  Returns ``(a_theta, b_theta)`` or ``None``.
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    BtB = np.asarray(BtB, dtype=float)
    M = g.shape[0]

    theta = _map_orth(a, b, g, n_iter)
    if theta <= 0 or not np.isfinite(theta):
        return None

    inv = 1.0 / g
    ggt = np.outer(inv, inv)
    R0 = ggt * BtB

    hp_curr = np.nan
    for _ in range(n_newtonsteps):
        P_theta = (1.0 + ggt / theta) * BtB
        try:
            PinvR = np.linalg.solve(P_theta, R0)
        except np.linalg.LinAlgError:
            return None
        tr1 = np.trace(PinvR)
        tr2 = np.trace(PinvR @ PinvR)

        h = -(a + M / 2.0) / theta + b / theta**2 + tr1 / (2.0 * theta**2)
        hp = (
            (a + M / 2.0) / theta**2
            - 2.0 * b / theta**3
            - tr1 / theta**3
            + tr2 / (2.0 * theta**4)
        )
        if hp == 0 or not np.isfinite(hp) or not np.isfinite(h):
            return None
        step = h / (theta * hp)
        theta = theta * np.exp(-step)
        hp_curr = hp
        if not np.isfinite(theta) or theta <= 0:
            return None

    if not np.isfinite(hp_curr) or hp_curr == 0:
        return None
    s2_theta = -1.0 / hp_curr
    return _inv_gamma_from_mode(theta, s2_theta, c)
