"""Fitted-model container for the adaptive KHAOS sampler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .basis import make_basis

__all__ = ["AdaptiveKhaos"]


@dataclass
class AdaptiveKhaos:
    """Posterior samples from :func:`khaos.adaptive_khaos`.

    Attributes
    ----------
    X, y : ndarray
        Training data.
    prior_type : {'ridge', 'gprior'}
    nbasis : ndarray of shape (n_keep,)
        Number of basis functions :math:`M` at each retained iteration.
    beta : ndarray of shape (n_keep, max_basis + 1)
        Coefficients, intercept first, NaN-padded beyond ``nbasis[i] + 1``.
    vars : ndarray of shape (n_keep, max_basis, max_order)
        0-based indices of the active inputs of each basis function
        (``-1`` where unused).
    degs : ndarray of shape (n_keep, max_basis, max_order)
        Matching univariate degrees.
    nint : ndarray of shape (n_keep, max_basis)
        Interaction order :math:`q(\\boldsymbol\\alpha_m)` of each term.
    dtot : ndarray of shape (n_keep, max_basis)
        Total degree :math:`d(\\boldsymbol\\alpha_m)` of each term.
    s2, lam : ndarray of shape (n_keep,)
        Error variance and Poisson-rate samples.
    g2 : ndarray or None
        Samples of :math:`g_0^2` (g-prior fits only).
    sum_sq : ndarray
        Residual sum of squares at each retained iteration.
    eta : ndarray of shape (p,)
        Final variable-inclusion counts driving the coin-flip proposal.
    count_accept, count_propose : dict
        Move-wise acceptance bookkeeping.
    B : ndarray
        Design matrix of the final state.
    """

    X: np.ndarray
    y: np.ndarray
    prior_type: str
    nbasis: np.ndarray
    beta: np.ndarray
    vars: np.ndarray
    degs: np.ndarray
    nint: np.ndarray
    dtot: np.ndarray
    s2: np.ndarray
    lam: np.ndarray
    g2: Optional[np.ndarray]
    sum_sq: np.ndarray
    eta: np.ndarray
    count_accept: dict
    count_propose: dict
    B: np.ndarray = field(repr=False, default=None)

    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.nbasis.shape[0])

    def acceptance_rates(self) -> dict:
        """Acceptance rate per move type."""
        return {
            k: (self.count_accept[k] / self.count_propose[k])
            if self.count_propose[k]
            else float("nan")
            for k in self.count_propose
        }

    def design_matrix(self, newdata, iteration: int) -> np.ndarray:
        """Design matrix ``[1, Psi_1, ..., Psi_M]`` for one posterior draw."""
        newdata = np.asarray(newdata, dtype=float)
        if newdata.ndim == 1:
            newdata = newdata[:, None]
        M = int(self.nbasis[iteration])
        B = np.ones((newdata.shape[0], M + 1))
        for j in range(M):
            k = int(self.nint[iteration, j])
            B[:, j + 1] = make_basis(
                self.vars[iteration, j, :k], self.degs[iteration, j, :k], newdata
            )
        return B

    def predict(
        self,
        newdata=None,
        mcmc_use=None,
        nugget: bool = False,
        nreps: int = 1,
        seed=None,
    ) -> np.ndarray:
        """Posterior predictive draws (R: ``predict.adaptive_khaos``).

        Parameters
        ----------
        newdata : array_like of shape (n_new, p), optional
            Defaults to the training inputs.
        mcmc_use : sequence of int, optional
            Which retained iterations to use; defaults to all of them.
        nugget : bool
            If ``True``, add observation noise :math:`N(0, \\sigma^2)`.
        nreps : int
            Draws per posterior sample (ignored when ``nugget`` is ``False``).

        Returns
        -------
        ndarray of shape (len(mcmc_use) * nreps, n_new)
            One row per predictive draw.
        """
        rng = np.random.default_rng(seed)
        if newdata is None:
            newdata = self.X
        newdata = np.asarray(newdata, dtype=float)
        if newdata.ndim == 1:
            newdata = newdata[:, None]
        if mcmc_use is None:
            mcmc_use = np.arange(self.n_samples)
        mcmc_use = np.atleast_1d(np.asarray(mcmc_use, dtype=int))
        if not nugget:
            nreps = 1

        n_new = newdata.shape[0]
        out = np.empty((mcmc_use.shape[0] * nreps, n_new))
        for cnt, i in enumerate(mcmc_use):
            M = int(self.nbasis[i])
            B = self.design_matrix(newdata, i)
            mu = B @ self.beta[i, : M + 1]
            if nugget:
                sd = np.sqrt(self.s2[i])
                block = np.broadcast_to(mu, (nreps, n_new)) + rng.normal(
                    0.0, sd, size=(nreps, n_new)
                )
                out[cnt * nreps : (cnt + 1) * nreps, :] = block
            else:
                out[cnt] = mu
        return out

    def sobol(self, plot: bool = False):
        """Sobol sensitivity indices; see :func:`khaos.sobol.sobol_khaos`."""
        from .sobol import sobol_khaos

        return sobol_khaos(self, plot=plot)

    # ------------------------------------------------------------------
    def plot(self, show: bool = True):
        """Four-panel diagnostic (R: ``plot.adaptive_khaos``).

        Traces of ``nbasis`` and ``sigma^2``, observed-versus-fitted with
        +/- 2 sd bars, and a residual histogram against a normal reference.
        """
        import matplotlib.pyplot as plt

        preds = self.predict(nugget=True, nreps=10)
        yhat = preds.mean(axis=0)
        ci = 2.0 * preds.std(axis=0, ddof=1)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].plot(self.nbasis, lw=0.8)
        axes[0, 0].set_ylabel("nbasis")
        axes[0, 1].plot(self.s2, lw=0.8)
        axes[0, 1].set_ylabel("s2")

        ax = axes[1, 0]
        ax.vlines(self.y, yhat - ci, yhat + ci, color="orange", lw=1)
        ax.plot(self.y, yhat, "o", ms=4, color="black")
        lims = [min(self.y.min(), yhat.min()), max(self.y.max(), yhat.max())]
        ax.plot(lims, lims, color="dodgerblue")
        ax.set_xlabel("y")
        ax.set_ylabel("yhat")

        resid = self.y - yhat
        ax = axes[1, 1]
        ax.hist(resid, bins="auto", density=True, color="lightgrey",
                edgecolor="white")
        grid = np.linspace(resid.min(), resid.max(), 200)
        sd = resid.std(ddof=1)
        if sd > 0:
            ax.plot(
                grid,
                np.exp(-0.5 * ((grid - resid.mean()) / sd) ** 2)
                / (sd * np.sqrt(2 * np.pi)),
                color="orange",
            )
        ax.set_xlabel("residual")

        fig.tight_layout()
        if show:
            plt.show()
        return fig
