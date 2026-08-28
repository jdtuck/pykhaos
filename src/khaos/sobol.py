"""Global sensitivity analysis from a fitted PCE.

Port of ``sobol_khaos`` (adaptive branch) from ``R/sobol_khaos.R``.

Because the basis is orthonormal on the unit hypercube, the Sobol
decomposition is available in closed form from the coefficients alone: each
basis function depends on exactly one subset :math:`u` of the inputs, so

.. math::

    V_u = \\sum_{m \\in \\mathcal{A}_u} \\beta_m^2,
    \\qquad S_u = \\frac{V_u}{\\mathrm{Var}(f(x))},
    \\qquad T_i = \\sum_{u \\ni i} S_u .

Following the R implementation, the denominator is
:math:`\\sum_m \\beta_m^2 + \\sigma^2`, i.e. the *observation* variance rather
than the function variance -- so ``leftover`` is the fraction of variance left
unexplained by the surrogate and every index is shrunk accordingly.  Indices
are computed once per retained MCMC iteration, giving a posterior sample of
each index rather than a point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SobolResult", "sobol_khaos"]


@dataclass
class SobolResult:
    """Posterior samples of Sobol indices.

    Attributes
    ----------
    S : ndarray of shape (n_keep, n_terms)
        Partial indices :math:`S_u`, one column per interaction set that
        appeared anywhere in the chain.
    labels : list of str
        Column names for ``S``, e.g. ``'x1'`` or ``'x1:x3'`` (1-based, matching
        the R output).
    T : ndarray of shape (n_keep, p)
        Total-effect indices :math:`T_i`.
    first_order : ndarray of shape (n_keep, p)
        Main-effect indices :math:`S_i`.
    leftover : ndarray of shape (n_keep,)
        Unexplained variance fraction :math:`\\sigma^2 / \\mathrm{Var}(y)`.
    """

    S: np.ndarray
    labels: list
    T: np.ndarray
    first_order: np.ndarray
    leftover: np.ndarray

    def summary(self, q=(0.05, 0.5, 0.95)):
        """Posterior quantiles of every index, as a dict of arrays."""
        return {
            "labels": list(self.labels),
            "S": np.nanquantile(self.S, q, axis=0),
            "T": np.nanquantile(self.T, q, axis=0),
            "first_order": np.nanquantile(self.first_order, q, axis=0),
            "leftover": np.nanquantile(self.leftover, q),
            "quantiles": tuple(q),
        }

    def to_dataframe(self):
        """``(S, T)`` as pandas DataFrames (requires pandas)."""
        import pandas as pd

        p = self.T.shape[1]
        names = [f"x{j + 1}" for j in range(p)]
        return (
            pd.DataFrame(self.S, columns=self.labels),
            pd.DataFrame(self.T, columns=names),
        )


def sobol_khaos(fit, plot: bool = False) -> SobolResult:
    """Compute Sobol indices from an :class:`~khaos.model.AdaptiveKhaos` fit.

    Parameters
    ----------
    fit : AdaptiveKhaos
        A fit returned by :func:`khaos.adaptive_khaos`.
    plot : bool
        Draw boxplots of the partial and total indices (requires matplotlib).

    Returns
    -------
    SobolResult
    """
    beta = np.asarray(fit.beta, dtype=float)
    nint = np.asarray(fit.nint)
    vars_ = np.asarray(fit.vars)
    s2 = np.asarray(fit.s2, dtype=float)

    n_iter = beta.shape[0]
    M = beta.shape[1] - 1
    p = fit.X.shape[1]

    first_order = np.full((n_iter, p), np.nan)
    total_effect = np.full((n_iter, p), np.nan)
    leftover = np.full(n_iter, np.nan)

    per_iter = [dict() for _ in range(n_iter)]
    all_labels: dict[str, None] = {}

    for it in range(n_iter):
        b = beta[it, 1:]  # drop intercept
        b2 = np.where(np.isfinite(b), b, 0.0) ** 2
        explained = float(np.nansum(b2))
        residual = float(s2[it])
        total_var = explained + residual
        if not np.isfinite(total_var) or total_var < 1e-12:
            continue

        # alpha_mat[m, j] > 0 iff basis m uses input j
        alpha_mat = np.zeros((M, p), dtype=np.int64)
        for m in range(M):
            k = int(nint[it, m]) if m < nint.shape[1] else 0
            if k <= 0:
                continue
            idx = vars_[it, m, :k]
            # np.unique also guards against a variable appearing twice in one
            # multi-index, which the reference variable-swap mutation allows
            # (see ``legacy_swap`` in :func:`khaos.adaptive_khaos`).
            idx = np.unique(idx[idx >= 0])
            if idx.size == 0:
                continue
            alpha_mat[m, idx] = 1
            label = ":".join(f"x{j + 1}" for j in idx)
            all_labels.setdefault(label, None)
            per_iter[it][label] = per_iter[it].get(label, 0.0) + b2[m]

        for label in per_iter[it]:
            per_iter[it][label] /= total_var

        uses = alpha_mat > 0
        other = uses.sum(axis=1)
        for j in range(p):
            only_j = uses[:, j] & (other == 1)
            any_j = uses[:, j]
            first_order[it, j] = b2[only_j].sum() / total_var
            total_effect[it, j] = b2[any_j].sum() / total_var

        leftover[it] = residual / total_var

    labels = sorted(all_labels, key=lambda s: (s.count(":") + 1, s))
    S = np.zeros((n_iter, len(labels)))
    pos = {lab: i for i, lab in enumerate(labels)}
    for it in range(n_iter):
        for lab, val in per_iter[it].items():
            S[it, pos[lab]] = val

    result = SobolResult(
        S=S, labels=labels, T=total_effect, first_order=first_order,
        leftover=leftover,
    )

    if plot:
        _plot_sobol(result)
    return result


def _plot_sobol(res: SobolResult):
    import matplotlib.pyplot as plt

    p = res.T.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot(
        np.column_stack([res.S, res.leftover]),
        tick_labels=list(res.labels) + ["leftover"],
    )
    axes[0].set_title("Sensitivity")
    axes[1].boxplot(
        np.column_stack([res.T, res.leftover]),
        tick_labels=[f"x{j + 1}" for j in range(p)] + ["leftover"],
    )
    axes[1].set_title("Total sensitivity")
    for ax in axes:
        ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    plt.show()
    return fig
