"""khaos: Bayesian adaptive polynomial chaos expansions in Python.

A NumPy/SciPy port of the adaptive (reversible-jump MCMC) sampler from the
``khaos`` R package, implementing

    Rumsey, K. N., Francom, D., Gibson, G., Tucker, J. D. and Huerta, G. (2026).
    "Bayesian Adaptive Polynomial Chaos Expansions." *Stat*, 15(1), e70151.
    https://doi.org/10.1002/sta4.70151

R reference implementation: https://github.com/knrumsey/khaos

Quick start
-----------
>>> import numpy as np
>>> from khaos import adaptive_khaos, sobol_khaos
>>> rng = np.random.default_rng(0)
>>> X = rng.random((150, 4))
>>> y = np.sin(np.pi * X[:, 0]) + 2 * (X[:, 1] - 0.5) ** 2 + rng.normal(0, .05, 150)
>>> fit = adaptive_khaos(X, y, prior_type="gprior", nmcmc=3000, nburn=2000,
...                      seed=1, verbose=False)
>>> yhat = fit.predict().mean(axis=0)
>>> idx = sobol_khaos(fit)
"""

from ._compat import HAVE_NUMBA
from .adaptive import (
    CoinPars,
    adaptive_khaos,
    adaptive_khaos_gprior,
    adaptive_khaos_ridge,
)
from .basis import legendre_poly, make_basis, make_basis_matrix, ss_legendre_poly
from .gprior import build_G, g_weight
from .model import AdaptiveKhaos
from .proposals import A_size, log_A_size, make_weights, random_partition
from .sobol import SobolResult, sobol_khaos
from .threads import single_threaded_blas

__version__ = "0.1.0"

__all__ = [
    "adaptive_khaos",
    "adaptive_khaos_ridge",
    "adaptive_khaos_gprior",
    "CoinPars",
    "AdaptiveKhaos",
    "sobol_khaos",
    "SobolResult",
    "legendre_poly",
    "ss_legendre_poly",
    "make_basis",
    "make_basis_matrix",
    "A_size",
    "log_A_size",
    "make_weights",
    "random_partition",
    "build_G",
    "g_weight",
    "single_threaded_blas",
    "HAVE_NUMBA",
    "__version__",
]
