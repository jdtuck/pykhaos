"""BLAS thread control.

The RJMCMC inner loop is thousands of tiny (``M+1`` square, typically < 100)
Cholesky factorisations and eigendecompositions.  Multi-threaded BLAS spends
more time synchronising threads than doing arithmetic at that size -- on a
typical machine, forcing a single thread makes the sampler roughly 4-5x
faster.  Set the environment variables before importing NumPy, or wrap the
fit::

    with khaos.single_threaded_blas():
        fit = khaos.adaptive_khaos(X, y)
"""

from __future__ import annotations

import contextlib

__all__ = ["single_threaded_blas"]


@contextlib.contextmanager
def single_threaded_blas(limit: int = 1):
    """Temporarily cap BLAS/LAPACK threads (no-op without ``threadpoolctl``).

    Parameters
    ----------
    limit : int
        Maximum threads per native pool inside the ``with`` block.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        import warnings

        warnings.warn(
            "threadpoolctl is not installed, so BLAS threads cannot be capped "
            "here. Install it, or set OMP_NUM_THREADS=1 / "
            "OPENBLAS_NUM_THREADS=1 before importing numpy.",
            stacklevel=2,
        )
        yield
        return

    with threadpool_limits(limits=limit):
        yield
