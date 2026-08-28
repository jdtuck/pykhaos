"""Optional-Numba shim.

The package is fully functional without Numba installed; ``njit`` degrades to a
no-op decorator and the pure-NumPy fallbacks are used instead.  When Numba *is*
available the hot inner kernels (Legendre evaluation, tensor-product basis
construction, coin-flip weights) are compiled.
"""

from __future__ import annotations

import os

__all__ = ["njit", "HAVE_NUMBA", "prange"]

_DISABLE = os.environ.get("KHAOS_NO_NUMBA", "").lower() in {"1", "true", "yes"}

try:  # pragma: no cover - trivial import branch
    if _DISABLE:
        raise ImportError("Numba disabled via KHAOS_NO_NUMBA")
    from numba import njit as _njit, prange  # type: ignore

    HAVE_NUMBA = True

    def njit(*args, **kwargs):
        kwargs.setdefault("cache", True)
        kwargs.setdefault("fastmath", False)
        return _njit(*args, **kwargs)

except Exception:  # pragma: no cover - exercised only when numba is absent
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        # Support both ``@njit`` and ``@njit(...)`` usage.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap
