"""Input rescaling onto the unit hypercube.

The polynomial chaos basis is orthonormal with respect to the *uniform measure
on* :math:`[0, 1]^p`, so the sampler needs inputs on that cube.  Rather than
making that the caller's problem, a fit carries an :class:`InputScaler` that
maps user-space inputs to the cube and is re-applied to every new ``X`` at
prediction time.

The map is affine and per-column,

.. math::

    z_j = \\frac{x_j - \\ell_j}{u_j - \\ell_j},

so it changes nothing about the model: a polynomial in :math:`z` of degree
:math:`d` is a polynomial in :math:`x` of degree :math:`d`, and the admissible
set, priors and proposals are untouched.  What it *does* fix is the measure the
Sobol decomposition is taken with respect to -- indices are shares of variance
under a uniform distribution over the box :math:`[\\ell, u]`, so choosing that
box deliberately (via ``x_range``) is worth doing when you know the real input
ranges.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ["InputScaler"]

_DEGENERATE_TOL = 1e-12


@dataclass
class InputScaler:
    """Affine per-column map from user space onto :math:`[0, 1]^p`.

    Attributes
    ----------
    lower, upper : ndarray of shape (p,)
        Box the inputs are mapped from.  ``upper`` is never below ``lower``.
    degenerate : ndarray of bool, shape (p,)
        Columns whose range collapsed (a constant input).  These are mapped to
        the constant 0.5 rather than dividing by zero.
    identity : bool
        ``True`` when the map is the no-op :math:`z = x`; used to keep already
        scaled data bit-identical to an unscaled fit.
    """

    lower: np.ndarray
    upper: np.ndarray
    degenerate: np.ndarray
    identity: bool = False

    # -- constructors ------------------------------------------------------
    @classmethod
    def identity_map(cls, p: int) -> "InputScaler":
        """The no-op map, for inputs already on the unit cube."""
        return cls(
            lower=np.zeros(p),
            upper=np.ones(p),
            degenerate=np.zeros(p, dtype=bool),
            identity=True,
        )

    @classmethod
    def from_data(cls, X: np.ndarray) -> "InputScaler":
        """Min-max map fitted to the columns of ``X``.

        Training points land exactly on :math:`[0, 1]`, endpoints included.
        Note that this makes the fit depend on the observed spread: two runs on
        different samples of the same system are scaled differently.  Pass
        ``x_range`` instead when the physical bounds are known.
        """
        X = np.asarray(X, dtype=float)
        lower = X.min(axis=0)
        upper = X.max(axis=0)
        return cls._build(lower, upper)

    @classmethod
    def from_range(cls, x_range, p: int) -> "InputScaler":
        """Map fitted to explicit bounds.

        Parameters
        ----------
        x_range : array_like
            Either two scalars ``(lower, upper)`` applied to every column, or
            an array of shape ``(2, p)`` whose rows are the per-column lower
            and upper bounds.
        """
        arr = np.asarray(x_range, dtype=float)
        if arr.shape == (2,):
            lower = np.full(p, arr[0])
            upper = np.full(p, arr[1])
        elif arr.shape == (2, p):
            lower, upper = arr[0].copy(), arr[1].copy()
        else:
            raise ValueError(
                "x_range must be (lower, upper) or an array of shape (2, p); "
                f"got shape {arr.shape} for p = {p}"
            )
        if np.any(upper < lower):
            raise ValueError("x_range upper bounds must not be below the lower bounds")
        return cls._build(lower, upper)

    @classmethod
    def _build(cls, lower: np.ndarray, upper: np.ndarray) -> "InputScaler":
        lower = np.asarray(lower, dtype=float).ravel()
        upper = np.asarray(upper, dtype=float).ravel()
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            raise ValueError("input bounds must be finite")
        degenerate = (upper - lower) <= _DEGENERATE_TOL
        if np.any(degenerate):
            warnings.warn(
                f"Input column(s) {np.flatnonzero(degenerate).tolist()} are "
                "constant over the given range; they are mapped to 0.5 and "
                "carry no information.",
                stacklevel=3,
            )
        return cls(lower=lower, upper=upper, degenerate=degenerate)

    # -- application -------------------------------------------------------
    @property
    def p(self) -> int:
        return int(self.lower.shape[0])

    def transform(self, X: np.ndarray, clip: bool = False) -> np.ndarray:
        """Map ``X`` onto the unit cube."""
        X = np.asarray(X, dtype=float)
        if self.identity:
            return np.ascontiguousarray(X)
        if X.shape[1] != self.p:
            raise ValueError(
                f"expected {self.p} columns to rescale, got {X.shape[1]}"
            )
        span = np.where(self.degenerate, 1.0, self.upper - self.lower)
        Z = (X - self.lower) / span
        if np.any(self.degenerate):
            Z[:, self.degenerate] = 0.5
        if clip:
            Z = np.clip(Z, 0.0, 1.0)
        return np.ascontiguousarray(Z)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map points on the unit cube back to user space."""
        Z = np.asarray(Z, dtype=float)
        if self.identity:
            return np.ascontiguousarray(Z)
        span = np.where(self.degenerate, 0.0, self.upper - self.lower)
        return np.ascontiguousarray(self.lower + Z * span)

    def out_of_range(self, Z: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """Boolean mask of rows of already-scaled ``Z`` that leave the cube."""
        Z = np.asarray(Z, dtype=float)
        return np.any((Z < -tol) | (Z > 1.0 + tol), axis=1)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.identity:
            return "InputScaler(identity)"
        return (
            f"InputScaler(lower={np.array2string(self.lower, precision=4)}, "
            f"upper={np.array2string(self.upper, precision=4)})"
        )


def resolve_scaler(X: np.ndarray, scale_inputs, x_range) -> InputScaler:
    """Pick the scaler implied by ``scale_inputs`` / ``x_range``.

    ``scale_inputs``
        ``"auto"`` (default)
            Leave inputs alone when they already lie in :math:`[0, 1]`;
            otherwise fit a min-max map to the data.  Keeps already-scaled
            problems bit-identical to an unscaled fit.
        ``True`` / ``"minmax"``
            Always fit a min-max map to the data.
        ``False`` / ``None``
            No rescaling; warn if the data leaves the cube.

    Passing ``x_range`` implies rescaling with those bounds and overrides
    ``scale_inputs`` unless it is explicitly ``False``.
    """
    p = X.shape[1]

    if x_range is not None:
        if scale_inputs is False or scale_inputs is None:
            raise ValueError("x_range was given but scale_inputs is False")
        scaler = InputScaler.from_range(x_range, p)
        Z = scaler.transform(X)
        outside = scaler.out_of_range(Z)
        if np.any(outside):
            warnings.warn(
                f"{int(outside.sum())} of {X.shape[0]} training rows fall "
                "outside the supplied x_range; the fit will extrapolate the "
                "polynomial basis there.",
                stacklevel=3,
            )
        return scaler

    if scale_inputs is False or scale_inputs is None:
        if X.max() > 1 or X.min() < 0:
            warnings.warn(
                "Inputs are expected to be scaled to [0, 1] and are not. "
                "Pass scale_inputs=True (or leave it at 'auto') to rescale "
                "them automatically.",
                stacklevel=3,
            )
        return InputScaler.identity_map(p)

    if scale_inputs is True or scale_inputs == "minmax":
        return InputScaler.from_data(X)

    if scale_inputs == "auto":
        if X.size and (X.max() > 1.0 or X.min() < 0.0):
            return InputScaler.from_data(X)
        return InputScaler.identity_map(p)

    raise ValueError(
        "scale_inputs must be 'auto', True/'minmax', or False; "
        f"got {scale_inputs!r}"
    )
