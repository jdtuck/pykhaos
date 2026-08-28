"""Adaptive Bayesian PCE on the Friedman function.

Fits both coefficient priors to n = 500 points of

    f(x) = 10 sin(pi x1 x2) + 20 (x3 - 1/2)^2 + 10 x4 + 5 x5 + 0 x6 + ... + 0 x10

(the last five inputs are inert), then reports held-out accuracy, coverage of
the 90% predictive interval, and posterior Sobol indices.

Run with::

    python examples/demo_friedman.py
"""

from __future__ import annotations

import time

import numpy as np

import khaos


def friedman(X: np.ndarray) -> np.ndarray:
    return (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
    )


def main(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n, p, noise = 500, 10, 1.0

    X = rng.random((n, p))
    y = friedman(X) + rng.normal(0.0, noise, n)
    X_test = rng.random((2000, p))
    y_test = friedman(X_test)                       # noise-free truth
    y_test_obs = y_test + rng.normal(0.0, noise, X_test.shape[0])

    print(f"n = {n}, p = {p} (5 active), noise sd = {noise}")
    print(f"numba acceleration: {khaos.HAVE_NUMBA}\n")

    fits = {}
    with khaos.single_threaded_blas():
        for prior in ("ridge", "gprior"):
            t0 = time.perf_counter()
            fit = khaos.adaptive_khaos(
                X, y,
                prior_type=prior,
                degree=10,
                order=3,
                nmcmc=20_000,
                nburn=10_000,
                thin=10,
                seed=seed + 1,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            fits[prior] = fit

            draws = fit.predict(X_test, nugget=True, nreps=2, seed=seed)
            mean = draws.mean(axis=0)
            lo, hi = np.quantile(draws, [0.05, 0.95], axis=0)

            rmse = np.sqrt(np.mean((y_test - mean) ** 2))
            # Coverage is checked against *observed* responses, since the
            # predictive interval includes the observation noise.
            coverage = np.mean((y_test_obs >= lo) & (y_test_obs <= hi))

            print(f"[{prior}] {elapsed:5.1f}s")
            print(f"    basis functions   {fit.nbasis.mean():5.1f} "
                  f"(range {fit.nbasis.min()}-{fit.nbasis.max()})")
            print(f"    sigma^2           {fit.s2.mean():5.2f}  "
                  f"(truth {noise ** 2:.2f})")
            print(f"    test RMSE         {rmse:5.3f}")
            print(f"    90% coverage      {coverage:5.1%}")
            if fit.g2 is not None:
                print(f"    g0^2 (median)     {np.median(fit.g2):5.3g}")
            rates = fit.acceptance_rates()
            print("    acceptance        "
                  + ", ".join(f"{k.split()[0][:5]} {v:.1%}"
                              for k, v in rates.items()))
            print()

    # ---- sensitivity analysis on the better-mixing fit --------------------
    fit = fits["gprior"]
    idx = khaos.sobol_khaos(fit)
    total = np.nanmean(idx.T, axis=0)
    print("Total-effect Sobol indices (posterior mean):")
    for j in np.argsort(-total):
        bar = "#" * int(round(50 * total[j]))
        print(f"    x{j + 1:<3d} {total[j]:6.3f}  {bar}")
    print(f"    unexplained {np.nanmean(idx.leftover):6.3f}")

    print("\nStrongest interaction terms:")
    means = np.nanmean(idx.S, axis=0)
    pairs = [(lab, m) for lab, m in zip(idx.labels, means) if ":" in lab]
    for lab, m in sorted(pairs, key=lambda t: -t[1])[:5]:
        print(f"    {lab:<12s} {m:6.3f}")


if __name__ == "__main__":
    main()
