# khaos (Python)

Bayesian **adaptive polynomial chaos expansions** via reversible-jump MCMC — a
NumPy/SciPy/Numba port of the adaptive sampler in the [`khaos` R
package](https://github.com/knrumsey/khaos), implementing

> Rumsey, K. N., Francom, D., Gibson, G., Tucker, J. D. and Huerta, G. (2026).
> "Bayesian Adaptive Polynomial Chaos Expansions." *Stat*, 15(1), e70151.
> <https://doi.org/10.1002/sta4.70151>

Ported: `adaptive_khaos` (both the ridge and the modified-g-prior variants),
`predict`, `plot`, and `sobol_khaos`. Not ported: `sparse_khaos`,
`ordinal_khaos`, `enrichment`.

## Install

```bash
pip install -e .            # numpy, scipy, numba, threadpoolctl
pip install -e ".[dev]"     # + pytest, matplotlib, pandas
pytest                      # 101 tests, ~25 s
```

Numba is optional at runtime: set `KHAOS_NO_NUMBA=1` to fall back to pure NumPy
kernels (same results, slower basis construction).

## Quick start

```python
import numpy as np
import khaos

rng = np.random.default_rng(0)
X = rng.random((500, 10))                    # inputs must be scaled to [0, 1]
y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1])
     + 20 * (X[:, 2] - 0.5) ** 2
     + 10 * X[:, 3] + 5 * X[:, 4]
     + rng.normal(0, 1, 500))

with khaos.single_threaded_blas():           # see "Performance" below
    fit = khaos.adaptive_khaos(
        X, y, prior_type="gprior",
        degree=10, order=3,                  # d_max, q_max
        nmcmc=20_000, nburn=10_000, thin=10,
        seed=1,
    )

draws = fit.predict(X_new, nugget=True, nreps=2)   # (n_draws, n_new)
mean, lo, hi = draws.mean(0), *np.quantile(draws, [.05, .95], axis=0)

idx = khaos.sobol_khaos(fit)                 # posterior samples of S_u and T_i
print(np.nanmean(idx.T, axis=0))
```

`examples/demo_friedman.py` runs the whole thing end to end. On the Friedman
function (n = 500, p = 10 with five inert inputs, noise sd 1):

| | basis fns | σ̂² | test RMSE | 90% coverage |
|---|---|---|---|---|
| ridge | 10.2 | 1.12 | 0.320 | 91.8% |
| g-prior | 13.7 | 1.14 | 0.364 | 92.1% |

and the recovered total-effect indices (x4 0.33, x2 0.25, x1 0.24, x3 0.10,
x5 0.09, inert inputs 0.000) match the analytic Sobol decomposition.

## The model

Inputs live on the unit hypercube, so the univariate basis is the standardised
shifted Legendre polynomial `ψ_α(x) = √(2α+1) P_α(2x−1)`, orthonormal under the
uniform measure. Basis functions are tensor products over the active inputs:

```
y_i = β₀ + Σ_m β_m Ψ_m(x_i | α_m) + ε_i,     ε_i ~ N(0, σ²)
α_m ~ Unif(A_{p,d_max,q_max}),  M | λ ~ Poisson(λ),  λ ~ Gamma(a_M, b_M)
```

Each MCMC sweep does one trans-dimensional move — **birth**, **death**, or
**mutate** (re-draw the degree partition, or swap one variable; the two
mutation types are chosen from their running acceptance rates) — then Gibbs
updates for β, σ², λ and, under the g-prior, a Metropolis–Hastings update of
g₀².

Birth proposals use the paper's **coin-flipping** scheme rather than the NKD
scheme of Nott et al. (2005): draw an expected interaction order `q₀`, build
per-variable inclusion probabilities that favour inputs already earning their
keep, flip one coin per input, and reject-and-retry when the draw is empty or
exceeds `q_max` (the delayed-rejection density is folded into the acceptance
ratio).

**Coefficient priors.** `prior_type="ridge"` uses `β | σ² ~ N(0, σ²τ²I)`.
`prior_type="gprior"` uses the paper's modified g-prior,

```
β | σ², g₀² ~ N(0, σ² g₀² D(g) (Ψ'Ψ)⁻¹ D(g)),   g_m = (1 + q(q + d − 2))^(−ζ/2)
```

so higher-degree, higher-order terms are shrunk harder; `ζ = 0` recovers the
standard g-prior. The posterior precision is `G ∘ Ψ'Ψ` with
`G_{mℓ} = 1 + 1/(g₀² g_m g_ℓ)`, and g₀² is updated by MH against a Laplace
approximation (`g2_sample` picks which one).

## Where this differs from the R reference

Three switches control behaviour where the reference implementation and the
paper disagree. **All three default to the paper's algorithm**; flip them to
reproduce R exactly.

| flag | default | R behaviour (`= reference`) |
|---|---|---|
| `sync_g2` | `True` | `False` — after the Gibbs update of g₀², R leaves the *current* model's `Σ`, `Q`, `ldet` at whatever g₀² was current when the last move was accepted, so each acceptance ratio compares two models at different g₀². On the Friedman benchmark this drops birth/death acceptance from ~26% to ~1%, leaves the chain stuck, and roughly triples test RMSE. |
| `legacy_swap` | `False` | `True` — the variable-swap mutation excludes only the variable being removed from the candidate pool, so it can put the *same* input into a basis function twice. The paper describes the move as swapping in an *inactive* variable. |
| `exact_marginal` | `False` | `False` — experimental. Adds the `−½log\|S₀\|` normalising term that R drops from the birth/death ratio (R *does* include the analogous `0.5·log(1/τ²)` in the ridge branch). It gives a genuine Bayes factor — with `ζ = 0`, the classical `(1+g₀²)^{−1/2}` Occam factor per term — but that penalty is weaker than what the reference effectively applies, so with the permissive default `b_M = 4/n` the model can grow very large. Tighten `b_M` if you use it. |

Smaller notes, all preserved rather than "fixed" so results match:

- **`coin_pars` argument order.** R calls
  `make_weights(eta, q0, coin_pars[[3]], coin_pars[[2]], coin_pars[[4]])`
  against the signature `make_weights(eta, p0, epsilon, alpha, num_passes)`, so
  `epsilon` and `alpha` arrive swapped relative to the R docs. `CoinPars` keeps
  the same wiring — with the defaults, base weights are `eta**1 + 2`.
- **Ridge quadratic form.** R uses `d = b_σ + y'y − β̂'V⁻¹β̂`, missing the
  factor of ½ the normal-inverse-gamma marginal calls for. It cancels out of
  every ratio when `b_sigma = 0` (the default Jeffreys prior), so the two agree
  in practice; `exact_marginal=True` uses the textbook form.
- **`dgsq_orth`.** R passes the unsquared weight vector into an argument named
  `gm_sq`. Preserved.
- **Laplace centring.** `laplace_orth` / `laplace_full` return `(a, b)` with the
  fit centred at `b/a` — R's `c = 0` matches neither the mode nor the mean of
  the resulting inverse-gamma.
- **`q_max == p`.** R's weight calibration divides by `log(1) = 0` and returns
  `NaN`; here the all-ones vector (the intended limit) is returned instead, so
  `order == ncol(X)` runs. It is still a degenerate corner — every birth then
  proposes all inputs.
- **`legacy` storage flag.** Not ported. It only controlled whether stale
  entries were left in R's ragged arrays beyond `nint`, which nothing reads.

Two changes are pure implementation, with no effect on the sampled chain:
sufficient statistics are updated incrementally (rank-one for birth/death,
single-column for mutations) instead of recomputing `crossprod(B)` each sweep,
and the coin-flip weights are cached per `eta`.

## Performance

The inner loop is thousands of tiny (`M+1`-square, usually < 100) Cholesky
factorisations and eigendecompositions. Multi-threaded BLAS spends more time
synchronising than computing at that size — **forcing a single thread makes the
sampler about 4–5× faster**:

```python
with khaos.single_threaded_blas():
    fit = khaos.adaptive_khaos(X, y)
```

or set `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` before importing NumPy.

Rough cost: 20 000 sweeps at n = 500, p = 10, M ≈ 15 takes ~9 s (ridge) or
~22 s (g-prior, which also samples g₀²).

## API

```
adaptive_khaos(X, y, prior_type="ridge", **kw)   -> AdaptiveKhaos
adaptive_khaos_ridge(X, y, ...)                  -> AdaptiveKhaos
adaptive_khaos_gprior(X, y, ...)                 -> AdaptiveKhaos
sobol_khaos(fit, plot=False)                     -> SobolResult

AdaptiveKhaos: .predict(newdata, mcmc_use, nugget, nreps, seed)
               .sobol()  .plot()  .design_matrix(newdata, i)
               .acceptance_rates()  .n_samples
               .nbasis .beta .vars .degs .nint .dtot .s2 .lam .g2 .eta

SobolResult:   .S (partial, per interaction set)  .labels
               .T (total-effect)  .first_order  .leftover
               .summary(q)  .to_dataframe()
```

Posterior arrays are indexed by *retained* iteration; `vars` holds 0-based
column indices (R uses 1-based), padded with `-1`.

## Verification

`pytest` covers the pieces that are easy to get subtly wrong, mostly against
independent references rather than the implementation itself:

- Legendre recurrence vs `scipy.special.eval_legendre`; orthonormality on
  [0, 1] by Gauss–Legendre quadrature.
- `A_size` vs brute-force enumeration of the admissible multi-index set.
- Coin-flip proposal probabilities sum to 1 over all 2^p subsets.
- `random_partition` hits exactly the set of valid compositions.
- **Collapsed marginal likelihoods vs an independent n×n computation** of the
  normal-inverse-gamma marginal, for both priors — the single most load-bearing
  test; also that `ζ = 0` reproduces the classical g-prior Bayes factor, and
  that a death exactly undoes the matching birth.
- Incremental `B'B` / `B'y` updates vs full recomputation.
- Laplace fits vs a grid search over the log posterior they approximate.
- Singular matrices are a *handled* outcome everywhere (the move is rejected),
  so the suite runs with `error::RuntimeWarning` — no floating-point warning is
  allowed to escape. Log-determinants go through `linalg.safe_logdet`, which
  tries Cholesky and only falls back to `slogdet` under `np.errstate`, since
  whether `slogdet` warns on a near-singular pivot depends on the LAPACK build.
- End to end: recovery of a known two-term expansion, of the active variable
  set, of σ², and of the analytic Sobol shares for an additive and a pure
  interaction target; seeded reproducibility; `degree`/`order`/`max_basis`
  limits honoured.

## License

BSD-3-Clause, matching the R package.
