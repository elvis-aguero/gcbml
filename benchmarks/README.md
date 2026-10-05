# Benchmark problems for gcbml

Seven cheap NumPy/SciPy solvers. The grid error of each is produced by a real discretisation (never a formula added to the answer). Each has a converged value at h = 0 that is known in closed form; the derivation is in the module docstring and a test checks it. Every problem implements `benchmarks.base.BenchmarkProblem`: `problem()`, `run(probe, seed)`, `truth(x_unit)`, `expected_order`, `notes`.

Run the tests on the allocation: `srun --jobid=<job> --overlap -c 4 uv run pytest tests/benchmarks -q` (about 10 s).

Conventions: `Probe.h` holds physical values (h = 1/N, so N = 1/h must be an integer); levels are h0 / 2^level. `RunResult.cost` is deterministic work in work units (see Cost below), not CPU time. `truth` takes unit inputs, with the v coordinate included for B3.

## The problems

| # | module | inputs (range) | QoI | truth | order measured | what it stresses |
|---|---|---|---|---|---|---|
| B1 | `b1_poisson` | amplitude [0.5, 2], rate [1, 3], anisotropy [0.1, 1] | integral of u over the square; `-(u_xx + kappa u_yy) = f`, u = A exp(w x + w y / 2), 5-point stencil, sparse LU | A (e^w - 1)/w * (e^{w/2} - 1)/(w/2) | 2.00 (to 3 digits) | the clean case; learned order close to 2 |
| B2 | `b2_upwind` | Peclet [2, 100], source [0.5, 2], outlet value [0.5, 1.5] | u(15/16) of `u' = u''/Pe + s`, u(0) = 0, u(1) = g, first-order upwind | `s (x - E) + g E`, E = expm1(x Pe)/expm1(Pe) | 1.0 at low Pe; at Pe = 100: 1.26, 1.42, 1.42, 1.32, 1.20, 1.11, 1.06 for levels 0-7 | order that depends on the input (pre-asymptotic at high Pe) |
| B3 | `b3_mixing` | diffusivity [0.5, 2], fill fraction [0.2, 0.5], **v = chi** [0.5, 0.95] (S1: n_controls 2 of 3) | time to reach chi = 0.5, 0.75, 0.9, 0.95 for 1-D diffusion of two layers; finite volumes + implicit Euler, dt = 0.2 h | root of chi(t) = v, chi from the cosine series (4000 modes) | 1.0 (0.98 to 1.10) | nested outputs, censoring (T_final = 0.2; unreached thresholds return y = T_final, `censored` True), error in space and in threshold detection |
| B4 | `b4_sde` | theta [0.5, 3], sigma [0.2, 1.5], x0 [0.5, 2] | E[X_T^2], T = 1, Euler-Maruyama, 2000 paths | x0^2 e^{-2 theta T} + sigma^2 (1 - e^{-2 theta T})/(2 theta) | weak order 1.01 to 1.11 (levels 2-6, exact scheme expectation) | aleatoric noise that depends on the inputs (sd ratio of about 6 between two test points); replicates are seeds |
| B5 | `b5_kink` | jump location [0.2, 0.8], frequency [1, 4], jump size [0.5, 2] | midpoint rule for sin(w x) + j H(x - a) | (1 - cos w)/w + j (1 - a) | none: error = j h xi(a/h) + O(h^2), xi in [-1/2, 1/2) | sign changes and non-monotone |error| as h halves (e.g. -8.6e-2, +6.2e-2, -1.9e-2, -1.9e-2, 2e-4 for one input) |
| B6 | `b6_heat` | horizon T [0.02, 0.1], mode-3 weight [0, 1], mode-5 weight [0, 1] | u(1/2, T) of `u_t = u_xx`, central differences + backward Euler, dt = T h_t | e^{-pi^2 T} - a e^{-9 pi^2 T} + b e^{-25 pi^2 T} | 2.0 in dx (1.95, 1.99, 2.00); 1.0 in dt (0.96, 0.98, 0.99) | k = 2 independent resolution components |
| B7 | `b7_thin_layer` | position [0.2, 0.8], amplitude [0.5, 2], width [3e-4, 9e-4] | outlet value of `u' = S(x)`, narrow Gaussian source, upwind | A w sqrt(pi/2) [erf + erf] | none; asymptotic range starts at level 8 (worst case over the region), 256 work units | the trap: levels 0-4 return about 0 (or 1e-74 apart), the truth is 2e-3. Must not claim success |

Level l has h = h0 / 2^l with h0 = 1/4 (B1, B4, B5, B6 both components), 1/8 (B3, B7), 1/16 (B2).

## Cost: deterministic work

`RunResult.cost` is `work(probe) * exp(sigma_c * N(0,1))`, in work units, with the normal draw seeded by (probe id, u, h, seed). `work` is the operation count of the discretisation, normalised so that the coarsest level of the problem costs 1 (so the coarsest run costs 1 work unit, level l costs 2^(gamma l) when every component is refined together). `sigma_c` is a class attribute (default 0.1; set it to 0 to switch the noise off and get the formula exactly). Measured CPU seconds are not the cost: use `BenchmarkProblem.measure_cpu(probe, seed)` for information. For microsecond 1-D solves CPU time is mostly call overhead and machine noise.

| problem | work formula (N = 1/h cells) | gamma | work at level 4 / 6 / 8 |
|---|---|---|---|
| b1_poisson | N^3 (sparse LU of the N^2-unknown 5-point matrix, O(n^1.5) flops) | 3 | 4096 / 262144 / 1.7e7 |
| b2_upwind | N (tridiagonal solve) | 1 | 16 / 64 / 256 |
| b3_mixing | N cells x N time steps | 2 | 256 / 4096 / 65536 |
| b4_sde | N time steps x 2000 paths | 1 | 16 / 64 / 256 |
| b5_kink | N midpoint evaluations | 1 | 16 / 64 / 256 |
| b6_heat | M cells x n steps (both refined together; 1 for each component alone) | 2 | 256 / 4096 / 65536 |
| b7_thin_layer | N (bidiagonal solve) | 1 | 16 / 64 / 256 |

**B7 and the budget.** Over the whole design region (checked on 40 positions at the narrowest, middle and widest source) every input is within 1% of the truth from level 8 on (`asymptotic_level = 8`, h = 1/2048), and some inputs are not at level 7. Level 8 costs 2^8 = 256 work units, against 1 for level 0. Levels 0 to 4 (cost 1 to 16 work units) return about 0 or a stray value that halves with h. So a budget that does not buy a level-8 run (about 256 work units, more with the lognormal noise) makes the trap real: the method can only see the misleading coarse levels. The measured CPU seconds of a level-8 run are about 3e-4 s; that number is why CPU time cannot be the cost here.

### Measured CPU seconds (information only)

Minimum of 3 runs at the centre of each design region, from `benchmarks/measure_costs.py`, on 4 cores of node1808 (the solvers are single-threaded). The last column is the fitted slope over the last 5 levels. It differs from the work gamma because of call and Python-loop overhead.

| problem | l0 | l1 | l2 | l3 | l4 | l5 | l6 | l7 | l8 | l9 | slope |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b1_poisson | 1.2e-03 | 1.4e-03 | 1.9e-03 | 3.7e-03 | 1.4e-02 | 6.7e-02 | 3.5e-01 | 2.3e+00 | | | 2.32 |
| b2_upwind | 6.2e-05 | 5.1e-05 | 4.9e-05 | 5.1e-05 | 5.7e-05 | 6.0e-05 | 7.4e-05 | 9.8e-05 | 1.6e-04 | 2.6e-04 | 0.53 |
| b3_mixing | 9.1e-04 | 1.0e-03 | 1.2e-03 | 1.7e-03 | 2.8e-03 | 5.8e-03 | 1.4e-02 | 3.9e-02 | 1.2e-01 | | 1.36 |
| b4_sde | 1.6e-04 | 2.9e-04 | 5.6e-04 | 1.1e-03 | 2.2e-03 | 4.4e-03 | 8.7e-03 | 1.7e-02 | 3.5e-02 | 7.0e-02 | 1.00 |
| b5_kink | 1.9e-05 | 1.6e-05 | 1.6e-05 | 1.7e-05 | 1.6e-05 | 1.7e-05 | 1.9e-05 | 2.5e-05 | 3.1e-05 | 4.7e-05 | 0.36 |
| b6_heat | 4.6e-04 | 4.8e-04 | 5.5e-04 | 6.5e-04 | 9.0e-04 | 1.5e-03 | 3.3e-03 | 8.9e-03 | 2.8e-02 | | 1.25 |
| b7_thin_layer | 6.6e-05 | 5.9e-05 | 5.7e-05 | 6.2e-05 | 5.7e-05 | 8.2e-05 | 1.1e-04 | 1.6e-04 | 2.7e-04 | 4.8e-04 | 0.64 |

## Notes per problem

- **B1.** Both the stencil and the trapezoid quadrature are second order. The error is proportional to the amplitude, so amplitude only scales it.
- **B2.** For the discrete solution the homogeneous solution is r^i with r = 1 + h/eps (exact exponential: e^{h/eps}), so the relative error is about h/(2 eps) and the first-order range starts when the cell Peclet number h Pe is small. At Pe = 100 and level 0 it is 6.25.
- **B3.** The reference is exact (series), not a Richardson value. No error bound is needed: the series is truncated at 4000 modes, and its tail at the times of interest is below 1e-300. chi uses the exact Var0 = L0 (1 - L0); using the discrete initial variance would add an O(h) error from the cell containing the interface. Truth is returned also for censored outputs.
- **B4.** `discrete_expectation` and `run_sd` are exact for the scheme (the scheme is linear and Gaussian), so the order and the spread are tested without sampling noise. The run-to-run sd of y is sqrt((2 v_h^2 + 4 m_h^2 v_h) / 2000).
- **B5.** For the test input a = 0.422, a/h has binary fraction 0.008 * 2^l, so the error doubles relative to h and is roughly constant for several levels: its size depends on the binary digits of a. The only guaranteed property is |error| <= j h / 2 plus the smooth O(h^2) part.
- **B6.** Pass `Probe.h = (dx, dt/T)`. `expected_order` is 1.0 (the lower one); `expected_orders` is (2.0, 1.0). Spatial order is only asymptotic from dx = 1/16 (2.59 at dx = 1/8 to 1/16 on one input); the dt order needs at least 16-32 steps per horizon (1.23, 1.15, 1.08 for 16 to 256 steps on that input; 0.96 to 0.99 for 64 to 1024 steps).
- **B7.** For example at a = 0.386, A = 1.25, w = 6e-4: levels 0-2 give about 1e-74, level 3 gives 2.4e-15, level 4 gives 7e-9, level 5 gives 2.4e-3, level 6 gives 1.5e-3, level 8 and up give the truth 1.88e-3. Successive coarse values agree to 1e-74 while the truth is 1.88e-3. When a stray node falls near the source, coarse levels instead show an apparent first-order convergence to zero (values halve with h) while the truth is 17 times larger than the last change.

# Benchmark machinery (PROTOCOL.md)

Modules: `oracle.py` (BenchmarkProblem behind quote/submit/poll; cost = work with seeded noise; a run at its cap returns cost = cap, `cost_censored`, no outputs; a function of (seed, probe_id)), `config.py` (units, eps, prior scales, cost prior), `design.py` (nested Sobol static designs, `fit_static`), `certificate.py`, `baselines.py` + `gp.py`, `metrics.py`, `runner.py`, `submit.py`, `report.py`.

Units: the oracle returns y / M (M = 5, 1, 0.3, 1, 0.5, 0.002 for B1, B2, B4, B5, B6, B7: one significant digit of the QoI scale), so the same prior scales (S_mu 1, S_c 0.5, S_delta 0.5, S_noise 0.02; B4: 0.1) serve both output transforms. The cost prior uses the problem's work exponent. B3 is not run: it has an output coordinate v and Campaign v1 requires n_controls == d.

## Tolerance table (proposal; one table)

| problem | eps (relative) | C* (work units) | certificate design | status |
|---|---|---|---|---|
| b1_poisson | 0.01 (placeholder) | not yet | - | scan in progress: cheapest design (n0 = 24, L = 2, cost 489) has max sigma_epi / abs(m_y) = 0.21 |
| b2_upwind | 0.01 (placeholder) | not yet | - | cheapest design (n0 = 24, L = 2, cost 75): 0.198, coverage 0.947 < 0.95 |
| b4_sde, b5_kink, b6_heat | 0.05, 0.01, 0.01 (placeholders) | not yet | - | not fitted |
| b7_thin_layer | 0.01 | none by construction | budget = 0.5 x cost of the n0 = 8 n_controls design with finest level 8 | |

`certificate.py` stores max sigma_epi / abs(m_y) per fitted design, so one fit answers for every eps (`derive`). One default-settings fit costs 13 to over 40 minutes on 4 cores, so the certificates are scanned (`--stride k`) and the eps chosen from the measured curve; they are provisional until gcbml stops changing.

## Methods and their settings (`baselines.py`)

All share `BenchmarkOracle`, the budget kappa C* and the tolerance. Fixed-design baselines plan with the noise-free work and spend on the finest affordable level: (a) 8 n_controls sites; (b) 4 / 8 n_controls sites at the two finest levels; (c) 8 n_controls sites at the three finest levels; (e) the dearest family design that fits. gcbml: `Campaign` + `run_campaign`, `CampaignSettings(seed=seed)`. (d): `h_kernels=("twy2",)`, prior on log p with median log(expected order) and sd 0.02 (p fixed through its prior; MAP is not available in the public API). (f): `h_kernels=("lb",)` (gamma learned, target h = 0: not Stroh as specified). (b) needs numpy < 2 and is left out of the manifests here.

## Running

`uv run python -m benchmarks.certificate --problem b1_poisson --stride 5`; `uv run python -m benchmarks.submit --mode smoke` (B1, kappa 1.5, seed 1, fast settings, results in `results/fast`); `--mode full` (not to be submitted until gcbml settles); `uv run python -m benchmarks.report [--fast]`.
