# Benchmark problems for gcbml

Seven cheap NumPy/SciPy solvers. The grid error of each is produced by a real discretisation (never a formula added to the answer). Each has a converged value at h = 0 that is known in closed form; the derivation is in the module docstring and a test checks it. Every problem implements `benchmarks.base.BenchmarkProblem`: `problem()`, `run(probe, seed)`, `truth(x_unit)`, `expected_order`, `notes`.

Run the tests on the allocation: `srun --jobid=<job> --overlap -c 4 uv run pytest tests/benchmarks -q` (about 9 s).

Conventions: `Probe.h` holds physical values (h = 1/N, so N = 1/h must be an integer); levels are h0 / 2^level. `RunResult.cost` is the CPU time of the solve in seconds (`time.process_time`), not core-hours. `truth` takes unit inputs, with the v coordinate included for B3.

## The problems

| # | module | inputs (range) | QoI | truth | order measured | what it stresses |
|---|---|---|---|---|---|---|
| B1 | `b1_poisson` | amplitude [0.5, 2], rate [1, 3], anisotropy [0.1, 1] | integral of u over the square; `-(u_xx + kappa u_yy) = f`, u = A exp(w x + w y / 2), 5-point stencil, sparse LU | A (e^w - 1)/w * (e^{w/2} - 1)/(w/2) | 2.00 (to 3 digits) | the clean case; learned order close to 2 |
| B2 | `b2_upwind` | Peclet [2, 100], source [0.5, 2], outlet value [0.5, 1.5] | u(15/16) of `u' = u''/Pe + s`, u(0) = 0, u(1) = g, first-order upwind | `s (x - E) + g E`, E = expm1(x Pe)/expm1(Pe) | 1.0 at low Pe; at Pe = 100: 1.26, 1.42, 1.42, 1.32, 1.20, 1.11, 1.06 for levels 0-7 | order that depends on the input (pre-asymptotic at high Pe) |
| B3 | `b3_mixing` | diffusivity [0.5, 2], fill fraction [0.2, 0.5], **v = chi** [0.5, 0.95] (S1: n_controls 2 of 3) | time to reach chi = 0.5, 0.75, 0.9, 0.95 for 1-D diffusion of two layers; finite volumes + implicit Euler, dt = 0.2 h | root of chi(t) = v, chi from the cosine series (4000 modes) | 1.0 (0.98 to 1.10) | nested outputs, censoring (T_final = 0.2; unreached thresholds return y = T_final, `censored` True), error in space and in threshold detection |
| B4 | `b4_sde` | theta [0.5, 3], sigma [0.2, 1.5], x0 [0.5, 2] | E[X_T^2], T = 1, Euler-Maruyama, 2000 paths | x0^2 e^{-2 theta T} + sigma^2 (1 - e^{-2 theta T})/(2 theta) | weak order 1.01 to 1.11 (levels 2-6, exact scheme expectation) | aleatoric noise that depends on the inputs (sd ratio of about 6 between two test points); replicates are seeds |
| B5 | `b5_kink` | jump location [0.2, 0.8], frequency [1, 4], jump size [0.5, 2] | midpoint rule for sin(w x) + j H(x - a) | (1 - cos w)/w + j (1 - a) | none: error = j h xi(a/h) + O(h^2), xi in [-1/2, 1/2) | sign changes and non-monotone |error| as h halves (e.g. -8.6e-2, +6.2e-2, -1.9e-2, -1.9e-2, 2e-4 for one input) |
| B6 | `b6_heat` | horizon T [0.02, 0.1], mode-3 weight [0, 1], mode-5 weight [0, 1] | u(1/2, T) of `u_t = u_xx`, central differences + backward Euler, dt = T h_t | e^{-pi^2 T} - a e^{-9 pi^2 T} + b e^{-25 pi^2 T} | 2.0 in dx (1.95, 1.99, 2.00); 1.0 in dt (0.96, 0.98, 0.99) | k = 2 independent resolution components |
| B7 | `b7_thin_layer` | position [0.2, 0.8], amplitude [0.5, 2], width [3e-4, 9e-4] | outlet value of `u' = S(x)`, narrow Gaussian source, upwind | A w sqrt(pi/2) [erf + erf] | none; asymptotic range starts at level 7 to 9 | the trap: levels 0-4 return about 0 (or 1e-74 apart), the truth is 2e-3. Must not claim success |

Level l has h = h0 / 2^l with h0 = 1/4 (B1, B4, B5, B6 both components), 1/8 (B3, B7), 1/16 (B2).

## Cost per level (CPU seconds, minimum of 3 runs, centre of the design region)

Measured with `benchmarks/measure_costs.py` on 4 cores of node1808 (the solvers are single-threaded, so 3 cores idle). `gamma` is the fitted slope of log2(cost) over the last 5 levels (cost ~ 2^{gamma l}).

| problem | l0 | l1 | l2 | l3 | l4 | l5 | l6 | l7 | l8 | l9 | gamma |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b1_poisson | 1.2e-03 | 1.4e-03 | 1.9e-03 | 3.7e-03 | 1.4e-02 | 6.7e-02 | 3.5e-01 | 2.3e+00 | | | 2.32 |
| b2_upwind | 6.2e-05 | 5.1e-05 | 4.9e-05 | 5.1e-05 | 5.7e-05 | 6.0e-05 | 7.4e-05 | 9.8e-05 | 1.6e-04 | 2.6e-04 | 0.53 |
| b3_mixing | 9.1e-04 | 1.0e-03 | 1.2e-03 | 1.7e-03 | 2.8e-03 | 5.8e-03 | 1.4e-02 | 3.9e-02 | 1.2e-01 | | 1.36 |
| b4_sde | 1.6e-04 | 2.9e-04 | 5.6e-04 | 1.1e-03 | 2.2e-03 | 4.4e-03 | 8.7e-03 | 1.7e-02 | 3.5e-02 | 7.0e-02 | 1.00 |
| b5_kink | 1.9e-05 | 1.6e-05 | 1.6e-05 | 1.7e-05 | 1.6e-05 | 1.7e-05 | 1.9e-05 | 2.5e-05 | 3.1e-05 | 4.7e-05 | 0.36 |
| b6_heat | 4.6e-04 | 4.8e-04 | 5.5e-04 | 6.5e-04 | 9.0e-04 | 1.5e-03 | 3.3e-03 | 8.9e-03 | 2.8e-02 | | 1.25 |
| b7_thin_layer | 6.6e-05 | 5.9e-05 | 5.7e-05 | 6.2e-05 | 5.7e-05 | 8.2e-05 | 1.1e-04 | 1.6e-04 | 2.7e-04 | 4.8e-04 | 0.64 |

Caveats on cost:
- The costs are small. The target of "about 1 ms at the coarsest level, up to 30 s at level 6-7" is met only by B1 (1 ms to 2.3 s at level 7). B3 and B6 start near 1 ms but reach only 0.1 s. B2, B5, B7 are 1D and cost microseconds; their cost is dominated by call overhead, so gamma over the first levels is below its asymptotic value (1). The tests use deeper levels (10-17) for B5 and B7, where gamma is 0.94 and 1.01.
- B3 and B6 have a Python loop over time steps, so gamma is about 1 (loop overhead) at small N and tends to 2 (steps x cells) at large N.
- B1 is the only problem with a cost that grows faster than 2^2 (sparse direct solve, about 2^2.3 here, 2^3 asymptotically).

## Notes per problem

- **B1.** Both the stencil and the trapezoid quadrature are second order. The error is proportional to the amplitude, so amplitude only scales it.
- **B2.** For the discrete solution the homogeneous solution is r^i with r = 1 + h/eps (exact exponential: e^{h/eps}), so the relative error is about h/(2 eps) and the first-order range starts when the cell Peclet number h Pe is small. At Pe = 100 and level 0 it is 6.25.
- **B3.** The reference is exact (series), not a Richardson value. No error bound is needed: the series is truncated at 4000 modes, and its tail at the times of interest is below 1e-300. chi uses the exact Var0 = L0 (1 - L0); using the discrete initial variance would add an O(h) error from the cell containing the interface. Truth is returned also for censored outputs.
- **B4.** `discrete_expectation` and `run_sd` are exact for the scheme (the scheme is linear and Gaussian), so the order and the spread are tested without sampling noise. The run-to-run sd of y is sqrt((2 v_h^2 + 4 m_h^2 v_h) / 2000).
- **B5.** For the test input a = 0.422, a/h has binary fraction 0.008 * 2^l, so the error doubles relative to h and is roughly constant for several levels: its size depends on the binary digits of a. The only guaranteed property is |error| <= j h / 2 plus the smooth O(h^2) part.
- **B6.** Pass `Probe.h = (dx, dt/T)`. `expected_order` is 1.0 (the lower one); `expected_orders` is (2.0, 1.0). Spatial order is only asymptotic from dx = 1/16 (2.59 at dx = 1/8 to 1/16 on one input); the dt order needs at least 16-32 steps per horizon (1.23, 1.15, 1.08 for 16 to 256 steps on that input; 0.96 to 0.99 for 64 to 1024 steps).
- **B7.** For example at a = 0.386, A = 1.25, w = 6e-4: levels 0-2 give about 1e-74, level 3 gives 2.4e-15, level 4 gives 7e-9, level 5 gives 2.4e-3, level 6 gives 1.5e-3, level 8 and up give the truth 1.88e-3. Successive coarse values agree to 1e-74 while the truth is 1.88e-3. When a stray node falls near the source, coarse levels instead show an apparent first-order convergence to zero (values halve with h) while the truth is 17 times larger than the last change.
