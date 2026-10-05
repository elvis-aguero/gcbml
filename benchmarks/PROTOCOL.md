# Benchmark protocol (the main evidence that gcbml works)

## Question
On problems with a known answer at h = 0, does gcbml reach the tolerance within a tight but sufficient
budget more often, or more cheaply, than the obvious alternatives, with calibrated uncertainty, and does
it refuse to claim success when success is impossible (B7)?

## 1. Tolerance and solvability certificate (benchmarks/certificate.py)
For each feasible problem B1-B6:
- Tolerance eps: relative, chosen per problem so that a certificate exists at a moderate cost (state it).
- Oracle static design: nested Sobol sites, n_l = max(3, n0 / 2^l) at levels 0..L, for n0 in
  {8, 16, 32} x n_controls and L in {2, ..., 7}. Fit gcbml (inference.fit, both h kernels, stacked).
- Certificate: the cheapest design with max over Sigma_N of sigma_epi / eps <= 1 AND |m_y - truth| <=
  2 sigma_epi at >= 95% of Sigma_N. Its cost is C*. No certificate -> the problem is re-tuned (eps).
- Budgets: C = kappa C*, kappa in {1.5, 2, 4}.
B7: no certificate exists below its asymptotic level by construction; budget = 0.5 x the cost of a
design whose finest level is the asymptotic level.

## 2. Methods (benchmarks/baselines.py; all get the same budget, oracle, tolerance)
- gcbml: Campaign + run_campaign with default settings.
- (a) single-fidelity GP at the finest level affordable with a space-filling design (plug-in ML-II).
- (b) Yi et al. KRR-LR-GPR with the two finest affordable levels (package mfbml, git dependency).
- (c) Eca-Hoekstra / Richardson at each site of a fixed ladder (least-squares power-law fit over >= 3 levels,
  F_s = 1.25 or 3 as in the GCI rules), then a GP over the extrapolated values for Sigma_N.
- (d) Boutelet-Sung style: p fixed at the problem's expected order, plug-in (MAP) hyperparameters, the same
  acquisition (variance reduction per cost), target h = 0.
- (e) gcbml without acquisition: the oracle's static design family, chosen by budget only.
- (f) Stroh style: Brownian kernel in h (LB with gamma = 0.5), full Bayes, MR-SUR, target = the finest
  level run (what their applications report), scored against the h = 0 truth.

## 3. Runs and metrics (benchmarks/runner.py, benchmarks/report.py)
20 seeds per (problem, kappa, method); each run is one SLURM task (array), 4 cores.
Per run (JSON): success (P1 met AND truth within m_y +- 2 sigma_epi at >= 95% of Sigma_N), claimed
success (P1 met, regardless of truth), cost used / C*, max |error| / eps, coverage of the 95% intervals,
z-scores of the truth, gate statuses, allocation per level, wall time.

## 4. Pass criteria for v1
- gcbml success >= 80% at kappa = 2 on B1-B4 and B6;
- 95% coverage within binomial limits (pooled over runs);
- false success (claimed but wrong, and any claimed success on B7) <= 5%;
- gcbml beats each baseline in success rate or cost on at least 4 of the 6 feasible problems.

## 5. Report
benchmarks/report.py: one figure per problem (cost / C* against max error / eps, per method), a coverage
table, a false-success table; a markdown summary.
