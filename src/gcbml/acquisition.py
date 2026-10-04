"""Step 5 of the algorithm (spec Section 3): the next runs, by maximum uncertainty reduction per cost.

    a* = argmax_{a admissible} (H_n - E J_n(a)) / E c(a)          (MR-SUR, Stroh et al. 2007.13553 eq 11)
    H_n = sum_{x in Sigma_N} (sigma_epi^2(x) / eps^2(x) - 1)_+     (hinge, mode "hinge"; zero iff P1 holds)
    H_n = beta^{-1} log sum_x exp(beta sigma_epi^2 / eps^2)         (mode "softmax", beta = 20; P2 criterion)

sigma_epi(x) is the PHYSICAL half-width (q84 - q16)/2 of the pooled posterior of f(x) = Lambda^{-1}(mu(x))
(spec 2.8): the mixture over posterior draws (and structures, with their stacking weights) of the
Gaussian conditionals of mu(x) at h = 0, its quantiles found in Lambda units (bisection on the mixture
CDF) and mapped through Lambda^{-1} (monotone; for a decreasing Lambda, q16 and q84 swap).

A candidate run a = (u, h): its outputs are the rows Xa (m, d) (m > 1 under module S1: one row per v),
all at the same hbar (k,). Order of new rows: v1 supports a shared order only (P columns constant);
new rows take P[0] of each draw. [assumption; varying order needs pi(x) at new x: later]

Fantasies (spec Step 5): for one candidate and posterior draws s = 1..S (pooled, weights w_s):
  1. pick a draw s ~ w; draw the fantasy outputs ya ~ joint predictive of the m outputs under draw s
     (model.joint_new: mean and covariance including noise and the within-run correlation);
  2. for every draw s', condition its GP on (data + ya) exactly: linalg.append on its factor;
  3. reweight: w'_s' proportional to w_s' * p(ya | data, draw s') (the joint predictive density);
     if the effective sample size (sum w')^2 / sum w'^2 < 50 [assumption, spec], keep w (conservative);
  4. recompute sigma_epi on Sigma_N with the conditioned draws and w', then H.
  E J_n(a) = average of H over the fantasies; gain = max(H_n - E J_n, 0) (negative = MC noise, clipped).
  Fantasies: start at 16, double until the MC standard error of the gain of the best candidate is below
  10% of the gap to the second best, or 256 is reached.
Stacking weights over structures are held fixed during fantasies [assumption, spec].

Budget (spec Step 5): a candidate is admissible only if cap(a) <= C_rem = budget - spent - sum caps of
pending runs. Pending runs enter as variance-only conditioning (append their rows to every draw's factor
with their predictive mean as the value: the posterior variance does not depend on the value).

Batches: greedy. After choosing a, add it to the pending set (variance-only), recompute, choose the next,
until q runs are chosen or nothing admissible is left.

TODO(W4-B): implement, and add model.joint_new (below) to model.py.

model.joint_new(params, data, z, cfg, Xn, Hn, Pn, same_run=True) -> (mean (m,), cov (m, m))
    Joint predictive of m NEW outputs of one new run in Lambda units, given the data, beta integrated,
    including the noise (noise_var_new (m,) argument) and, if cfg.within_run, their within-run correlation.
    Plus the cross-covariance needed to append them to the factor (return a small NamedTuple).

Interfaces in this module:
Candidate: NamedTuple(Xa (m, d) unit, hbar (k,), levels (k,) int, cost_mean, cost_cap, noise_var (m,) per
    draw-free estimate or (S, m))
StructurePosterior: NamedTuple(params: ModelParams with leading axis S, z (S, n_pad), cfg: ModelConfig,
    transform: transforms.Transform, weight: float)
sigma_epi_physical(structures, data, Xs) -> (s,)
H_value(sigma_epi, eps, mode="hinge", beta=20.0) -> scalar
expected_gain(key, structures, data, cand, Xs, eps, n_fantasy, mode) -> (gain, mc_se)
select_batch(key, structures, data, candidates, Xs, eps, q, budget_remaining, pending=(), mode="hinge",
             max_draws=64) -> (chosen indices, table of (gain, cost_mean, ratio, admissible) per candidate)
    max_draws: subsample this many pooled draws (by weight, systematic resampling) for speed; document it.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class Candidate(NamedTuple):
    Xa: Any
    hbar: Any
    levels: Any
    cost_mean: float
    cost_cap: float
    noise_var: Any


class StructurePosterior(NamedTuple):
    params: Any
    z: Any
    cfg: Any
    transform: Any
    weight: float


def sigma_epi_physical(structures, data, Xs):
    raise NotImplementedError("W4-B")


def H_value(sigma_epi, eps, mode: str = "hinge", beta: float = 20.0):
    raise NotImplementedError("W4-B")


def expected_gain(key, structures, data, cand: Candidate, Xs, eps, n_fantasy: int, mode: str = "hinge"):
    raise NotImplementedError("W4-B")


def select_batch(
    key,
    structures,
    data,
    candidates,
    Xs,
    eps,
    q: int,
    budget_remaining: float,
    pending=(),
    mode: str = "hinge",
    max_draws: int = 64,
):
    raise NotImplementedError("W4-B")
