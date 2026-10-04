"""Step 4 of the algorithm (spec Section 3): forecast of the greedy policy, at bounded cost.

Simulate the greedy policy of Step 5 with 20 posterior draws (spread over the structures by their
weights), for at most 50 steps or until success, with VARIANCE-ONLY updates (no fantasies, no
reweighting): at each step every draw's factor is extended by the chosen run's rows (linalg.append),
and the run's cost is its expected cost. Success is judged on sigma_epi of the POOLED 20 draws (so the
spread between draws and between structures stays in; variance-only updates cannot shrink it, so the
forecast is conservative about success). Known limit (spec): when the between-structure spread exceeds
eps, the forecast says "P1 infeasible" from the first step and carries no information.

forecast(key, structures, data, candidates, Xs, eps, budget_remaining, n_draws=20, max_steps=50)
    -> ForecastResult(success: bool, steps: int, cost: float, p_exceed: float, trajectory: list of
       (chosen candidate index, H after the step, cumulative cost), infeasible: bool)
    p_exceed: fraction of the 20 draws whose own path (judged per draw) has not met P1 when the budget
    is used; the Step 4 rule reports the forecast to the user when p_exceed > 0.5 and switches the
    acquisition to mode "softmax" (P2) when the pooled forecast says P1 is infeasible.

How it is computed (W4-B):
  * The posterior of mu(Xs) jointly with the outputs of every candidate is built once per draw
    (acquisition._build_pool). Observing a run without its value is then a rank-m update of that joint
    covariance (exact for a Gaussian: the posterior covariance does not depend on the value), identical to
    extending the draw's factor with linalg.append and reading the variance, and it needs no new
    factorisation and no new array shape per step (tests/test_acquisition.py checks it against predict_mu
    on the enlarged data).
  * Draws: n_draws of the pooled draws, by systematic resampling at the fixed offset 1/2 if there are more
    (acquisition._select_draws); `key` is accepted for interface stability and not used (deterministic).
  * Greedy rule: the candidate with the best (H_now - H_after) / cost_mean among those with cost_cap <= what
    is left (budget_remaining minus the expected costs spent so far), as in select_batch with hinge H;
    each candidate is used at most once. The loop stops when P1 holds (H = 0), nothing is admissible, no
    candidate has a positive gain, or max_steps is reached.
  * infeasible: before the first step, the pooled sigma_epi is computed with every draw's own variance set
    to zero (the limit of any amount of variance-only information). If H > 0 there, no variance-only
    path can reach P1: the result is success False, steps 0, cost 0, trajectory [], infeasible True.
    [assumption: this is the documented known limit; the stub's ForecastResult has no field for it, so the
    field `infeasible` (default False) is ADDED at the end.]
  * p_exceed: at the end of the path (success, budget or max_steps), the share (by pooled weight) of draws
    that have some x in Sigma_N with their own half-width above eps(x).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from gcbml import acquisition as acq


class ForecastResult(NamedTuple):
    success: bool
    steps: int
    cost: float
    p_exceed: float
    trajectory: Any
    infeasible: bool = False


def _p_exceed(pool, eps) -> float:
    s = pool.s
    m = jnp.concatenate([a[:, :s] for a in pool.mean])
    v = jnp.concatenate([jnp.diagonal(c, axis1=1, axis2=2)[:, :s] for c in pool.cov])
    own = acq._draw_sigma(m, jnp.sqrt(jnp.maximum(v, 0.0)), pool.segs)
    miss = jnp.any(own > jnp.asarray(eps, dtype=float), axis=1)
    W = acq._pool_W(pool)
    return float(jnp.sum(W * miss) / jnp.sum(W))


def _H_now(pool, eps) -> float:
    return float(acq.H_value(acq._pool_sigma(pool), eps, "hinge"))


def forecast(
    key,
    structures,
    data,
    candidates,
    Xs,
    eps,
    budget_remaining: float,
    n_draws: int = 20,
    max_steps: int = 50,
) -> ForecastResult:
    cands = list(candidates)
    n = len(cands)
    pool = acq._build_pool(structures, data, Xs, cands, n_draws)
    # the limit of any variance-only information: every draw's own variance -> 0
    floor = acq._halfwidth(
        jnp.concatenate([a[:, : pool.s] for a in pool.mean]),
        jnp.zeros((sum(a.shape[0] for a in pool.mean), pool.s)),
        acq._pool_W(pool),
        pool.segs,
    )
    if float(acq.H_value(floor, eps, "hinge")) > 0.0:
        return ForecastResult(False, 0, 0.0, _p_exceed(pool, eps), [], True)
    caps = np.array([float(c.cost_cap) for c in cands])
    costs = np.array([float(c.cost_mean) for c in cands])
    ms = np.array([len(pool.rows[i]) for i in range(n)])
    available = np.ones(n, dtype=bool)
    H = _H_now(pool, eps)
    spent, traj = 0.0, []
    while H > 0.0 and len(traj) < max_steps:
        adm = available & (caps <= float(budget_remaining) - spent)
        if not adm.any():
            break
        H_after = np.full(n, np.inf)
        for m in np.unique(ms[adm]):
            ids = np.flatnonzero(adm & (ms == m))
            idx_mat = jnp.asarray(np.stack([pool.rows[i] for i in ids]))
            H_after[ids] = np.asarray(
                acq._variance_only_H(
                    pool.mean, pool.cov, pool.w, pool.sw, idx_mat, jnp.asarray(eps, dtype=float),
                    s=pool.s, mode="hinge", segs=pool.segs,
                )
            )  # fmt: skip
        ratio = np.where(adm & (H - H_after > 0.0), (H - H_after) / costs, -np.inf)
        if not np.isfinite(ratio.max()):
            break
        best = int(np.argmax(ratio))
        available[best] = False
        spent += costs[best]
        pool = acq._condition_variance_only(pool, pool.rows[best])
        H = _H_now(pool, eps)
        traj.append((best, H, spent))
    return ForecastResult(H <= 0.0, len(traj), spent, _p_exceed(pool, eps), traj, False)
