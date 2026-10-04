"""Step 4 of the algorithm (spec Section 3): forecast of the greedy policy, at bounded cost.

Simulate the greedy policy of Step 5 with 20 posterior draws (spread over the structures by their
weights), for at most 50 steps or until success, with VARIANCE-ONLY updates (no fantasies, no
reweighting): at each step every draw's factor is extended by the chosen run's rows (linalg.append),
and the run's cost is its expected cost. Success is judged on sigma_epi of the POOLED 20 draws (so the
spread between draws and between structures stays in; variance-only updates cannot shrink it, so the
forecast is conservative about success). Known limit (spec): when the between-structure spread exceeds
eps, the forecast says "P1 infeasible" from the first step and carries no information.

TODO(W4-B): implement.

forecast(key, structures, data, candidates, Xs, eps, budget_remaining, n_draws=20, max_steps=50)
    -> ForecastResult(success: bool, steps: int, cost: float, p_exceed: float, trajectory: list of
       (chosen candidate index, H after the step, cumulative cost))
    p_exceed: fraction of the 20 draws whose own path (judged per draw) has not met P1 when the budget
    is used; the Step 4 rule reports the forecast to the user when p_exceed > 0.5 and switches the
    acquisition to mode "softmax" (P2) when the pooled forecast says P1 is infeasible.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class ForecastResult(NamedTuple):
    success: bool
    steps: int
    cost: float
    p_exceed: float
    trajectory: Any


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
    raise NotImplementedError("W4-B")
