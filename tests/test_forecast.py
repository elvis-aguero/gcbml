"""Step 4: the variance-only forecast of the greedy policy."""

import jax
import numpy as np
from test_acquisition import XS, cand, deterministic_gain, make_params, world

import gcbml  # noqa: F401
from gcbml import acquisition as acq
from gcbml.forecast import ForecastResult, forecast


def candidates():
    xs = np.linspace(0.0, 1.0, 7)
    cheap = [cand(x, 1.0, cost=1.0, cap=1.5, level=0) for x in xs]
    fine = [cand(x, 0.5, cost=4.0, cap=6.0, level=1) for x in xs]
    return cheap + fine


def test_forecast_feasible_problem_reports_success_within_steps_and_budget():
    structs, data = world()
    sig0 = float(np.asarray(acq.sigma_epi_physical(structs, data, XS)).max())
    eps = 0.6 * sig0
    cands = candidates()
    res = forecast(jax.random.key(0), structs, data, cands, XS, eps, budget_remaining=200.0)
    assert isinstance(res, ForecastResult)
    assert res.success and not res.infeasible
    assert 1 <= res.steps <= 50 and len(res.trajectory) == res.steps
    assert res.cost <= 200.0
    idx, H, cum = zip(*res.trajectory, strict=True)
    assert H[-1] == 0.0 and all(h > 0 for h in H[:-1])  # P1 holds exactly at the last step
    assert all(a >= b for a, b in zip(H, H[1:], strict=False))  # variance-only updates never raise H
    assert all(a < b for a, b in zip(cum, cum[1:], strict=False))
    np.testing.assert_allclose(cum[-1], res.cost)
    assert res.p_exceed == 0.0


def test_forecast_first_step_is_the_best_ratio_of_the_deterministic_gains():
    structs, data = world()
    sig0 = float(np.asarray(acq.sigma_epi_physical(structs, data, XS)).max())
    eps = 0.6 * sig0
    cands = candidates()[::3]
    res = forecast(jax.random.key(0), structs, data, cands, XS, eps, 200.0)
    ratios = [deterministic_gain(structs, data, c, XS, eps) / c.cost_mean for c in cands]
    assert abs(ratios[res.trajectory[0][0]] - max(ratios)) < 1e-9 * max(ratios)  # (ties: x = 0 and 1)


def test_forecast_huge_between_draw_spread_is_infeasible_from_step_one():
    structs, data = world([make_params(24, c0=c) for c in (0.0, 6.0)])
    eps = 0.2
    res = forecast(jax.random.key(0), structs, data, candidates(), XS, eps, budget_remaining=1e6)
    assert not res.success and res.infeasible
    assert res.steps == 0 and res.cost == 0.0 and res.trajectory == []


def test_forecast_budget_too_small_is_not_success_and_never_overspends():
    structs, data = world()
    sig0 = float(np.asarray(acq.sigma_epi_physical(structs, data, XS)).max())
    res = forecast(jax.random.key(0), structs, data, candidates(), XS, 0.2 * sig0, budget_remaining=3.0)
    assert not res.success and not res.infeasible
    assert res.cost <= 3.0 and res.steps >= 1
    assert res.p_exceed == 1.0  # the single draw has not met P1 when the money is gone


def test_forecast_when_p1_already_holds_it_stops_at_step_zero():
    structs, data = world()
    res = forecast(jax.random.key(0), structs, data, candidates(), XS, 1e3, budget_remaining=10.0)
    assert res.success and res.steps == 0 and res.cost == 0.0 and res.trajectory == []
    assert res.p_exceed == 0.0
