"""Baselines (a)-(f) on the toy problem: valid records, budget respected, known-answer pieces."""

import math

import numpy as np
import pytest
from toy_problem import family_cost, toy_setup

from benchmarks import baselines, config
from benchmarks.baselines import METHODS, gci_site, mfbml_usable, richardson, run_method
from benchmarks.metrics import score

SETUP = toy_setup(rel_tol=0.1)
TRUTH = SETUP.truth(SETUP.problem().sigma_n)
N_SIGMA = len(TRUTH)


def check_valid(oc, budget):
    assert oc.m_y is not None and oc.m_y.shape == (N_SIGMA,) and np.all(np.isfinite(oc.m_y))
    assert (
        oc.sigma_epi.shape == (N_SIGMA,) and np.all(oc.sigma_epi >= 0) and np.all(np.isfinite(oc.sigma_epi))
    )
    assert 0 < oc.spent <= budget * (1 + 1e-9)
    assert oc.allocation and sum(oc.allocation.values()) == pytest.approx(oc.spent)
    s = score(oc.m_y, oc.sigma_epi, TRUTH, SETUP.rel_tol)
    assert s["has_estimate"] and 0.0 <= s["coverage95"] <= 1.0 and len(s["z"]) == N_SIGMA


def test_richardson_recovers_an_exact_power_law():
    hbar = np.array([1.0, 0.5, 0.25])
    f = 1.4755 + 0.7865 * hbar**1.5
    f0, p, resid, good = richardson(hbar, f)
    assert good and f0 == pytest.approx(1.4755, abs=1e-6) and p == pytest.approx(1.5, abs=0.01)
    val, sd = gci_site(hbar, f)
    # GCI: U = 1.25 |f_fine - f0|; sd = U / 2
    assert val == pytest.approx(1.4755, abs=1e-6)
    assert sd == pytest.approx(0.5 * 1.25 * 0.7865 * 0.25**1.5, rel=1e-3)


def test_richardson_falls_back_to_the_finest_value_when_not_monotone():
    hbar = np.array([1.0, 0.5, 0.25])
    f = np.array([1.0, 1.3, 1.1])
    val, sd = gci_site(hbar, f)
    assert val == 1.1
    assert sd == pytest.approx(0.5 * 3.0 * 0.3)


def test_method_a_uses_the_finest_affordable_level():
    budget = 2000.0
    oc = run_method("a", SETUP, budget, seed=1, settings=config.fast_settings())
    check_valid(oc, budget)
    n = 8 * SETUP.n_controls
    top = max(t for t in range(12) if n * 2**t <= budget)
    assert oc.extra["level"] == top and set(oc.allocation) == {top}
    # the finest level has error 0.5 h^2 = 0.5 / 4^(2+top) relative to ~1-3: a GP at that level is that close
    assert np.max(np.abs(oc.m_y - TRUTH)) < 0.05


def test_method_c_extrapolates_the_h_squared_error_away():
    budget = 3000.0
    oc = run_method("c", SETUP, budget, seed=1, settings=config.fast_settings())
    check_valid(oc, budget)
    assert len(oc.extra["levels"]) == 3
    assert np.max(np.abs(oc.m_y - TRUTH)) < 0.02


def test_method_e_picks_the_most_expensive_affordable_family_design():
    nc = SETUP.n_controls
    budget = 400.0
    costs = {(a, t): family_cost(a * nc, t) for a in config.N0_PER_CONTROL for t in config.LEVELS}
    best = max((c, k) for k, c in costs.items() if c <= budget)[1]
    # method e fits gcbml; here only the choice of design is under test, so the fit is replaced by a stub
    import benchmarks.baselines as bl

    class Rep:
        m_y, sigma_epi, spent, allocation, status, notes, gates = (
            TRUTH,
            np.full(N_SIGMA, 0.01),
            0.0,
            {},
            "running",
            [],
            [],
        )

    orig = bl.fit_static
    bl.fit_static = lambda setup, res, settings: Rep()
    try:
        oc = run_method("e", SETUP, budget, seed=1, settings=config.fast_settings())
    finally:
        bl.fit_static = orig
    assert tuple(oc.extra["design"]) == best
    assert oc.spent == pytest.approx(costs[best]) and oc.spent <= budget


def test_tight_budget_is_never_exceeded_and_nothing_affordable_is_reported():
    oc = run_method("a", SETUP, 5.0, seed=1, settings=config.fast_settings())
    assert oc.m_y is None and oc.notes
    s = score(oc.m_y, oc.sigma_epi, TRUTH, SETUP.rel_tol)
    assert not s["claimed"] and not s["success"]


def test_unknown_method_name_raises():
    with pytest.raises(KeyError):
        run_method("zz", SETUP, 100.0, 1, config.fast_settings())


def test_method_names_are_those_of_the_protocol():
    assert METHODS == ("gcbml", "a", "b", "c", "d", "e", "f")


@pytest.mark.skipif(not mfbml_usable()[0], reason=mfbml_usable()[1])
def test_method_b_two_finest_levels():
    budget = 3000.0
    oc = run_method("b", SETUP, budget, seed=1, settings=config.fast_settings())
    check_valid(oc, budget)
    assert set(oc.allocation) == {oc.extra["level"] - 1, oc.extra["level"]}
    assert math.isfinite(oc.extra["n_hi"])


@pytest.mark.slow
@pytest.mark.parametrize("name", ["gcbml", "d", "e", "f"])
def test_gcbml_configurations_return_valid_records(name):
    budget = 600.0
    oc = run_method(name, SETUP, budget, seed=1, settings=config.fast_settings())
    check_valid(oc, budget)
    assert oc.gates is not None and oc.status is not None


def test_method_d_fixes_the_order_through_the_prior(monkeypatch):
    seen = {}

    def fake(setup, budget, seed, settings, scales=None):
        seen.update(scales=scales, kernels=settings.h_kernels)
        return baselines.Outcome()

    monkeypatch.setattr(baselines, "_campaign", fake)
    run_method("d", SETUP, 100.0, 1, config.fast_settings())
    assert seen["kernels"] == ("twy2",)
    assert seen["scales"].log_p_mean == pytest.approx(math.log(2.0)) and seen["scales"].log_p_sd <= 0.05
    run_method("f", SETUP, 100.0, 1, config.fast_settings())
    assert seen["kernels"] == ("lb",)
