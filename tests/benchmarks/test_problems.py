"""Benchmark problems B1-B7: orders, truth, determinism, cost growth, interface.

Every test compares against an independent reference: a closed form (the truth), a self-convergence
ratio (no truth needed), or an exact discrete recursion (B4).
"""

import numpy as np
import pytest

from benchmarks import ALL_PROBLEMS
from benchmarks.base import BenchmarkProblem, observed_orders, self_convergence_orders, values_at_levels
from gcbml.data import Probe, RunResult
from gcbml.problem import Problem

NAMES = sorted(ALL_PROBLEMS)


def make(name: str) -> BenchmarkProblem:
    return ALL_PROBLEMS[name]()


def random_unit_inputs(bp: BenchmarkProblem, n: int, seed: int) -> np.ndarray:
    d = bp.problem().inputs.d
    return np.random.default_rng(seed).uniform(0.1, 0.9, size=(n, d))


def probe_at(bp: BenchmarkProblem, z: np.ndarray, level: int, pid: str = "p") -> Probe:
    p = bp.problem()
    x = p.inputs.from_unit(z)
    h = p.resolution.h_at((level,) * p.resolution.k)
    return Probe(pid, tuple(float(t) for t in x[: p.inputs.n_controls]), tuple(float(t) for t in h))


def valid_mask(bp: BenchmarkProblem, z: np.ndarray) -> np.ndarray:
    """Outputs that are not censored by the time window (all of them unless the problem says otherwise)."""
    if hasattr(bp, "valid_outputs"):
        return bp.valid_outputs(z)
    return np.ones(bp.n_outputs, bool)


# ---------------------------------------------------------------- interface


@pytest.mark.parametrize("name", NAMES)
def test_problem_returns_valid_gcbml_objects(name):
    bp = make(name)
    p = bp.problem()
    assert isinstance(p, Problem)
    assert 2 <= p.inputs.d <= 6
    assert p.resolution.k == bp.n_resolution
    assert all(c.refine_ratio == 2.0 for c in p.resolution.components)
    assert p.sigma_n.shape == (100 * p.inputs.d, p.inputs.d)
    assert isinstance(bp.name, str) and bp.name
    assert isinstance(bp.notes, str) and len(bp.notes) > 20


@pytest.mark.parametrize("name", NAMES)
def test_run_returns_runresult_with_right_shapes(name):
    bp = make(name)
    p = bp.problem()
    z = random_unit_inputs(bp, 1, 0)[0]
    probe = probe_at(bp, z, 1)
    r = bp.run(probe, seed=0)
    assert isinstance(r, RunResult)
    m = len(r.y)
    assert m == bp.n_outputs
    assert r.x.shape == (m, p.inputs.d)
    assert r.censored.shape == (m,) and r.censored.dtype == bool
    assert np.all(r.x[:, : p.inputs.n_controls] == np.asarray(probe.u))
    assert np.all(np.isfinite(r.y))
    assert r.cost > 0 and r.cost_censored is False
    assert bp.truth(p.inputs.to_unit(r.x)).shape == (m,)


def test_s1_problem_has_controls_fewer_than_inputs():
    p = make("b3_mixing").problem()
    assert p.inputs.n_controls < p.inputs.d
    assert make("b3_mixing").n_outputs == 4


@pytest.mark.parametrize("name", NAMES)
def test_determinism_given_probe_and_seed(name):
    bp = make(name)
    probe = probe_at(bp, random_unit_inputs(bp, 1, 3)[0], 2)
    a, b = bp.run(probe, seed=7), bp.run(probe, seed=7)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.censored, b.censored)


# ---------------------------------------------------------------- orders and convergence

ORDER_LEVELS = {"b1_poisson": (3, 4, 5, 6), "b2_upwind": (4, 5, 6, 7), "b3_mixing": (3, 4, 5, 6)}


@pytest.mark.parametrize("name", ["b1_poisson", "b2_upwind", "b3_mixing"])
def test_observed_order_matches_expected_within_15_percent(name):
    bp = make(name)
    levels = ORDER_LEVELS[name]
    for z in random_unit_inputs(bp, 3, 11):
        if name == "b2_upwind":
            z[0] *= 0.15  # low-Peclet end of the range, where the scheme is asymptotic
        vals = values_at_levels(bp, z, levels)  # (n_levels, m)
        err = np.abs(vals - bp.truth_for_controls(z))[:, valid_mask(bp, z)]
        p_obs = observed_orders(err)
        assert np.all(np.abs(p_obs / bp.expected_order - 1) < 0.15), (z, p_obs)


@pytest.mark.parametrize("name", ["b1_poisson", "b2_upwind", "b3_mixing", "b6_heat"])
def test_finest_level_closer_to_truth_than_coarsest(name):
    bp = make(name)
    k = bp.problem().resolution.k
    for z in random_unit_inputs(bp, 3, 5):
        coarse, fine = values_at_levels(bp, z, [(0,) * k, (5,) * k])
        t = bp.truth_for_controls(z)
        ok = valid_mask(bp, z)
        assert np.all(np.abs(fine - t)[ok] < np.abs(coarse - t)[ok])


def test_b2_high_peclet_is_preasymptotic_on_coarse_grids():
    bp = make("b2_upwind")
    z = np.array([1.0, 0.5, 0.5])  # Pe at the top of the range
    err = np.abs(values_at_levels(bp, z, range(0, 8)) - bp.truth_for_controls(z))[:, 0]
    p_obs = observed_orders(err[:, None])[:, 0]
    assert abs(p_obs[0] / 1.0 - 1) > 0.15  # coarse pair: not first order
    assert abs(p_obs[-1] / 1.0 - 1) < 0.15  # finest pair: asymptotic


def test_b4_weak_order_one_from_exact_discrete_expectation():
    bp = make("b4_sde")
    for z in random_unit_inputs(bp, 3, 2):
        x = bp.problem().inputs.from_unit(z)
        h = bp.problem().resolution.h_at((0,))[0] / 2.0 ** np.arange(2, 7)
        err = np.abs([bp.discrete_expectation(x, hh) - bp.truth_for_controls(z)[0] for hh in h])
        p_obs = observed_orders(err[:, None])
        assert np.all(np.abs(p_obs / 1.0 - 1) < 0.15), (z, p_obs)


def test_b4_mean_over_seeds_matches_exact_discrete_expectation():
    bp = make("b4_sde")
    z = np.array([0.4, 0.6, 0.5])
    probe = probe_at(bp, z, 1)
    ys = np.array([bp.run(probe, seed=s).y[0] for s in range(300)])
    mean_exact = bp.discrete_expectation(bp.problem().inputs.from_unit(z), probe.h[0])
    se = bp.run_sd(bp.problem().inputs.from_unit(z), probe.h[0]) / np.sqrt(len(ys))
    assert abs(ys.mean() - mean_exact) < 4 * se


def test_b4_seeds_differ_and_spread_is_heteroscedastic_as_documented():
    bp = make("b4_sde")
    sds = []
    for z in ([0.4, 0.2, 0.5], [0.4, 0.9, 0.9]):
        probe = probe_at(bp, np.array(z), 1)
        ys = np.array([bp.run(probe, seed=s).y[0] for s in range(300)])
        assert len(np.unique(ys)) == len(ys)
        pred = bp.run_sd(bp.problem().inputs.from_unit(np.array(z)), probe.h[0])
        assert abs(ys.std(ddof=1) / pred - 1) < 0.2
        sds.append(ys.std(ddof=1))
    assert sds[1] / sds[0] > 3  # the spread depends strongly on the inputs


def test_b5_error_changes_sign_and_matches_closed_form():
    bp = make("b5_kink")
    p = bp.problem()
    for z in random_unit_inputs(bp, 3, 4):
        x = p.inputs.from_unit(z)
        a, j = x[0], x[2]
        hs = p.resolution.h_at((0,))[0] / 2.0 ** np.arange(0, 12)
        vals = values_at_levels(bp, z, range(12))[:, 0]
        err = vals - bp.truth_for_controls(z)[0]
        xi = a / hs - np.floor(a / hs + 0.5)
        smooth = err - j * hs * xi  # what is left is the smooth-part error, <= (h^2/24) |f'(1) - f'(0)|
        assert np.all(np.abs(smooth) < 0.1 * x[1] * hs**2), smooth
        assert (err > 0).any() and (err < 0).any()  # sign change
        assert np.any(np.diff(np.abs(err)) > 0)  # and |error| is not monotone


def test_b6_orders_are_two_in_dx_and_one_in_dt():
    bp = make("b6_heat")
    for z in random_unit_inputs(bp, 2, 8):
        space = [values_at_levels(bp, z, [(lx, 9)])[0, 0] for lx in range(2, 7)]
        time = [values_at_levels(bp, z, [(8, lt)])[0, 0] for lt in range(4, 9)]
        for series, expected in ((space, 2.0), (time, 1.0)):
            p_obs = self_convergence_orders(np.asarray(series))
            assert np.all(np.abs(p_obs / expected - 1) < 0.15), (z, expected, p_obs)


def test_b6_components_refine_independently():
    bp = make("b6_heat")
    z = random_unit_inputs(bp, 1, 1)[0]
    a, b = values_at_levels(bp, z, [(2, 5), (5, 2)])
    assert a[0] != b[0]
    assert bp.problem().resolution.k == 2


def test_b7_coarse_levels_look_converged_but_are_wrong():
    bp = make("b7_thin_layer")
    for z in random_unit_inputs(bp, 3, 6):
        t = bp.truth_for_controls(z)[0]
        v = values_at_levels(bp, z, range(0, 4))[:, 0]
        apparent = max(abs(v[3] - v[2]), abs(v[2] - v[1]))
        actual = abs(v[3] - t)
        assert actual > 10 * apparent, (v, t)
        fine = values_at_levels(bp, z, [9])[0, 0]  # the asymptotic range is reached only here
        assert abs(fine - t) < 1e-3 * abs(t)


# ---------------------------------------------------------------- truth verified independently


def test_truth_b1_integral_by_fine_quadrature():
    bp = make("b1_poisson")
    z = np.array([0.3, 0.7, 0.4])
    amp, w, _ = bp.problem().inputs.from_unit(z)
    n = 4000
    s = (np.arange(n) + 0.5) / n
    q = amp * np.mean(np.exp(w * s)) * np.mean(np.exp(0.5 * w * s))
    assert bp.truth_for_controls(z)[0] == pytest.approx(q, rel=1e-6)


def test_truth_b3_matches_direct_fourier_variance():
    bp = make("b3_mixing")
    z = np.array([0.5, 0.6, 0.5])  # D, L0, v
    t = bp.truth(z[None, :])[0]
    x = bp.problem().inputs.from_unit(z)
    D, L0 = x[0], x[1]
    # direct: sample the exact cosine series on a fine grid and compute the variance
    xs = (np.arange(20000) + 0.5) / 20000
    n = np.arange(1, 3000)
    a = 2 * np.sin(n * np.pi * L0) / (n * np.pi)
    c = L0 + (
        a[:, None] * np.cos(np.pi * n[:, None] * xs[None, :]) * np.exp(-D * (n * np.pi) ** 2 * t)[:, None]
    ).sum(0)
    chi = 1 - c.var() / (L0 * (1 - L0))
    assert chi == pytest.approx(x[2], abs=2e-3)


def test_truth_b2_satisfies_the_ode_and_boundary_conditions():
    from benchmarks.problems.b2_upwind import exact_u

    pe, s, g = 7.0, 1.3, 0.9
    x, d = 0.6, 1e-4
    u = lambda t: exact_u(t, pe, s, g)  # noqa: E731
    u1 = (u(x + d) - u(x - d)) / (2 * d)
    u2 = (u(x + d) - 2 * u(x) + u(x - d)) / d**2
    assert u1 - u2 / pe == pytest.approx(s, abs=1e-5)
    assert u(0.0) == pytest.approx(0.0, abs=1e-14) and u(1.0) == pytest.approx(g, rel=1e-12)


def test_truth_b4_is_the_limit_of_the_exact_euler_maruyama_expectation():
    bp = make("b4_sde")
    z = np.array([0.5, 0.5, 0.5])
    x = bp.problem().inputs.from_unit(z)
    assert bp.discrete_expectation(x, 2.0**-16) == pytest.approx(bp.truth_for_controls(z)[0], rel=1e-4)


def test_truth_b6_matches_a_very_fine_run():
    bp = make("b6_heat")
    z = np.array([0.5, 0.5, 0.5])
    y = values_at_levels(bp, z, [(9, 9)])[0, 0]
    assert y == pytest.approx(bp.truth_for_controls(z)[0], rel=2e-3)


# ---------------------------------------------------------------- cost growth

WORK_GAMMA = {
    "b1_poisson": 3.0,
    "b2_upwind": 1.0,
    "b3_mixing": 2.0,
    "b4_sde": 1.0,
    "b5_kink": 1.0,
    "b6_heat": 2.0,
    "b7_thin_layer": 1.0,
}


@pytest.mark.parametrize("name", NAMES)
def test_cost_is_the_documented_work_formula_when_noise_is_off(name):
    bp = make(name)
    bp.sigma_c = 0.0
    assert bp.work_gamma == WORK_GAMMA[name]
    z = random_unit_inputs(bp, 1, 9)[0]
    levels = np.arange(0, 6)
    cost = np.array([bp.run(probe_at(bp, z, int(lev)), seed=0).cost for lev in levels])
    assert cost[0] == pytest.approx(1.0)  # the coarsest level costs 1 work unit
    assert np.polyfit(levels, np.log2(cost), 1)[0] == pytest.approx(WORK_GAMMA[name], abs=1e-9)
    assert np.allclose(cost, 2.0 ** (WORK_GAMMA[name] * levels))


@pytest.mark.parametrize("name", NAMES)
def test_cost_noise_is_lognormal_and_deterministic_given_the_seed(name):
    bp = make(name)
    assert bp.sigma_c == 0.1
    probe = probe_at(bp, random_unit_inputs(bp, 1, 2)[0], 1)
    w = 2.0**bp.work_gamma
    c = np.array([bp.run(probe, seed=s).cost for s in range(400)])
    assert bp.run(probe, seed=5).cost == bp.run(probe, seed=5).cost
    logr = np.log(c / w)
    assert abs(logr.mean()) < 4 * 0.1 / np.sqrt(len(c))
    assert logr.std(ddof=1) == pytest.approx(0.1, rel=0.2)


@pytest.mark.parametrize("name", NAMES)
def test_measure_cpu_reports_positive_seconds(name):
    bp = make(name)
    probe = probe_at(bp, random_unit_inputs(bp, 1, 2)[0], 1)
    assert bp.measure_cpu(probe, seed=0) > 0


def test_b7_asymptotic_level_and_its_work_cost_are_documented_correctly():
    bp = make("b7_thin_layer")
    p = bp.problem()
    lev = bp.asymptotic_level
    bp.sigma_c = 0.0
    assert bp.run(probe_at(bp, np.full(3, 0.5), lev), 0).cost == 2.0**lev
    # from `lev` on every input is within 1% of the truth; one level earlier some input is not (worst case)
    pos = np.random.default_rng(0).uniform(0.05, 0.95, 40)
    zs = [np.array([a, 0.5, w]) for a in pos for w in (0.0, 0.5, 1.0)]
    rel = np.array(
        [values_at_levels(bp, z, [lev - 1, lev])[:, 0] / bp.truth_for_controls(z)[0] - 1 for z in zs]
    )
    assert np.all(np.abs(rel[:, 1]) < 0.01)
    assert np.any(np.abs(rel[:, 0]) > 0.01)
    assert p.resolution.components[0].level_value(lev) < 9e-4  # resolution comparable to the narrowest layer
