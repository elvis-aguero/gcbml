"""Synthetic truths: the stated formulas, the problem they describe, the oracle (results, costs, caps)."""

import numpy as np

import gcbml  # noqa: F401
from gcbml.data import Probe
from gcbml.synthetic import A12Truth, MaternDraw, OscillatingTruth, RichardsonToy, matern52


def run_probe(truth, u, h, probe_id="p0", cap=1e12, seed=0):
    o = truth.oracle(seed=seed)
    o.submit([Probe(probe_id, tuple(u), (h,))], [cap])
    (res,) = o.poll()
    return res


def test_a12_problem_is_valid_and_levels_are_the_documented_ones():
    t = A12Truth(3, d=2)
    p = t.problem()
    assert p.inputs.d == 2 and p.inputs.n_controls == 2
    assert p.resolution.k == 1
    hs = [p.resolution.components[0].level_value(lev) for lev in range(5)]
    np.testing.assert_allclose(hs, [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16])
    assert p.resolution.hbar(np.array([0.25]))[0] == 0.25
    assert p.sigma_n.shape == (200, 2)
    assert p.transforms == ("identity",)


def test_a12_oracle_follows_the_formula_at_given_x_and_h():
    t = A12Truth(5, d=2)
    x = np.array([[0.3, 0.7]])
    for h in (1.0, 0.5, 0.25, 0.125):
        expect = t.f0(x)[0] + t.a(x)[0] * h**t.p
        assert abs(t.value(x, h) - expect) < 1e-14
        # the oracle adds noise of sd 0.01 on top of the formula: 400 replicates, mean within 5 s.e.
        o = t.oracle(seed=1)
        probes = [Probe(f"r{i}", tuple(x[0]), (h,)) for i in range(400)]
        o.submit(probes, [1e12] * 400)
        y = np.array([r.y[0] for r in o.poll()])
        assert abs(y.mean() - expect) < 5 * 0.01 / np.sqrt(400)
        assert abs(y.std(ddof=1) - 0.01) < 0.002
    assert 0.7 <= t.p <= 2.5
    np.testing.assert_allclose(t.truth(x), t.f0(x))  # truth(x_unit) is f0


def test_a12_p_is_uniform_on_0p7_2p5_over_truths():
    p = np.array([A12Truth(s).p for s in range(300)])
    assert p.min() >= 0.7 and p.max() <= 2.5
    assert abs(p.mean() - 1.6) < 0.1


def test_a12_gp_draws_have_matern52_variance_and_correlation():
    pts = np.array([[0.2, 0.2], [0.2, 0.2 + 0.15], [0.2, 0.2 + 0.3]])
    vals = np.array(
        [MaternDraw(np.random.default_rng(s), 2, 1.0, 0.3)(pts) for s in range(1500)]
    )  # (draws, 3)
    assert abs(vals[:, 0].var() - 1.0) < 0.12
    for j, r in ((1, 0.15), (2, 0.3)):
        corr = np.mean(vals[:, 0] * vals[:, j]) / np.sqrt(np.mean(vals[:, 0] ** 2) * np.mean(vals[:, j] ** 2))
        assert abs(corr - matern52(r, 0.3)) < 0.08, (r, corr, matern52(r, 0.3))
    # the amplitude argument: sd 0.5 gives a quarter of the variance
    v5 = np.array([MaternDraw(np.random.default_rng(s), 2, 0.5, 0.3)(pts[:1])[0] for s in range(1500)])
    assert abs(v5.var() - 0.25) < 0.04


def test_a12_oracle_cost_is_2_to_3l_times_lognormal_0p1():
    t = A12Truth(1, d=2)
    o = t.oracle(seed=2)
    n = 300
    for lev, h in ((0, 1.0), (2, 0.25), (4, 0.0625)):
        probes = [Probe(f"c{lev}_{i}", (0.5, 0.5), (h,)) for i in range(n)]
        o.submit(probes, [1e30] * n)
        c = np.array([r.cost for r in o.poll()])
        lc = np.log(c) - 3 * lev * np.log(2.0)
        assert abs(lc.mean()) < 4 * 0.1 / np.sqrt(n)
        assert abs(lc.std(ddof=1) - 0.1) < 0.02


def test_oracle_result_depends_on_probe_id_not_on_submission_order():
    t = A12Truth(2, d=1)
    a = Probe("a", (0.4,), (0.5,))
    b = Probe("b", (0.4,), (0.5,))
    o1, o2 = t.oracle(seed=3), t.oracle(seed=3)
    o1.submit([a, b], [1e9, 1e9])
    o2.submit([b, a], [1e9, 1e9])
    r1 = {r.probe.probe_id: (r.y[0], r.cost) for r in o1.poll()}
    r2 = {r.probe.probe_id: (r.y[0], r.cost) for r in o2.poll()}
    assert r1 == r2
    assert r1["a"][0] != r1["b"][0]  # replicates differ


def test_oracle_stops_a_run_at_its_cap_and_reports_a_censored_cost():
    t = A12Truth(2, d=1)
    res = run_probe(t, (0.4,), 0.125, cap=1.0)  # level 3 costs ~512
    assert res.cost == 1.0 and res.cost_censored
    assert res.y.size == 0 and res.x.shape == (0, 1)
    ok = run_probe(t, (0.4,), 1.0, cap=1e6)
    assert not ok.cost_censored and ok.y.size == 1 and ok.x.shape == (1, 1)


def test_richardson_toy_reproduces_the_guide_sequence_to_4_decimals():
    t = RichardsonToy()
    got = [t.value(np.array([[0.5]]), h) for h in (1.0, 0.5, 0.25, 0.125)]
    np.testing.assert_allclose(np.round(got, 4), [2.2620, 1.7536, 1.5738, 1.5103])
    assert abs(float(t.truth(np.array([[0.5]]))[0]) - 1.4755) < 1e-12
    for h, want in zip((1.0, 0.5, 0.25, 0.125), (2.2620, 1.7536, 1.5738, 1.5103)):
        res = run_probe(t, (0.5,), h)  # noise-free by default
        assert abs(res.y[0] - want) < 5e-5


def test_oscillating_truth_matches_its_formula():
    t = OscillatingTruth()
    for h in (1.0, 0.75, 0.5, 0.25, 0.125, 0.0):
        want = 1.0 + 0.4 * h**1.5 * np.cos(2 * np.pi * h)
        assert abs(t.value(np.array([[0.5]]), h) - want) < 1e-14
    assert abs(t.value(np.array([[0.5]]), 0.5) - (1 - 0.4 * 0.5**1.5)) < 1e-14
    assert t.truth(np.array([[0.1]]))[0] == 1.0
    assert abs(run_probe(t, (0.5,), 0.75).y[0] - 1.0) < 1e-12  # 0.4 * 0.75^1.5 cos(1.5 pi) = 0
