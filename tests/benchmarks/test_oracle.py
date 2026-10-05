"""BenchmarkOracle: determinism given (seed, probe_id), cap behaviour, and the work cost law."""

import numpy as np

from benchmarks import ALL_PROBLEMS
from benchmarks.oracle import BenchmarkOracle, run_seed
from gcbml.data import Probe


def probe(bp, pid="p1", level=2, z=0.4):
    p = bp.problem()
    u = p.inputs.from_unit(np.full(p.inputs.d, z))[: p.inputs.n_controls]
    return Probe(
        pid, tuple(float(t) for t in u), tuple(float(t) for t in p.resolution.h_at((level,) * p.resolution.k))
    )


def test_result_is_a_function_of_seed_and_probe_id_only():
    bp = ALL_PROBLEMS["b4_sde"]()
    a, b = probe(bp, "a"), probe(bp, "b")
    o1, o2 = BenchmarkOracle(bp, seed=3), BenchmarkOracle(bp, seed=3)
    o1.submit([a, b], [1e9, 1e9])
    o2.submit([b, a], [1e9, 1e9])  # other order
    r1 = {r.probe.probe_id: r for r in o1.poll()}
    r2 = {r.probe.probe_id: r for r in o2.poll()}
    for k in "ab":
        assert np.array_equal(r1[k].y, r2[k].y) and r1[k].cost == r2[k].cost
    # replicates (same u and h, different probe id) differ; another seed differs
    assert r1["a"].y[0] != r1["b"].y[0]
    o3 = BenchmarkOracle(bp, seed=4)
    o3.submit([a], [1e9])
    assert o3.poll()[0].y[0] != r1["a"].y[0]


def test_cost_is_the_work_times_seeded_noise():
    bp = ALL_PROBLEMS["b2_upwind"]()
    pr = probe(bp, "x", level=3)
    r = BenchmarkOracle(bp, seed=1).run(pr)
    assert r.cost == bp.cost(pr, run_seed(1, "x"))
    z = np.log(r.cost / bp.work(pr)) / bp.sigma_c
    assert abs(z) < 5
    assert not r.cost_censored and r.y.shape == (1,)
    bp.sigma_c = 0.0
    try:
        assert BenchmarkOracle(bp, seed=1).run(pr).cost == bp.work(pr) == 2.0**3
    finally:
        del bp.sigma_c


def test_run_stopped_at_its_cap_returns_cap_censored_and_no_outputs():
    bp = ALL_PROBLEMS["b2_upwind"]()
    pr = probe(bp, "x", level=4)
    full = BenchmarkOracle(bp, seed=1).run(pr)
    o = BenchmarkOracle(bp, seed=1)
    cap = 0.5 * full.cost
    o.submit([pr], [cap])
    r = o.poll()[0]
    assert r.cost == cap and r.cost_censored
    assert r.y.shape == (0,) and r.x.shape == (0, len(pr.u)) and r.censored.shape == (0,)
    o.submit([pr], [1.5 * full.cost])  # a cap above the cost: the full result, charged the cost
    r = o.poll()[0]
    assert r.cost == full.cost and not r.cost_censored and np.array_equal(r.y, full.y)
    assert o.poll() == []


def test_quote_is_none_unless_asked():
    bp = ALL_PROBLEMS["b2_upwind"]()
    pr = probe(bp)
    assert BenchmarkOracle(bp).quote([pr]) == [None]
    assert BenchmarkOracle(bp, quote_work=True).quote([pr]) == [bp.work(pr)]


def test_y_scale_is_a_choice_of_units():
    bp = ALL_PROBLEMS["b2_upwind"]()
    pr = probe(bp, "x", level=3)
    a = BenchmarkOracle(bp, seed=1).run(pr)
    b = BenchmarkOracle(bp, seed=1, y_scale=2.0).run(pr)
    assert np.allclose(b.y, a.y / 2.0) and b.cost == a.cost
