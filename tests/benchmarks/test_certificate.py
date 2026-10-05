"""Certificate: the cheapest design that passes, with a known C* (toy problem, hand-computed costs)."""

import json

import pytest
from toy_problem import family_cost, toy_setup

from benchmarks import certificate, config
from benchmarks.certificate import budget_for, certify, reference_cost
from benchmarks.design import family_designs


def test_family_has_18_designs_with_the_hand_computed_costs():
    setup = toy_setup()
    designs = family_designs(setup, seed=0)
    assert len(designs) == 3 * 6
    nc = setup.n_controls
    from benchmarks.design import expected_cost

    for (a, top), probes in designs:
        assert expected_cost(setup, probes) == pytest.approx(family_cost(a * nc, top))
    costs = [expected_cost(setup, p) for _, p in designs]
    assert costs == sorted(costs)


def test_certificate_is_the_cheapest_passing_design(tmp_path, monkeypatch):
    """With a stub fit that passes exactly when cost >= 300, C* is the smallest family cost >= 300."""
    setup = toy_setup()
    nc = setup.n_controls
    costs = sorted(family_cost(a * nc, top) for a in config.N0_PER_CONTROL for top in config.LEVELS)
    threshold = 300.0
    expected = min(c for c in costs if c >= threshold)
    assert expected > costs[0]  # the stub must reject some designs

    def stub(setup, results, settings):
        c = float(sum(r.cost for r in results))
        return {
            "cost": c,
            "n_runs": len(results),
            "passed": c >= threshold,
            "success": c >= threshold,
            "max_sigma_over_eps": 0.5 if c >= threshold else 2.0,
            "coverage95": 0.97,
            "inside2": 0.99,
            "fit_seconds": 0.0,
            "gates": {},
        }

    monkeypatch.setattr(certificate, "evaluate_design", stub)
    st = certify(setup, settings=None, fast=True, root=tmp_path, log=lambda *_: None)
    assert st["certificate"]["C_star"] == expected
    assert [e["passed"] for e in st["evaluated"]][-1] and not any(e["passed"] for e in st["evaluated"][:-1])
    # stored as JSON, readable by budget_for (kappa C*) and reference_cost
    saved = json.loads(certificate.cert_path("toy", True, tmp_path).read_text())
    assert saved["certificate"]["design"]["n0"] == saved["certificate"]["design"]["n0_per_control"] * nc
    assert reference_cost("toy", True, tmp_path, setup) == expected
    assert budget_for("toy", 2.0, True, tmp_path, setup) == 2.0 * expected


def test_no_certificate_when_no_design_passes(tmp_path, monkeypatch):
    setup = toy_setup()
    monkeypatch.setattr(
        certificate,
        "evaluate_design",
        lambda s, r, st: {
            "cost": 1.0,
            "passed": False,
            "max_sigma_over_eps": 9.0,
            "inside2": 0.0,
            "fit_seconds": 0.0,
        },
    )
    st = certify(setup, None, True, root=tmp_path, log=lambda *_: None)
    assert st["certificate"] is None and len(st["evaluated"]) == 18
    with pytest.raises(RuntimeError, match="no certificate"):
        reference_cost("toy", True, tmp_path, setup)


def test_resume_does_not_refit_evaluated_designs(tmp_path, monkeypatch):
    setup = toy_setup()
    calls = []

    def stub(s, r, st):
        calls.append(len(r))
        return {
            "cost": float(sum(x.cost for x in r)),
            "passed": len(calls) >= 3,
            "success": True,
            "max_sigma_over_eps": 0.1,
            "coverage95": 1.0,
            "inside2": 1.0,
            "fit_seconds": 0.0,
        }

    monkeypatch.setattr(certificate, "evaluate_design", stub)
    certify(setup, None, True, root=tmp_path, log=lambda *_: None)
    n = len(calls)
    certify(setup, None, True, root=tmp_path, log=lambda *_: None)
    assert len(calls) == n


@pytest.mark.slow
def test_real_fit_on_the_toy_certifies_the_cheapest_design(tmp_path):
    """Large tolerance, exact smooth truth: the cheapest design (n0 = 16, top level 2, cost 48) must pass."""
    setup = toy_setup(rel_tol=0.5)
    st = certify(setup, config.fast_settings(), True, root=tmp_path, log=lambda *_: None)
    assert st["certificate"] is not None
    assert st["certificate"]["C_star"] == family_cost(16, 2) == 48.0
    assert st["certificate"]["max_sigma_over_eps"] <= 1.0
