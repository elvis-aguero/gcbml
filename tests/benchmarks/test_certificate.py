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


def stub_eval(threshold, calls=None):
    """A fit that passes a design exactly when its cost is at least ``threshold`` (at the toy's tolerance)."""

    def stub(setup, results, settings):
        c = float(sum(r.cost for r in results))
        if calls is not None:
            calls.append(c)
        ok = c >= threshold
        rel = (0.5 if ok else 2.0) * setup.rel_tol
        return {
            "cost": c,
            "n_runs": len(results),
            "max_rel_sigma": rel,
            "max_rel_error": 0.0,
            "coverage95": 0.97,
            "inside2": 0.99,
            "fit_seconds": 0.0,
            "gates": {},
            "status": "running",
        }

    return stub


def test_certificate_is_the_cheapest_passing_design(tmp_path, monkeypatch):
    """With a stub fit that passes exactly when cost >= 300, C* is the smallest family cost >= 300."""
    setup = toy_setup()
    nc = setup.n_controls
    costs = sorted(family_cost(a * nc, top) for a in config.N0_PER_CONTROL for top in config.LEVELS)
    threshold = 300.0
    expected = min(c for c in costs if c >= threshold)
    assert expected > costs[0]  # the stub must reject some designs
    calls = []
    monkeypatch.setattr(certificate, "evaluate_design", stub_eval(threshold, calls))
    st = certify(setup, settings=None, fast=True, root=tmp_path, log=lambda *_: None)
    cert = st["certificate"]
    assert cert["C_star"] == expected and cert["verified_cheapest"]
    assert calls == sorted(calls) and calls[-1] == expected and all(c < expected for c in calls[:-1])
    saved = json.loads(certificate.cert_path("toy", True, tmp_path).read_text())
    assert saved["certificate"]["design"]["n0"] == saved["certificate"]["design"]["n0_per_control"] * nc
    assert reference_cost("toy", True, tmp_path, setup) == expected
    assert budget_for("toy", 2.0, True, tmp_path, setup) == 2.0 * expected


def test_one_fit_serves_every_tolerance(tmp_path, monkeypatch):
    """The stored max sigma/|m_y| gives the certificate for another eps without a new fit."""
    setup = toy_setup()
    state_calls = []
    monkeypatch.setattr(certificate, "evaluate_design", stub_eval(300.0, state_calls))
    st = certify(setup, None, True, root=tmp_path, log=lambda *_: None)
    n_fits = len(state_calls)
    # a 4x larger tolerance: 2 rel_tol / (4 rel_tol) = 0.5 <= 1, so every design passes: C* is the cheapest
    loose = toy_setup(rel_tol=4 * setup.rel_tol)
    assert certificate.derive(st, loose)["C_star"] == min(e["cost"] for e in st["evaluated"])
    assert len(state_calls) == n_fits


def test_no_certificate_when_no_design_passes(tmp_path, monkeypatch):
    setup = toy_setup()
    monkeypatch.setattr(certificate, "evaluate_design", stub_eval(1e18))
    st = certify(setup, None, True, root=tmp_path, log=lambda *_: None)
    assert st["certificate"] is None and len(st["evaluated"]) == 18
    with pytest.raises(RuntimeError, match="no certificate"):
        reference_cost("toy", True, tmp_path, setup)


def test_resume_does_not_refit_evaluated_designs(tmp_path, monkeypatch):
    setup = toy_setup()
    calls = []
    monkeypatch.setattr(certificate, "evaluate_design", stub_eval(300.0, calls))
    certify(setup, None, True, root=tmp_path, log=lambda *_: None)
    n = len(calls)
    certify(setup, None, True, root=tmp_path, log=lambda *_: None)
    assert len(calls) == n


def test_scan_fits_every_kth_design_and_flags_an_unverified_certificate(tmp_path, monkeypatch):
    setup = toy_setup()
    calls = []
    monkeypatch.setattr(certificate, "evaluate_design", stub_eval(300.0, calls))
    st = certify(setup, None, True, root=tmp_path, log=lambda *_: None, stride=4)
    assert len(calls) == len({0, 4, 8, 12, 16, 17})
    assert st["certificate"] is not None and not st["certificate"]["verified_cheapest"]
    certify(setup, None, True, root=tmp_path, log=lambda *_: None)  # exact search fills in the gaps
    st = json.loads(certificate.cert_path("toy", True, tmp_path).read_text())
    assert st["certificate"]["verified_cheapest"]


@pytest.mark.slow
def test_real_fit_on_the_toy_certifies_the_cheapest_design(tmp_path):
    """Large tolerance, exact smooth truth: the cheapest design (n0 = 16, top level 2, cost 48) must pass."""
    setup = toy_setup(rel_tol=0.5)
    st = certify(setup, config.fast_settings(), True, root=tmp_path, log=lambda *_: None)
    assert st["certificate"] is not None
    assert st["certificate"]["C_star"] == family_cost(16, 2) == 48.0
    assert st["certificate"]["max_sigma_over_eps"] <= 1.0
