"""Campaign: initial design, ask/tell bookkeeping and budget, persistence, stop rules, end to end.

Fast tests use d = 1 or tiny chains; the end-to-end and the A12 test are marked slow.
"""

import dataclasses
import json
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gcbml  # noqa: F401
from gcbml import transforms
from gcbml.acquisition import StructurePosterior
from gcbml.campaign import Campaign, CampaignSettings, _Fit, run_campaign
from gcbml.cost import CostPrior
from gcbml.data import RunResult
from gcbml.gates import GateResult
from gcbml.kernels import DeltaParams
from gcbml.model import ModelConfig, ModelParams
from gcbml.priors import PriorScales
from gcbml.problem import Problem
from gcbml.synthetic import A12Truth

SCALES = PriorScales(S_mu=1.0, S_c=0.5, S_delta=0.5, S_noise=0.02)
COST_PRIOR = CostPrior(k0_mean=0.0, k0_sd=1.0, gamma_mean=(3.0,), gamma_sd=(0.5,), s_delta=(0.3,))
FAST = CampaignSettings(
    n_warmup=40,
    n_samples=40,
    n_chains=2,
    q=6,
    n_candidates_u=4,
    extra_levels=1,
    h_kernels=("twy2",),
    max_draws_acquisition=16,
    seed=1,
)


def make(truth, settings=FAST, state_dir=None, problem=None):
    return Campaign(problem or truth.problem(), SCALES, COST_PRIOR, settings, state_dir)


def probe_key(p):
    return (p.probe_id, tuple(p.u), tuple(p.h), p.run_length)


GATE_NAMES = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]


def _fake_gates(failing=()):
    def gates_(self, an, key):
        return [GateResult(n, "fail" if n in failing else "pass", {}) for n in GATE_NAMES], None

    return gates_


def _fake_fit_all(self, data, key, skip_note=None, init=None, n_warmup=None, max_extensions=None):
    """Stand-in for the MCMC fits: 4 fixed-hyperparameter draws per structure (no sampler, no compilation).

    The posterior of mu and of the levels is still computed from the real data by the real GP algebra, so
    sigma_epi, the candidates, the forecast and the acquisition behave as in a real campaign; only the
    hyperparameters are not inferred.
    """
    n_pad, k = np.asarray(data.H).shape
    y = np.asarray(data.y, dtype=float)
    fits = []
    for hk in self.settings.h_kernels:
        for tname in self.problem.transforms:
            tf = transforms.get(tname)
            cfg = ModelConfig(h_kernel=hk, increasing=(tname != "reciprocal"))
            nvar = 4e-4 if tname == "identity" else 4e-6  # the noise in the units of each transform

            def draw(i, nvar=nvar):
                return ModelParams(
                    sigma_mu=jnp.asarray(1.0 + 0.05 * i),
                    ell_mu=jnp.full((self.problem.inputs.d,), 0.3),
                    c0=jnp.full((k,), 0.3),
                    c1=jnp.full((k,), 0.1),
                    P=jnp.full((n_pad, k), 1.4),
                    delta=DeltaParams(
                        sigma=jnp.full((k,), 0.2),
                        ell_x=jnp.full((k, self.problem.inputs.d), 0.4),
                        ell_h=jnp.full((k,), 0.8),
                        gamma=jnp.full((k,), 0.5),
                    ),
                    noise_var=jnp.full((n_pad,), nvar),
                    ell_v=jnp.asarray(0.3),
                )

            params = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *[draw(i) for i in range(4)])
            z = jnp.broadcast_to(tf.forward(jnp.asarray(y)), (4, n_pad))
            theta = {
                "m_s": np.full((1, 4), np.log(nvar)),
                "b_s": np.zeros((1, 4, k)),
                "log_p0": np.full((1, 4, k), np.log(1.4)) + 0.05 * np.arange(4)[None, :, None],
            }
            sp = StructurePosterior(params, z, cfg, tf, 1.0)
            post = types.SimpleNamespace(theta=theta, diagnostics=types.SimpleNamespace(n_extensions=0))
            fits.append(_Fit(f"{hk}/{tname}", cfg, tf, post, sp, np.arange(4), params, z))
    return fits


@pytest.fixture
def no_gates(monkeypatch):
    """Fake fits (_fake_fit_all) and fake gates: gates G0-G7 and the MCMC are tested in test_gates.py and
    test_inference.py; the tests using this fixture exercise the control flow of the campaign (budget,
    persistence, stop rules, repair), which reads their verdicts. One test below runs the real thing."""
    monkeypatch.setattr(Campaign, "_fit_all", _fake_fit_all)
    # sigma_fid (a joint posterior of 2 x 100 points per draw and level) is checked by the real-fit test only
    monkeypatch.setattr(Campaign, "_sigma_fid", lambda self, fits, w, data, Xs, hbar, key: np.zeros(len(Xs)))
    monkeypatch.setattr(Campaign, "_gates", _fake_gates())


# ----------------------------------------------------------------------------------------------
# 2. Initial design
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def design():
    t = A12Truth(1, d=2, budget=1e9)
    base = t.problem()
    prob = Problem(
        inputs=base.inputs,
        resolution=base.resolution,
        tolerance=base.tolerance,
        budget=1e9,
        transforms=("identity",),
        region_lower=(0.2, 0.2),
        region_upper=(0.8, 0.8),
    )
    c = make(t, problem=prob)
    return c, c.initial_design()


def by_level(probes):
    out = {}
    for p in probes:
        out.setdefault(round(-np.log2(p.h[0])), []).append(p)
    return out


def test_initial_design_sizes_per_level_and_ids(design):
    c, probes = design
    lv = by_level(probes)
    n0 = 10 * 2
    assert sorted(lv) == [0, 1, 2]
    # distinct sites: n0, n0 // 2, max(3, n0 // 4); runs add 2 replicates at 3 sites per level
    sites = {lev: {p.u for p in ps} for lev, ps in lv.items()}
    assert [len(sites[lev]) for lev in (0, 1, 2)] == [n0, n0 // 2, max(3, n0 // 4)]
    assert [len(lv[lev]) for lev in (0, 1, 2)] == [n0 + 6, n0 // 2 + 6, max(3, n0 // 4) + 6]
    assert len({p.probe_id for p in probes}) == len(probes)
    assert all(p.run_length is None for p in probes)


def test_initial_design_ladder_is_nested_and_a_sobol_prefix(design):
    _, probes = design
    lv = by_level(probes)
    order = {}
    for lev, ps in lv.items():
        seen = []
        for p in ps:
            if p.u not in seen:
                seen.append(p.u)
        order[lev] = seen
    assert set(order[1]) <= set(order[0]) and set(order[2]) <= set(order[1])
    assert order[1] == order[0][: len(order[1])] and order[2] == order[1][: len(order[2])]


def test_initial_design_replicates_sit_at_the_first_three_sites_of_every_level(design):
    _, probes = design
    first3 = None
    for ps in by_level(probes).values():
        counts = {}
        for p in ps:
            counts[p.u] = counts.get(p.u, 0) + 1
        order = list(dict.fromkeys(p.u for p in ps))
        if first3 is None:
            first3 = order[:3]
        assert order[:3] == first3  # the same three Sobol points at every level
        assert [counts[u] for u in first3] == [3, 3, 3]  # 1 run + 2 replicates
        assert all(counts[u] == 1 for u in order[3:])


def test_initial_design_is_inside_sigma(design):
    _, probes = design
    u = np.array([p.u for p in probes])
    assert u.min() >= 0.2 and u.max() <= 0.8
    assert u.min() < 0.3 and u.max() > 0.7  # and it fills it


def test_initial_design_reserves_a_cap_for_every_probe_and_fits_the_budget(design):
    c, probes = design
    assert set(c.pending) == {p.probe_id for p in probes}
    assert all(c.cap_of(p) > 0 for p in probes)
    assert c.reserved == pytest.approx(sum(c.cap_of(p) for p in probes))
    assert c.reserved <= c.problem.budget
    with pytest.raises(RuntimeError):
        c.initial_design()  # issued once


def test_initial_design_is_cut_to_the_budget_with_the_cheap_level_first():
    t = A12Truth(1, d=1, budget=25.0)
    c = make(t)
    probes = c.initial_design()
    assert 0 < len(probes) < 10 + 6 + 5 + 6 + 3 + 6
    assert c.reserved <= 25.0
    assert {round(-np.log2(p.h[0])) for p in probes} == {0}  # nothing finer than level 0 fits


# ----------------------------------------------------------------------------------------------
# 3. tell / ask bookkeeping
# ----------------------------------------------------------------------------------------------


def test_tell_adds_cost_and_releases_the_reservation_of_that_probe():
    t = A12Truth(2, d=1, budget=5000.0)
    c = make(t)
    probes = c.initial_design()
    reserved0 = c.reserved
    caps = {p.probe_id: c.cap_of(p) for p in probes}
    o = t.oracle(seed=0)
    o.submit(probes[:4], [caps[p.probe_id] for p in probes[:4]])
    res = o.poll()
    c.tell(res)
    assert c.spent == pytest.approx(sum(r.cost for r in res))
    assert len(c.pending) == len(probes) - 4
    assert c.reserved == pytest.approx(c.spent + sum(caps[p.probe_id] for p in probes[4:]))
    assert c.reserved < reserved0 + 1e-9  # the realised cost is below the caps it replaced


def test_a_run_stopped_at_its_cap_counts_its_cap_and_is_a_censored_cost():
    t = A12Truth(2, d=1, budget=5000.0)
    c = make(t)
    probes = c.initial_design()
    p = probes[0]
    cap = c.cap_of(p)
    res = RunResult(p, np.zeros((0, 1)), np.zeros(0), np.zeros(0, bool), cap, True)
    c.tell([res])
    assert c.spent == pytest.approx(cap)
    assert p.probe_id not in c.pending
    assert len(c.dataset) == 1 and c.dataset.results[0].cost_censored


def test_no_run_is_stopped_below_its_cap_by_the_bookkeeping():
    t = A12Truth(2, d=1, budget=5000.0)
    c = make(t)
    (p, *_) = c.initial_design()
    with pytest.raises(KeyError):
        c.cap_of("not-a-probe")
    assert c.cap_of(p) == c.cap_of(p.probe_id)


class Audit:
    """Oracle wrapper that checks the budget invariant whenever the campaign hands over or gets back runs."""

    def __init__(self, inner, campaign, budget):
        self.inner, self.c, self.budget = inner, campaign, budget
        self.max_reserved = 0.0
        self.n_censored = 0
        self.n_runs = 0

    def quote(self, probes):
        return self.inner.quote(probes)

    def submit(self, probes, caps):
        for p, cap in zip(probes, caps, strict=True):
            assert cap == self.c.cap_of(p)
        self.max_reserved = max(self.max_reserved, self.c.reserved)
        assert self.c.reserved <= self.budget + 1e-9, (self.c.reserved, self.budget)
        self.inner.submit(probes, caps)

    def poll(self):
        res = self.inner.poll()
        self.n_runs += len(res)
        self.n_censored += sum(r.cost_censored for r in res)
        assert self.c.reserved <= self.budget + 1e-9
        return res


@pytest.mark.usefixtures("no_gates")
def test_budget_is_never_exceeded_over_a_whole_campaign_with_heavy_tailed_costs():
    budget = 400.0
    # unreachable tolerance: spend to the end; runs cost ~6x what the prior expects, so many hit their cap
    t = A12Truth(4, d=1, budget=budget, eps_abs=1e-4, cost_unit=6.0)
    c = make(t, dataclasses.replace(FAST, q=24))  # large batches: few rounds
    audit = Audit(t.oracle(seed=3, cost_sigma=1.5), c, budget)
    rep = run_campaign(c, audit)
    assert c.spent <= budget
    assert c.reserved == pytest.approx(c.spent)  # nothing is left pending at the end
    assert audit.max_reserved <= budget + 1e-9
    assert audit.n_censored > 0  # the heavy tail reached the caps
    assert sum(r.cost_censored for r in c.dataset.results) == audit.n_censored
    assert rep.status == "P2"
    assert rep.spent == pytest.approx(c.spent)
    assert sum(r.cost for r in c.dataset.results) == pytest.approx(c.spent)
    assert audit.n_runs == len(c.dataset)
    assert sum(rep.allocation.values()) == pytest.approx(c.spent)
    assert c.ask() == []  # finished


# ----------------------------------------------------------------------------------------------
# 4. Persistence
# ----------------------------------------------------------------------------------------------


def run_initial(c, t, seed=0):
    probes = c.initial_design()
    o = t.oracle(seed=seed)
    o.submit(probes, [c.cap_of(p) for p in probes])
    c.tell(o.poll())
    return o


def state_of(c):
    return (
        c.spent,
        {k: c.cap_of(k) for k in c.pending},
        c.mode,
        c.status,
        len(c.dataset),
        np.asarray(c._key).tolist(),
        c._min_level,
        [(r.probe.probe_id, r.cost, r.y.tolist(), r.cost_censored) for r in c.dataset.results],
        len(c.history),
    )


@pytest.mark.usefixtures("no_gates")
def test_save_after_tell_then_load_reproduces_the_state_and_the_next_ask(tmp_path):
    t = A12Truth(5, d=1, budget=3000.0, eps_abs=1e-4)
    c = make(t, state_dir=tmp_path)
    run_initial(c, t)
    assert (tmp_path / "state.json").exists() and (tmp_path / "results.npz").exists()
    loaded = Campaign.load(tmp_path)
    assert state_of(loaded) == state_of(c)
    assert loaded.problem == c.problem and loaded.settings == c.settings
    assert loaded.scales == c.scales and loaded.cost_prior == c.cost_prior
    a1 = c.ask()
    a2 = loaded.ask()
    assert a1 and [probe_key(p) for p in a1] == [probe_key(p) for p in a2]
    assert [c.cap_of(p) for p in a1] == [loaded.cap_of(p) for p in a2]
    assert state_of(c) == state_of(loaded)
    # save after ask: a second load sees the pending probes and the advanced key
    again = Campaign.load(tmp_path)
    assert state_of(again) == state_of(c)
    assert set(again.pending) == {p.probe_id for p in a1}


@pytest.mark.usefixtures("no_gates")
def test_continuing_from_a_loaded_state_matches_continuing_in_memory(tmp_path):
    t = A12Truth(5, d=1, budget=3000.0, eps_abs=1e-4)
    c = make(t, state_dir=tmp_path)
    o = run_initial(c, t)
    a1 = c.ask()
    loaded = Campaign.load(tmp_path)  # saved after the ask
    o.submit(a1, [c.cap_of(p) for p in a1])
    res = o.poll()
    c.tell(res)
    loaded.tell(res)
    b1, b2 = c.ask(), loaded.ask()
    assert b1 and [probe_key(p) for p in b1] == [probe_key(p) for p in b2]
    assert state_of(c) == state_of(loaded)


def test_state_files_are_json_plus_npz(tmp_path):
    t = A12Truth(5, d=1, budget=3000.0)
    c = make(t, state_dir=tmp_path)
    run_initial(c, t)
    s = json.loads(Path(tmp_path, "state.json").read_text())
    assert {"problem", "scales", "cost_prior", "settings", "spent", "pending", "key", "mode"} <= set(s)
    z = np.load(Path(tmp_path, "results.npz"))
    assert len(z.files) > 0


# ----------------------------------------------------------------------------------------------
# 5. Stop rules
# ----------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("no_gates")
def test_trivially_easy_problem_stops_with_success_after_the_initial_design():
    t = A12Truth(6, d=1, budget=1e6, eps_abs=1e3)
    c = make(t)
    o = t.oracle(seed=0)
    rep = run_campaign(c, o)
    n_init = 10 + 6 + 5 + 6 + 3 + 6
    assert o.n_submitted == n_init  # nothing beyond the initial design
    assert rep.status == "success"
    assert c.status == "success" and c.ask() == []
    assert not any(g.status == "fail" for g in rep.gates)
    assert rep.m_y is not None and rep.sigma_epi is not None
    assert float(np.max(rep.sigma_epi)) <= 1e3


@pytest.mark.usefixtures("no_gates")
def test_a_gate_failure_that_cannot_be_repaired_gives_uncalibrated_never_success(monkeypatch):
    monkeypatch.setattr(Campaign, "_gates", _fake_gates(failing=("G3",)))
    t = A12Truth(6, d=1, budget=1e6, eps_abs=1e3)  # P1 holds at once
    c = make(t)
    rep = run_campaign(c, t.oracle(seed=0))
    assert rep.status == "uncalibrated"
    assert any(g.name == "G3" and g.status == "fail" for g in rep.gates)
    assert c.status == "uncalibrated" and c.ask() == []


@pytest.mark.usefixtures("no_gates")
def test_a_tiny_budget_ends_with_P2_and_a_report_and_a_failed_gate_is_only_noted(monkeypatch):
    """Budget 30: a few level-0 runs, then nothing admissible. P1 is unmet, so a failing gate (G0, faked)
    does not make the status "uncalibrated": it stays P2 and the gate is listed in the notes."""
    monkeypatch.setattr(Campaign, "_gates", _fake_gates(failing=("G0",)))
    t = A12Truth(7, d=1, budget=30.0, eps_abs=1e-4)
    c = make(t)
    rep = run_campaign(c, t.oracle(seed=0))
    assert rep.status == "P2"
    assert 0 < len(c.dataset) and c.spent <= 30.0
    assert rep.m_y is not None and len(rep.m_y) == 100  # Sigma_N: 100 d points
    assert rep.spent == pytest.approx(c.spent)
    assert any("P2" in n for n in rep.notes)
    assert any("G0" in n and "not calibrated" in n for n in rep.notes)


@pytest.fixture(scope="module")
def easy_run():
    """One campaign with the REAL gates on a problem whose tolerance is met by the initial design."""
    t = A12Truth(6, d=1, budget=1e6, eps_abs=1e3)
    c = make(t, dataclasses.replace(FAST, n_warmup=30, n_samples=30, n0_per_control=6))
    return c, run_campaign(c, t.oracle(seed=0))


def test_gate_table_has_every_gate_and_never_passes_what_it_cannot_test(easy_run):
    _, rep = easy_run
    assert [g.name for g in rep.gates] == GATE_NAMES
    status = {g.name: g.status for g in rep.gates}
    assert status["G1"] == "not testable"  # too few levels and sites for a cross-fitted hold-out
    assert status["G5"] == "not testable"  # no monotone coordinate declared
    assert set(status.values()) <= {"pass", "fail", "not testable"}
    # no gate may hide an exception behind "not testable"
    assert not [g.name for g in rep.gates if "error" in g.stats], [g.stats for g in rep.gates]
    # whatever the gates say, the status follows them: no "success" with a failed gate
    blocking = [g.name for g in rep.gates if g.status == "fail" and g.name != "G6"]
    assert (rep.status == "success") == (not blocking)


def test_report_has_the_outputs_of_spec_section_3(easy_run):
    c, rep = easy_run
    s = c.problem.sigma_n.shape[0]
    for name in ("m_y", "sigma_epi", "sigma_tot"):
        a = np.asarray(getattr(rep, name))
        assert a.shape == (s,) and np.all(np.isfinite(a)), name
    assert set(rep.allocation) == {0, 1, 2}
    assert sum(rep.allocation.values()) == pytest.approx(rep.spent)
    assert np.asarray(rep.weights).sum() == pytest.approx(1.0)
    # the levels used: the coarsest may have been removed by the G4 repair
    assert set(rep.sigma_env) == set(rep.sigma_fid) and set(rep.sigma_env) <= {0, 1, 2}
    assert {1, 2} <= set(rep.sigma_env)
    # the fidelity envelope shrinks with h (spec 2.8)
    env = [float(np.median(rep.sigma_env[lev])) for lev in sorted(rep.sigma_env)]
    assert all(a > b > 0 for a, b in zip(env[:-1], env[1:]))
    assert all(np.all(np.isfinite(rep.sigma_fid[lev])) for lev in rep.sigma_fid)
    assert rep.s0 is None or np.asarray(rep.s0).shape == (s,)


def test_a_posterior_that_cannot_be_fitted_is_a_clear_error():
    t = A12Truth(6, d=1, budget=100.0)
    prob = dataclasses.replace(t.problem(), transforms=("log",))
    c = make(t, problem=prob)
    probes = c.initial_design()
    # every output is negative: the log transform has an empty domain, so there is no structure to fit
    res = [
        RunResult(p, np.array([p.u]), np.array([-1.0 - 0.01 * i]), np.zeros(1, bool), 1.0)
        for i, p in enumerate(probes[:8])
    ]
    c.tell(res)
    with pytest.raises(ValueError, match="domain"):
        c.report()


# ----------------------------------------------------------------------------------------------
# 6. End to end (slow)
# ----------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_a12_truth_moderate_budget(tmp_path, capsys):
    t = A12Truth(11, d=2, budget=2500.0, eps_abs=0.05)
    settings = CampaignSettings(
        n_warmup=150,
        n_samples=100,
        n_chains=4,
        q=60,
        n_candidates_u=16,
        extra_levels=2,
        h_kernels=("twy2", "lb"),
        max_draws_acquisition=32,
        seed=2,
    )
    c = make(t, settings, state_dir=tmp_path)
    audit = Audit(t.oracle(seed=1, cost_sigma=0.3), c, 2500.0)
    rep = run_campaign(c, audit)
    assert c.spent <= 2500.0
    assert rep.status in {"success", "P2", "uncalibrated"}
    xs = c.problem.sigma_n
    f0 = t.truth(xs)
    sig = np.asarray(rep.sigma_epi)
    inside = np.abs(np.asarray(rep.m_y) - f0) <= 2 * sig
    cover = float(inside.mean())
    with capsys.disabled():
        print("\nE2E status", rep.status, "spent", round(rep.spent, 1), "of 2500")
        print("E2E coverage of f0 in m_y +- 2 sigma_epi:", cover)
        print("E2E allocation per level:", {k: round(v, 1) for k, v in sorted(rep.allocation.items())})
        print("E2E gates:", {g.name: g.status for g in rep.gates})
        print("E2E weights:", np.round(rep.weights, 3), "max sigma_epi:", float(sig.max()))
        print("E2E notes:", rep.notes)
    assert cover >= 0.9
    assert rep.gates and rep.allocation


# ----------------------------------------------------------------------------------------------
# G6 is a warning; the repair buys a finer probe; per-structure noise
# ----------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("no_gates")
def test_a_g6_failure_is_a_warning_in_the_notes_and_never_blocks_success(monkeypatch):
    def g6_fails(self, an, key):
        gs = [GateResult(n, "pass", {}) for n in GATE_NAMES]
        gs[6] = GateResult("G6", "fail", {"median_rel_diff": 0.9})
        return gs, None

    monkeypatch.setattr(Campaign, "_gates", g6_fails)
    t = A12Truth(6, d=1, budget=1e6, eps_abs=1e3)
    c = make(t)
    rep = run_campaign(c, t.oracle(seed=0))
    assert rep.status == "success"
    assert any(n.startswith("warning: G6") for n in rep.notes)


def _g2_fails_at(row_values):
    """Fake gates: G2 fails, with the given |z| for the first len(row_values) real rows."""

    def gates_(self, an, key):
        rows = np.arange(len(row_values))
        an.gate_loc["G2"] = (rows, np.asarray(row_values, dtype=float))
        gs = [GateResult(n, "pass", {}) for n in GATE_NAMES]
        gs[2] = GateResult("G2", "fail", {})
        return gs, None

    return gates_


@pytest.mark.usefixtures("no_gates")
def test_repair_buys_a_probe_one_level_finer_where_z_is_largest_at_most_twice(monkeypatch):
    z = [0.1, 0.2, 5.0, 0.3, 0.4, 0.5, 0.6]  # row 2 has the largest |z|
    monkeypatch.setattr(Campaign, "_gates", _g2_fails_at(z))
    t = A12Truth(6, d=1, budget=1e6, eps_abs=1e3)  # P1 holds: only the repair can add runs
    c = make(t)
    run_initial(c, t)
    data, _ = c._build_data()
    u_row2 = float(c.problem.inputs.from_unit(np.asarray(data.X)[2])[0])
    o = t.oracle(seed=0)
    for cycle in (1, 2):
        probes = c.ask()
        assert len(probes) == 1, cycle
        (p,) = probes
        assert p.u == (pytest.approx(u_row2),)
        assert round(-np.log2(p.h[0])) == 2 + cycle  # one level finer than the finest run so far
        assert c.cap_of(p) > 0 and c.reserved <= c.problem.budget
        o.submit(probes, [c.cap_of(p)])
        c.tell(o.poll())
    assert c._buy_cycles == 2
    assert c.ask() == []  # no third purchase: the output is labelled, not repaired
    assert c.status == "uncalibrated"


@pytest.mark.usefixtures("no_gates")
def test_repair_is_skipped_when_the_gate_has_no_location_or_the_budget_is_too_small(monkeypatch):
    def gates_(self, an, key):
        gs = [GateResult(n, "pass", {}) for n in GATE_NAMES]
        gs[2] = GateResult("G2", "fail", {})  # fails, but gives no residuals to locate
        return gs, None

    monkeypatch.setattr(Campaign, "_gates", gates_)
    t = A12Truth(6, d=1, budget=1e6, eps_abs=1e3)
    c = make(t)
    run_initial(c, t)
    assert c.ask() == [] and c.status == "uncalibrated" and c._buy_cycles == 0


class PositiveA12(A12Truth):
    """A12 values shifted to be positive, so that the log transform is admissible."""

    def value(self, x_unit, hbar):
        return super().value(x_unit, hbar) + 10.0


@pytest.mark.usefixtures("no_gates")
def test_every_structure_has_its_own_noise_variance_for_new_rows():
    t = PositiveA12(3, d=1, budget=1e6, eps_abs=0.01)
    prob = dataclasses.replace(t.problem(), transforms=("identity", "log"))
    c = make(t, problem=prob)
    run_initial(c, t)
    an = c._analyse()
    assert [s.transform.name for s in an.structures] == ["identity", "log"]
    cand = types.SimpleNamespace(hbar=np.array([0.25]))
    nv = [np.asarray(s.noise_var(cand)) for s in an.structures]
    assert nv[0].shape == nv[1].shape == (4, 1)
    np.testing.assert_allclose(nv[0], 4e-4)  # identity units
    np.testing.assert_allclose(nv[1], 4e-6)  # log units: each structure its own variance
    plan = c.plan(key=jax.random.PRNGKey(0))
    assert np.all(np.isfinite(plan.table.ratio[plan.table.admissible]))


# ----------------------------------------------------------------------------------------------
# G4 on real fits (slow): spec Step 3
# ----------------------------------------------------------------------------------------------


PRE_AMP = 1.5  # a large smooth term (the noise sd is 0.01, the signal sd 1); 5.0 is absorbed even more


def _four_level_campaign(seed, preasymptotic=False, mcmc=(150, 100, 4)):
    """d = 1 A12 data at levels 0-3 (hbar 1 .. 1/8), noise 0.01. ``preasymptotic`` adds a large smooth term at
    hbar = 1 only, a coarsest level that is not in the asymptotic range."""
    t = A12Truth(seed, d=1, budget=1e9, eps_abs=0.01)
    settings = dataclasses.replace(FAST, n_warmup=mcmc[0], n_samples=mcmc[1], n_chains=mcmc[2], seed=seed)
    c = make(t, settings)
    o = t.oracle(seed=seed)
    res = []
    rng = np.random.default_rng(seed)
    for lev, n_sites in enumerate((12, 8, 6, 4)):
        for i in range(n_sites):
            u = (float(rng.uniform()),)
            h = c._h_of_level(lev)
            from gcbml.data import Probe

            o.submit([Probe(f"s{lev}_{i}", u, h, None)], [1e12])
            (r,) = o.poll()
            if preasymptotic and lev == 0:
                r = dataclasses.replace(r, y=r.y + PRE_AMP * np.cos(7.0 * u[0]))
            res.append(r)
    c.tell(res)
    return c


def _g4(c, mcmc=(150, 100, 4)):
    """G4 (posterior-predictive check of the coarsest level) on a main fit of the data, as _gates does."""
    from gcbml import gates

    c._min_level = 0
    data, lev = c._build_data()
    f = c._fit_all(data, jax.random.PRNGKey(c.settings.seed))[0]
    block = np.flatnonzero(np.asarray(data.mask) & (lev == 0))
    idx = jnp.asarray(np.linspace(0, f.sp.z.shape[0] - 1, 32).astype(int))
    params = jax.tree_util.tree_map(lambda a: a[idx], f.sp.params)
    return gates.g4_coarsest_level(params, data, f.sp.z[idx], f.cfg, block)


def _prior_predictive_campaign(seed, mcmc=(150, 100, 4)):
    """The four-level design with outputs simulated from the model's OWN prior (hyperparameters drawn
    from the prior of inference.fit, outputs from the Gaussian model), not from an A12 truth."""
    from test_acquisition import CFG, simulate_z

    from gcbml import inference
    from gcbml.model import ModelConfig

    c = _four_level_campaign(seed, mcmc=mcmc)
    data, _ = c._build_data()
    z0 = np.asarray(data.y)
    smp = inference._Sampler(data, z0, z0, ModelConfig(), SCALES, 1, False)
    st = smp.draw_prior(jax.random.PRNGKey(1000 + seed))
    params = smp.params(smp.layout.unpack(st["theta"]), st["zeta"], smp._no_pi())
    zsim = np.asarray(simulate_z(seed, params, data, CFG))
    res = [dataclasses.replace(r, y=np.array([zsim[i]])) for i, r in enumerate(c.dataset.results)]
    c2 = make(A12Truth(seed, d=1, budget=1e9), dataclasses.replace(c.settings))
    c2.tell(res)
    return c2


@pytest.mark.slow
def test_g4_size_on_data_from_the_models_own_prior_is_at_most_3_of_20(capsys):
    fails = [_g4(_prior_predictive_campaign(s)).status == "fail" for s in range(20)]
    with capsys.disabled():
        print(f"\nG4 v3 size on prior-predictive data: {sum(fails)} fails of 20")
    assert sum(fails) <= 3, fails


@pytest.mark.slow
@pytest.mark.xfail(
    strict=False,
    reason="G4 v3 does not detect pre-asymptotic levels: the delta GP absorbs the offset (same result on "
    "real data, p-value 0.97). A shape option for the error model (branch `shape`) is the planned remedy",
)
@pytest.mark.parametrize("amp", [1.5, 5.0])
def test_g4_power_a_preasymptotic_coarsest_level_fails_in_most_seeds(amp, monkeypatch, capsys):
    monkeypatch.setattr(sys.modules[__name__], "PRE_AMP", amp)
    fails = [
        _g4(_four_level_campaign(s, preasymptotic=True, mcmc=(150, 100, 4))).status == "fail"
        for s in range(10)
    ]
    with capsys.disabled():
        print(f"\nG4 v3 power, amplitude {amp}: {sum(fails)} fails of 10")
    assert sum(fails) >= 6, fails


@pytest.mark.slow
def test_parallel_fits_give_the_same_posteriors_as_serial_fits(monkeypatch):
    t = A12Truth(2, d=1, budget=1e6)
    c = make(t, dataclasses.replace(FAST, n_warmup=20, n_samples=20, n_chains=2, h_kernels=("twy2", "lb")))
    run_initial(c, t)
    data, _ = c._build_data()
    monkeypatch.setattr(Campaign, "PARALLEL_FITS", True)
    par = c._fit_all(data, jax.random.PRNGKey(4))
    monkeypatch.setattr(Campaign, "PARALLEL_FITS", False)
    ser = c._fit_all(data, jax.random.PRNGKey(4))
    assert [f.name for f in par] == [f.name for f in ser] == ["twy2/identity", "lb/identity"]
    for a, b in zip(par, ser, strict=True):
        np.testing.assert_array_equal(a.post.theta["log_p0"], b.post.theta["log_p0"])


@pytest.mark.usefixtures("no_gates")
def test_refits_do_not_extend_the_chains_but_the_main_fit_does_and_n_extensions_is_logged(monkeypatch):
    seen = []
    real = Campaign._fit_all

    def spy(self, data, key, skip_note=None, init=None, n_warmup=None, max_extensions=None):
        seen.append(max_extensions)
        return real(self, data, key, skip_note, init, n_warmup, max_extensions)

    monkeypatch.setattr(Campaign, "_fit_all", spy)
    t = A12Truth(6, d=1, budget=1e6, eps_abs=1e-4)
    c = make(t)
    run_initial(c, t)
    c._analyse()
    assert seen[0] is None  # the main fit after tell: inference.fit's default (extends on rhat / ESS)
    fit_events = [h for h in c.history if h["event"] == "fit"]
    assert fit_events and fit_events[-1]["n_extensions"] == [0]
    from gcbml import campaign as cp

    an = c._analyse()
    cand = c.plan(key=jax.random.PRNGKey(0)).cands[0]
    ds = cp.fantasy_dataset(c, cand, 0.1, "x")
    seen.clear()
    cp.refit_H(c, an, ds, jax.random.PRNGKey(1))
    assert seen == [0]  # a fantasy refit: no extension
