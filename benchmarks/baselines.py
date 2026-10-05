"""The methods of PROTOCOL Section 2. All get the same oracle, budget and tolerance.

``run_method(name, setup, budget, seed, settings)`` returns an ``Outcome``: the estimate of the converged
value on Sigma_N (``m_y`` and the epistemic sd ``sigma_epi``, in the units of the oracle), the cost spent
(never above the budget: every run is capped at what remains), the cost per level, and, for gcbml
configurations, the gate statuses. Metrics are computed from the Outcome by benchmarks.metrics.score.

Sizing of the fixed-design baselines [proposal]: a baseline spends the budget on the finest level it can
afford with a small fixed number of sites, planned with the noise-free work law E[cost] (a user of a
fixed ladder knows the cost scaling; gcbml has to learn it). Site counts: (a) 8 n_controls; (b) 4
n_controls fine and 8 n_controls coarse; (c) 8 n_controls sites at the three finest affordable levels;
(e) the family design of benchmarks.design. The leftover budget is not spent.

  gcbml   Campaign + run_campaign, default settings.
  a       single-fidelity GP (benchmarks.gp, ML-II plug-in) at the finest affordable level, space-filling
          sites.
  b       Yi et al. KRR-LR-GPR (mfbml.methods.krr_lr_gpr) on the two finest affordable levels.
  c       Richardson per site (least-squares power law f0 + a h^p over three levels, p in [0.5, 4], GCI factor
          1.25 when the fit is monotone and p is interior, else 3), then a GP over the extrapolated values
          with the GCI half-band as known noise sd.
  d       gcbml with p fixed (prior on log p: median expected order, sd 0.02), twy2 kernel only.
          NOT Boutelet-Sung as specified: the public API has no MAP option, so the other hyperparameters
          stay full Bayes.
  e       gcbml fit (no acquisition) of the most expensive static family design that fits the budget.
  f       gcbml with the lb kernel only. NOT Stroh as specified: gamma is learned (not fixed at 0.5) and the
          target is h = 0, not the finest level run: the public API has neither option.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from benchmarks.config import Setup
from benchmarks.design import (
    expected_cost,
    family_designs,
    fit_static,
    level_of,
    make_probe,
    run_all,
    sobol_sites,
)
from benchmarks.gp import PlugInGP
from benchmarks.oracle import BenchmarkOracle
from gcbml.campaign import Campaign, CampaignSettings, run_campaign
from gcbml.data import RunResult

METHODS = ("gcbml", "a", "b", "c", "d", "e", "f")
P_GRID = np.linspace(0.5, 4.0, 351)
GCI_FS_GOOD, GCI_FS_BAD = 1.25, 3.0


@dataclass
class Outcome:
    m_y: np.ndarray | None = None
    sigma_epi: np.ndarray | None = None
    spent: float = 0.0
    allocation: dict = field(default_factory=dict)
    gates: dict | None = None
    status: str | None = None
    notes: list = field(default_factory=list)
    n_capped: int = 0
    extra: dict = field(default_factory=dict)


def _allocation(setup: Setup, results) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in results:
        lev = level_of(setup, r.probe)
        out[lev] = out.get(lev, 0.0) + float(r.cost)
    return dict(sorted(out.items()))


def _unit(setup: Setup, u: np.ndarray) -> np.ndarray:
    p = setup.problem()
    nc = p.inputs.n_controls
    lo, hi = np.asarray(p.inputs.lower)[:nc], np.asarray(p.inputs.upper)[:nc]
    return (np.asarray(u, float) - lo) / (hi - lo)


def _level_cost(setup: Setup, level: int) -> float:
    probe = make_probe(setup, "plan", np.asarray(setup.problem().inputs.lower)[: setup.n_controls], level)
    return expected_cost(setup, [probe])


def _finest_affordable(setup: Setup, budget: float, per_level_runs: dict[int, int], top_min: int = 0):
    """Largest top level whose plan fits the budget. ``per_level_runs`` maps offset-from-top -> runs
    (0 = top level, 1 = one level below, ...). Returns None if even top = top_min does not fit."""
    best = None
    for top in range(top_min, 40):
        offs = per_level_runs
        if min(top - o for o in offs) < 0:
            continue
        cost = sum(n * _level_cost(setup, top - o) for o, n in offs.items())
        if cost > budget:
            break
        best = top
    return best


def _sigma_n(setup: Setup) -> np.ndarray:
    return setup.problem().sigma_n


def _fit_gp_best_transform(X, y, known_sd=None, fit_nugget=True):
    """Plug-in GP on y or on log y, whichever has the larger likelihood of the raw y."""
    n = len(y)
    g_id = PlugInGP.fit(X, y, known_sd, fit_nugget)
    ll_id = g_id.loglik - n * math.log(g_id.ys)
    if np.all(y > 0):
        ksd = None if known_sd is None else np.asarray(known_sd) / y
        g_lg = PlugInGP.fit(X, np.log(y), ksd, fit_nugget)
        ll_lg = g_lg.loglik - n * math.log(g_lg.ys) - float(np.sum(np.log(y)))
        if ll_lg > ll_id:
            return g_lg, "log"
    return g_id, "identity"


def _predict(gp: PlugInGP, tf: str, Xs):
    m, s = gp.predict(Xs)
    if tf == "log":
        return np.exp(m), np.exp(m) * s
    return m, s


# ------------------------------------------------------------------------------------------------ (a)


def method_a(setup, budget, seed, settings) -> Outcome:
    n = 8 * setup.n_controls
    top = _finest_affordable(setup, budget, {0: n})
    if top is None:
        return Outcome(notes=["nothing affordable"])
    U = sobol_sites(setup, n, seed)
    probes = [make_probe(setup, f"a-s{i}", U[i], top) for i in range(n)]
    oracle = BenchmarkOracle(setup.bp, seed, y_scale=setup.y_scale)
    res = run_all(oracle, probes, budget)
    ok = [r for r in res if len(r.y)]
    if len(ok) < 4:
        return Outcome(
            spent=sum(r.cost for r in res), notes=["too few runs finished"], n_capped=oracle.n_capped
        )
    X = _unit(setup, np.array([r.probe.u for r in ok]))
    y = np.array([r.y[0] for r in ok])
    gp, tf = _fit_gp_best_transform(X, y)
    m, s = _predict(gp, tf, _unit_sigma(setup))
    return Outcome(
        m,
        s,
        sum(r.cost for r in res),
        _allocation(setup, res),
        n_capped=oracle.n_capped,
        extra={"level": top, "transform": tf, "n_sites": len(ok)},
    )


def _unit_sigma(setup: Setup) -> np.ndarray:
    """Sigma_N in unit coordinates of the controls (the first n_controls columns)."""
    return _sigma_n(setup)[:, : setup.n_controls]


# ------------------------------------------------------------------------------------------------ (b)


def method_b(setup, budget, seed, settings) -> Outcome:
    from mfbml.methods.krr_lr_gpr import KernelRidgeLinearGaussianProcess

    nc = setup.n_controls
    n_h, n_l = 4 * nc, 8 * nc
    top = _finest_affordable(setup, budget, {0: n_h, 1: n_l}, top_min=1)
    if top is None:
        return Outcome(notes=["nothing affordable"])
    U = sobol_sites(setup, n_l, seed)
    probes = [make_probe(setup, f"b-lo-s{i}", U[i], top - 1) for i in range(n_l)]
    probes += [make_probe(setup, f"b-hi-s{i}", U[i], top) for i in range(n_h)]
    oracle = BenchmarkOracle(setup.bp, seed, y_scale=setup.y_scale)
    res = run_all(oracle, probes, budget)
    lo = [r for r in res if len(r.y) and r.probe.probe_id.startswith("b-lo")]
    hi = [r for r in res if len(r.y) and r.probe.probe_id.startswith("b-hi")]
    spent = sum(r.cost for r in res)
    if len(hi) < 3 or len(lo) < 4:
        return Outcome(
            spent=spent,
            allocation=_allocation(setup, res),
            notes=["too few runs finished"],
            n_capped=oracle.n_capped,
        )
    x_lo, x_hi = (_unit(setup, np.array([r.probe.u for r in rr])) for rr in (lo, hi))
    y_lo = np.array([r.y[0] for r in lo])[:, None]
    y_hi = np.array([r.y[0] for r in hi])[:, None]
    model = KernelRidgeLinearGaussianProcess(
        design_space=np.array([[0.0, 1.0]] * nc), optimizer_restart=3, seed=seed
    )
    model.train([x_hi, x_lo], [y_hi, y_lo])
    mean, total = model.predict(_unit_sigma(setup), return_std=True)
    epi = np.asarray(model.epistemic).ravel()  # epistemic part of the package's total sd
    del total
    return Outcome(
        np.asarray(mean).ravel(),
        epi,
        spent,
        _allocation(setup, res),
        n_capped=oracle.n_capped,
        extra={"level": top, "n_hi": len(hi), "n_lo": len(lo)},
    )


# ------------------------------------------------------------------------------------------------ (c)


def richardson(hbar: np.ndarray, f: np.ndarray) -> tuple[float, float, float, bool]:
    """Least-squares power law f(h) = f0 + a h^p over the levels (p on a grid in [0.5, 4]).

    Returns (f0, p, residual, good): good = monotone differences and p strictly inside the grid.
    """
    best = None
    for p in P_GRID:
        A = np.column_stack([np.ones_like(hbar), hbar**p])
        coef, *_ = np.linalg.lstsq(A, f, rcond=None)
        r = float(np.sum((A @ coef - f) ** 2))
        if best is None or r < best[0] - 1e-300:
            best = (r, float(coef[0]), float(p))
    r, f0, p = best
    d = np.diff(f)
    monotone = bool(np.all(d > 0) or np.all(d < 0))
    good = monotone and P_GRID[0] < p < P_GRID[-1]
    return f0, p, r, good


def gci_site(hbar: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    """(extrapolated value, half-band sd) at one site: U = F_s |f_fine - f0|, sd = U / 2 (a 95% band)."""
    f0, _, _, good = richardson(hbar, f)
    if good:
        fs, val = GCI_FS_GOOD, f0
        u = fs * abs(f[-1] - f0)
    else:
        fs, val = GCI_FS_BAD, float(f[-1])
        u = fs * float(np.max(np.abs(np.diff(f))))
    return val, 0.5 * max(u, 1e-12 * abs(val))


def method_c(setup, budget, seed, settings) -> Outcome:
    n = 8 * setup.n_controls
    top = _finest_affordable(setup, budget, {0: n, 1: n, 2: n}, top_min=2)
    if top is None:
        return Outcome(notes=["nothing affordable"])
    levels = [top - 2, top - 1, top]
    U = sobol_sites(setup, n, seed)
    probes = [make_probe(setup, f"c-l{lv}-s{i}", U[i], lv) for lv in levels for i in range(n)]
    oracle = BenchmarkOracle(setup.bp, seed, y_scale=setup.y_scale)
    res = run_all(oracle, probes, budget)
    spent = sum(r.cost for r in res)
    by = {(level_of(setup, r.probe), int(r.probe.probe_id.rsplit("-s", 1)[1])): r for r in res if len(r.y)}
    sites = [i for i in range(n) if all((lv, i) in by for lv in levels)]
    if len(sites) < 4:
        return Outcome(
            spent=spent,
            allocation=_allocation(setup, res),
            notes=["too few complete sites"],
            n_capped=oracle.n_capped,
        )
    hbar = np.array([2.0 ** -(lv - levels[0]) for lv in levels])
    vals, sds = [], []
    for i in sites:
        f = np.array([by[(lv, i)].y[0] for lv in levels])
        v, s = gci_site(hbar, f)
        vals.append(v)
        sds.append(s)
    vals, sds = np.array(vals), np.array(sds)
    X = _unit(setup, U[sites])
    gp, tf = _fit_gp_best_transform(X, vals, known_sd=sds, fit_nugget=False)
    m, s = _predict(gp, tf, _unit_sigma(setup))
    return Outcome(
        m,
        s,
        spent,
        _allocation(setup, res),
        n_capped=oracle.n_capped,
        extra={"levels": levels, "n_sites": len(sites), "transform": tf},
    )


# ------------------------------------------------------------------------------------------------ gcbml


def _from_report(rep, spent: float, notes: list | None = None) -> Outcome:
    return Outcome(
        m_y=None if rep.m_y is None else np.asarray(rep.m_y),
        sigma_epi=None if rep.sigma_epi is None else np.asarray(rep.sigma_epi),
        spent=float(rep.spent),
        allocation={int(k): float(v) for k, v in rep.allocation.items()},
        gates={g.name: g.status for g in rep.gates},
        status=rep.status,
        notes=list(rep.notes if notes is None else notes),
    )


def _campaign(setup, budget, seed, settings, scales=None) -> Outcome:
    camp = Campaign(
        setup.problem(budget),
        scales or setup.scales,
        setup.cost_prior,
        dataclasses.replace(settings, seed=seed),
    )
    oracle = BenchmarkOracle(setup.bp, seed, y_scale=setup.y_scale)
    rep = run_campaign(camp, oracle)
    out = _from_report(rep, camp.spent)
    out.spent = float(camp.spent)
    out.n_capped = oracle.n_capped
    out.extra = {"mode": camp.mode, "n_runs": len(camp.dataset), "history_len": len(camp.history)}
    return out


def method_gcbml(setup, budget, seed, settings) -> Outcome:
    return _campaign(setup, budget, seed, settings)


def method_d(setup, budget, seed, settings) -> Outcome:
    order = setup.bp.expected_order or 1.0
    scales = dataclasses.replace(setup.scales, log_p_mean=math.log(order), log_p_sd=0.02)
    out = _campaign(setup, budget, seed, dataclasses.replace(settings, h_kernels=("twy2",)), scales)
    out.extra["fixed_order"] = order
    return out


def method_f(setup, budget, seed, settings) -> Outcome:
    return _campaign(setup, budget, seed, dataclasses.replace(settings, h_kernels=("lb",)))


def method_e(setup, budget, seed, settings) -> Outcome:
    designs = [(k, p) for k, p in family_designs(setup, seed) if expected_cost(setup, p) <= budget]
    if not designs:
        return Outcome(notes=["nothing affordable"])
    key, probes = designs[-1]  # the most expensive design that fits
    oracle = BenchmarkOracle(setup.bp, seed, y_scale=setup.y_scale)
    res: list[RunResult] = run_all(oracle, probes, budget)
    spent = sum(r.cost for r in res)
    rep = fit_static(setup, [r for r in res], dataclasses.replace(settings, seed=seed))
    out = _from_report(rep, spent)
    out.spent, out.allocation, out.n_capped = spent, _allocation(setup, res), oracle.n_capped
    out.extra = {"design": list(key), "n_runs": len(res)}
    return out


_DISPATCH = {
    "gcbml": method_gcbml,
    "a": method_a,
    "b": method_b,
    "c": method_c,
    "d": method_d,
    "e": method_e,
    "f": method_f,
}


def run_method(name: str, setup: Setup, budget: float, seed: int, settings: CampaignSettings) -> Outcome:
    if name not in _DISPATCH:
        raise KeyError(f"unknown method {name!r}; known: {METHODS}")
    return _DISPATCH[name](setup, float(budget), int(seed), settings)


def mfbml_usable() -> tuple[bool, str]:
    """(ok, reason): mfbml imports and its KRR-LR-GPR runs only with numpy < 2 (its pinned 1.26.4)."""
    if int(np.__version__.split(".")[0]) >= 2:
        return False, f"mfbml's KRR-LR-GPR fails under numpy {np.__version__} (it pins numpy 1.26.4)"
    try:
        import mfbml.methods.krr_lr_gpr  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        return False, f"mfbml not importable: {e!r}"
    return True, ""
