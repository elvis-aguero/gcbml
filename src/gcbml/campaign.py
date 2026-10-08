"""The algorithm of spec Section 3 as an ask/tell loop (Steps 0-6), with persistent state.

    c = Campaign(problem, scales, cost_prior, settings)
    probes = c.initial_design()                      # Step 1
    loop:  c.tell(results); probes = c.ask()         # Steps 2-5; [] when finished
    rep = c.report()                                 # the output of spec Section 3

Nothing here runs a simulation. run_campaign(campaign, oracle) is a convenience loop over an Oracle
(submit, poll) that respects the budget reservations.

Step 1, initial design [assumption; every number is a CampaignSettings field]:
  cheapest level 0: a scrambled Sobol set of n0 = 10 * n_controls points in Sigma (unit u);
  ladder: levels 1 and 2 use nested prefixes of that Sobol set of sizes max(3, n0 // 2^l);
  replicates: 2 extra runs at 3 sites (the first 3 Sobol points) at levels 0, 1 and 2;
  the A12 prerequisite is a test in tests/, not a runtime check.
Step 2: fit every structure (inference.fit) on all data; stacking weights (stacking.py) from the
  finest-level hold-out when testable, else equal weights. Structures = product of settings.h_kernels and
  problem.transforms (levels: all) [v1].
Step 3: gates G0-G4, G6, G7 (gates.py; G5 when settings.monotone is set). Repair (spec): at most 2 cycles:
  remove the coarsest level on a G4 fail (only while at least 3 levels remain); buy one probe a level finer
  than any run so far where |z| of a failed G1/G2 is largest; on other fails, label the result
  "uncalibrated" (never silently calibrated). G6 is a warning in the notes, never a blocking gate.
  Gate results go in the report.
Step 4: success if max over Sigma_N of sigma_epi / eps <= 1 and the gates pass -> ask() returns [];
  if the spent cost plus the cheapest admissible cap exceeds the budget -> P2 report, ask() returns [];
  else forecast (forecast.py): if infeasible switch the acquisition to mode "softmax" (P2) and record it;
  if p_exceed > 0.5 record a warning for the user.
Step 5: candidates = settings.n_candidates_u Sobol points in Sigma plus the existing sites, at every level
  from 0 to (finest level run so far + settings.extra_levels); cost of each from cost.fit_cost (refitted
  every ask) with quotes from the caller (ask(quotes=...)); acquisition.select_batch with q = settings.q.
State: everything needed to resume (results, spent cost, pending probes and their caps, RNG key, mode,
  history of reports) is saved to state_dir after every tell/ask as JSON + NPZ; Campaign.load(state_dir).

Implementation notes and assumptions (each marked [assumption] is a choice the stubs did not fix).
  * Interface additions: Campaign.cap_of(probe | probe_id) is the cost cap of a pending probe (a Probe has
    no cap field; run_campaign passes these to Oracle.submit); Campaign.pending {probe_id: cap};
    Campaign.spent; Campaign.reserved = spent + the caps of the pending probes (the quantity that must
    never exceed the budget); Campaign.status; Campaign.history; Campaign.save().
  * ask(quotes=...): ``quotes`` is a callable probes -> sequence of float | None (an Oracle's ``quote``), or
    None. The candidates are only known inside ask, so the caller cannot align a list in advance.
    [assumption] initial_design() has no quotes argument and its caps come from the cost prior alone.
  * One output per run (no output coordinates v): a campaign with n_controls < d raises NotImplementedError
    (module S1 needs the within-run noise and the censored outputs in the candidates; v1 omits it).
    Every resolution component is refined together: level l means level l in every component. [assumption]
  * Determinism: every fit is a function of (settings.seed, number of results, the coarsest level kept), so
    ask(), report() and a campaign reloaded from state_dir fit the same posterior. The only RNG state that
    advances is the acquisition key (fantasies), saved after every ask.
  * Initial design under a budget: probes are issued in the order level 0 (sites, then replicates), level 1,
    level 2, each with its cap from the cost prior; a probe whose cap does not fit in the remaining budget
    is skipped, so the reservation never exceeds the budget. [assumption]
  * "Not testable" gates do not block "success" (G1 is not testable until a cross-fitted hold-out exists);
    a "fail" gate does. When the budget ends with P1 unmet, the status stays "P2" and the failed gates are
    listed in the notes ("uncalibrated" is the label of an output that would otherwise be "success").
    [assumption]
  * G2 uses the highest-weight structure, at most 32 draws and 20 runs; G3 the same structure's posterior
    noise variance; G0 and G7 pool the draws of all structures by their weights. G7 cannot refit: a scenario
    with ESS < 400 is "not testable" (gates.g7_prior), so with few draws G7 is mostly "not testable".
  * Noise variance of new rows is per structure (StructurePosterior.noise_var): exp(m_s + b_s . hbar) of each
    kept draw of that structure, in ITS Lambda units (the latent field zeta stays at its mean).
  * The Step 4 forecast switches the acquisition to "softmax" whenever it does not reach P1 within the
    remaining budget (infeasible or budget-bound), and back to "hinge" when a later forecast succeeds.

State layout in state_dir: state.json (problem, scales, cost prior, settings, counters, pending probes and
caps, RNG key, mode, status, history, result metadata) and results.npz (x_i, y_i, censored_i per result).
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr, ndtri
from scipy.stats import qmc

from gcbml import acquisition as acq
from gcbml import cost, forecast, gates, inference, model, predict, priors, stacking, transforms
from gcbml._config import bucket
from gcbml.acquisition import Candidate, StructurePosterior
from gcbml.data import Dataset, PaddedData, Probe, RunResult
from gcbml.model import ModelConfig
from gcbml.priors import PriorScales
from gcbml.problem import InputSpace, Problem, ResolutionComponent, ResolutionSpec, Tolerance

MIN_ROWS = 4  # fewer usable output rows than this: nothing to fit
N_ACQ_DRAWS = 256  # draws per structure kept for sigma_epi, hold-out scores, report (thinned evenly)
N_COST_DRAWS = 128  # draws of the cost posterior used to price candidates
N_SAMPLE_TARGET = 4000  # physical samples per structure for medians and quantile spreads
G2_MAX_BLOCKS, G2_MAX_DRAWS = 20, 32
FID_MAX_DRAWS = 32
MAX_REPAIR = 2
MIN_SITES_HOLDOUT = 12  # two halves of >= 6 held-out sites (spec 2.7)
CAP_QUANTILE = 0.95
PRICE_PAD = 64  # candidate runs are priced in blocks of this many (fewer recompilations)


@dataclass(frozen=True)
class CampaignSettings:
    n_warmup: int = 500
    n_samples: int = 500
    n_chains: int = 4
    q: int = 1
    n0_per_control: int = 10
    n_replicate_sites: int = 3
    n_replicates: int = 2
    n_candidates_u: int = 64
    extra_levels: int = 2
    h_kernels: tuple[str, ...] = ("twy2", "lb")
    varying_order: bool = False
    monotone: tuple[int, str] | None = None  # (coordinate index, "increasing" | "decreasing") for G5
    max_draws_acquisition: int = 64
    seed: int = 0


@dataclass
class Report:
    status: str  # "success" | "P2" | "uncalibrated" | "running"
    m_y: Any = None
    sigma_epi: Any = None
    s0: Any = None
    sigma_tot: Any = None
    sigma_env: Any = None
    sigma_fid: Any = None
    gates: list = field(default_factory=list)
    weights: Any = None
    spent: float = 0.0
    allocation: dict = field(default_factory=dict)  # level -> cost spent
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------------------------------
# State serialisation
# ----------------------------------------------------------------------------------------------


def _problem_to_dict(p: Problem) -> dict:
    return {
        "inputs": {
            "names": list(p.inputs.names),
            "lower": list(p.inputs.lower),
            "upper": list(p.inputs.upper),
            "n_controls": p.inputs.n_controls,
        },
        "components": [
            {"name": c.name, "values": list(c.values), "refine_ratio": c.refine_ratio}
            for c in p.resolution.components
        ],
        "tolerance": {"rel": p.tolerance.rel, "abs": p.tolerance.abs},
        "budget": p.budget,
        "transforms": list(p.transforms),
        "region_lower": None if p.region_lower is None else list(p.region_lower),
        "region_upper": None if p.region_upper is None else list(p.region_upper),
        "seed": p.seed,
    }


def _problem_from_dict(d: dict) -> Problem:
    i = d["inputs"]
    inputs = InputSpace(tuple(i["names"]), tuple(i["lower"]), tuple(i["upper"]), i["n_controls"])
    res = ResolutionSpec(
        tuple(ResolutionComponent(c["name"], tuple(c["values"]), c["refine_ratio"]) for c in d["components"])
    )
    rl, ru = d["region_lower"], d["region_upper"]
    return Problem(
        inputs=inputs,
        resolution=res,
        tolerance=Tolerance(**d["tolerance"]),
        budget=d["budget"],
        transforms=tuple(d["transforms"]),
        region_lower=None if rl is None else tuple(rl),
        region_upper=None if ru is None else tuple(ru),
        seed=d["seed"],
    )


def _settings_from_dict(d: dict) -> CampaignSettings:
    d = dict(d)
    d["h_kernels"] = tuple(d["h_kernels"])
    if d.get("monotone") is not None:
        d["monotone"] = tuple(d["monotone"])
    return CampaignSettings(**d)


def _cost_prior_from_dict(d: dict) -> cost.CostPrior:
    d = dict(d)
    d["gamma_mean"], d["gamma_sd"] = tuple(d["gamma_mean"]), tuple(d["gamma_sd"])
    if "q_sd" in d or "t_sd" in d:
        raise ValueError(
            "state saved with the polynomial cost prior (q_sd, t_sd); it was replaced by the random walk "
            "(s_delta, spec 2.6): set CostPrior.s_delta and start a new campaign state"
        )
    if isinstance(d["s_delta"], list):
        d["s_delta"] = tuple(d["s_delta"])
    return cost.CostPrior(**d)


def _probe_to_dict(p: Probe) -> dict:
    return {"probe_id": p.probe_id, "u": list(p.u), "h": list(p.h), "run_length": p.run_length}


def _probe_from_dict(d: dict) -> Probe:
    return Probe(d["probe_id"], tuple(d["u"]), tuple(d["h"]), d["run_length"])


def _thin_idx(n: int, m: int) -> np.ndarray:
    """m indices spread evenly over range(n) (all of them if n <= m)."""
    return np.arange(n) if n <= m else np.unique(np.linspace(0, n - 1, m).astype(int))


# ----------------------------------------------------------------------------------------------
# Jitted pieces
# ----------------------------------------------------------------------------------------------


def _pd(X, H, run, mask, zz):
    return PaddedData(X=X, H=H, y=zz, censored=jnp.zeros_like(mask), run=run, mask=mask)


@functools.partial(jax.jit, static_argnames=("cfg",))
def _mu_moments(params, z, X, H, run, mask, Xs, cfg):
    """(mean, var) of mu(Xs) per draw, (S, s) each; sequential over draws in batches (bounded memory)."""

    def one(args):
        p, zz = args
        return model.predict_mu(p, _pd(X, H, run, mask, zz), zz, cfg, Xs)

    return jax.lax.map(one, (params, z), batch_size=32)


@functools.partial(jax.jit, static_argnames=("cfg",))
def _fid_moments(params, z, X, H, run, mask, Xs, hbar, cfg):
    """Joint posterior of (mu(x), f(x, h)) per draw and x: both means, both variances and the covariance.

    A new, noise-free output at hbar = 0 is mu(x) exactly (rho0 = 0, rho1 = 1, delta = 0 there), so the joint
    predictive of the rows [(x, 0); (x, h)] (model.joint_new) gives the pair.
    """
    s, k = Xs.shape[0], hbar.shape[0]
    Xn = jnp.concatenate([Xs, Xs])
    Hn = jnp.concatenate([jnp.zeros((s, k)), jnp.broadcast_to(hbar, (s, k))])

    def one(args):
        p, zz = args
        Pn = jnp.broadcast_to(p.P[0], Hn.shape)
        jn = model.joint_new(p, _pd(X, H, run, mask, zz), zz, cfg, Xn, Hn, Pn, same_run=False)
        d = jnp.diagonal(jn.cov)
        return jn.mean[:s], jn.mean[s:], d[:s], d[s:], jnp.diagonal(jn.cov[:s, s:])

    return jax.lax.map(one, (params, z), batch_size=8)


@functools.partial(jax.jit, static_argnames=("cfg",))
def _level_moments(params, z, X, H, run, mask, Xr, Hr, cfg):
    """Posterior predictive (noise included) of rows (Xr, Hr) per draw, in Lambda units: (S, n) each."""

    def one(args):
        p, zz = args
        Pr = jnp.broadcast_to(p.P[0], Hr.shape)
        return model.predict_level(p, _pd(X, H, run, mask, zz), zz, cfg, Xr, Hr, Pr, None)

    return jax.lax.map(one, (params, z), batch_size=32)


# ----------------------------------------------------------------------------------------------
# Fits and analysis
# ----------------------------------------------------------------------------------------------


@dataclass
class _Fit:
    name: str
    cfg: ModelConfig
    tf: Any
    post: Any  # inference.Posterior
    sp: StructurePosterior  # thinned draws, weight 1
    idx: np.ndarray  # indices of the kept draws in the flattened posterior
    full_params: Any = None
    full_z: Any = None


@dataclass
class _Analysis:
    key: tuple
    data: PaddedData
    levels: np.ndarray  # (n_pad,) level of each row, -1 on padding
    fits: list
    weights: np.ndarray
    structures: list  # StructurePosterior with weights
    m_y: np.ndarray
    sigma_epi: np.ndarray
    eps: np.ndarray
    moments: list  # per fit (mean (S, s), var (S, s)) of mu at Sigma_N
    gates: list
    notes: list
    weight_note: str = ""
    g1: tuple | None = None
    skipped: list = field(default_factory=list)
    gate_loc: dict = field(default_factory=dict)  # gate name -> (data rows, standardised residuals)
    fid_cache: dict = field(default_factory=dict)  # level -> sigma_fid (Sigma_N,), computed once per analysis


@dataclass
class _Plan:
    an: _Analysis
    cands: list
    probes: list
    caps: np.ndarray
    costs: np.ndarray
    table: Any
    chosen: list
    mode: str


def _arrays(data: PaddedData):
    return (
        jnp.asarray(data.X, dtype=float),
        jnp.asarray(data.H, dtype=float),
        jnp.asarray(data.run),
        jnp.asarray(data.mask, dtype=bool),
    )


class Campaign:
    WARM_START = True  # start refits at the last posterior draws (inference.fit(init=...))
    WARM_FRACTION = 0.25  # warm-up of a warm-started fit = max(WARM_MIN, round(WARM_FRACTION * n_warmup))
    WARM_MIN = 40
    PARALLEL_FITS = True  # independent fits of one ask run in threads (structures; main fit and hold-out)

    def __init__(
        self, problem: Problem, scales: PriorScales, cost_prior, settings: CampaignSettings | None = None,
        state_dir=None,
    ):  # fmt: skip
        if problem.inputs.n_controls != problem.inputs.d:
            raise NotImplementedError("campaign v1 has no output coordinates v (module S1): n_controls == d")
        self.problem, self.scales, self.cost_prior = problem, scales, cost_prior
        self.settings = settings or CampaignSettings()
        self.state_dir = None if state_dir is None else Path(state_dir)
        self.dataset = Dataset()
        self.spent = 0.0
        self._pending: dict[str, dict] = {}
        self._counter = 0
        self._key = np.asarray(jax.random.PRNGKey(self.settings.seed + 104729), dtype=np.uint32)
        self.mode = "hinge"
        self.status = "running"
        self.history: list[dict] = []
        self.notes: list[str] = []
        self._design_issued = False
        self._min_level = 0
        self._removals = 0
        self._buy_cycles = 0
        self._warm: dict[str, dict] = {}  # structure name -> last draw of every chain (theta arrays)
        self._cache: _Analysis | None = None
        self._report_cache: Report | None = None
        self._report_cache_key: tuple = ()
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ properties and small helpers

    @property
    def pending(self) -> dict[str, float]:
        """{probe_id: cap} of the probes asked for and not yet told."""
        return {k: v["cap"] for k, v in self._pending.items()}

    @property
    def reserved(self) -> float:
        """spent + the caps of all pending probes: never more than the budget (spec Step 5)."""
        return self.spent + sum(v["cap"] for v in self._pending.values())

    def cap_of(self, probe) -> float:
        pid = probe if isinstance(probe, str) else probe.probe_id
        return float(self._pending[pid]["cap"])

    @property
    def _k(self) -> int:
        return self.problem.resolution.k

    @property
    def _nc(self) -> int:
        return self.problem.inputs.n_controls

    def _region(self) -> tuple[np.ndarray, np.ndarray]:
        inp = self.problem.inputs
        lo = np.asarray(
            self.problem.region_lower if self.problem.region_lower is not None else inp.lower, float
        )
        hi = np.asarray(
            self.problem.region_upper if self.problem.region_upper is not None else inp.upper, float
        )
        return lo[: self._nc], hi[: self._nc]

    def _h_of_level(self, lev: int) -> tuple[float, ...]:
        return tuple(float(t) for t in self.problem.resolution.h_at((lev,) * self._k))

    def _level_of(self, h) -> int:
        levs = []
        for comp, hj in zip(self.problem.resolution.components, h, strict=True):
            for lev in range(60):
                if math.isclose(comp.level_value(lev), hj, rel_tol=1e-9):
                    levs.append(lev)
                    break
            else:
                raise ValueError(f"h = {hj} is not a level of component {comp.name!r}")
        return max(levs)

    def _new_id(self, tag: str) -> str:
        self._counter += 1
        return f"{tag}{self._counter:05d}"

    def _next_key(self):
        k, sub = jax.random.split(jnp.asarray(self._key, dtype=jnp.uint32))
        self._key = np.asarray(k, dtype=np.uint32)
        return sub

    def _fit_key(self, purpose: int):
        base = jax.random.PRNGKey(self.settings.seed)
        return jax.random.fold_in(base, len(self.dataset) * 1000 + self._min_level * 100 + purpose)

    def _eps(self, m_y: np.ndarray) -> np.ndarray:
        tol = self.problem.tolerance
        eps = tol.rel * np.abs(m_y) if tol.rel is not None else np.full(np.shape(m_y), float(tol.abs))
        return np.maximum(eps, 1e-12)

    # ------------------------------------------------------------------ Step 1: initial design

    def initial_design(self) -> list[Probe]:
        if self._design_issued:
            raise RuntimeError("initial_design() was already issued")
        s, nc = self.settings, self._nc
        n0 = s.n0_per_control * nc
        m = max(1, math.ceil(math.log2(n0)))
        sob = qmc.Sobol(nc, scramble=True, seed=s.seed).random_base2(m)[:n0]
        lo, hi = self._region()
        U = lo + sob * (hi - lo)
        sizes = [n0] + [min(n0, max(3, n0 // 2**lev)) for lev in (1, 2)]
        n_rep = min(s.n_replicate_sites, min(sizes))
        wanted: list[tuple[int, int, int]] = []  # (level, site, replicate index; 0 = the base run)
        for lev, n_sites in enumerate(sizes):
            wanted += [(lev, i, 0) for i in range(n_sites)]
            wanted += [(lev, i, r) for i in range(n_rep) for r in range(1, s.n_replicates + 1)]
        probes = [
            Probe(self._new_id("d"), tuple(float(t) for t in U[i]), self._h_of_level(lev), None)
            for lev, i, r in wanted
        ]
        # prior-only caps (an empty cost dataset), levels are few so price each level once per site
        post = self._cost_posterior()
        mean, cap = self._price(post, [p.u for p in probes], [self._level_of(p.h) for p in probes], None)
        del mean
        kept, reserved = [], self.reserved
        for p, c in zip(probes, cap, strict=True):
            if reserved + c <= self.problem.budget:
                kept.append(p)
                reserved += c
                self._pending[p.probe_id] = {"probe": p, "cap": float(c)}
        self._design_issued = True
        self._record("initial_design", n_probes=len(kept))
        self.save()
        return kept

    # ------------------------------------------------------------------ tell

    def tell(self, results) -> None:
        for r in results:
            self.dataset.add(r)
            self.spent += float(r.cost)
            entry = self._pending.pop(r.probe.probe_id, None)
            if entry is not None and r.cost > entry["cap"] * (1 + 1e-9) + 1e-12:
                self.notes.append(f"oracle exceeded the cap of {r.probe.probe_id}: {r.cost} > {entry['cap']}")
        self._cache = None
        self._report_cache = None
        self.save()

    # ------------------------------------------------------------------ cost model

    def _cost_data(self) -> cost.CostData:
        res = self.dataset.results
        n = len(res)
        n_pad = bucket(n)
        lo, hi = self._region_full()
        U = np.zeros((n_pad, self._nc))
        L = np.zeros((n_pad, self._k))
        log2c = np.zeros(n_pad)
        cens = np.zeros(n_pad, bool)
        log2q = np.zeros(n_pad)
        has_q = np.zeros(n_pad, bool)
        mask = np.zeros(n_pad, bool)
        for i, r in enumerate(res):
            U[i] = (np.asarray(r.probe.u, float) - lo) / (hi - lo)
            L[i] = self._level_vec(r.probe.h)
            log2c[i] = math.log2(max(float(r.cost), 1e-300))
            cens[i] = bool(r.cost_censored)
            if r.cost_hint is not None and r.cost_hint > 0:
                log2q[i], has_q[i] = math.log2(r.cost_hint), True
            mask[i] = True
        return cost.CostData(U, L, log2c, cens, log2q, has_q, mask)

    def _region_full(self):
        inp = self.problem.inputs
        return np.asarray(inp.lower, float)[: self._nc], np.asarray(inp.upper, float)[: self._nc]

    def _level_vec(self, h) -> list[float]:
        out = []
        for comp, hj in zip(self.problem.resolution.components, h, strict=True):
            for lev in range(60):
                if math.isclose(comp.level_value(lev), hj, rel_tol=1e-9):
                    out.append(float(lev))
                    break
            else:
                raise ValueError(f"h = {hj} is not a level of component {comp.name!r}")
        return out

    def _cost_posterior(self) -> cost.CostPosterior:
        s = self.settings
        key = jax.random.fold_in(jax.random.PRNGKey(s.seed), 7_000_000 + len(self.dataset))
        post = cost.fit_cost(key, self._cost_data(), self.cost_prior, s.n_warmup, s.n_samples, s.n_chains)
        idx = _thin_idx(int(post.log2c.shape[0]), N_COST_DRAWS)
        return post._replace(
            hyper={k: v[idx] for k, v in post.hyper.items()}, log2c=jnp.asarray(post.log2c)[idx]
        )

    def _price(self, post, us, levels, log2q):
        """(E[min(c, cap)], cap) of runs at controls ``us`` (physical) and diagonal levels."""
        lo, hi = self._region_full()
        U = (np.asarray(us, float).reshape(len(us), self._nc) - lo) / (hi - lo)
        L = np.repeat(np.asarray(levels, float)[:, None], self._k, axis=1)
        if log2q is None:
            lq, hq = np.zeros(len(U)), np.zeros(len(U), bool)
        else:
            lq, hq = log2q
        # pad the runs to a multiple of PRICE_PAD (copies of the first row) so that the compiled predictors
        # are reused while the candidate set grows with the sites; the padding is sliced off again
        m = len(U)
        mp = PRICE_PAD * math.ceil(max(m, 1) / PRICE_PAD)
        idx = np.concatenate([np.arange(m), np.zeros(mp - m, dtype=int)])
        mean, var = cost.predict_log2(post, U[idx], L[idx], np.asarray(lq)[idx], np.asarray(hq)[idx])
        w = jnp.ones(mean.shape[0]) / mean.shape[0]
        cap = cost.cost_cap(mean, var, w, CAP_QUANTILE)
        # a run is stopped at its cap, so it is charged E[min(c, cap)], not E[c] (spec 2.6, Step 5)
        capped = np.asarray(cost.expected_capped_cost(mean, var, w, cap))
        cap = np.asarray(cap)
        # exp(m + s^2/2) overflows to inf for an enormous predictive variance and inf * 0 is nan in
        # cost.expected_capped_cost; the price can never exceed the cap, which is the fallback and the bound
        capped = np.where(np.isfinite(capped), np.minimum(capped, cap), cap)
        return capped[:m], cap[:m]

    # ------------------------------------------------------------------ data

    def _n_rows(self) -> int:
        return int(sum(len(r.y) for r in self.dataset.results))

    def _build_data(self, dataset: Dataset | None = None) -> tuple[PaddedData, np.ndarray]:
        ds = self.dataset if dataset is None else dataset
        inp, res = self.problem.inputs, self.problem.resolution
        pd = ds.padded(inp.to_unit, res.hbar)
        lev = np.concatenate(
            [np.full(len(r.y), self._level_of(r.probe.h)) for r in ds.results]
            + [np.full(pd.mask.size - pd.n, -1)]
        ).astype(int)
        mask = np.asarray(pd.mask) & (lev >= self._min_level)
        return dataclasses.replace(pd, mask=mask), lev

    # ------------------------------------------------------------------ Step 2: fits and weights

    def _fit_all(
        self, data: PaddedData, key, skip_note: list | None = None, init: list | dict | None = None,
        n_warmup: int | None = None,
    ) -> list[_Fit]:  # fmt: skip
        """Fit every structure (in parallel threads when PARALLEL_FITS: the fits are independent, each one
        keyed by fold_in(key, i), so the result does not depend on the scheduling)."""
        s = self.settings
        n_warmup = s.n_warmup if n_warmup is None else n_warmup
        y = np.asarray(data.y, dtype=float)
        jobs = []
        for hk in s.h_kernels:
            for tname in self.problem.transforms:
                tf = transforms.get(tname)
                if not bool(np.all(np.asarray(tf.domain_ok(jnp.asarray(y))))):
                    if skip_note is not None:
                        skip_note.append(
                            f"structure {hk}/{tname} dropped: outputs outside the transform domain"
                        )
                    continue
                name = f"{hk}/{tname}"
                i = len(jobs)
                ini = init.get(name) if isinstance(init, dict) else (None if init is None else init[i])
                jobs.append((i, name, hk, tf, ini))
        if not jobs:
            raise ValueError(
                "no structure can be fitted: the outputs are outside the domain of every transform"
            )

        def one(job):
            i, name, hk, tf, ini = job
            cfg = ModelConfig(h_kernel=hk, increasing=(tf.name != "reciprocal"))
            z = np.asarray(tf.forward(jnp.asarray(y)))
            post = inference.fit(
                jax.random.fold_in(key, i), data, z, z, cfg, self.scales, self._nc,
                n_warmup, s.n_samples, s.n_chains, s.varying_order, init=ini,
            )  # fmt: skip
            params, zs = inference.flatten(post)
            idx = _thin_idx(int(zs.shape[0]), N_ACQ_DRAWS)
            take = jnp.asarray(idx)
            sp = StructurePosterior(
                jax.tree_util.tree_map(lambda a, take=take: a[take], params), zs[take], cfg, tf, 1.0
            )
            return _Fit(name, cfg, tf, post, sp, idx, params, zs)

        if self.PARALLEL_FITS and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                return list(ex.map(one, jobs))
        return [one(j) for j in jobs]

    def _moments(self, fits, data, Xs):
        X, H, run, mask = _arrays(data)
        out = []
        for f in fits:
            m, v = _mu_moments(f.sp.params, f.sp.z, X, H, run, mask, jnp.asarray(Xs, dtype=float), f.cfg)
            out.append((m, jnp.maximum(v, 0.0)))
        return out

    def _pool(self, fits, weights, moments, key, extra_var=None, n_rep=None, center=None):
        """Physical samples pooled over structures: (draws (N, s), weights (N,)).

        Draw j of structure M is N(mean, var + extra_var) in Lambda units, mapped back with Lambda^{-1};
        ``center`` replaces the mean (used for s0). ``extra_var`` (S, s) is a noise variance for sigma_tot.
        """
        out, ws = [], []
        for i, (f, w, (m, v)) in enumerate(zip(fits, weights, moments, strict=True)):
            S = int(m.shape[0])
            nr = n_rep or max(1, N_SAMPLE_TARGET // S)
            mean = m if center is None else jnp.broadcast_to(center[i], m.shape)
            var = v if extra_var is None else (v if center is None else 0.0 * v) + extra_var[i]
            z, wr = predict.sample_gaussian_draws(jax.random.fold_in(key, i), mean, var, nr)
            out.append(np.asarray(f.tf.inverse(z)))
            ws.append(float(w) * np.asarray(wr))
        w = np.concatenate(ws)
        return np.concatenate(out), w / w.sum()

    def _holdout_plan(self, data, levels, M: int):
        """(train, rows, finest) of the extrapolation hold-out, or (None, None, reason) if not testable."""
        mask = np.asarray(data.mask, dtype=bool)
        active = np.unique(levels[mask])
        if M == 1:
            return None, None, "one structure"
        if len(active) < 4:
            return (
                None,
                None,
                f"equal weights: {len(active)} levels (< 4), p is not identifiable after a hold-out",
            )
        finest = int(active.max())
        train, rows = stacking.holdout(data, levels, finest)
        sites = np.unique(np.asarray(data.X)[rows], axis=0) if rows.size else np.zeros((0, 1))
        if len(sites) < MIN_SITES_HOLDOUT:
            return None, None, f"equal weights: {len(sites)} held-out sites (< {MIN_SITES_HOLDOUT})"
        return train, rows, finest

    def _stacking(self, data, levels, fits, key, plan=None, fits_tr=None):
        """(weights, note, g1) of spec 2.7: cross-fitted extrapolation hold-out when testable, else equal.

        ``plan`` / ``fits_tr``: the hold-out plan and its fits when the caller already ran them (concurrently
        with the main fit)."""
        M = len(fits)
        eq = np.full(M, 1.0 / M)
        train, rows, finest = plan if plan is not None else self._holdout_plan(data, levels, M)
        if train is None:
            return eq, finest, None
        if fits_tr is None:
            fits_tr = self._fit_all(train, jax.random.fold_in(key, 77), **self._refit_kw(fits))
        sps = [f.sp for f in fits_tr]
        scores = stacking.log_scores(sps, train, rows)
        w = stacking.stack_weights(scores)
        cf = stacking.cross_fitted_weights(jax.random.fold_in(key, 78), scores, np.asarray(data.X)[rows])
        g1 = None
        if cf.testable:
            comp = self._pit_components(sps, train, rows)  # (M, n)
            u = np.empty(len(rows))
            u[cf.idx_B] = (np.asarray(cf.w_A)[:, None] * comp[:, cf.idx_B]).sum(axis=0)  # A-weights on B
            u[cf.idx_A] = (np.asarray(cf.w_B)[:, None] * comp[:, cf.idx_A]).sum(axis=0)  # B-weights on A
            zz = np.asarray(ndtri(jnp.asarray(np.clip(u, 1e-12, 1 - 1e-12))))
            g1 = (zz, (u > 0.025) & (u < 0.975), np.asarray(rows))
        return np.asarray(w), f"stacking weights from {len(rows)} held-out level-{finest} runs", g1

    @staticmethod
    def _pit_components(sps, train, rows) -> np.ndarray:
        """P(Y <= y_i) under each structure's pooled predictive (noise included), (M, n)."""
        r = jnp.asarray(np.asarray(rows, dtype=int))
        X, H, run, mask = _arrays(train)
        y = jnp.asarray(train.y, dtype=float)[r]
        out = []
        for sp in sps:
            mean, var = _level_moments(sp.params, sp.z, X, H, run, mask, X[r], H[r], sp.cfg)
            nv = sp.params.noise_var[:, r]
            c = jnp.mean(ndtr((sp.transform.forward(y)[None, :] - mean) / jnp.sqrt(var + nv)), axis=0)
            out.append(np.asarray(c if sp.cfg.increasing else 1.0 - c))
        return np.stack(out)

    # ------------------------------------------------------------------ Step 3: analysis and gates

    def _analyse(self, repair: bool = True) -> _Analysis:
        ck = (len(self.dataset), self._min_level)
        if self._cache is not None and self._cache.key == ck:
            return self._cache
        reuse = None
        for _ in range(MAX_REPAIR + 1):
            an = self._analyse_once(reuse)
            g4 = next((g for g in an.gates if g.name == "G4"), None)
            active = np.unique(an.levels[np.asarray(an.data.mask, dtype=bool)])
            if (
                repair
                and g4 is not None
                and g4.status == "fail"
                and self._removals < MAX_REPAIR
                and len(active) >= 4  # at least 3 levels must remain
            ):
                self._min_level = int(active.min()) + 1
                self._removals += 1
                reuse = None  # refit on the data without the level (warm-started from this fit's draws)
                self._warm = self._last_draws(an.fits)
                self.notes.append(
                    f"G4 failed: coarsest level removed (levels < {self._min_level} are ignored)"
                )
                continue
            break
        an.key = (len(self.dataset), self._min_level)
        self._cache = an
        self._warm = self._last_draws(an.fits)
        return an

    def _refit_kw(self, fits) -> dict:
        """Warm start of a derived refit (hold-out, G4) from the draws of the main fit of this analysis."""
        if not self.WARM_START:
            return {}
        init = {n: types.SimpleNamespace(theta=d) for n, d in self._last_draws(fits).items()}
        return {"init": init, "n_warmup": self._warm_warmup()}

    def _warm_warmup(self) -> int:
        return max(self.WARM_MIN, round(self.WARM_FRACTION * self.settings.n_warmup))

    def _warm_inits(self):
        """{name: last draws} of the latest posterior, or None (cold start): see inference.fit(init=...)."""
        if not self.WARM_START or not self._warm:
            return None
        return {n: types.SimpleNamespace(theta=d) for n, d in self._warm.items()}

    @staticmethod
    def _last_draws(fits) -> dict:
        return {f.name: {k: np.asarray(v)[:, -1:] for k, v in f.post.theta.items()} for f in fits}

    def _noise_fn(self, f: _Fit):
        """Candidate -> (S_k, 1) noise variance of a new row in THIS structure's Lambda units, per draw.

        exp(m_s + b_s . hbar) of each kept draw (the latent field zeta is left at its mean). The acquisition
        needs it per structure because the units follow the transform (identity: physical, log: relative).
        """
        ms = np.asarray(f.post.theta["m_s"]).reshape(-1)[f.idx]
        bs = np.asarray(f.post.theta["b_s"]).reshape(-1, self._k)[f.idx]

        def fn(cand):
            return np.exp(ms + bs @ np.asarray(cand.hbar, dtype=float))[:, None]

        return fn

    def _weighted(self, fits, weights):
        return [
            f.sp._replace(weight=float(w), noise_var=self._noise_fn(f))
            for f, w in zip(fits, weights, strict=True)
        ]

    def _analyse_once(self, reuse: list | None) -> _Analysis:
        data, levels = self._build_data()
        if int(np.sum(data.mask)) < MIN_ROWS:
            raise ValueError("too few output rows to fit")
        Xs = self.problem.sigma_n
        key = self._fit_key(0)
        skipped: list[str] = []
        warm = self._warm_inits()
        kw_main = {} if warm is None else {"init": warm, "n_warmup": self._warm_warmup()}
        plan = self._holdout_plan(data, levels, len(self.settings.h_kernels) * len(self.problem.transforms))
        fits_tr = None
        if reuse is not None:
            fits = reuse
        elif plan[0] is not None and self.PARALLEL_FITS:
            # the main fit and the hold-out fit are independent (both start at the previous analysis' draws)
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_main = ex.submit(self._fit_all, data, key, skipped, **kw_main)
                f_tr = ex.submit(self._fit_all, plan[0], jax.random.fold_in(key, 77), **kw_main)
                fits, fits_tr = f_main.result(), f_tr.result()
        else:
            fits = self._fit_all(data, key, skipped, **kw_main)
        if plan[0] is not None and len(fits) != len(self.settings.h_kernels) * len(self.problem.transforms):
            plan, fits_tr = self._holdout_plan(data, levels, len(fits)), None  # a structure was dropped
        weights, wnote, g1 = self._stacking(data, levels, fits, key, plan, fits_tr)
        structures = self._weighted(fits, weights)
        moments = self._moments(fits, data, Xs)
        draws, w = self._pool(fits, weights, moments, jax.random.fold_in(key, 5))
        m_y = np.asarray(predict.weighted_quantile(draws, w, 0.5))
        sigma = np.asarray(acq.sigma_epi_physical(structures, data, Xs))
        an = _Analysis(
            (len(self.dataset), self._min_level), data, levels, fits, weights, structures, m_y, sigma,
            self._eps(m_y), moments, [], [], wnote, g1, skipped,
        )  # fmt: skip
        an.gates, _ = self._gates(an, key)
        return an

    def _gates(self, an: _Analysis, key):
        s, data, fits = self.settings, an.data, an.fits
        top = int(np.argmax(an.weights))
        mask = np.asarray(data.mask, dtype=bool)
        cens = np.asarray(data.censored, dtype=bool)
        out: list[gates.GateResult] = []

        def safe(name, fn):
            try:
                return fn()
            except Exception as e:  # a gate that cannot run is "not testable" with the error, never "pass"
                return gates.GateResult(name, "not testable", {"error": f"{type(e).__name__}: {e}"})

        def pooled_theta(name, n=1000):
            parts = []
            for f, w in zip(fits, an.weights, strict=True):
                a = np.asarray(f.post.theta[name])
                a = a.reshape((-1,) + a.shape[2:])
                parts.append(a[_thin_idx(len(a), max(1, int(round(n * w))))])
            return np.concatenate(parts)

        # G0
        def g0():
            sp = None
            if s.varying_order:
                for f in fits:
                    if "log_sigma_pi" in f.post.theta:
                        a = np.asarray(f.post.theta["log_sigma_pi"])
                        sp = np.exp(a.reshape((-1,) + a.shape[2:]))
                        break
            return gates.g0_order(pooled_theta("log_p0"), None, sp, prior_sd=self.scales.log_p_sd)

        out.append(safe("G0", g0))
        # G1
        if an.g1 is None:
            out.append(gates.GateResult("G1", "not testable", {"reason": an.weight_note}))
        else:
            an.gate_loc["G1"] = (an.g1[2], np.asarray(an.g1[0]))
            out.append(safe("G1", lambda: gates.g1_level_holdout(an.g1[0], an.g1[1], True)))
        # G2 (highest-weight structure, a subset of draws and runs)
        ft = fits[top]

        def g2():
            rows = np.flatnonzero(mask & ~cens)
            rows = rows[_thin_idx(len(rows), G2_MAX_BLOCKS)]
            blocks = [np.array([r]) for r in rows]
            idx = _thin_idx(int(ft.sp.z.shape[0]), G2_MAX_DRAWS)
            params = jax.tree_util.tree_map(lambda a: a[jnp.asarray(idx)], ft.sp.params)
            res = gates.g2_block_loo(params, data, ft.sp.z[jnp.asarray(idx)], ft.cfg, blocks)
            if "z_marginal" in res.stats:
                an.gate_loc["G2"] = (rows, np.asarray(res.stats["z_marginal"]))
            return res

        out.append(safe("G2", g2))

        # G3
        def g3():
            X, H, y = np.asarray(data.X), np.asarray(data.H), np.asarray(data.y)
            groups: dict[tuple, list[int]] = {}
            for i in np.flatnonzero(mask & ~cens):
                groups.setdefault(tuple(X[i]) + tuple(H[i]), []).append(int(i))
            reps = [g for g in groups.values() if len(g) >= 2]
            zt = np.asarray(ft.tf.forward(jnp.asarray(y)))
            nv = np.asarray(ft.sp.params.noise_var)  # (S, n_pad)
            return gates.g3_noise([zt[g] for g in reps], [float(nv[:, g[0]].mean()) for g in reps])

        out.append(safe("G3", g3))
        # G4 (refit without the coarsest level)
        active = np.unique(an.levels[mask])

        def g4():
            if len(active) < 2:
                return gates.GateResult("G4", "not testable", {"reason": "one level"})
            block = np.flatnonzero(mask & ~cens & (an.levels == int(active.min())))
            idx = jnp.asarray(_thin_idx(int(ft.sp.z.shape[0]), G2_MAX_DRAWS))
            params = jax.tree_util.tree_map(lambda a: a[idx], ft.sp.params)
            return gates.g4_coarsest_level(params, data, ft.sp.z[idx], ft.cfg, block)

        out.append(safe("G4", g4))
        # G5
        out.append(safe("G5", lambda: self._g5(an)))

        # G6 and G7 share physical draws of f(x): one Gaussian sample per posterior draw for G7
        def g6():
            d, w = self._pool(fits, an.weights, an.moments, jax.random.fold_in(key, 6))
            return gates.g6_shape(d, w)

        out.append(safe("G6", g6))
        out.append(safe("G7", lambda: self._g7(an, key)))
        return out, None

    def _g5(self, an: _Analysis):
        if self.settings.monotone is None:
            return gates.g5_monotone(None, None)
        coord, direction = self.settings.monotone
        n_lines, n_pts = 5, 11
        lo, hi = self._region_full()
        inp = self.problem.inputs
        region_lo, region_hi = (
            np.asarray(
                self.problem.region_lower if self.problem.region_lower is not None else inp.lower, float
            ),
            np.asarray(
                self.problem.region_upper if self.problem.region_upper is not None else inp.upper, float
            ),
        )
        grid = (
            region_lo[coord] + np.linspace(0, 1, n_pts) * (region_hi[coord] - region_lo[coord]) - lo[coord]
        ) / (hi[coord] - lo[coord])
        base = self.problem.sigma_n[:n_lines]
        pts = np.repeat(base[:, None, :], n_pts, axis=1)
        pts[:, :, coord] = grid[None, :]
        pts = pts.reshape(-1, base.shape[1])
        mom = self._moments(an.fits, an.data, pts)
        d, w = self._pool(an.fits, an.weights, mom, jax.random.PRNGKey(5))
        med = np.asarray(predict.weighted_quantile(d, w, 0.5)).reshape(n_lines, n_pts)
        tol = 0.1 * float(np.median(an.sigma_epi))
        rs = [gates.g5_monotone(med[i], direction, tol=tol) for i in range(n_lines)]
        bad = sum(r.stats["n_violations"] for r in rs)
        return gates.GateResult("G5", "fail" if bad else "pass", {"n_violations": int(bad), "tol": tol})

    def _g7(self, an: _Analysis, key):
        sc = self.scales
        fits = an.fits
        X, H, run, mask = _arrays(an.data)
        Xs = jnp.asarray(self.problem.sigma_n, dtype=float)
        draws, thetas, ws = [], [], []
        for i, (f, w) in enumerate(zip(fits, an.weights, strict=True)):
            m, v = _mu_moments(f.full_params, f.full_z, X, H, run, mask, Xs, f.cfg)
            z = m + jnp.sqrt(jnp.maximum(v, 0.0)) * jax.random.normal(
                jax.random.fold_in(key, 300 + i), m.shape
            )
            draws.append(np.asarray(f.tf.inverse(z)))
            n = z.shape[0]
            ws.append(np.full(n, float(w) / n))
            th = {}
            for name in ("log_sigma_mu", "log_ell_mu", "c0", "c1", "log_sigma_delta", "log_ell_x", "m_s"):
                a = np.asarray(f.post.theta[name])
                th[name] = a.reshape((-1,) + a.shape[2:])
            thetas.append(th)
        theta = {k: np.concatenate([t[k] for t in thetas]) for k in thetas[0]}
        D, W = np.concatenate(draws), np.concatenate(ws)
        W = W / W.sum()
        fields = {"S_mu": sc.S_mu, "S_c": sc.S_c, "S_delta": sc.S_delta, "S_noise": sc.S_noise}

        def log_ratio(scenario, th):
            name, factor = scenario
            new = dataclasses.replace(sc, **{name: fields[name] * factor})

            def lp(scales, t):
                if name == "S_mu":
                    return priors.pc_matern_logpdf(
                        t["log_sigma_mu"], t["log_ell_mu"], scales.S_mu, scales.ell0
                    )
                if name == "S_c":
                    return priors.normal_logpdf(t["c0"], 0.0, scales.S_c) + priors.normal_logpdf(
                        t["c1"], 0.0, scales.S_c
                    )
                if name == "S_delta":
                    return jnp.sum(
                        jax.vmap(lambda a, b: priors.pc_matern_logpdf(a, b, scales.S_delta, scales.ell0))(
                            t["log_sigma_delta"], t["log_ell_x"]
                        )
                    )
                return priors.normal_logpdf(t["m_s"], 2.0 * math.log(scales.S_noise), math.log(10.0))

            tj = {k: jnp.asarray(v) for k, v in th.items()}
            return jax.vmap(lambda t: lp(new, t) - lp(sc, t))(tj)

        scenarios = [(n, f) for n in fields for f in (0.5, 2.0)]
        return gates.g7_prior(log_ratio, D, W, transforms.get("identity"), scenarios, theta)

    # ------------------------------------------------------------------ report

    def _alloc(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for r in self.dataset.results:
            lev = self._level_of(r.probe.h)
            out[lev] = out.get(lev, 0.0) + float(r.cost)
        return dict(sorted(out.items()))

    def report(self) -> Report:
        if self._report_cache is not None and self._report_cache_key == (
            len(self.dataset),
            self._min_level,
            self.status,
        ):
            return self._report_cache
        notes = list(self.notes)
        if self._n_rows() < MIN_ROWS:
            rep = Report(
                self.status, spent=self.spent, allocation=self._alloc(), notes=notes + ["no data to fit"]
            )
            return rep
        an = self._analyse()
        rep = self._build_report(an, notes)
        self._report_cache, self._report_cache_key = rep, (len(self.dataset), self._min_level, self.status)
        return rep

    def _build_report(self, an: _Analysis, notes: list) -> Report:
        fits, w = an.fits, an.weights
        key = self._fit_key(50)
        top = int(np.argmax(w))
        S0 = self.problem.sigma_n
        data = an.data
        mask = np.asarray(data.mask, dtype=bool)
        cens = np.asarray(data.censored, dtype=bool)
        notes += list(an.skipped)
        if an.weight_note:
            notes.append(an.weight_note)
        # noise at h = 0: exp(m_s) is the variance when hbar = 0 (the latent field is left at its mean)
        sd0 = []
        for f in fits:
            ms = np.asarray(f.post.theta["m_s"]).reshape(-1)[f.idx]
            sd0.append(jnp.asarray(np.broadcast_to(np.exp(0.5 * ms)[:, None], (len(ms), len(S0)))))
        d, ww = self._pool(fits, w, an.moments, jax.random.fold_in(key, 1), extra_var=[s**2 for s in sd0])
        sigma_tot = np.asarray(
            0.5 * (predict.weighted_quantile(d, ww, 0.84) - predict.weighted_quantile(d, ww, 0.16))
        )
        # s0: identified?
        rep_levels = set()
        groups: dict[tuple, list[int]] = {}
        X, H = np.asarray(data.X), np.asarray(data.H)
        lev = an.levels
        for i in np.flatnonzero(mask & ~cens):
            groups.setdefault(tuple(X[i]) + tuple(H[i]), []).append(int(i))
        for g in groups.values():
            if len(g) >= 2:
                rep_levels.add(int(lev[g[0]]))
        bs = np.asarray(fits[top].post.theta["b_s"])
        bs = bs.reshape(-1, bs.shape[-1])
        s0 = None
        if len(rep_levels) < 3:
            notes.append(f"s0 not identified: replicates at {len(rep_levels)} levels (< 3)")
        elif np.any(bs.std(axis=0, ddof=1) > 0.5 * self.scales.b_s_sd):
            notes.append("s0 not identified: the noise trend b_s is prior-dominated")
        else:
            m_y = jnp.asarray(an.m_y)
            centers = [
                jnp.broadcast_to(f.tf.forward(m_y)[None, :], s.shape) for f, s in zip(fits, sd0, strict=True)
            ]
            d0, w0 = self._pool(
                fits, w, an.moments, jax.random.fold_in(key, 2), extra_var=[s**2 for s in sd0], center=centers
            )
            s0 = np.asarray(
                0.5 * (predict.weighted_quantile(d0, w0, 0.84) - predict.weighted_quantile(d0, w0, 0.16))
            )
        # fidelity envelope (highest-weight structure, Lambda units) and posterior RMS error, per level
        levels = [int(x) for x in np.unique(lev[mask])]
        env, fid = {}, {}
        ft = fits[top]
        p = ft.sp.params
        mu_m, mu_v = an.moments[top]
        mu = mu_m + jnp.sqrt(mu_v) * jax.random.normal(jax.random.fold_in(key, 3), mu_m.shape)
        Sd, s = mu.shape
        Ph = jnp.broadcast_to(p.P[:, 0, :][:, None, :], (Sd, s, self._k))
        wt = jnp.ones(Sd) / Sd
        for lv in levels:
            hbar = jnp.asarray(self.problem.resolution.hbar(np.asarray(self._h_of_level(lv))), dtype=float)
            env[lv] = np.asarray(
                predict.sigma_env(
                    p.c0, p.c1, Ph, p.delta.sigma, jnp.ones((Sd, s, self._k)), mu,
                    jnp.broadcast_to(hbar, (s, self._k)), wt,
                )
            )  # fmt: skip
            if lv not in an.fid_cache:
                an.fid_cache[lv] = self._sigma_fid(fits, w, data, S0, hbar, jax.random.fold_in(key, 10 + lv))
            fid[lv] = an.fid_cache[lv]
        # posteriors of the order and the transfer coefficients
        p0 = np.exp(np.concatenate([np.asarray(f.post.theta["log_p0"]).reshape(-1, self._k) for f in fits]))
        q = np.quantile(p0, [0.16, 0.5, 0.84], axis=0)
        notes.append(
            f"order p0: median {np.round(q[1], 3).tolist()}, 16-84% "
            f"{np.round(q[0], 3).tolist()}..{np.round(q[2], 3).tolist()}"
        )
        c0, c1 = np.asarray(ft.sp.params.c0), np.asarray(ft.sp.params.c1)
        notes.append(
            f"c0 median {np.round(np.median(c0, 0), 3).tolist()}, "
            f"c1 median {np.round(np.median(c1, 0), 3).tolist()} ({ft.name})"
        )
        for g in an.gates:
            if "error" in g.stats:
                notes.append(f"{g.name} could not run: {g.stats['error']}")
        failed = self._blocking_fails(an.gates)
        for g in an.gates:
            if g.name == "G6" and g.status == "fail":
                notes.append(
                    f"warning: G6 shape: median deviation of the 2.5/97.5% quantiles from m +- 1.96 s is "
                    f"{g.stats['median_rel_diff']:.2f} s (> 0.2); the posterior is not Gaussian"
                )
        if failed and self.status != "success":
            notes.append(f"gates failed: {failed}; the output is not calibrated")
        ratio = float(np.max(an.sigma_epi / an.eps))
        if self.status == "P2":
            notes.append(f"P2: tolerance not met, max sigma_epi/eps = {ratio:.3g}")
        return Report(
            status=self.status, m_y=an.m_y, sigma_epi=an.sigma_epi, s0=s0, sigma_tot=sigma_tot, sigma_env=env,
            sigma_fid=fid, gates=list(an.gates), weights=np.asarray(w), spent=self.spent,
            allocation=self._alloc(), notes=notes,
        )  # fmt: skip

    def _sigma_fid(self, fits, weights, data, Xs, hbar, key):
        """sigma_fid(x, h) of spec 2.8, pooled: RMS of Lambda^{-1}(f(x,h)) - Lambda^{-1}(f(x))."""
        X, H, run, mask = _arrays(data)
        Xs = jnp.asarray(Xs, dtype=float)
        lv, f0, ws = [], [], []
        for i, (f, w) in enumerate(zip(fits, weights, strict=True)):
            idx = jnp.asarray(_thin_idx(int(f.sp.z.shape[0]), FID_MAX_DRAWS))
            params = jax.tree_util.tree_map(lambda a, idx=idx: a[idx], f.sp.params)
            m0, ml, v0, vl, c = _fid_moments(params, f.sp.z[idx], X, H, run, mask, Xs, hbar, f.cfg)
            k1, k2 = jax.random.split(jax.random.fold_in(key, i))
            e1, e2 = jax.random.normal(k1, m0.shape), jax.random.normal(k2, m0.shape)
            a0 = m0 + jnp.sqrt(jnp.maximum(v0, 0.0)) * e1
            beta = c / jnp.maximum(v0, 1e-300)
            al = ml + beta * (a0 - m0) + jnp.sqrt(jnp.maximum(vl - beta * c, 0.0)) * e2
            lv.append(al)
            f0.append(a0)
            ws.append(np.full(a0.shape[0], float(w) / a0.shape[0]))
        tot = np.zeros(Xs.shape[0])
        norm = float(sum(w_.sum() for w_ in ws))
        for al, a0, f, w_ in zip(lv, f0, fits, ws, strict=True):
            dd = np.asarray(f.tf.inverse(al) - f.tf.inverse(a0))
            tot += (w_[:, None] * dd**2).sum(axis=0)
        return np.sqrt(tot / norm)

    # ------------------------------------------------------------------ Steps 4 and 5: ask

    def _noise_var(self, an: _Analysis, hbar) -> np.ndarray:
        """Candidate noise variance (1,): median over draws of exp(m_s + b_s . hbar) of the top structure."""
        f = an.fits[int(np.argmax(an.weights))]
        ms = np.asarray(f.post.theta["m_s"]).reshape(-1)
        bs = np.asarray(f.post.theta["b_s"]).reshape(-1, self._k)
        return np.array([float(np.median(np.exp(ms + bs @ np.asarray(hbar))))])

    def _site_list(self) -> np.ndarray:
        s = self.settings
        lo, hi = self._region()
        m = max(1, math.ceil(math.log2(max(s.n_candidates_u, 1))))
        sob = qmc.Sobol(self._nc, scramble=True, seed=s.seed + 1).random_base2(m)[: s.n_candidates_u]
        sites = [lo + sob * (hi - lo)] + [np.asarray(r.probe.u, float)[None, :] for r in self.dataset.results]
        allu = np.vstack(sites)
        _, first = np.unique(np.round(allu, 12), axis=0, return_index=True)
        return allu[np.sort(first)]

    def _candidates(self, an: _Analysis, post, quotes):
        sites = self._site_list()
        data_levels = an.levels[np.asarray(an.data.mask, dtype=bool)]
        finest = int(data_levels.max())
        levs = list(range(self._min_level, finest + self.settings.extra_levels + 1))
        probes = [
            Probe("cand", tuple(float(t) for t in u), self._h_of_level(lv), None)
            for lv in levs
            for u in sites
        ]
        lvl = np.array([self._level_of(p.h) for p in probes])
        q = None
        if quotes is not None:
            raw = list(quotes(probes))
            hq = np.array([x is not None and x > 0 for x in raw])
            lq = np.array([math.log2(x) if h else 0.0 for x, h in zip(raw, hq, strict=True)])
            q = (lq, hq)
        mean, cap = self._price(post, [p.u for p in probes], lvl, q)
        lo, hi = self._region_full()
        cands = []
        for p, c, cp_, lv in zip(probes, mean, cap, lvl, strict=True):
            hbar = self.problem.resolution.hbar(np.asarray(p.h))
            xu = self.problem.inputs.to_unit(np.asarray(p.u, float))[None, :]
            cands.append(
                Candidate(
                    xu, hbar, np.full(self._k, int(lv)), float(c), float(cp_), self._noise_var(an, hbar)
                )
            )
        return cands, probes, cap, mean

    def _pending_candidates(self, an: _Analysis) -> list[Candidate]:
        out = []
        for e in self._pending.values():
            p = e["probe"]
            hbar = self.problem.resolution.hbar(np.asarray(p.h))
            xu = self.problem.inputs.to_unit(np.asarray(p.u, float))[None, :]
            out.append(
                Candidate(
                    xu,
                    hbar,
                    np.full(self._k, self._level_of(p.h)),
                    e["cap"],
                    e["cap"],
                    self._noise_var(an, hbar),
                )
            )
        return out

    def plan(self, quotes=None, key=None) -> _Plan:
        """Steps 4-5 without side effects on the budget: candidates, prices, gain table, batch."""
        an = self._analyse()
        post = self._cost_posterior()
        cands, probes, caps, costs = self._candidates(an, post, quotes)
        key = self._next_key() if key is None else key
        k_sel = jax.random.fold_in(key, 1)
        Xs = self.problem.sigma_n
        chosen, table = acq.select_batch(
            k_sel, an.structures, an.data, cands, Xs, an.eps, self.settings.q,
            self.problem.budget - self.spent, self._pending_candidates(an), self.mode,
            self.settings.max_draws_acquisition,
        )  # fmt: skip
        return _Plan(an, cands, probes, caps, costs, table, list(chosen), self.mode)

    def ask(self, quotes=None) -> list[Probe]:
        if self.status != "running":
            return []
        if not self._design_issued:
            return self.initial_design()
        if self._n_rows() < MIN_ROWS:
            if self._pending:
                return []
            self._finalize("P2", "P2: no usable data, the budget did not allow a fit")
            return []
        an = self._analyse()
        ratio = float(np.max(an.sigma_epi / an.eps))
        fails = self._blocking_fails(an.gates)
        self._record("ask", max_ratio=ratio, gates_failed=fails)
        if not self._pending and self._buy_cycles < MAX_REPAIR:
            probe = self._repair_probe(an, [n for n in fails if n in ("G1", "G2")], quotes)
            if probe is not None:
                return [probe]
        if ratio <= 1.0:
            if self._pending:
                self.save()
                return []
            if fails:
                self._finalize(
                    "uncalibrated", f"P1 holds but gates {fails} fail after repair: not calibrated"
                )
            else:
                self._finalize("success", "P1 holds and no gate fails")
            return []
        post = self._cost_posterior()
        cands, probes, caps, costs = self._candidates(an, post, quotes)
        c_rem = self.problem.budget - self.reserved
        adm = np.flatnonzero(caps <= c_rem)
        if adm.size == 0:
            if self._pending:
                self.save()
                return []
            self._finalize(
                "P2", f"budget used: cheapest cap {caps.min():.4g} exceeds the remaining {c_rem:.4g}"
            )
            return []
        key = self._next_key()
        fc = forecast.forecast(
            jax.random.fold_in(key, 0), an.structures, an.data, [cands[i] for i in adm], self.problem.sigma_n,
            an.eps, c_rem,
        )  # fmt: skip
        new_mode = "hinge" if fc.success else "softmax"
        if new_mode != self.mode:
            self.notes.append(
                f"forecast: P1 {'reachable' if fc.success else 'not reachable'} within the remaining budget"
                f"{' (variance-only floor above eps)' if fc.infeasible else ''}: acquisition mode {new_mode}"
            )
        self.mode = new_mode
        if fc.p_exceed > 0.5:
            self.notes.append(f"warning: forecast probability of exceeding the budget is {fc.p_exceed:.2f}")
        chosen = self._select(an, cands, key)
        if not chosen:  # the other criterion may still find a positive gain (Monte Carlo noise in the gains)
            self.mode = "softmax" if self.mode == "hinge" else "hinge"
            chosen = self._select(an, cands, key)
        if not chosen:
            if self._pending:
                self.save()
                return []
            self._finalize("P2", "P2: no admissible candidate has a positive expected gain")
            return []
        out = []
        for i in chosen:
            p = dataclasses.replace(probes[i], probe_id=self._new_id("a"))
            self._pending[p.probe_id] = {"probe": p, "cap": float(caps[i])}
            out.append(p)
        self._record("issued", n_probes=len(out), mode=self.mode)
        self.save()
        return out

    @staticmethod
    def _blocking_fails(gate_list) -> list[str]:
        """Names of failed gates that block "success". G6 (shape) is a warning only (spec Step 3)."""
        return [g.name for g in gate_list if g.status == "fail" and g.name != "G6"]

    def _repair_probe(self, an: _Analysis, names: list[str], quotes) -> Probe | None:
        """Spec Step 3 repair: buy a probe one level finer than any run so far, where |z| is largest.

        z is the standardised residual of the failed G1 (hold-out) or G2 (block LOO) gate; ``gate_loc`` holds
        the data rows and their z. At most MAX_REPAIR such purchases per campaign. Returns the (reserved)
        probe, or None when no failed gate has a location or the probe's cap does not fit the budget.
        """
        best = None
        for name in names:
            loc = an.gate_loc.get(name)
            if loc is None or len(loc[0]) == 0:
                continue
            rows, z = np.asarray(loc[0]), np.abs(np.asarray(loc[1], dtype=float))
            i = int(np.nanargmax(z))
            if best is None or z[i] > best[0]:
                best = (float(z[i]), int(rows[i]), name)
        if best is None:
            return None
        _, row, name = best
        inp = self.problem.inputs
        u = tuple(float(t) for t in inp.from_unit(np.asarray(an.data.X)[row])[: self._nc])
        finest = int(an.levels[np.asarray(an.data.mask, dtype=bool)].max())
        probe = Probe(self._new_id("r"), u, self._h_of_level(finest + 1), None)
        q = None
        if quotes is not None:
            raw = list(quotes([probe]))
            ok = raw[0] is not None and raw[0] > 0
            q = (np.array([math.log2(raw[0]) if ok else 0.0]), np.array([ok]))
        _, cap = self._price(self._cost_posterior(), [probe.u], [finest + 1], q)
        if self.reserved + float(cap[0]) > self.problem.budget:
            return None
        self._buy_cycles += 1
        self._pending[probe.probe_id] = {"probe": probe, "cap": float(cap[0])}
        self.notes.append(
            f"repair {self._buy_cycles}: {name} failed, probe at level {finest + 1} where |z| is largest"
        )
        self._record("repair", gate=name)
        self.save()
        return probe

    def _select(self, an: _Analysis, cands, key) -> list[int]:
        chosen, _ = acq.select_batch(
            jax.random.fold_in(key, 1), an.structures, an.data, cands, self.problem.sigma_n, an.eps,
            self.settings.q, self.problem.budget - self.spent, self._pending_candidates(an), self.mode,
            self.settings.max_draws_acquisition,
        )  # fmt: skip
        return [int(i) for i in chosen]

    def _finalize(self, status: str, note: str) -> None:
        self.status = status
        self.notes.append(note)
        self._report_cache = None
        self._record("finished", status=status)
        self.save()

    def _record(self, event: str, **kw) -> None:
        self.history.append(
            {"event": event, "n_results": len(self.dataset), "spent": self.spent, "mode": self.mode, **kw}
        )

    # ------------------------------------------------------------------ persistence

    def save(self) -> None:
        if self.state_dir is None:
            return
        res = self.dataset.results
        arrays = {}
        meta = []
        for i, r in enumerate(res):
            arrays[f"x_{i}"], arrays[f"y_{i}"], arrays[f"c_{i}"] = (
                np.asarray(r.x, float),
                np.asarray(r.y, float),
                np.asarray(r.censored, bool),
            )
            meta.append(
                {
                    "probe": _probe_to_dict(r.probe),
                    "cost": float(r.cost),
                    "cost_censored": bool(r.cost_censored),
                    "cost_hint": None if r.cost_hint is None else float(r.cost_hint),
                    "replicate_of": r.replicate_of,
                }
            )
        state = {
            "problem": _problem_to_dict(self.problem),
            "scales": dataclasses.asdict(self.scales),
            "cost_prior": dataclasses.asdict(self.cost_prior),
            "settings": dataclasses.asdict(self.settings),
            "spent": self.spent,
            "pending": [
                {"probe": _probe_to_dict(v["probe"]), "cap": v["cap"]} for v in self._pending.values()
            ],
            "counter": self._counter,
            "key": [int(t) for t in self._key],
            "mode": self.mode,
            "status": self.status,
            "history": self.history,
            "notes": self.notes,
            "design_issued": self._design_issued,
            "min_level": self._min_level,
            "removals": self._removals,
            "buy_cycles": self._buy_cycles,
            "results": meta,
        }
        for name, d in self._warm.items():
            for k, v in d.items():
                arrays[f"warm|{name}|{k}"] = np.asarray(v)
        npz_tmp = self.state_dir / "results.tmp.npz"
        json_tmp = self.state_dir / "state.tmp.json"
        np.savez(npz_tmp, **arrays)
        json_tmp.write_text(json.dumps(state))
        os.replace(npz_tmp, self.state_dir / "results.npz")
        os.replace(json_tmp, self.state_dir / "state.json")

    @classmethod
    def load(cls, state_dir) -> Campaign:
        d = Path(state_dir)
        s = json.loads((d / "state.json").read_text())
        c = cls(
            _problem_from_dict(s["problem"]),
            PriorScales(**s["scales"]),
            _cost_prior_from_dict(s["cost_prior"]),
            _settings_from_dict(s["settings"]),
            d,
        )
        z = np.load(d / "results.npz")
        for key in z.files:
            if key.startswith("warm|"):
                _, name, k = key.split("|")
                c._warm.setdefault(name, {})[k] = z[key]
        for i, m in enumerate(s["results"]):
            c.dataset.add(
                RunResult(
                    probe=_probe_from_dict(m["probe"]),
                    x=z[f"x_{i}"],
                    y=z[f"y_{i}"],
                    censored=z[f"c_{i}"],
                    cost=m["cost"],
                    cost_censored=m["cost_censored"],
                    cost_hint=m["cost_hint"],
                    replicate_of=m["replicate_of"],
                )
            )
        c.spent = s["spent"]
        c._pending = {
            p["probe"]["probe_id"]: {"probe": _probe_from_dict(p["probe"]), "cap": p["cap"]}
            for p in s["pending"]
        }
        c._counter = s["counter"]
        c._key = np.asarray(s["key"], dtype=np.uint32)
        c.mode, c.status = s["mode"], s["status"]
        c.history, c.notes = s["history"], s["notes"]
        c._design_issued = s["design_issued"]
        c._min_level, c._removals = s["min_level"], s["removals"]
        c._buy_cycles = s.get("buy_cycles", 0)
        return c


# ----------------------------------------------------------------------------------------------
# run_campaign
# ----------------------------------------------------------------------------------------------


def run_campaign(campaign: Campaign, oracle, max_rounds: int = 1000, poll_sleep: float = 0.0) -> Report:
    """ask -> submit -> poll -> tell until the campaign finishes (or ``max_rounds`` rounds).

    Each round submits what ask() returned, with the campaign's caps, waits for every pending run (an
    asynchronous oracle is polled until its results arrive), tells them, and asks again. The reservation
    spent + pending caps never exceeds the budget, so neither does the spend.
    """
    import time

    quotes = oracle.quote
    probes = campaign.ask(quotes)
    for _ in range(max_rounds):
        if probes:
            oracle.submit(probes, [campaign.cap_of(p) for p in probes])
        while campaign.pending:
            res = oracle.poll()
            if res:
                campaign.tell(res)
            elif poll_sleep:
                time.sleep(poll_sleep)
        if not probes and campaign.status != "running":
            break
        probes = campaign.ask(quotes)
        if not probes and not campaign.pending and campaign.status != "running":
            break
    return campaign.report()


# ----------------------------------------------------------------------------------------------
# The A12 oracle: full MCMC refits on fantasy outcomes
# ----------------------------------------------------------------------------------------------


def _fantasy_outcome(truth, campaign: Campaign, an: _Analysis, cand: Candidate, key, source: str) -> float:
    """One fantasy outcome (physical units) of the probe ``cand``: the posterior predictive of a random draw
    ("model": what Step 5 does) or the truth's f(x, h) plus noise ("truth")."""
    if source == "truth":
        return float(truth.value(np.asarray(cand.Xa), float(cand.hbar[0]))) + truth.noise_sd * float(
            jax.random.normal(jax.random.fold_in(key, 3))
        )
    X, H, run, mask = _arrays(an.data)
    k1, k2, k3 = jax.random.split(key, 3)
    s_i = int(jax.random.choice(k1, len(an.fits), p=jnp.asarray(an.weights)))
    f = an.fits[s_i]
    j = int(jax.random.randint(k2, (), 0, int(f.sp.z.shape[0])))
    p = jax.tree_util.tree_map(lambda a, j=j: a[j], f.sp.params)
    Hr = jnp.asarray(cand.hbar, dtype=float)[None, :]
    Pr = jnp.broadcast_to(p.P[0], Hr.shape)
    xa = jnp.asarray(cand.Xa, dtype=float)
    nv = jnp.asarray(an.structures[s_i].noise_var(cand)[j], dtype=float)
    mean, var = model.predict_level(p, _pd(X, H, run, mask, f.sp.z[j]), f.sp.z[j], f.cfg, xa, Hr, Pr, nv)
    return float(f.tf.inverse(mean[0] + jnp.sqrt(var[0]) * jax.random.normal(k3)))


def refit_H(campaign: Campaign, an: _Analysis, dataset: Dataset, key, warm: bool = False, n_warmup=None):
    """Full MCMC refit of every structure on ``dataset`` with the stacking weights of ``an`` held fixed
    (spec Step 5); returns (H_n of the refitted pool, max over structures of rhat of log p0).

    ``warm``: chains start at the last draws of the base posterior ``an`` (inference.fit(init=...)), with
    ``n_warmup`` iterations of warm-up (default the campaign's)."""
    from gcbml.mcmc import diagnostics as dg

    data2, _ = campaign._build_data(dataset)
    init = [f.post for f in an.fits] if warm else None
    fits2 = campaign._fit_all(data2, key, init=init, n_warmup=n_warmup)
    structs2 = campaign._weighted(fits2, an.weights)
    sig2 = np.asarray(acq.sigma_epi_physical(structs2, data2, campaign.problem.sigma_n))
    rhat = max(float(dg.rhat(np.asarray(f.post.theta["log_p0"])[..., 0])) for f in fits2)
    return float(acq.H_value(sig2, an.eps, campaign.mode)), rhat


def fantasy_dataset(campaign: Campaign, cand: Candidate, y: float, tag: str) -> Dataset:
    inp = campaign.problem.inputs
    u = inp.from_unit(np.asarray(cand.Xa, float))[0]
    h = tuple(float(t) for t in campaign.problem.resolution.h_at(tuple(int(t) for t in cand.levels)))
    probe = Probe(f"fantasy-{tag}", tuple(float(t) for t in u), h, None)
    res = RunResult(
        probe, np.asarray(u, float)[None, :], np.array([y]), np.zeros(1, bool), float(cand.cost_mean)
    )
    return Dataset(list(campaign.dataset.results) + [res])


def oracle_values(
    truth, campaign: Campaign, candidates, n_refit_draws: int, key=None, source: str = "model",
    se_target: float | None = None, n_max: int | None = None, H_base: float | None = None,
    max_h_ratio: float = 10.0, warm: bool = False, paired: bool = False, refit_warmup: int | None = None,
    max_rhat: float | None = None,
):  # fmt: skip
    """See synthetic.a12_oracle_ranking. Imported lazily by it so that campaign does not import synthetic.

    n_refit_draws fantasies per candidate; with ``se_target`` the number grows (doubling, up to ``n_max``)
    until the Monte Carlo s.e. of the mean gain is below se_target x the mean gain. ``H_base`` replaces H
    of the current posterior as the reference (e.g. the mean over refits of the unchanged data).

    A fantasy is DROPPED from the mean (and counted) when (a) a refit has rhat(log p0) > ``max_rhat``
    (default: no filter), or (b) its gain is divergent: |gain| > (max_h_ratio - 1) x |H_ref| on EITHER side
    (a divergent refit of the fantasy gives a huge negative gain, a divergent reference refit a huge positive
    one; seen at H ~ 1e51). The raw gains and rhats of every fantasy are returned. [assumption]

    ``warm``: every refit starts at the base posterior (``refit_warmup`` warm-up iterations).
    ``paired``: the gain of fantasy number j is H(refit of the unchanged data) - H(refit with the fantasy),
    both refits with the SAME key (common random numbers: the Monte Carlo error of the two largely cancels);
    the key of fantasy number j is shared by all candidates, so the reference refit of j is computed once.
    """
    from gcbml.synthetic import OracleRank

    an = campaign._analyse()
    key = jax.random.PRNGKey(0) if key is None else key
    H_now = float(acq.H_value(an.sigma_epi, an.eps, campaign.mode)) if H_base is None else float(H_base)
    out, refs = [], {}
    for i, cand in enumerate(candidates):
        gains, rhats, n_drawn = [], [], 0
        n_target = n_refit_draws
        while True:
            while n_drawn < n_target:
                kf = jax.random.fold_in(key, i * 100000 + n_drawn)
                y = _fantasy_outcome(truth, campaign, an, cand, kf, source)
                ds = fantasy_dataset(campaign, cand, y, f"{i}-{n_drawn}")
                if paired:
                    kj = jax.random.fold_in(key, 888000 + n_drawn)
                    if n_drawn not in refs:
                        refs[n_drawn] = refit_H(campaign, an, campaign.dataset, kj, warm, refit_warmup)
                    H_ref, rh_ref = refs[n_drawn]
                else:
                    kj, H_ref, rh_ref = jax.random.fold_in(kf, 9), H_now, 0.0
                H_after, rh = refit_H(campaign, an, ds, kj, warm, refit_warmup)
                n_drawn += 1
                gains.append(H_ref - H_after)
                rhats.append(max(rh, rh_ref))
            raw, rh_all = np.asarray(gains), np.asarray(rhats)
            ok = np.isfinite(raw) & (np.abs(raw) <= (max_h_ratio - 1.0) * abs(H_now))
            n_div = int(np.sum(~ok))
            if max_rhat is not None:
                n_rhat = int(np.sum(ok & (rh_all > max_rhat)))
                ok &= rh_all <= max_rhat
            else:
                n_rhat = 0
            g = raw[ok]
            se = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("inf")
            rel = se / abs(g.mean()) if len(g) and g.mean() != 0 else float("inf")
            if se_target is None or rel < se_target or n_drawn >= (n_max or n_drawn):
                break
            n_target = min(2 * n_drawn, n_max) if n_max else 2 * n_drawn
        out.append(
            OracleRank(
                i, float(g.mean() / cand.cost_mean), float(g.mean()), se / cand.cost_mean, len(g), rel,
                float(rh_all.max()), tuple(float(x) for x in raw), tuple(float(x) for x in rh_all),
                n_div, n_rhat,
            )
        )  # fmt: skip
    return sorted(out, key=lambda t: -t.value)
