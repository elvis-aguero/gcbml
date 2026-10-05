"""Static designs, their cost, and the gcbml fit of a fixed data set (shared by certificate and baselines).

Static design (PROTOCOL Section 1): nested scrambled-Sobol sites in the region, n_l = max(3, n0 // 2^l) at
levels 0..L (the sites of level l are the first n_l of level 0). Level l means level l in every resolution
component. ``expected_cost`` is the noise-free work times exp(sigma_c^2 / 2), the mean of the lognormal
cost: what a user who knows the cost law plans with.

``fit_static`` fits gcbml to given results with the public Campaign API (tell, then report): the full
default analysis (both h kernels, both transforms, stacking, gates), nothing is acquired.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import qmc

from benchmarks.config import Setup
from benchmarks.oracle import BenchmarkOracle
from gcbml.campaign import Campaign, CampaignSettings, Report
from gcbml.data import Probe, RunResult


def sobol_sites(setup: Setup, n: int, seed: int) -> np.ndarray:
    """First n points of a scrambled Sobol set in the region (physical units), shape (n, n_controls)."""
    p = setup.problem()
    nc = p.inputs.n_controls
    m = max(1, math.ceil(math.log2(n)))
    unit = qmc.Sobol(nc, scramble=True, seed=seed).random_base2(m)[:n]
    lo, hi = np.asarray(p.inputs.lower)[:nc], np.asarray(p.inputs.upper)[:nc]
    return lo + unit * (hi - lo)


def h_of_level(setup: Setup, level: int) -> tuple[float, ...]:
    p = setup.problem()
    return tuple(float(t) for t in p.resolution.h_at((level,) * p.resolution.k))


def make_probe(setup: Setup, pid: str, u: Sequence[float], level: int) -> Probe:
    return Probe(pid, tuple(float(t) for t in u), h_of_level(setup, level))


def static_design(setup: Setup, n0: int, top: int, seed: int, tag: str = "s") -> list[Probe]:
    """Nested design: n_l = max(3, n0 // 2^l) sites at level l, l = 0..top."""
    U = sobol_sites(setup, n0, seed)
    out = []
    for lev in range(top + 1):
        for i in range(min(n0, max(3, n0 // 2**lev))):
            out.append(make_probe(setup, f"{tag}-l{lev}-s{i}", U[i], lev))
    return out


def level_of(setup: Setup, probe: Probe) -> int:
    p = setup.problem()
    comp = p.resolution.components[0]
    return next(lev for lev in range(60) if math.isclose(comp.level_value(lev), probe.h[0], rel_tol=1e-9))


def expected_cost(setup: Setup, probes: Sequence[Probe]) -> float:
    bp = setup.bp
    return float(sum(bp.work(p) for p in probes) * math.exp(0.5 * bp.sigma_c**2))


def family() -> list[tuple[int, int]]:
    """(n0 per control, top level) of the oracle static design family."""
    from benchmarks.config import LEVELS, N0_PER_CONTROL

    return [(a, top) for a in N0_PER_CONTROL for top in LEVELS]


def family_designs(setup: Setup, seed: int = 0) -> list[tuple[tuple[int, int], list[Probe]]]:
    """Every design of the family, cheapest (expected cost) first."""
    out = [
        ((a, top), static_design(setup, a * setup.n_controls, top, seed, tag=f"f{a}_{top}"))
        for a, top in family()
    ]
    return sorted(out, key=lambda t: expected_cost(setup, t[1]))


def run_all(oracle: BenchmarkOracle, probes: Sequence[Probe], budget: float = math.inf) -> list[RunResult]:
    """Run probes in order; each is capped at the budget that remains, so the total never exceeds it."""
    results, spent = [], 0.0
    for p in probes:
        rem = budget - spent
        if rem <= 0:
            break
        r = oracle.run(p, rem)
        spent += r.cost
        results.append(r)
    return results


def fit_static(setup: Setup, results: Sequence[RunResult], settings: CampaignSettings) -> Report:
    """gcbml posterior of the given runs (spec Steps 2-3), through the public Campaign API."""
    spent = sum(r.cost for r in results)
    camp = Campaign(setup.problem(max(spent, 1.0)), setup.scales, setup.cost_prior, settings)
    camp.tell(list(results))
    return camp.report()
