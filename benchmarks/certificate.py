"""Solvability certificate (PROTOCOL Section 1) and the budgets C = kappa C*.

For a feasible problem: run every design of the static family (n0 in {8, 16, 32} x n_controls, top level
L in 2..7, nested Sobol sites, n_l = max(3, n0 / 2^l) at level l) once with a fixed oracle seed, order the
designs by their realised cost, and fit gcbml (Campaign.tell + report: both h kernels, both transforms,
stacked) to them from the cheapest up. The certificate is the first design with
    max over Sigma_N of sigma_epi / eps <= 1  AND  |m_y - truth| <= 2 sigma_epi at >= 95% of Sigma_N.
Its cost is C*. Every evaluated design is written to the JSON as it is fitted (the run can be resumed).

B7 has no certificate by construction: its budget is 0.5 x the realised cost of the design with n0 = 8
n_controls whose finest level is the asymptotic level of the problem (``asymptotic_level``).

    uv run python -m benchmarks.certificate --problem b1_poisson [--fast] [--rel-tol 0.01]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmarks import config
from benchmarks.config import Setup, get_setup
from benchmarks.design import family_designs, fit_static, run_all, static_design
from benchmarks.metrics import score
from benchmarks.oracle import BenchmarkOracle

CERT_SEED = 0
RESULTS = Path(__file__).parent / "results"


def cert_path(name: str, fast: bool = False, root: Path | None = None) -> Path:
    return (root or RESULTS) / "certificates" / f"{name}{'_fast' if fast else ''}.json"


def evaluate_design(setup: Setup, results, settings) -> dict:
    """Fit gcbml to the runs of one design and score it against the truth on Sigma_N."""
    t0 = time.time()
    rep = fit_static(setup, results, settings)
    z = setup.problem().sigma_n
    sc = score(rep.m_y, rep.sigma_epi, setup.truth(z), setup.rel_tol)
    sc.pop("z")
    sc["cost"] = float(sum(r.cost for r in results))
    sc["n_runs"] = len(results)
    sc["passed"] = bool(sc["success"])
    sc["fit_seconds"] = time.time() - t0
    sc["gates"] = {g.name: g.status for g in rep.gates}
    return sc


def certify(setup: Setup, settings, fast: bool = False, root=None, log=print) -> dict:
    name = setup.name
    path = cert_path(name, fast, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(path.read_text()) if path.exists() else {}
    if state.get("rel_tol") != setup.rel_tol or state.get("fast") != fast:
        state = {"problem": name, "rel_tol": setup.rel_tol, "fast": fast, "seed": CERT_SEED, "evaluated": []}
    done = {tuple(e["design"]): e for e in state["evaluated"]}
    oracle = BenchmarkOracle(setup.bp, seed=CERT_SEED, y_scale=setup.y_scale)
    runs = []
    for key, probes in family_designs(setup, CERT_SEED):
        res = run_all(oracle, probes)
        runs.append((sum(r.cost for r in res), key, res))
    runs.sort(key=lambda t: t[0])
    state["certificate"] = None
    for cost, key, res in runs:
        e = done.get(key)
        if e is None:
            e = {"design": list(key), **evaluate_design(setup, res, settings)}
            state["evaluated"].append(e)
            path.write_text(json.dumps(state, indent=1))
            log(
                f"{name} design {key}: cost {cost:.4g} max sigma/eps {e['max_sigma_over_eps']:.3g} "
                f"inside2 {e['inside2']:.3f} passed {e['passed']} ({e['fit_seconds']:.0f} s)"
            )
        if e["passed"]:
            a, top = key
            state["certificate"] = {
                "design": {"n0_per_control": a, "n0": a * setup.n_controls, "top_level": top},
                "C_star": e["cost"],
                "max_sigma_over_eps": e["max_sigma_over_eps"],
                "coverage95": e["coverage95"],
                "inside2": e["inside2"],
            }
            break
    path.write_text(json.dumps(state, indent=1))
    return state


def trap_budget(setup: Setup, seed: int = CERT_SEED) -> dict:
    """B7: half the realised cost of the cheapest-n0 design whose finest level is the asymptotic level."""
    top = int(setup.bp.asymptotic_level)
    probes = static_design(setup, config.N0_PER_CONTROL[0] * setup.n_controls, top, seed, tag="trap")
    res = run_all(BenchmarkOracle(setup.bp, seed=seed, y_scale=setup.y_scale), probes)
    full = float(sum(r.cost for r in res))
    return {"problem": setup.name, "top_level": top, "design_cost": full, "budget": 0.5 * full}


def reference_cost(name: str, fast: bool = False, root=None, setup: Setup | None = None) -> float:
    """C* of a feasible problem (from its certificate file) or the cost of the B7 reference design."""
    if name == config.TRAP:
        return trap_budget(setup or get_setup(name))["design_cost"]
    state = json.loads(cert_path(name, fast, root).read_text())
    if state.get("certificate") is None:
        raise RuntimeError(f"{name}: no certificate at eps = {state['rel_tol']}; re-tune eps")
    return float(state["certificate"]["C_star"])


def budget_for(name: str, kappa: float, fast: bool = False, root=None, setup: Setup | None = None) -> float:
    """Budget of one (problem, kappa) cell: kappa C* (B7: 0.5 x the asymptotic-level design, any kappa)."""
    if name == config.TRAP:
        return trap_budget(setup or get_setup(name))["budget"]
    return kappa * reference_cost(name, fast, root, setup)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--rel-tol", type=float, default=None)
    a = ap.parse_args(argv)
    settings = config.fast_settings() if a.fast else config.default_settings()
    st = certify(get_setup(a.problem, a.rel_tol), settings, a.fast)
    print(json.dumps(st["certificate"], indent=1))
    np.set_printoptions(precision=4)


if __name__ == "__main__":
    main()
