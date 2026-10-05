"""Solvability certificate (PROTOCOL Section 1) and the budgets C = kappa C*.

For a feasible problem: run every design of the static family (n0 in {8, 16, 32} x n_controls, top level
L in 2..7, nested Sobol sites, n_l = max(3, n0 / 2^l) at level l) once with a fixed oracle seed, order the
designs by their realised cost, and fit gcbml (Campaign.tell + report: both h kernels, both transforms,
stacked) to them from the cheapest up. The certificate is the first design with
    max over Sigma_N of sigma_epi / eps <= 1  AND  |m_y - truth| <= 2 sigma_epi at >= 95% of Sigma_N.
Its cost is C*. Every evaluated design is written to the JSON as it is fitted (the run can be resumed).
One fit serves every tolerance: the file stores max sigma_epi / |m_y|, and ``derive`` applies eps afterwards.

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
from benchmarks.oracle import BenchmarkOracle

CERT_SEED = 0
COVER_FRACTION = 0.95
RESULTS = Path(__file__).parent / "results"


def cert_path(name: str, fast: bool = False, root: Path | None = None) -> Path:
    return (root or RESULTS) / "certificates" / f"{name}{'_fast' if fast else ''}.json"


def evaluate_design(setup: Setup, results, settings) -> dict:
    """Fit gcbml to the runs of one design; keep what does not depend on the tolerance.

    max_rel_sigma = max over Sigma_N of sigma_epi / |m_y|, so max sigma_epi / eps = max_rel_sigma / rel_tol
    for any relative tolerance: one fit answers the question for every eps.
    """
    t0 = time.time()
    rep = fit_static(setup, results, settings)
    z = setup.problem().sigma_n
    truth = setup.truth(z)
    m, sig = np.asarray(rep.m_y), np.asarray(rep.sigma_epi)
    err = np.abs(m - truth)
    return {
        "cost": float(sum(r.cost for r in results)),
        "n_runs": len(results),
        "max_rel_sigma": float(np.max(sig / np.abs(m))),
        "max_rel_error": float(np.max(err / np.abs(truth))),
        "inside2": float(np.mean(err <= 2.0 * sig)),
        "coverage95": float(np.mean(err <= 1.96 * sig)),
        "fit_seconds": time.time() - t0,
        "gates": {g.name: g.status for g in rep.gates},
        "status": rep.status,
    }


def passes(e: dict, rel_tol: float) -> bool:
    """PROTOCOL Section 1: max sigma_epi / eps <= 1 and the truth within 2 sigma_epi at >= 95% of Sigma_N."""
    return bool(e["max_rel_sigma"] / rel_tol <= 1.0 and e["inside2"] >= COVER_FRACTION)


def derive(state: dict, setup: Setup) -> dict | None:
    """The certificate at setup.rel_tol from the evaluated designs: the cheapest passing one.

    ``verified`` is True when every design cheaper than it was evaluated (and failed), so that it is the
    cheapest of the whole family; False when the search skipped some cheaper designs (a scan).
    """
    done = {tuple(e["design"]): e for e in state["evaluated"]}
    skipped = False
    for key in state["order"]:
        e = done.get(tuple(key))
        if e is None:
            skipped = True
            continue
        if passes(e, setup.rel_tol):
            a, top = key
            return {
                "design": {"n0_per_control": a, "n0": a * setup.n_controls, "top_level": top},
                "rel_tol": setup.rel_tol, "C_star": e["cost"],
                "max_sigma_over_eps": e["max_rel_sigma"] / setup.rel_tol,
                "coverage95": e["coverage95"], "inside2": e["inside2"], "verified_cheapest": not skipped,
            }  # fmt: skip
    return None


def certify(setup: Setup, settings, fast: bool = False, root=None, log=print, stride: int = 0) -> dict:
    """Evaluate designs in cost order and derive the certificate at setup.rel_tol.

    stride = 0: from the cheapest up, stop at the first design that passes (the exact definition).
    stride = k > 0: a scan of every k-th design of the cost order (and the dearest), all of them, no stop;
    the certificate is then the cheapest passing design among those (``verified_cheapest`` False if
    cheaper ones were skipped). Designs already in the file are not fitted again.
    """
    name = setup.name
    path = cert_path(name, fast, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(path.read_text()) if path.exists() else {}
    if state.get("fast") != fast or "order" not in state:
        state = {"problem": name, "fast": fast, "seed": CERT_SEED, "evaluated": []}
    done = {tuple(e["design"]): e for e in state["evaluated"]}
    oracle = BenchmarkOracle(setup.bp, seed=CERT_SEED, y_scale=setup.y_scale)
    runs = []
    for key, probes in family_designs(setup, CERT_SEED):
        res = run_all(oracle, probes)
        runs.append((sum(r.cost for r in res), key, res))
    runs.sort(key=lambda t: t[0])
    state["order"] = [list(k) for _, k, _ in runs]
    chosen = range(len(runs)) if stride <= 0 else sorted({*range(0, len(runs), stride), len(runs) - 1})
    for i in chosen:
        cost, key, res = runs[i]
        e = done.get(key)
        if e is None:
            e = {"design": list(key), **evaluate_design(setup, res, settings)}
            state["evaluated"].append(e)
            path.write_text(json.dumps(state, indent=1))
            log(
                f"{name} design {key}: cost {cost:.4g} max sigma/|m_y| {e['max_rel_sigma']:.3g} "
                f"inside2 {e['inside2']:.3f} ({e['fit_seconds']:.0f} s)"
            )
        if stride <= 0 and passes(e, setup.rel_tol):
            break
    state["certificate"] = derive(state, setup)
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
    setup = setup or get_setup(name)
    state = json.loads(cert_path(name, fast, root).read_text())
    cert = derive(state, setup)
    if cert is None:
        raise RuntimeError(f"{name}: no certificate at eps = {setup.rel_tol}; re-tune eps")
    return float(cert["C_star"])


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
    ap.add_argument("--stride", type=int, default=0, help="scan every k-th design of the cost order")
    a = ap.parse_args(argv)
    settings = config.fast_settings() if a.fast else config.default_settings()
    st = certify(get_setup(a.problem, a.rel_tol), settings, a.fast, stride=a.stride)
    print(json.dumps(st["certificate"], indent=1))


if __name__ == "__main__":
    main()
