"""Run one benchmark task: one (problem, kappa, method, seed). One SLURM array task = one call.

    uv run python -m benchmarks.runner --problem b1_poisson --kappa 2 --method gcbml --seed 1
    uv run python -m benchmarks.runner --manifest results/manifest_full.json --task-id 17

Result: benchmarks/results/<problem>/k<kappa>/<method>/seed<seed>.json (skipped when it exists, so a
resubmitted array only does the missing tasks). Failures write <seed>.error.txt and no result.
The budget is kappa x C* from the certificate (B7: from its asymptotic-level design), in work units.
``--fast`` uses short chains and the fast certificate (smoke runs, tests), written under results/fast/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from benchmarks import config
from benchmarks.certificate import RESULTS, budget_for, reference_cost
from benchmarks.config import get_setup

KEY_FIELDS = ("problem", "kappa", "method", "seed")


def kappa_label(kappa: float) -> str:
    return f"{kappa:g}"


def result_path(root: Path, problem: str, kappa: float, method: str, seed: int) -> Path:
    return root / problem / f"k{kappa_label(kappa)}" / method / f"seed{seed}.json"


def results_root(fast: bool, root: Path | None = None) -> Path:
    return (root or RESULTS) / ("fast" if fast else "full")


def enable_jax_cache() -> None:
    """Share compiled programs between tasks (the first compile of a fit is minutes)."""
    import jax

    d = os.environ.get("GCBML_JAX_CACHE", str(RESULTS / "jax_cache"))
    Path(d).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", d)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    return o


def run_task(
    problem: str,
    kappa: float,
    method: str,
    seed: int,
    fast: bool = False,
    root: Path | None = None,
    cert_root: Path | None = None,
    setup=None,
) -> Path:
    """Run and write one task; return the result path (an existing result is not recomputed)."""
    from benchmarks.baselines import run_method
    from benchmarks.metrics import score

    out = result_path(results_root(fast, root), problem, kappa, method, seed)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    setup = setup or get_setup(problem)
    settings = config.fast_settings() if fast else config.default_settings()
    cert = cert_root if cert_root is not None else root
    budget = budget_for(problem, kappa, fast, cert, setup)
    c_star = reference_cost(problem, fast, cert, setup)
    t0 = time.time()
    try:
        oc = run_method(method, setup, budget, seed, settings)
    except Exception:
        out.with_suffix(".error.txt").write_text(traceback.format_exc())
        raise
    wall = time.time() - t0
    truth = setup.truth(setup.problem().sigma_n)
    sc = score(oc.m_y, oc.sigma_epi, truth, setup.rel_tol)
    rec = {
        "problem": problem,
        "kappa": kappa,
        "method": method,
        "seed": seed,
        "fast": fast,
        "rel_tol": setup.rel_tol,
        "budget": budget,
        "C_star": c_star,
        "cost_used": oc.spent,
        "cost_over_Cstar": oc.spent / c_star,
        "wall_seconds": wall,
        "allocation": oc.allocation,
        "gates": oc.gates,
        "status": oc.status,
        "n_capped": oc.n_capped,
        "notes": oc.notes[-12:],
        "extra": oc.extra,
        **sc,
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(_jsonable(rec)))
    tmp.rename(out)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem")
    ap.add_argument("--kappa", type=float)
    ap.add_argument("--method")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--manifest")
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    enable_jax_cache()
    if a.manifest:
        m = json.loads(Path(a.manifest).read_text())
        t = m["tasks"][a.task_id]
        fast = m["fast"]
    else:
        t = {k: getattr(a, k) for k in KEY_FIELDS}
        fast = a.fast
    p = run_task(t["problem"], t["kappa"], t["method"], t["seed"], fast)
    print(p)


if __name__ == "__main__":
    main()
