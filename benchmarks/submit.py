"""Build the task manifest and submit the SLURM array (one task per problem x kappa x method x seed).

    uv run python -m benchmarks.submit --mode smoke --dry-run    # write manifest + script, print the command
    uv run python -m benchmarks.submit --mode full --time 12:00:00

Modes. smoke: B1, seed 1, kappa 1.5, every method, fast settings and the fast certificate, results under
results/fast/. full: PROTOCOL Section 3: B1, B2, B4, B5, B6 x kappa {1.5, 2, 4} x methods x seeds 1..20, and
B7 (one budget) x methods x seeds; default settings. Tasks that already have a result are left out, so a
second submission is a resume. The array is throttled to at most 100 concurrent tasks, 4 cores each,
partition batch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from benchmarks import config
from benchmarks.baselines import METHODS
from benchmarks.certificate import RESULTS
from benchmarks.runner import result_path, results_root

MAX_CONCURRENT = 100
SEEDS = tuple(range(1, 21))


def all_tasks(mode: str) -> list[dict]:
    if mode == "smoke":
        return [{"problem": "b1_poisson", "kappa": config.KAPPAS[0], "method": m, "seed": 1} for m in METHODS]
    tasks = []
    for p in (*config.FEASIBLE, config.TRAP):
        kappas = config.KAPPAS if p != config.TRAP else (1.0,)
        for k in kappas:
            for m in METHODS:
                tasks += [{"problem": p, "kappa": k, "method": m, "seed": s} for s in SEEDS]
    return tasks


def pending_tasks(mode: str, root: Path | None = None) -> list[dict]:
    r = results_root(mode == "smoke", root)
    return [
        t
        for t in all_tasks(mode)
        if not result_path(r, t["problem"], t["kappa"], t["method"], t["seed"]).exists()
    ]


def write_manifest(mode: str, root: Path | None = None) -> Path:
    root = root or RESULTS
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"manifest_{mode}.json"
    path.write_text(json.dumps({"mode": mode, "fast": mode == "smoke", "tasks": pending_tasks(mode, root)}))
    return path


def sbatch_script(manifest: Path, n: int, walltime: str, mem: str, name: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time={walltime}
#SBATCH --array=0-{n - 1}%{MAX_CONCURRENT}
#SBATCH --output={manifest.parent}/slurm/%x_%A_%a.out
#SBATCH --error={manifest.parent}/slurm/%x_%A_%a.err

cd {Path(__file__).resolve().parent.parent}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
uv run python -m benchmarks.runner --manifest {manifest} --task-id $SLURM_ARRAY_TASK_ID
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--time", default="12:00:00", help="wall time per task (from a measured run)")
    ap.add_argument("--mem", default="16G")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    manifest = write_manifest(a.mode)
    n = len(json.loads(manifest.read_text())["tasks"])
    if n == 0:
        print("nothing to do: every task has a result")
        return
    (RESULTS / "slurm").mkdir(parents=True, exist_ok=True)
    script = manifest.with_suffix(".sh")
    script.write_text(sbatch_script(manifest, n, a.time, a.mem, f"gcbml-{a.mode}"))
    cmd = ["sbatch", str(script)]
    print(f"{n} tasks; {' '.join(cmd)}")
    if not a.dry_run:
        print(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())


if __name__ == "__main__":
    main()
