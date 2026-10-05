"""Submit the A12 test as a SLURM array (one task per truth, 4 cores each, partition batch).

    uv run python benchmarks/a12/submit_a12.py --n 20 --time 06:00:00 --out results/a12 [--dry-run]

Writes the batch script next to the results and runs ``sbatch``. Each task runs
``python -m benchmarks.a12.run_truth --seed $SLURM_ARRAY_TASK_ID``; aggregate with
``python -m benchmarks.a12.aggregate_a12 <out>`` when the array is done. Extra arguments after ``--`` go to
run_truth (MCMC sizes, fantasies).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def script(out: Path, time: str, mem: str, extra: str, n: int, first: int) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=a12
#SBATCH --partition=batch
#SBATCH --array={first}-{first + n - 1}
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={out}/slurm_%a.out
#SBATCH --error={out}/slurm_%a.err
cd {ROOT}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
export OMP_NUM_THREADS=4
uv run python -m benchmarks.a12.run_truth --seed $SLURM_ARRAY_TASK_ID --out {out} {extra}
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--time", required=True, help="wall time per truth, e.g. 06:00:00")
    ap.add_argument("--mem", default="16G")
    ap.add_argument("--out", default="results/a12")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs="*", help="arguments for run_truth")
    args = ap.parse_args(argv)
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    f = out / "a12.sbatch"
    f.write_text(script(out, args.time, args.mem, " ".join(args.extra), args.n, args.first))
    print(f"wrote {f}")
    if not args.dry_run:
        print(subprocess.run(["sbatch", str(f)], capture_output=True, text=True, check=True).stdout.strip())


if __name__ == "__main__":
    main()
