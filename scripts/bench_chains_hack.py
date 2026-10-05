"""bench_chains with the affinity trick: create the XLA CPU client while pinned to one cpu, then restore."""

import os
import runpy
import sys

full = os.sched_getaffinity(0)
os.sched_setaffinity(0, {min(full)})
import jax  # noqa: E402

jax.devices()
os.sched_setaffinity(0, full)
sys.argv = ["bench_chains.py"] + sys.argv[1:]
runpy.run_path("scripts/bench_chains.py", run_name="__main__")
