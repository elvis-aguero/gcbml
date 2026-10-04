"""Print a markdown table of the CPU cost (seconds) per level, at the centre of each design region.

Run on the allocation: ``srun --jobid=<job> --overlap -c 4 uv run python -m benchmarks.measure_costs``.
"""

import numpy as np

from benchmarks import ALL_PROBLEMS
from benchmarks.base import values_at_levels  # noqa: F401  (keeps the import graph honest)
from gcbml.data import Probe

MAX_LEVEL = {"b1_poisson": 7, "b3_mixing": 8, "b6_heat": 8}

for name, cls in ALL_PROBLEMS.items():
    bp = cls()
    p = bp.problem()
    z = np.full(p.inputs.d, 0.5)
    u = tuple(float(t) for t in p.inputs.from_unit(z)[: p.inputs.n_controls])
    row = []
    for lev in range(MAX_LEVEL.get(name, 9) + 1):
        h = tuple(float(t) for t in p.resolution.h_at((lev,) * p.resolution.k))
        row.append(min(bp.run(Probe("c", u, h), 0).cost for _ in range(3)))
    gamma = np.polyfit(np.arange(len(row))[-5:], np.log2(row[-5:]), 1)[0]
    print(f"| {name} | " + " | ".join(f"{c:.1e}" for c in row) + f" | {gamma:.2f} |")
