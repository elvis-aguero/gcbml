"""Print measured CPU seconds per level (information only; the cost of ``run`` is the work formula).

Run on the allocation: ``srun --jobid=<job> --overlap -c 4 uv run python -m benchmarks.measure_costs``.
"""

import numpy as np

from benchmarks import ALL_PROBLEMS
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
        row.append(min(bp.measure_cpu(Probe("c", u, h), 0) for _ in range(3)))
    gamma = np.polyfit(np.arange(len(row))[-5:], np.log2(row[-5:]), 1)[0]
    print(f"| {name} | " + " | ".join(f"{c:.1e}" for c in row) + f" | {gamma:.2f} |")
