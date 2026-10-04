"""Print observed orders (against the truth) for the given problem at a few random inputs."""

import sys

import numpy as np

from benchmarks import ALL_PROBLEMS
from benchmarks.base import observed_orders, values_at_levels

name, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
bp = ALL_PROBLEMS[name]()
rng = np.random.default_rng(0)
for z in rng.uniform(0.1, 0.9, size=(4, bp.problem().inputs.d)):
    vals = values_at_levels(bp, z, range(lo, hi + 1))
    err = np.abs(vals - bp.truth_for_controls(z))
    with np.errstate(all="ignore"):
        print(np.round(z, 2), "\n  err", err[:, 0], "\n  p", np.round(observed_orders(err)[:, 0], 3))
