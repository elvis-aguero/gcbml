"""Issue 3 (random-walk prior): realised-cost coverage of the 0.95 cap vs the prior s_delta, 24 seeds.

usage: exp_cost_walk.py LADDER S_DELTA [N_SEEDS]    LADDER: family | accel | decel | stress
"""

import sys

import numpy as np

sys.path.insert(0, "tests")
import test_cost as tc  # noqa: E402
from gcbml.cost import CostPrior  # noqa: E402

name, sdl = sys.argv[1], float(sys.argv[2])
ns = int(sys.argv[3]) if len(sys.argv) > 3 else 24
if name == "stress":
    prior = CostPrior(float(tc.STRESS_LEV[0]), 3.0, (3.0,), (1.0,), (sdl,))
else:
    prior = CostPrior(0.0, 1.0, (2.5,), (1.5,), (sdl,))
res = []
for sd in range(ns):
    if name == "family":
        lev = tc.prior_family_ladder(np.random.default_rng(sd))
    elif name == "accel":
        lev = np.concatenate([[0.0], np.cumsum([3.0, 3.5, 4.0, 4.5])])
    elif name == "decel":
        lev = np.concatenate([[0.0], np.cumsum([4.0, 3.5, 3.0, 2.5])])
    else:
        lev = tc.STRESS_LEV
    res.append(tc.ladder_coverage(sd, lev, prior))
res = np.array(res)
print(f"{name} s_delta={sdl}: first {res[:, 0].mean():.3f} (min {res[:, 0].min():.2f}) second {res[:, 1].mean():.3f}")
