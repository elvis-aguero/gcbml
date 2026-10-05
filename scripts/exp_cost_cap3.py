"""Issue 3: E[c] and 0.95 cap (core-hours) at levels 0-4 with zero, 1-level and 3-level data.

usage: exp_cost_cap3.py VARIANT N_SEEDS     (VARIANT: base | q01 | cubic ; as in exp_cost_cap2.py)
True ladder: log2[0.015, 0.25, 6, 300] (levels 0-3); 10 runs per fitted level, noise sd 0.3 (log2). Medians over seeds.
Prior: k0 = log2 0.015, sd 3; gamma 3 +- 1; q_sd 0.5 (base). Controls u = (0.5, 0.5).
"""

import sys

import jax
import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "tests")
import exp_cost_cap2 as e2  # noqa: E402
from test_cost import FAST, CostPrior, bucket, pad_data  # noqa: E402

from gcbml import cost  # noqa: E402


def one(seed, levels, q_sd, n_per=10, noise=0.3):
    rng = np.random.default_rng(1000 + seed)
    lev = e2.LEV
    if levels:
        L = np.repeat(np.array(levels), n_per).astype(float)[:, None]
        U = rng.random((L.shape[0], 2))
        y = lev[L[:, 0].astype(int)] + 0.8 * (U[:, 0] - 0.5) + noise * rng.standard_normal(L.shape[0])
    else:
        L, U, y = np.zeros((0, 1)), np.zeros((0, 2)), np.zeros(0)
    n = len(y)
    prior = CostPrior(float(lev[0]), 3.0, (3.0,), (1.0,), q_sd=q_sd)
    data = pad_data(U, L, y, np.zeros(n, bool), np.zeros(n), np.zeros(n, bool), bucket(max(n, 1)))
    post = cost.fit_cost(jax.random.key(seed), data, prior, **FAST)
    Ln = np.arange(5.0)[:, None]
    mean, var = cost.predict_log2(post, np.full((5, 2), 0.5), Ln, np.zeros(5), np.zeros(5, bool))
    w = np.full(mean.shape[0], 1.0 / mean.shape[0])
    return np.asarray(cost.expected_cost(mean, var, w)), np.asarray(cost.cost_cap(mean, var, w, 0.95))


if __name__ == "__main__":
    variant, ns = sys.argv[1], int(sys.argv[2])
    q_sd = 0.1 if variant == "q01" else 0.5
    if variant == "cubic":
        e2.patch_cubic(0.1)
    print(f"variant={variant}  true cost levels 0-3: 0.015 0.25 6 300 core-h (median over {ns} seeds)")
    for name, levels in (("zero data", []), ("1 level (0)", [0]), ("3 levels (0-2)", [0, 1, 2])):
        r = [one(s, levels, q_sd) for s in range(ns)]
        ec = np.median([a for a, _ in r], axis=0)
        cap = np.median([b for _, b in r], axis=0)
        print(f"{name:15s} E[c]: " + " ".join(f"{x:10.3g}" for x in ec))
        print(f"{'':15s} cap : " + " ".join(f"{x:10.3g}" for x in cap))
