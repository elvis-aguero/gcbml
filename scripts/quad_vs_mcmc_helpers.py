"""Shared dataset builders of the profiling scripts (A12 data, d = 2, levels 1, 1/2, 1/4)."""

import numpy as np
import test_inference as ti

from gcbml.priors import PriorScales
from gcbml.synthetic import A12Truth

SCALES = PriorScales(S_mu=1.0, S_c=0.5, S_delta=0.5, S_noise=0.02)


def big_dataset(n0_per_control, seed=3):
    """3 levels, n0 = 2 n0_per_control sites at level 0, half at level 1, a quarter at level 2, replicates."""
    from scipy.stats import qmc

    n0 = 2 * n0_per_control
    m = int(np.ceil(np.log2(n0)))
    sob = qmc.Sobol(2, scramble=True, seed=seed).random_base2(m)[:n0]
    t = A12Truth(seed, d=2)
    rng = np.random.default_rng(seed)
    X, H = [], []
    for lev, n in enumerate([n0, n0 // 2, n0 // 4]):
        u = sob[:n]
        r = np.repeat(u[:3], 2, axis=0)
        X += [u, r]
        H += [np.full(len(u) + len(r), 0.5**lev)]
    X = np.vstack(X)
    H = np.concatenate(H)[:, None]
    y = np.array(
        [t.value(x[None, :], hh) for x, hh in zip(X, H[:, 0], strict=True)]
    ) + 0.01 * rng.standard_normal(len(X))
    return ti.pad(X, H, y)
