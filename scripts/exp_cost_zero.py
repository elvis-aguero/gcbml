"""Zero-data E[c], E[min(c, cap)], cap at levels 0-4 for the random-walk prior (gamma 3, s_gamma 1, s_delta 0.5)."""

import jax
import numpy as np

from gcbml import cost
from gcbml.cost import CostData, CostPrior

n_pad = 16
z = np.zeros
data = CostData(z((n_pad, 2)), z((n_pad, 1)), z(n_pad), z(n_pad, bool), z(n_pad), z(n_pad, bool), z(n_pad, bool))
prior = CostPrior(float(np.log2(0.015)), 3.0, (3.0,), (1.0,), (0.5,))
res = []
for seed in range(6):
    post = cost.fit_cost(jax.random.key(seed), data, prior, 150, 150, 4)
    mean, var = cost.predict_log2(post, np.full((5, 2), 0.5), np.arange(5.0)[:, None], z(5), z(5, bool))
    w = np.full(mean.shape[0], 1.0 / mean.shape[0])
    cap = cost.cost_cap(mean, var, w, 0.95)
    res.append((cost.expected_cost(mean, var, w), cost.expected_capped_cost(mean, var, w, cap), cap))
for i, nm in enumerate(["E[c]", "E[min(c,cap)]", "cap"]):
    print(f"{nm:14s}", " ".join(f"{x:10.3g}" for x in np.median([np.asarray(r[i]) for r in res], axis=0)))
