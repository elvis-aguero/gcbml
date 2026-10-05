"""Item 1 (speed2): coverage of the true f0 by the full MCMC on many datasets (prior draws and A12 truths).

    PYTHONPATH=tests:scripts:. uv run python scripts/coverage_many.py OUT.json [N_PER_CELL]

Cells: (prior | a12) x (3 | 4 levels), N_PER_CELL datasets each (default 10), ids 100 + k. One MCMC fit per
dataset (500 warm-up, 500 samples, 4 chains); a dataset with rhat >= 1.05 for log p0 or the predictive mean
is kept but flagged. Stores z = (f0 - m) / sigma at the 100 points of Sigma_N, with m and sigma the median and
the half-width (q84 - q16) / 2 of the Gaussian-mixture predictive of mu (exact over the draws), and log p0 of
the truth against the posterior.
"""

import json
import re
import sys
import time

import jax
import numpy as np

import gcbml  # noqa: F401
import quad_vs_mcmc as qv
from gcbml import inference, quadrature
from gcbml.mcmc import diagnostics as dg

out, n_cell = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10
rows, k = [], 0
for kind in ("prior", "a12"):
    for nl in (3, 4):
        for _ in range(n_cell):
            i = 100 + k
            k += 1
            data, zp, Xs, f0, label, _ = qv.make_dataset(i, kind, nl)
            t0 = time.perf_counter()
            post = inference.fit(jax.random.key(i), data, zp, zp, qv.CFG, qv.SCALES, 2, 500, 500, 4)
            mean, var = quadrature.mu_moments(post, data, qv.CFG, Xs)
            w = np.full(len(mean), 1.0 / len(mean))
            m, s, *_ = quadrature.summarize(w, mean, var)
            S = mean.shape[0]
            rm = max(dg.rhat(mean.reshape(4, S // 4, -1)[:, :, j]) for j in range(mean.shape[1]))
            lp = np.asarray(post.theta["log_p0"])
            rp = dg.rhat(lp[..., 0])
            ptrue = float(re.search(r"p=([0-9.]+)", label).group(1))
            z = (f0 - m) / s
            rows.append(dict(id=i, kind=kind, n_levels=nl, label=label, z=z.tolist(), rhat_p=rp, rhat_m=rm,
                             log_p_true=float(np.log(ptrue)), log_p_mean=float(lp.mean()), log_p_sd=float(lp.std()),
                             seconds=time.perf_counter() - t0))  # fmt: skip
            print(f"[{i}] {kind} {nl}: cov {np.mean(abs(z) < 2):.2f} mean z {z.mean():.2f} sd z {z.std():.2f} "
                  f"rhat {max(rp, rm):.3f} {rows[-1]['seconds']:.0f}s", flush=True)
            json.dump(rows, open(out, "w"))
