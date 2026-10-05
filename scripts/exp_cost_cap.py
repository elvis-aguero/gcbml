"""Experiment for issue 3: why does the 0.95 cap under-cover REALISED costs at an unprobed level?

Ladder log2[0.015, 0.25, 6, 300], levels 0-2 fitted (10 runs per level), noise sd 0.3 (log2), prior as in
tests/test_cost.py::ladder_coverage. For each seed and for levels 2 (probed) and 3 (unprobed) it reports
  cov_real : fraction of 200 fresh REALISED log2 costs below the 0.95 cap
  cov_clean: same for the noise-free cost
  ratio    : median cap / median realised cost
  bias     : mean over draws of predictive mean minus the noise-free truth (log2)
  psd      : predictive sd of the pooled mixture (log2);  s_eta, sigma_w: posterior medians
usage: exp_cost_cap.py N_SEEDS [q_sd]
"""

import sys

import jax
import numpy as np

sys.path.insert(0, "tests")
from test_cost import FAST, CostPrior, bucket, pad_data  # noqa: E402

from gcbml import cost  # noqa: E402

LEV = np.log2([0.015, 0.25, 6.0, 300.0])


def run(seed, q_sd=0.5, n_per=10, noise=0.3, n_new=200, prior_kw=None):
    rng = np.random.default_rng(1000 + seed)
    L = np.repeat(np.arange(3), n_per).astype(float)[:, None]
    U = rng.random((L.shape[0], 2))
    y = LEV[L[:, 0].astype(int)] + 0.8 * (U[:, 0] - 0.5) + noise * rng.standard_normal(L.shape[0])
    prior = CostPrior(float(LEV[0]), 3.0, (3.0,), (1.0,), q_sd=q_sd, **(prior_kw or {}))
    n = len(y)
    data = pad_data(U, L, y, np.zeros(n, bool), np.zeros(n), np.zeros(n, bool), bucket(n))
    post = cost.fit_cost(jax.random.key(seed), data, prior, **FAST)
    out = {"s_eta": float(np.median(post.hyper["s_eta"])), "sigma_w": float(np.median(post.hyper["sigma_w"]))}
    for lv in (2, 3):
        Un = rng.random((n_new, 2))
        clean = LEV[lv] + 0.8 * (Un[:, 0] - 0.5)
        real = clean + noise * rng.standard_normal(n_new)
        mean, var = cost.predict_log2(
            post, Un, np.full((n_new, 1), float(lv)), np.zeros(n_new), np.zeros(n_new, bool)
        )
        w = np.full(mean.shape[0], 1.0 / mean.shape[0])
        cap = np.log2(np.asarray(cost.cost_cap(mean, var, w, 0.95)))
        mm = np.asarray(mean)
        out[lv] = dict(
            cov_real=float(np.mean(cap >= real)),
            cov_clean=float(np.mean(cap >= clean)),
            ratio=float(2 ** (np.median(cap) - np.median(real))),
            bias=float(np.mean(mm.mean(0) - clean)),
            psd=float(np.sqrt(np.mean(np.asarray(var).mean(0) + mm.var(0)))),
        )
    return out


if __name__ == "__main__":
    ns = int(sys.argv[1])
    q_sd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    res = [run(s, q_sd) for s in range(ns)]
    print(f"q_sd={q_sd} seeds={ns}")
    print(
        f"s_eta median {np.median([r['s_eta'] for r in res]):.3f} (truth 0.3); sigma_w median {np.median([r['sigma_w'] for r in res]):.3f}"
    )
    for lv in (2, 3):
        a = {k: np.array([r[lv][k] for r in res]) for k in res[0][lv]}
        print(
            f"level {lv}: cov_real mean {a['cov_real'].mean():.3f} (min {a['cov_real'].min():.2f}) "
            f"cov_clean {a['cov_clean'].mean():.3f} ratio median {np.median(a['ratio']):.2f} "
            f"bias mean {a['bias'].mean():+.2f} sd {a['bias'].std():.2f} psd median {np.median(a['psd']):.2f}"
        )
