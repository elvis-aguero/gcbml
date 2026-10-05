"""Issue 3, candidate fixes: coverage of REALISED costs at the first unprobed level for model variants.

usage: exp_cost_cap2.py VARIANT N_SEEDS
Variants (the model in src is NOT changed; variants patch cost._design/_beta_prior inside this process):
  base      : q_sd = 0.5 (current default)
  q1, q2    : q_sd = 1.0, 2.0
  cubic     : adds a cubic coefficient c ~ N(0, 0.1^2) per level^3 to the mean (q_sd 0.5)
  cubic02   : same with c_sd 0.2
  quadtruth : data from a ladder whose log2 cost is exactly quadratic in level (control; current model)
Ladder: log2[0.015, 0.25, 6, 300]; levels 0-2 fitted, 10 runs per level, noise sd 0.3; level 2 probed, 3 not.
Prints per level: coverage of 200 fresh REALISED costs by the 0.95 cap (mean, min over seeds), median cap/realised,
mean bias of the predictive mean vs the noise-free truth, predictive sd.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "tests")
from test_cost import FAST, CostPrior, bucket, pad_data  # noqa: E402

from gcbml import cost  # noqa: E402

LEV = np.log2([0.015, 0.25, 6.0, 300.0])


def patch_cubic(c_sd):
    orig_design, orig_beta = cost._design, cost._beta_prior

    def design(L, log2q, has_q, qbar):
        A = orig_design(L, log2q, has_q, qbar)
        k = jnp.asarray(L).shape[1]
        return jnp.concatenate([A[:, : 2 + 2 * k - 1 + 0], A[:, 1 : 1 + k] ** 3, A[:, -1:]], axis=1)

    def beta(prior):
        b0, bsd = orig_beta(prior)
        k = len(prior.gamma_mean)
        return jnp.concatenate([b0[:-1], jnp.zeros(k), b0[-1:]]), jnp.concatenate(
            [bsd[:-1], jnp.full(k, c_sd), bsd[-1:]]
        )

    cost._design, cost._beta_prior = design, beta


def run(seed, lev, q_sd, n_per=10, noise=0.3, n_new=200):
    rng = np.random.default_rng(1000 + seed)
    L = np.repeat(np.arange(3), n_per).astype(float)[:, None]
    U = rng.random((L.shape[0], 2))
    y = lev[L[:, 0].astype(int)] + 0.8 * (U[:, 0] - 0.5) + noise * rng.standard_normal(L.shape[0])
    prior = CostPrior(float(lev[0]), 3.0, (3.0,), (1.0,), q_sd=q_sd)
    n = len(y)
    data = pad_data(U, L, y, np.zeros(n, bool), np.zeros(n), np.zeros(n, bool), bucket(n))
    post = cost.fit_cost(jax.random.key(seed), data, prior, **FAST)
    out = {}
    for lv in (2, 3):
        Un = rng.random((n_new, 2))
        clean = lev[lv] + 0.8 * (Un[:, 0] - 0.5)
        real = clean + noise * rng.standard_normal(n_new)
        mean, var = cost.predict_log2(
            post, Un, np.full((n_new, 1), float(lv)), np.zeros(n_new), np.zeros(n_new, bool)
        )
        w = np.full(mean.shape[0], 1.0 / mean.shape[0])
        cap = np.log2(np.asarray(cost.cost_cap(mean, var, w, 0.95)))
        mm = np.asarray(mean)
        out[lv] = dict(
            cov_real=float(np.mean(cap >= real)),
            ratio=float(2 ** (np.median(cap) - np.median(real))),
            bias=float(np.mean(mm.mean(0) - clean)),
            psd=float(np.sqrt(np.mean(np.asarray(var).mean(0) + mm.var(0)))),
        )
    return out


if __name__ == "__main__":
    variant, ns = sys.argv[1], int(sys.argv[2])
    q_sd, lev = 0.5, LEV
    if variant == "q01":
        q_sd = 0.1
    elif variant == "q1":
        q_sd = 1.0
    elif variant == "q2":
        q_sd = 2.0
    elif variant == "cubic":
        patch_cubic(0.1)
    elif variant == "cubic02":
        patch_cubic(0.2)
    elif variant == "quadtruth":
        d1, d2 = LEV[1] - LEV[0], LEV[2] - LEV[1]
        lev = np.concatenate([LEV[:3], [LEV[2] + d2 + (d2 - d1)]])
    res = [run(s, lev, q_sd) for s in range(ns)]
    print(f"variant={variant} seeds={ns}")
    for lv in (2, 3):
        a = {k: np.array([r[lv][k] for r in res]) for k in res[0][lv]}
        print(
            f"  level {lv}: cov_real mean {a['cov_real'].mean():.3f} min {a['cov_real'].min():.2f} "
            f"median {np.median(a['cov_real']):.2f} | cap/realised median {np.median(a['ratio']):.2f} "
            f"| bias mean {a['bias'].mean():+.2f} sd {a['bias'].std():.2f} | psd median {np.median(a['psd']):.2f}"
        )
