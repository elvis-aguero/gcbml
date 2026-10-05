"""Diagnose one prior-draw dataset with bad coverage: true parameters against the posterior.

PYTHONPATH=tests:scripts:. uv run python scripts/diag_prior_ds.py ID N_LEVELS
"""

import sys

import jax
import numpy as np
import quad_vs_mcmc as qv

import gcbml  # noqa: F401
from gcbml import inference, quadrature

i, nl = int(sys.argv[1]), int(sys.argv[2])
data, zp, Xs, f0, label, _ = qv.make_dataset(i, "prior", nl)
T = qv.LAST_T
print(label)
for k, v in T.items():
    print(f"  true {k:12s} {np.round(np.asarray(v), 3).tolist()}")
post = inference.fit(jax.random.key(i), data, zp, zp, qv.CFG, qv.SCALES, 2, 500, 500, 4)
th = post.theta
for name in (
    "log_sigma_mu",
    "log_ell_mu",
    "c0",
    "c1",
    "log_p0",
    "log_sigma_delta",
    "log_ell_x",
    "log_ell_h",
    "m_s",
    "b_s",
):
    a = np.asarray(th[name]).reshape(-1, int(np.prod(th[name].shape[2:])) if th[name].ndim > 2 else 1)
    print(f"  post {name:16s} mean {np.round(a.mean(0), 2).tolist()} sd {np.round(a.std(0), 2).tolist()}")
mean, var = quadrature.mu_moments(post, data, qv.CFG, Xs)
w = np.full(len(mean), 1.0 / len(mean))
m, s, *_ = quadrature.summarize(w, mean, var)
print("f0 range", f0.min(), f0.max(), "m range", m.min(), m.max(), "sigma median", np.median(s))
print(
    "y by level:",
    [
        (h, np.round(np.asarray(zp)[np.asarray(data.H)[:, 0] == h][:6], 2).tolist())
        for h in (1.0, 0.5, 0.25, 0.125)
    ],
)
