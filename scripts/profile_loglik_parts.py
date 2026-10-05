"""Time the parts of one model.log_marginal call at a given n_pad (Cholesky, kernels, solves).

PYTHONPATH=tests:scripts:. uv run python scripts/profile_loglik_parts.py N0_PER_CONTROL
"""

import sys
import time

import jax
import jax.numpy as jnp
import quad_vs_mcmc_helpers as h

import gcbml  # noqa: F401
from gcbml import inference, linalg, model
from gcbml.kernels import ard_matern, delta_cov

data, zp = h.big_dataset(int(sys.argv[1]))
cfg = model.ModelConfig()
smp = inference._Sampler(data, zp, zp, cfg, h.SCALES, 2, False)
st = smp.draw_prior(jax.random.key(0))
p = smp.constrain(st)
z = jnp.asarray(zp)
n = data.X.shape[0]
print("n_pad", n)


def bench(name, f, *a, reps=100):
    g = jax.jit(f)
    jax.block_until_ready(g(*a))
    t = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(g(*a))
    print(f"{name:34s} {(time.perf_counter() - t) / reps * 1e3:7.3f} ms")


mask = jnp.asarray(data.mask)
X, H = jnp.asarray(data.X), jnp.asarray(data.H)
bench("log_marginal", lambda p, z: model.log_marginal(p, data, z, cfg), p, z)
bench("covariance (kernels only)", lambda p: model.covariance(p, data, cfg), p)
K = model.covariance(p, data, cfg)
bench("ard_matern X,X", lambda: ard_matern(X, X, p.ell_mu, 2.5))
bench("delta_cov", lambda: delta_cov(X, H, X, H, p.P, p.P, p.delta, "twy2", 2.5, 1.5))
bench("linalg.factor(K)", lambda K: linalg.factor(K, mask).L, K)
bench("jnp cholesky only", lambda K: jnp.linalg.cholesky(K + jnp.eye(n)), K)
F = linalg.factor(K, mask)
bench("solve(F, r)", lambda F, r: linalg.solve(F, r), F, z)
