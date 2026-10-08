"""Cost of value vs value-and-gradient of the theta log density at n_pad = 240 (single thread), and of its parts.

PYTHONPATH=tests:scripts:. uv run python scripts/profile_grad.py [N0_PER_CONTROL]
"""

import sys
import time

import jax
import jax.numpy as jnp
import quad_vs_mcmc_helpers as h

import gcbml  # noqa: F401
from gcbml import inference, linalg, model

data, zp = h.big_dataset(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
cfg = model.ModelConfig()
smp = inference._Sampler(data, zp, zp, cfg, h.SCALES, 2, False)
st = smp.draw_prior(jax.random.key(0))
pi = smp._no_pi()
n = data.X.shape[0]
print("n_pad", n)


def bench(name, f, *a, reps=100):
    g = jax.jit(f)
    jax.block_until_ready(g(*a))
    t = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(g(*a))
    dt = (time.perf_counter() - t) / reps * 1e3
    print(f"{name:34s} {dt:7.3f} ms")
    return dt


ld = lambda v: smp._theta_logdensity(v, st["hz"], st["zeta"], pi, st["z"])  # noqa: E731
v0 = st["theta"]
a = bench("theta logdensity", ld, v0)
b = bench("value_and_grad (current JVP)", jax.value_and_grad(ld), v0)
print(f"ratio {b / a:.1f}")
p = smp.constrain(st)
K = model.covariance(p, data, cfg)
mask = jnp.asarray(data.mask)
bench("factor(K)", lambda K: linalg.factor(K, mask).L, K)
bench("grad of sum(factor(K).L)", jax.grad(lambda K: jnp.sum(linalg.factor(K, mask).L)), K)
bench("jnp cholesky", lambda K: jnp.linalg.cholesky(K + jnp.eye(n)), K)
bench("grad of sum(cholesky)", jax.grad(lambda K: jnp.sum(jnp.linalg.cholesky(K + jnp.eye(n)))), K)
