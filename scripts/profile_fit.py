"""Cost per likelihood evaluation inside inference.fit against a bare jitted log_marginal call.

PYTHONPATH=tests:. uv run python scripts/profile_fit.py N0_PER_CONTROL [ITER]
"""

import sys
import time

import jax
import quad_vs_mcmc_helpers as h  # noqa: E402

import gcbml  # noqa: F401
from gcbml import inference, model

n0, iters = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 20
data, zp = h.big_dataset(n0)
print("n real", data.n, "n_pad", data.X.shape[0])
cfg = model.ModelConfig()
t = time.perf_counter()
post = inference.fit(jax.random.key(0), data, zp, zp, cfg, h.SCALES, 2, iters, iters, 4)
jax.block_until_ready(post.params)
t1 = time.perf_counter() - t
t = time.perf_counter()
post = inference.fit(jax.random.key(1), data, zp, zp, cfg, h.SCALES, 2, iters, iters, 4)
jax.block_until_ready(post.params)
t2 = time.perf_counter() - t
print(
    f"fit cold {t1:.1f}s warm {t2:.1f}s, n_evals {post.n_evals}, per eval (sum over chains) {t2 / post.n_evals * 1e3:.2f} ms,"
    f" per iteration per chain {t2 / (2 * iters) * 1e3:.1f} ms; evals/iter/chain {post.n_evals / (4 * 2 * iters):.0f}"
)
p = jax.tree_util.tree_map(lambda a: a[0, 0], post.params)
z = post.z[0, 0]
f = jax.jit(lambda p, z: model.log_marginal(p, data, z, cfg))
f(p, z).block_until_ready()
t = time.perf_counter()
for _ in range(50):
    f(p, z).block_until_ready()
print(f"bare log_marginal: {(time.perf_counter() - t) / 50 * 1e3:.2f} ms per call")
