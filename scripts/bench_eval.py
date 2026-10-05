"""Time one jitted log-density evaluation of the sampler at n_pad (single chain). usage: bench_eval.py N_PAD"""

import sys
import time

import jax
import numpy as np

sys.path.insert(0, "tests")
import test_inference as ti  # noqa: E402

from gcbml import inference  # noqa: E402
from gcbml.model import ModelConfig  # noqa: E402

n_pad = int(sys.argv[1])
rng = np.random.default_rng(0)
X, H = ti.design_1d()
z, _ = ti.simulate(rng, ti.TRUTH, X, H, ti.XS)
data, zp = ti.pad(X, H, z, n_pad=n_pad)
smp = inference._Sampler(data, zp, zp, ModelConfig(), ti.SCALES, 1, False)
st = smp.init_chains(jax.random.key(0), 1)
st = jax.tree_util.tree_map(lambda a: a[0], st)
f = jax.jit(smp.log_density)
f(st).block_until_ready()
t = time.perf_counter()
for _ in range(50):
    f(st).block_until_ready()
print(f"n_pad={n_pad} eval {(time.perf_counter() - t) / 50 * 1e3:.2f} ms")
