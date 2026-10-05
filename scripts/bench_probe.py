"""One-shot timing probe: compile + run at n_pad, printing the phases. usage: bench_probe.py N_PAD NS"""

import sys
import time

import jax
import numpy as np

sys.path.insert(0, "tests")
import test_inference as ti  # noqa: E402

from gcbml import inference  # noqa: E402
from gcbml.model import ModelConfig  # noqa: E402

n_pad, ns = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(0)
X, H = ti.design_1d()
z, _ = ti.simulate(rng, ti.TRUTH, X, H, ti.XS)
data, zp = ti.pad(X, H, z, n_pad=n_pad)
for i in range(2):
    t = time.perf_counter()
    post = inference.fit(
        jax.random.key(i), data, zp, zp, ModelConfig(), ti.SCALES, 1, n_warmup=0, n_samples=ns
    )
    jax.block_until_ready(post.params)
    print(
        f"call {i}: {time.perf_counter() - t:.1f}s n_evals={post.n_evals} devices={len(jax.devices())}",
        flush=True,
    )
