"""Time init and single-chain Gibbs steps (jit, one chain). usage: bench_step.py N_PAD"""

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
init = jax.jit(smp.init_state)
t = time.perf_counter()
st, n = init(jax.random.key(0))
jax.block_until_ready(st)
print(f"init compile+run {time.perf_counter() - t:.1f}s evals={int(n)}", flush=True)
t = time.perf_counter()
st, n = init(jax.random.key(1))
jax.block_until_ready(st)
print(f"init run {time.perf_counter() - t:.2f}s evals={int(n)}", flush=True)
step = jax.jit(smp.make_step(smp.init_widths()))
t = time.perf_counter()
st, info = step(jax.random.key(0), st)
jax.block_until_ready(st)
print(f"step compile+run {time.perf_counter() - t:.1f}s", flush=True)
ts, ne = [], []
for i in range(10):
    t = time.perf_counter()
    st, info = step(jax.random.key(i), st)
    jax.block_until_ready(st)
    ts.append(time.perf_counter() - t)
    ne.append(int(info["n_evals"]))
print(
    f"step median {np.median(ts):.3f}s evals/step {np.mean(ne):.0f} -> {np.median(ts) / np.mean(ne) * 1e3:.2f} ms/eval"
)
