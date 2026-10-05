"""Per-chain timing of the threaded phase at n_pad. usage: probe_threads.py N_PAD NS [hack]"""

import os
import sys
import time

if len(sys.argv) > 3:
    full = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(full)})
    import jax

    jax.devices()
    os.sched_setaffinity(0, full)

import jax
import numpy as np

sys.path.insert(0, "tests")
import test_inference as ti  # noqa: E402

from gcbml import inference  # noqa: E402
from gcbml.mcmc import chains  # noqa: E402
from gcbml.model import ModelConfig  # noqa: E402

n_pad, ns = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(0)
X, H = ti.design_1d()
z, _ = ti.simulate(rng, ti.TRUTH, X, H, ti.XS)
data, zp = ti.pad(X, H, z, n_pad=n_pad)
smp = inference._Sampler(data, zp, zp, ModelConfig(), ti.SCALES, 1, False)
t = time.perf_counter()
st = smp.init_chains(jax.random.key(0), 4)  # vmapped init, once
print(f"init(vmap) {time.perf_counter() - t:.1f}s", flush=True)
phase = chains.make_phase(smp.make_step, ns, 4)
keys = jax.random.split(jax.random.key(1), 4)
w = smp.init_widths()
for rep in range(3):
    t = time.perf_counter()
    out = phase(w, keys, st, ns, 1)
    jax.block_until_ready(out)
    ev = np.asarray(out[1]["n_evals"])
    print(f"rep {rep}: {time.perf_counter() - t:.2f}s evals/chain {ev}", flush=True)
