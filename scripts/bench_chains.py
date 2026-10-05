"""Seconds per Gibbs iteration of inference.fit (4 chains) at a given n_pad.

usage: python scripts/bench_chains.py N_PAD [REPEATS]
Per-iteration cost = (t(HI) - t(LO)) / (HI - LO), LO=10, HI=30 after a compile run, so that the starting
rule and compilation drop out. Environment (GCBML_HOST_DEVICES, GCBML_CHAIN_SCHEME) selects the scheme.
"""

import os
import statistics
import sys
import time

import jax
import numpy as np

sys.path.insert(0, "tests")
import test_inference as ti  # noqa: E402

from gcbml import inference  # noqa: E402
from gcbml.model import ModelConfig  # noqa: E402


def main():
    n_pad = int(sys.argv[1])
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    LO, HI = int(os.environ.get("BENCH_LO", 10)), int(os.environ.get("BENCH_HI", 30))
    rng = np.random.default_rng(0)
    X, H = ti.design_1d()
    z, _ = ti.simulate(rng, ti.TRUTH, X, H, ti.XS)
    data, zp = ti.pad(X, H, z, n_pad=n_pad)
    cfg = ModelConfig()

    def run(ns, seed=1):
        t = time.perf_counter()
        post = inference.fit(jax.random.key(seed), data, zp, zp, cfg, ti.SCALES, 1, n_warmup=0, n_samples=ns)
        jax.block_until_ready(post.params)
        return time.perf_counter() - t, post

    run(LO)
    run(HI)
    out = []
    for r in range(reps):
        t20, _ = run(LO, r)
        t80, _ = run(HI, r)
        out.append((t80 - t20) / (HI - LO))
    print(
        f"n_pad={n_pad} devices={len(jax.devices())} s/iter median={statistics.median(out):.5f} "
        f"min={min(out):.5f} max={max(out):.5f}"
    )


if __name__ == "__main__":
    main()
