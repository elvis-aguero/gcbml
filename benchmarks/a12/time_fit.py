"""Timing and convergence of one full fit for A12 (picks warm-up/samples): prints seconds and rhat of log p0."""

import sys
import time

import jax

from benchmarks.a12.run_truth import COST_PRIOR, SCALES, A12Campaign
from gcbml.campaign import CampaignSettings
from gcbml.mcmc import diagnostics as dg
from gcbml.synthetic import A12Truth

seed = int(sys.argv[1])
for wu, ns in [(int(a.split(",")[0]), int(a.split(",")[1])) for a in sys.argv[2:]]:
    t = A12Truth(seed, d=2, budget=1e9, eps_abs=0.01)
    s = CampaignSettings(n_warmup=wu, n_samples=ns, n_chains=4, h_kernels=("twy2",), seed=seed + 1000)
    c = A12Campaign(t.problem(), SCALES, COST_PRIOR, s)
    pr = c.initial_design()
    o = t.oracle(seed=seed)
    o.submit(pr, [c.cap_of(p) for p in pr])
    c.tell(o.poll())
    data, _ = c._build_data()
    for rep in range(2):
        t0 = time.time()
        fits = c._fit_all(data, jax.random.PRNGKey(rep))
        lp = fits[0].post.theta["log_p0"][..., 0]
        print(f"warmup {wu} samples {ns} rep {rep}: {time.time() - t0:.0f}s rhat(log p0) {dg.rhat(lp):.3f} true p {t.p:.2f}", flush=True)
