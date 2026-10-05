# ruff: noqa
"""Time of Campaign._analyse at n ~ 200: cold refits (before) against warm-started refits (after).

    uv run python scripts/bench_warm_analyse.py
Analysis 1 (cold, both) on ~200 rows with 4 levels; then 6 more runs are told and analysis 2 is timed twice
from the same state: WARM_START False (cold main fit, cold hold-out and G4 refits) and True.
"""

import copy
import dataclasses
import time

import jax
import numpy as np

import gcbml
from benchmarks.a12.run_truth import COST_PRIOR, SCALES
from gcbml.campaign import Campaign, CampaignSettings
from gcbml.data import Probe
from gcbml.synthetic import A12Truth

truth = A12Truth(3, d=2, budget=1e9, eps_abs=0.05)
s = CampaignSettings(
    n_warmup=300, n_samples=200, n_chains=4, n0_per_control=50, h_kernels=("twy2", "lb"), seed=7
)
c = Campaign(truth.problem(), SCALES, COST_PRIOR, s)
probes = c.initial_design()
o = truth.oracle(seed=1)
o.submit(probes, [c.cap_of(p) for p in probes])
c.tell(o.poll())
rng = np.random.default_rng(0)
extra = [Probe(f"x3_{i}", tuple(rng.uniform(size=2)), c._h_of_level(3), None) for i in range(14)]
o.submit(extra, [1e12] * len(extra))
c.tell(o.poll())
print("n rows", c._n_rows(), flush=True)
Campaign.WARM_START = True
t0 = time.time()
an = c._analyse()
t1 = time.time() - t0
print(f"analysis 1 (cold): {t1:.0f}s; weights note: {an.weight_note}; G4: {[g.status for g in an.gates if g.name == 'G4']}", flush=True)
warm1 = copy.deepcopy(c._warm)
state = (c._min_level, c._removals, c._buy_cycles)
more = [Probe(f"y_{i}", tuple(rng.uniform(size=2)), c._h_of_level(0), None) for i in range(6)]
o.submit(more, [1e12] * 6)
c.tell(o.poll())
out = {}
for warm in (False, True):
    Campaign.WARM_START = warm
    c._cache = None
    c._warm = copy.deepcopy(warm1)
    c._min_level, c._removals, c._buy_cycles = state
    t0 = time.time()
    an2 = c._analyse()
    out[warm] = time.time() - t0
    print(f"analysis 2, WARM_START={warm}: {out[warm]:.0f}s; sigma_epi median {np.median(an2.sigma_epi):.4f}", flush=True)
print(f"speed-up {out[False] / out[True]:.2f}x")
