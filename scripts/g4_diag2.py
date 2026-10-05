# ruff: noqa
"""G4 internals: how often does the (0.1 sigma_f)^2 floor decide the denominator, and what is z^2 elsewhere?

uv run python scripts/g4_diag2.py prior 0 1 7 15 17 18 2 3     # prior-predictive datasets (as g4_diag.py c)
uv run python scripts/g4_diag2.py a12 4 14 18 2 5               # A12 truths
"""

from __future__ import annotations

import dataclasses
import sys

import jax
import numpy as np

sys.path.insert(0, "tests")
import test_acquisition as ta  # noqa: E402
import test_campaign as tc  # noqa: E402

from gcbml import gates, inference  # noqa: E402
from gcbml.model import ModelConfig  # noqa: E402

MC = (300, 200, 4)
CAP = {}
_orig = gates.g4_pre_asymptotic


def spy(m_full, sf, m_wo, sw=None, **kw):
    CAP["a"] = (
        np.asarray(m_full),
        np.asarray(sf),
        None if m_wo is None else np.asarray(m_wo),
        None if sw is None else np.asarray(sw),
    )
    return _orig(m_full, sf, m_wo, sw, **kw)


gates.g4_pre_asymptotic = spy


def prior_campaign(s):
    c = tc._four_level_campaign(s, mcmc=MC)
    data, _ = c._build_data()
    z0 = np.asarray(data.y)
    smp = inference._Sampler(data, z0, z0, ModelConfig(), tc.SCALES, 1, False)
    stt = smp.draw_prior(jax.random.PRNGKey(1000 + s))
    t = smp.layout.unpack(stt["theta"])
    params = smp.params(t, stt["zeta"], smp._no_pi())
    zsim = np.asarray(ta.simulate_z(s, params, data, ta.CFG))
    res = [dataclasses.replace(r, y=np.array([zsim[i]])) for i, r in enumerate(c.dataset.results)]
    c2 = tc.make(
        tc.A12Truth(s, d=1, budget=1e9),
        dataclasses.replace(tc.FAST, n_warmup=MC[0], n_samples=MC[1], n_chains=MC[2], seed=s),
    )
    c2.tell(res)
    return c2


kind, seeds = sys.argv[1], [int(x) for x in sys.argv[2:]]
for s in seeds:
    c = prior_campaign(s) if kind == "prior" else tc._four_level_campaign(s, mcmc=MC)
    g = tc._g4(c)
    mf, sf, mw, sw = CAP["a"]
    var_dm = sw**2 - sf**2
    floor = var_dm < (0.1 * sf) ** 2
    dm = mw - mf
    z = dm / np.sqrt(np.maximum(var_dm, (0.1 * sf) ** 2))
    ok = ~floor
    print(
        f"{kind} {s}: {g.status} frac {g.stats['frac_exceed']:.2f} | floor active at {floor.mean():.2f} of points | "
        f"median sigma_w/sigma_f {np.median(sw / sf):.2f} | mean z^2 off-floor {np.mean(z[ok] ** 2) if ok.any() else float('nan'):.2f} "
        f"| mean (dm/sigma_f)^2 {np.mean((dm / sf) ** 2):.2f} | mean |dm| {np.mean(np.abs(dm)):.3f} sigma_f {np.mean(sf):.3f}",
        flush=True,
    )
