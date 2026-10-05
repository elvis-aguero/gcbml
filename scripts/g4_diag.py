# ruff: noqa
"""Why does G4 fail on A12 data? Falsifiable checks (a) MCMC health, (b) p0 with/without level 0,
(c) the z statistic on data simulated from the model's own prior.

    uv run python scripts/g4_diag.py a 4 14 18 2 5      # (a), (b) for these seeds
    uv run python scripts/g4_diag.py c 20               # (c): 20 prior-predictive datasets
"""

from __future__ import annotations

import dataclasses
import sys

import jax
import numpy as np

sys.path.insert(0, "tests")
import test_acquisition as ta  # noqa: E402
import test_campaign as tc  # noqa: E402

from gcbml.mcmc import diagnostics as dg  # noqa: E402

MC = (300, 200, 4)


def fits_both(c):
    c._min_level = 0
    an = c._analyse(repair=False)
    return an


def part_ab(seeds):
    for s in seeds:
        c = tc._four_level_campaign(s, mcmc=MC)
        an = fits_both(c)
        g4 = next(g for g in an.gates if g.name == "G4")
        z = np.asarray(g4.stats["z"])
        ix = int(np.argmax(np.abs(z)))
        data, lev = c._build_data()
        mask = np.asarray(data.mask)
        data_wo = dataclasses.replace(data, mask=mask & (an.levels > 0))
        fits_wo = c._fit_all(data_wo, jax.random.PRNGKey(91))
        print(
            f"== seed {s}: G4 {g4.status} frac {g4.stats['frac_exceed']:.2f} max|z| {np.abs(z).max():.1f} worst x index {ix}"
        )
        for name, fits, d in (("full", an.fits, data), ("without level 0", fits_wo, data_wo)):
            f = fits[0]
            th = f.post.theta
            row = []
            for key in ("log_p0", "c0", "c1"):
                a = np.asarray(th[key])[..., 0]
                row.append(f"{key}: rhat {dg.rhat(a):.3f} ess {dg.bulk_ess(a):.0f}")
            m, v = c._moments([f], d, c.problem.sigma_n)[0]
            nch = np.asarray(th["log_p0"]).shape[0]
            mu = np.asarray(m)[:, ix].reshape(nch, -1)
            row.append(f"mu(x*): rhat {dg.rhat(mu):.3f} ess {dg.bulk_ess(mu):.0f}")
            p0 = np.exp(np.asarray(th["log_p0"])[..., 0])
            q = np.quantile(p0, [0.05, 0.25, 0.5, 0.75, 0.95])
            per_chain = np.median(p0, axis=1)
            print(f"  {name:16s} | " + "; ".join(row))
            print(
                f"  {'':16s} | p0 quantiles 5/25/50/75/95 {np.round(q, 2).tolist()}; per-chain medians "
                f"{np.round(per_chain, 2).tolist()}; mu(x*) mean {float(np.mean(m[:, ix])):.3f} sd of mean "
                f"{float(np.std(m[:, ix])):.3f} mean post sd {float(np.mean(np.sqrt(v[:, ix]))):.3f}"
            )
        print(f"  sigma_f(x*) {an.sigma_epi[ix]:.4f}; sigma_w(x*) from gate: dm/z -> ", end="")
        sw = None
        print("see z; true p", tc.A12Truth(s, d=1).p)


def part_c(n):
    from gcbml import inference
    from gcbml.model import ModelConfig

    fr, st = [], []
    for s in range(n):
        c = tc._four_level_campaign(s, mcmc=MC)
        data, _ = c._build_data()
        cfg = ModelConfig()
        z0 = np.asarray(data.y)
        smp = inference._Sampler(data, z0, z0, cfg, tc.SCALES, 1, False)
        stt = smp.draw_prior(jax.random.PRNGKey(1000 + s))
        t = smp.layout.unpack(stt["theta"])
        params = smp.params(t, stt["zeta"], smp._no_pi())
        zsim = np.asarray(ta.simulate_z(s, params, data, ta.CFG))
        n_real = int(np.sum(np.asarray(data.mask)))
        res = []
        for i, r in enumerate(c.dataset.results):
            res.append(dataclasses.replace(r, y=np.array([zsim[i]])))
        c2 = tc.make(
            tc.A12Truth(s, d=1, budget=1e9),
            dataclasses.replace(tc.FAST, n_warmup=MC[0], n_samples=MC[1], n_chains=MC[2], seed=s),
        )
        c2.tell(res)
        g = tc._g4(c2)
        fr.append(g.stats.get("frac_exceed", np.nan))
        z = np.asarray(g.stats.get("z", [np.nan]))
        st.append((float(np.mean(z)), float(np.std(z)), float(np.median(np.abs(z)))))
        print(
            f"seed {s}: {g.status} frac {fr[-1]:.2f} z mean {st[-1][0]:.2f} sd {st[-1][1]:.2f} median|z| {st[-1][2]:.2f}",
            flush=True,
        )
    sds = np.array([x[1] for x in st])
    print(
        "pass",
        sum(f <= 0.1 for f in fr),
        "of",
        n,
        "; median over seeds of sd(z):",
        float(np.median(sds)),
        "(1.0 expected if z ~ N(0,1))",
    )


if __name__ == "__main__":
    if sys.argv[1] == "a":
        part_ab([int(x) for x in sys.argv[2:]])
    else:
        part_c(int(sys.argv[2]))
