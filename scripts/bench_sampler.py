"""Item 2 (speed2): Gibbs/slice (inference.fit) against NUTS-in-Gibbs (nuts.fit_nuts) on the same datasets.

    PYTHONPATH=tests:scripts:. uv run python scripts/bench_sampler.py OUTDIR NAME [NAME ...]
        [--gibbs 500 500] [--nuts 300 500]

NAME: a12_<id> and prior_<id> (ids 0-11 of scripts/quad_vs_mcmc.py), prior4_<id> (prior draw, 4 levels, any id),
n200 (193 rows, A12 seed 3, 3 levels). Each sampler runs twice with the same settings: the first call
compiles, the second is timed (wall seconds, warm-up + sampling, 4 chains in 4 threads, -c 4).
Metrics per sampler: wall time; evaluations (Gibbs: likelihood evaluations; NUTS: gradient evaluations +
likelihood evaluations of the slice/elliptical moves); bulk-ESS of log p0, log sigma_mu and the per-draw
conditional mean of mu(x) (h = 0) at 3 points of Sigma_N; ESS per second and per 1000 evaluations; the
largest rhat over all settings. Agreement: two-sample KS statistic of the pooled log p0 draws and the median over
Sigma_N of |m_nuts - m_gibbs| / sigma_gibbs (m, sigma: median and half-width (q84 - q16)/2 of the exact mixture
of the per-draw Gaussians), and the coverage of the true f0.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
import quad_vs_mcmc as qv
import quad_vs_mcmc_helpers as hp
from scipy import stats

import gcbml  # noqa: F401
from gcbml import inference, nuts, quadrature
from gcbml.mcmc import diagnostics as dg


def dataset(name):
    kind, _, i = name.partition("_")
    if kind == "n200":
        data, zp = hp.big_dataset(50)
        return data, zp, qv.sigma_n(0), None, "n200 A12 seed 3"
    if kind in ("a12", "prior"):
        data, zp, Xs, f0, label, _ = qv.make_dataset(int(i))
    else:  # prior4
        data, zp, Xs, f0, label, _ = qv.make_dataset(int(i), "prior", 4)
    return data, zp, Xs, f0, label


def measure(post, data, Xs, pts, wall, evals_total, evals_note):
    cfg = qv.CFG
    mean, var = quadrature.mu_moments(post, data, cfg, Xs)
    S = mean.shape[0]
    nc = post.params.c0.shape[0]
    ch = mean.reshape(nc, S // nc, -1)
    lp = np.asarray(post.theta["log_p0"])[..., 0]
    ls = np.asarray(post.theta["log_sigma_mu"])
    series = {"log_p0": lp, "log_sigma_mu": ls}
    for j in pts:
        series[f"mu[{j}]"] = ch[:, :, j]
    ess = {k: float(dg.bulk_ess(v)) for k, v in series.items()}
    rhats = {k: float(dg.rhat(v)) for k, v in series.items()}
    rh_all = max(v[0] for v in post.diagnostics.values())
    w = np.full(S, 1.0 / S)
    med, sig, *_ = quadrature.summarize(w, mean, var)
    return dict(
        seconds=wall, evals=evals_total, evals_note=evals_note, ess=ess, rhat=rhats, rhat_max_all=float(rh_all),
        ess_min=min(ess.values()), ess_per_s={k: v / wall for k, v in ess.items()},
        ess_min_per_s=min(ess.values()) / wall, ess_min_per_1000_evals=1000 * min(ess.values()) / evals_total,
    ), med, sig, lp.reshape(-1)  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--gibbs", type=int, nargs=2, default=[500, 500])
    ap.add_argument("--nuts", type=int, nargs=2, default=[300, 500])
    ap.add_argument("--skip-gibbs", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.names:
        data, zp, Xs, f0, label = dataset(name)
        pts = [0, len(Xs) // 3, 2 * len(Xs) // 3]
        res = dict(name=name, label=label, n=int(data.n), gibbs_settings=a.gibbs, nuts_settings=a.nuts)
        runs = {}
        if not a.skip_gibbs:
            for rep in range(2):
                t0 = time.perf_counter()
                post = inference.fit(
                    jax.random.key(10 + rep), data, zp, zp, qv.CFG, qv.SCALES, 2, *a.gibbs, 4
                )
                jax.block_until_ready(post.params)
                wall = time.perf_counter() - t0
            runs["gibbs"] = measure(post, data, Xs, pts, wall, int(post.n_evals), "likelihood evaluations")
        for rep in range(2):
            t0 = time.perf_counter()
            post = nuts.fit_nuts(jax.random.key(20 + rep), data, zp, zp, qv.CFG, qv.SCALES, 2, *a.nuts, 4)
            jax.block_until_ready(post.params)
            wall = time.perf_counter() - t0
        i = post.info
        res["nuts_info"] = {k: np.asarray(v).tolist() for k, v in i.items()}
        runs["nuts"] = measure(
            post, data, Xs, pts, wall, int(post.n_evals),
            f"grad {int(i['grad_evals'].sum() + i['warm_grad_evals'].sum())} + lik {int(i['lik_evals'].sum() + i['warm_lik_evals'].sum())}",
        )  # fmt: skip
        for k, (m, med, sig, lp) in runs.items():
            res[k] = m
            res[k]["median"], res[k]["sigma"] = med.tolist(), sig.tolist()
            res[k]["log_p0_draws"] = lp.tolist()
            if f0 is not None:
                res[k]["coverage_f0"] = float(np.mean(np.abs(f0 - med) <= 2 * sig))
        if "gibbs" in runs:
            g, n = runs["gibbs"], runs["nuts"]
            res["agreement"] = dict(
                ks_log_p0=float(stats.ks_2samp(g[3], n[3]).statistic),
                median_dm_over_sigma=float(np.median(np.abs(n[1] - g[1]) / g[2])),
                p90_dm_over_sigma=float(np.percentile(np.abs(n[1] - g[1]) / g[2], 90)),
                sigma_ratio_median=float(np.median(n[2] / g[2])),
            )
        (out / f"{name}.json").write_text(json.dumps(res))
        print(name, label, "n", res["n"], flush=True)
        for k in ("gibbs", "nuts"):
            if k in res:
                r = res[k]
                print(
                    f"  {k:5s} {r['seconds']:7.1f}s  evals {r['evals']:>9d} ({r['evals_note']})  ESS "
                    + " ".join(f"{e:6.0f}" for e in r["ess"].values())
                    + f"  min/s {r['ess_min_per_s']:.2f}  min/1000ev {r['ess_min_per_1000_evals']:.2f} rhat_all {r['rhat_max_all']:.3f}",
                    flush=True,
                )
        if "agreement" in res:
            print("  agreement", {k: round(v, 3) for k, v in res["agreement"].items()}, flush=True)


if __name__ == "__main__":
    main()
