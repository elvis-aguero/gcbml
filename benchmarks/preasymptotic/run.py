"""Pre-asymptotic benchmark harness. Fits model A (L6-L10) and model O (L8-L10) with the CURRENT model.

Fit configuration is copied from multi-fidelity-bioreactor/scripts/gcbml_hydro_fit_v3.py.

Run (from the repo root):
    uv run python -m benchmarks.preasymptotic.run fit pre      # writes results/fits_pre.json
    uv run python -m benchmarks.preasymptotic.run fit control  # writes results/fits_control.json
    uv run python -m benchmarks.preasymptotic.run report       # writes baseline.{json,md,png}
Options for fit: --seeds a:b (default: the family's pre-registered seeds), --warmup, --samples.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gcbml  # noqa: F401  (enables x64)
from benchmarks.preasymptotic import config as C
from gcbml import inference, model, predict, transforms
from gcbml._config import bucket
from gcbml.data import PaddedData
from gcbml.priors import PriorScales
from gcbml.synthetic import preasymptotic_truth

RESULTS = Path(__file__).resolve().parent / "results"
SCALES = PriorScales(S_mu=0.8, S_c=0.5, S_delta=0.5, S_noise=0.02, log_p_mean=math.log(1.2), log_p_sd=0.45)
CFG = model.ModelConfig(h_kernel="twy2", mean_basis="constant", beta_prior=None, increasing=True)
RHAT_MAX = 1.05
MAX_TRIES = 6

PRE_CRITERIA = (
    "C1: pooled coverage of model A on the pre-asymptotic family >= 0.90 (pooled over 20 truths x 9 x = 180 indicators).",  # noqa: E501
    "C2: median over truths of W_A / W_O on the pre-asymptotic family <= 1.25.",
    "C3: median over truths of W_A / W_O on the CONTROL family <= 1.00, and pooled coverage of model A on control >= 0.90.",  # noqa: E501
    "C4: pooled coverage of model O on the pre-asymptotic family >= 0.90.",
)


def padded(levels, x_grid, y_by_level) -> PaddedData:
    """Rows: every (level, x). hbar = 2^-(L - Lmin of the subset), as in gcbml_hydro_fit_v3.py."""
    lmin = min(levels)
    X, H, y = [], [], []
    for L, row in zip(levels, y_by_level):
        for xi, yi in zip(x_grid, row):
            X.append(xi)
            H.append(2.0 ** -(L - lmin))
            y.append(yi)
    n = len(y)
    npad = bucket(n)
    Xp, Hp, yp = np.zeros((npad, 1)), np.ones((npad, 1)), np.ones(npad)
    Xp[:n, 0], Hp[:n, 0], yp[:n] = X, H, y
    Xp[n:], Hp[n:], yp[n:] = Xp[0], Hp[0], yp[0]
    mask = np.arange(npad) < n
    return PaddedData(
        X=Xp, H=Hp, y=yp, censored=np.zeros(npad, bool), run=np.where(mask, np.arange(npad), -1), mask=mask
    )


def fit_subset(levels, x_grid, y_by_level, key, n_warmup, n_samples):
    """One fit; returns (summary dict of f(x, 0) = exp(mu), max rhat, n_extensions)."""
    data = padded(levels, x_grid, y_by_level)
    tf = transforms.get("log")
    z = jnp.asarray(tf.forward(jnp.asarray(data.y)))
    post = inference.fit(key, data, z, z, CFG, SCALES, n_controls=1, n_warmup=n_warmup, n_samples=n_samples)
    params, zs = inference.flatten(post)
    xs = np.asarray(x_grid)[:, None]
    mean, var = jax.vmap(lambda p, zz: model.predict_mu(p, data, zz, CFG, jnp.asarray(xs)))(params, zs)
    k1 = jax.random.fold_in(key, 1)
    draws, w = predict.sample_gaussian_draws(k1, mean, var, 4)
    epi = predict.summarize(draws, w, tf)
    worst = max(float(v[0]) for v in post.diagnostics.values())
    lp = np.asarray(post.theta["log_p0"]).reshape(-1)
    out = dict(
        m=np.asarray(epi["m"]).tolist(),
        q025=np.asarray(epi["q025"]).tolist(),
        q975=np.asarray(epi["q975"]).tolist(),
        p_median=float(np.median(np.exp(lp))),
    )
    return out, worst, int(getattr(post.diagnostics, "n_extensions", 0))


def fit_robust(levels, x_grid, ys, base_key, n_warmup, n_samples):
    """Re-run with a new key until max rhat <= RHAT_MAX. Returns (summary, n_dropped, converged)."""
    dropped = 0
    for t in range(MAX_TRIES):
        out, worst, n_ext = fit_subset(
            levels, x_grid, ys, jax.random.fold_in(base_key, 1000 * t), n_warmup, n_samples
        )
        if worst <= RHAT_MAX:
            return dict(out, max_rhat=worst, n_ext=n_ext), dropped, True
        dropped += 1
    return dict(out, max_rhat=worst, n_ext=n_ext), dropped, False


def run_truth(seed, family, n_warmup, n_samples):
    t = preasymptotic_truth(seed, **family)
    x = C.X_GRID
    ys = np.exp(t.observe(x, C.HBAR, C.NOISE_SD, seed=10_000 + seed))  # raw scale; the model takes the log
    f0 = np.exp(t.z0(x))
    rec = dict(seed=seed, truth=dict(p=t.p, h_s=t.h_s, m=t.m, mu1=t.mu1, a0=t.a0, a1=t.a1))
    for name, idx in (("A", slice(0, 5)), ("O", slice(2, 5))):
        t0 = time.time()
        out, dropped, ok = fit_robust(C.LEVELS[idx], x, ys[idx], jax.random.key(seed), n_warmup, n_samples)
        m, lo, hi = (np.asarray(out[k]) for k in ("m", "q025", "q975"))
        rec[name] = dict(
            covered=((lo <= f0) & (f0 <= hi)).tolist(),
            W=float(np.mean((hi - lo) / m)),
            rel_err=float(np.mean(np.abs(m - f0) / f0)),
            p_median=out["p_median"],
            max_rhat=out["max_rhat"],
            n_ext=out["n_ext"],
            dropped=dropped,
            converged=ok,
            seconds=time.time() - t0,
        )
    return rec


def fit_family(which, seeds, n_warmup=600, n_samples=600):
    family = C.FAMILY if which == "pre" else C.CONTROL
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"fits_{which}.json"
    done = {r["seed"]: r for r in json.load(open(path))} if path.exists() else {}
    for s in seeds:
        if s in done:
            continue
        done[s] = run_truth(s, family, n_warmup, n_samples)
        json.dump(sorted(done.values(), key=lambda r: r["seed"]), open(path, "w"), indent=1)
        print(which, s, {k: round(done[s][k]["W"], 3) for k in "AO"}, flush=True)


def load_results() -> dict:
    return {w: json.load(open(RESULTS / f"fits_{w}.json")) for w in ("pre", "control")}


def criteria(res: dict) -> dict:
    def cov(rs, k):
        return float(np.mean([c for r in rs for c in r[k]["covered"]]))

    def ratio(rs):
        return np.array([r["A"]["W"] / r["O"]["W"] for r in rs])

    pre, ctl = res["pre"], res["control"]
    return dict(
        C1=cov(pre, "A"),
        C2=float(np.median(ratio(pre))),
        C3_ratio=float(np.median(ratio(ctl))),
        C3_cov=cov(ctl, "A"),
        C4=cov(pre, "O"),
        ratio_pre_p10_p90=np.percentile(ratio(pre), [10, 90]).tolist(),
        ratio_ctl_p10_p90=np.percentile(ratio(ctl), [10, 90]).tolist(),
        p_med_A_pre=float(np.median([r["A"]["p_median"] for r in pre])),
        p_med_O_pre=float(np.median([r["O"]["p_median"] for r in pre])),
        cov_O_ctl=cov(ctl, "O"),
        dropped=int(sum(r[k]["dropped"] for rs in (pre, ctl) for r in rs for k in "AO")),
        unconverged=int(sum(not r[k]["converged"] for rs in (pre, ctl) for r in rs for k in "AO")),
        seconds=float(sum(r[k]["seconds"] for rs in (pre, ctl) for r in rs for k in "AO")),
    )


def report():
    res = load_results()
    c = criteria(res)
    json.dump(dict(criteria=c, fits=res), open(RESULTS / "baseline.json", "w"), indent=1)
    ok = dict(
        C1=c["C1"] >= 0.90,
        C2=c["C2"] <= 1.25,
        C3=c["C3_ratio"] <= 1.00 and c["C3_cov"] >= 0.90,
        C4=c["C4"] >= 0.90,
    )
    lines = [
        "| criterion | value | pass |",
        "|---|---|---|",
        f"| C1 coverage of A, pre-asymptotic (>= 0.90) | {c['C1']:.3f} | {ok['C1']} |",
        f"| C2 median W_A/W_O, pre-asymptotic (<= 1.25) | {c['C2']:.3f} (10-90%: {c['ratio_pre_p10_p90'][0]:.2f}-{c['ratio_pre_p10_p90'][1]:.2f}) | {ok['C2']} |",  # noqa: E501
        f"| C3 median W_A/W_O, control (<= 1.00) and coverage of A, control (>= 0.90) | {c['C3_ratio']:.3f} (10-90%: {c['ratio_ctl_p10_p90'][0]:.2f}-{c['ratio_ctl_p10_p90'][1]:.2f}); coverage {c['C3_cov']:.3f} | {ok['C3']} |",  # noqa: E501
        f"| C4 coverage of O, pre-asymptotic (>= 0.90) | {c['C4']:.3f} | {ok['C4']} |",
        "",
        f"Median posterior p, pre-asymptotic family: A {c['p_med_A_pre']:.2f}, O {c['p_med_O_pre']:.2f}.",
        f"Coverage of O on control: {c['cov_O_ctl']:.3f}.",
        f"Dropped fits (rhat > 1.05, re-run): {c['dropped']}; fits still unconverged after {MAX_TRIES} tries: {c['unconverged']}.",  # noqa: E501
        f"Total fit wall time (sum over fits): {c['seconds'] / 3600:.2f} h.",
    ]
    (RESULTS / "baseline.md").write_text("\n".join(lines) + "\n")
    plot(res)
    print("\n".join(lines))


def plot(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for which, label, col in (("pre", "pre-asymptotic", "tab:orange"), ("control", "control", "tab:blue")):
        ax.scatter(
            [r["O"]["W"] for r in res[which]], [r["A"]["W"] for r in res[which]], s=22, color=col, label=label
        )
    lim = [0.02, 5.0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(lim, lim, color="k", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("relative width W, model O (L8-L10)")
    ax.set_ylabel("relative width W, model A (L6-L10)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.savefig(RESULTS / "baseline.png", dpi=150, bbox_inches="tight")


def main(argv):
    if argv[0] == "report":
        return report()
    which = argv[1]
    seeds = C.SEEDS_PRE if which == "pre" else C.SEEDS_CONTROL
    kw = {}
    for i, a in enumerate(argv):
        if a == "--seeds":
            lo, hi = argv[i + 1].split(":")
            seeds = range(int(lo), int(hi))
        if a == "--warmup":
            kw["n_warmup"] = int(argv[i + 1])
        if a == "--samples":
            kw["n_samples"] = int(argv[i + 1])
    fit_family(which, seeds, **kw)


if __name__ == "__main__":
    main(sys.argv[1:])
