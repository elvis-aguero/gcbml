"""Pre-asymptotic benchmark harness. Fits model A (L6-L10) and model O (L8-L10) with the CURRENT model.

Fit configuration is copied from multi-fidelity-bioreactor/scripts/gcbml_hydro_fit_v3.py.

Run (from the repo root):
    uv run python -m benchmarks.preasymptotic.run fit pre      # writes results/fits_pre.json
    uv run python -m benchmarks.preasymptotic.run fit control  # writes results/fits_control.json
    uv run python -m benchmarks.preasymptotic.run report       # writes baseline.{json,md,png}
Round 2: fit pre|control --candidate base|ref|sat|two (seeds 200-219 | 300-319), then report2.
Options for fit: --seeds a:b (default: the family's pre-registered seeds), --warmup, --samples.
"""

from __future__ import annotations

import dataclasses
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


def fit_subset(levels, x_grid, y_by_level, key, n_warmup, n_samples, cfg=CFG):
    """One fit; returns (summary dict of f(x, 0) = exp(mu), max rhat, n_extensions)."""
    data = padded(levels, x_grid, y_by_level)
    tf = transforms.get("log")
    z = jnp.asarray(tf.forward(jnp.asarray(data.y)))
    post = inference.fit(key, data, z, z, cfg, SCALES, n_controls=1, n_warmup=n_warmup, n_samples=n_samples)
    params, zs = inference.flatten(post)
    xs = np.asarray(x_grid)[:, None]
    mean, var = jax.vmap(lambda p, zz: model.predict_mu(p, data, zz, cfg, jnp.asarray(xs)))(params, zs)
    k1 = jax.random.fold_in(key, 1)
    draws, w = predict.sample_gaussian_draws(k1, mean, var, 4)
    epi = predict.summarize(draws, w, tf)
    worst = max(float(v[0]) for v in post.diagnostics.values())
    lp = np.asarray(post.theta["log_p0"]).reshape(-1)
    aux = {}
    if cfg.shape != "power":
        a = np.asarray(post.theta["aux"])
        aux["aux_q"] = np.percentile(a.reshape(-1, a.shape[-1]), [5, 50, 95], axis=0).T.tolist()
        if cfg.shape == "saturating":  # support of log h_s: the prior interval (inference module docstring)
            Hr = np.asarray(data.H)[np.asarray(data.mask)]
            pos = Hr[Hr > 0]
            aux["hs_prior"] = [float(np.log(pos.min() / 2.0)), float(np.log(8.0 * pos.max()))]
    out = dict(
        **aux,
        m=np.asarray(epi["m"]).tolist(),
        q025=np.asarray(epi["q025"]).tolist(),
        q975=np.asarray(epi["q975"]).tolist(),
        p_median=float(np.median(np.exp(lp))),
    )
    return out, worst, int(getattr(post.diagnostics, "n_extensions", 0))


def fit_robust(levels, x_grid, ys, base_key, n_warmup, n_samples, cfg=CFG):
    """Re-run with a new key until max rhat <= RHAT_MAX. Returns (summary, n_dropped, converged)."""
    dropped = 0
    for t in range(MAX_TRIES):
        out, worst, n_ext = fit_subset(
            levels, x_grid, ys, jax.random.fold_in(base_key, 1000 * t), n_warmup, n_samples, cfg
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


# ---------------------------------------------------------------------------------------------------
# Round 2 (README, "Round 2"): error-shape candidates. Same generator, noise and MCMC settings as round 1.
# ---------------------------------------------------------------------------------------------------

SEEDS_PRE_R2 = tuple(range(200, 220))
SEEDS_CONTROL_R2 = tuple(range(300, 320))
CANDIDATES = {  # name -> (ModelConfig, level slice)
    "base": (CFG, slice(0, 5)),
    "ref": (CFG, slice(2, 5)),
    "sat": (dataclasses.replace(CFG, shape="saturating"), slice(0, 5)),
    "two": (dataclasses.replace(CFG, shape="two_term"), slice(0, 5)),
}
ROUND2_CRITERIA = (
    "C1: pooled coverage of X on the pre-asymptotic family >= 0.90.",
    "C2: median over truths of W_X / W_ref <= 1.25 (pre-asymptotic family).",
    "C3: median over truths of W_X / W_ref <= 1.00 and pooled coverage >= 0.90 on the control family.",
    "C5: median over truths of relErr_X / relErr_ref <= 1.50 (pre-asymptotic family). Added AFTER round 1.",
)


def run_truth_r2(seed, family, cand, n_warmup, n_samples):
    t = preasymptotic_truth(seed, **family)
    x = C.X_GRID
    ys = np.exp(t.observe(x, C.HBAR, C.NOISE_SD, seed=10_000 + seed))
    f0 = np.exp(t.z0(x))
    cfg, idx = CANDIDATES[cand]
    t0 = time.time()
    out, dropped, ok = fit_robust(C.LEVELS[idx], x, ys[idx], jax.random.key(seed), n_warmup, n_samples, cfg)
    m, lo, hi = (np.asarray(out[k]) for k in ("m", "q025", "q975"))
    rec = dict(
        seed=seed,
        truth=dict(p=t.p, h_s=None if math.isinf(t.h_s) else t.h_s, m=t.m),
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
    for k in ("aux_q", "hs_prior"):
        if k in out:
            rec[k] = out[k]
    return rec


def fit_round2(which, cand, seeds, n_warmup=600, n_samples=600):
    family = C.FAMILY if which == "pre" else C.CONTROL
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"round2_{cand}_{which}.json"
    done = {r["seed"]: r for r in json.load(open(path))} if path.exists() else {}
    for s in seeds:
        if s in done:
            continue
        done[s] = run_truth_r2(s, family, cand, n_warmup, n_samples)
        json.dump(sorted(done.values(), key=lambda r: r["seed"]), open(path, "w"), indent=1)
        print(which, cand, s, round(done[s]["W"], 3), round(done[s]["rel_err"], 4), flush=True)


def load_round2() -> dict:
    return {
        c: {w: json.load(open(RESULTS / f"round2_{c}_{w}.json")) for w in ("pre", "control")}
        for c in CANDIDATES
    }


def criteria2(res: dict) -> dict:
    """Per candidate: C1, C2, C3 (ratio, coverage), C5, medians. Pairs by seed with the reference."""

    def cov(rs):
        return float(np.mean([c for r in rs for c in r["covered"]]))

    out = {}
    for c, r in res.items():
        ref = res["ref"]
        d = dict(
            C1=cov(r["pre"]),
            C2=float(np.median([a["W"] / b["W"] for a, b in zip(r["pre"], ref["pre"])])),
            C3_ratio=float(np.median([a["W"] / b["W"] for a, b in zip(r["control"], ref["control"])])),
            C3_cov=cov(r["control"]),
            C5=float(np.median([a["rel_err"] / b["rel_err"] for a, b in zip(r["pre"], ref["pre"])])),
            rel_err_med=float(np.median([a["rel_err"] for a in r["pre"]])),
            p_med=float(np.median([a["p_median"] for a in r["pre"]])),
            seconds=float(np.mean([a["seconds"] for w in ("pre", "control") for a in r[w]])),
            dropped=int(sum(a["dropped"] for w in ("pre", "control") for a in r[w])),
            unconverged=int(sum(not a["converged"] for w in ("pre", "control") for a in r[w])),
        )
        d["pass"] = dict(
            C1=d["C1"] >= 0.90,
            C2=d["C2"] <= 1.25,
            C3=d["C3_ratio"] <= 1.00 and d["C3_cov"] >= 0.90,
            C5=d["C5"] <= 1.50,
        )
        out[c] = d
    out["_p_true_med"] = float(np.median([a["truth"]["p"] for a in res["ref"]["pre"]]))
    if "sat" in res:
        out["_sat"] = sat_summary(res["sat"]["pre"])
    return out


def sat_summary(rs) -> dict:
    """How informative the posterior of log h_s is, and how it relates to the true h_s (pre family)."""
    narrow, med, true = [], [], []
    for r in rs:
        lo, hi = r["hs_prior"]
        q5, q50, q95 = r["aux_q"][0]
        narrow.append((q95 - q5) < 0.5 * (hi - lo))
        med.append(q50)
        true.append(math.log(r["truth"]["h_s"]))
    rel = [
        (r["aux_q"][0][1] - r["hs_prior"][0]) / (r["hs_prior"][1] - r["hs_prior"][0]) for r in rs
    ]  # posterior median position inside the prior interval (0 = lower bound, 1 = upper bound)
    return dict(
        frac_narrow=float(np.mean(narrow)),
        corr_logmedian_logtrue=float(np.corrcoef(med, true)[0, 1]),
        rel_pos_median_quartiles=np.percentile(rel, [25, 50, 75]).tolist(),
    )


def report2():
    res = load_round2()
    c = criteria2(res)
    json.dump(dict(criteria=c, fits=res), open(RESULTS / "round2.json", "w"), indent=1)
    names = {
        "base": "base (power, L6-L10)",
        "ref": "ref (power, L8-L10)",
        "sat": "sat (saturating, L6-L10)",
        "two": "two (two-term, L6-L10)",
    }  # noqa: E501
    lines = [
        "| candidate | C1 coverage, pre (>= 0.90) | C2 median W/W_ref, pre (<= 1.25) | C3 median W/W_ref, control (<= 1.00) ; coverage control (>= 0.90) | C5 median relErr/relErr_ref, pre (<= 1.50) | median relErr, pre | median posterior p, pre | mean s per fit |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in ("base", "sat", "two", "ref"):
        d = c[k]
        ps = d["pass"]
        lines.append(
            f"| {names[k]} | {d['C1']:.3f} ({ps['C1']}) | {d['C2']:.3f} ({ps['C2']}) | "
            f"{d['C3_ratio']:.3f} ; {d['C3_cov']:.3f} ({ps['C3']}) | {d['C5']:.3f} ({ps['C5']}) | "
            f"{d['rel_err_med']:.4f} | {d['p_med']:.2f} | {d['seconds']:.0f} |"
        )
    lines += [
        "",
        f"True p, median over the pre-asymptotic truths: {c['_p_true_med']:.2f}.",
        "C5 was added AFTER the round-1 results were seen (round 1: base at 2.67 on its own seeds); C1-C3 are the round-1 criteria applied to the candidate X in place of A and O replaced by ref.",  # noqa: E501
        "Seeds: pre-asymptotic 200-219, control 300-319. Ref is a fit of its own on each truth.",
        f"Dropped fits (rhat > 1.05, re-run): {sum(c[k]['dropped'] for k in CANDIDATES)}; unconverged after {MAX_TRIES} tries: {sum(c[k]['unconverged'] for k in CANDIDATES)}.",  # noqa: E501
    ]
    if "_sat" in c:
        sa = c["_sat"]
        lines.append(
            f"sat, log h_s posterior: fraction of truths with 90% interval < half the prior log-range {sa['frac_narrow']:.2f}; "  # noqa: E501
            f"corr(posterior median, true) over log h_s {sa['corr_logmedian_logtrue']:.2f}; "
            f"posterior-median position in the prior interval (quartiles) {np.round(sa['rel_pos_median_quartiles'], 2).tolist()}."  # noqa: E501
        )
    (RESULTS / "round2.md").write_text("\n".join(lines) + "\n")
    plot2(res)
    print("\n".join(lines))


def plot2(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ref = res["ref"]["pre"]
    for k, col, lab in (
        ("base", "tab:gray", "base"),
        ("sat", "tab:orange", "sat"),
        ("two", "tab:blue", "two"),
    ):
        ax.scatter(
            [r["rel_err"] for r in ref], [r["rel_err"] for r in res[k]["pre"]], s=22, color=col, label=lab
        )
    lim = [3e-4, 1.0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(lim, lim, color="k", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("relative error of f(x, 0), ref (L8-L10)")
    ax.set_ylabel("relative error of f(x, 0), candidate")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.savefig(RESULTS / "round2.png", dpi=150, bbox_inches="tight")


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
    if argv[0] == "report2":
        return report2()
    which = argv[1]
    seeds = C.SEEDS_PRE if which == "pre" else C.SEEDS_CONTROL
    cand = None
    kw = {}
    for i, a in enumerate(argv):
        if a == "--seeds":
            lo, hi = argv[i + 1].split(":")
            seeds = range(int(lo), int(hi))
        if a == "--candidate":
            cand = argv[i + 1]
            if "--seeds" not in argv:
                seeds = SEEDS_PRE_R2 if which == "pre" else SEEDS_CONTROL_R2
        if a == "--warmup":
            kw["n_warmup"] = int(argv[i + 1])
        if a == "--samples":
            kw["n_samples"] = int(argv[i + 1])
    if cand is not None:
        return fit_round2(which, cand, seeds, **kw)
    fit_family(which, seeds, **kw)


if __name__ == "__main__":
    main(sys.argv[1:])
