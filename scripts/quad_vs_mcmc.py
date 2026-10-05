"""Item 3: full MCMC against p-quadrature on the same synthetic data.

    PYTHONPATH=tests:. uv run python scripts/quad_vs_mcmc.py --out DIR --ids 0 1 2 ...

Dataset ids 0-11: 0-3 A12Truth 3 levels (seeds 101-104), 4-7 A12Truth 4 levels (seeds 105-108),
8-9 prior draws with 3 levels, 10-11 prior draws with 4 levels (data drawn from the model's own prior with
the generative simulator of tests/test_inference.py). d = 2 inputs, one resolution component, levels
hbar = 1, 1/2, 1/4 (, 1/8), 12/6/3(/3) sites per level and 3 replicate sites x 2 repeats per level.
Per dataset one JSON with: settings, wall times, convergence (rhat of log p0 and of the predictive mean),
the variance decomposition of the full-MCMC h = 0 prediction, and the per-x arrays of both methods.

Full MCMC settings start at the campaign default (500 warm-up, 500 samples, 4 chains) and double (up to
2000/2000) until rhat < 1.05 for log p0 and for the predictive mean of mu at every x of Sigma_N.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
from scipy.stats import qmc

import gcbml  # noqa: F401
import test_inference as ti
from gcbml import inference, quadrature
from gcbml.mcmc import diagnostics as dg
from gcbml.model import ModelConfig
from gcbml.priors import PriorScales
from gcbml.synthetic import A12Truth

SCALES = PriorScales(S_mu=1.0, S_c=0.5, S_delta=0.5, S_noise=0.02)  # the A12 benchmark scales
CFG = ModelConfig()
SIZES = [12, 6, 3, 3]
N_REP_SITES, N_REPEATS = 3, 2
N_SIGMA = 100


def design(seed: int, n_levels: int):
    """X (n, 2) unit coordinates and H (n, 1) hbar: nested Sobol sites per level, plus repeated runs."""
    sob = qmc.Sobol(2, scramble=True, seed=seed).random_base2(4)[: SIZES[0]]
    X, H = [], []
    for lev in range(n_levels):
        u = sob[: SIZES[lev]]
        reps = np.repeat(u[:N_REP_SITES], N_REPEATS, axis=0)
        X += [u, reps]
        H += [np.full(len(u) + len(reps), 0.5**lev)]
    X = np.vstack(X)
    return X, np.concatenate(H)[:, None]


def sigma_n(seed: int) -> np.ndarray:
    return qmc.Sobol(2, scramble=True, seed=seed + 9999).random_base2(7)[:N_SIGMA]


def make_dataset(i: int):
    """(data, z, Xs, f0_true, label, n_levels)."""
    rng = np.random.default_rng(7000 + i)
    n_levels = 3 if i in (0, 1, 2, 3, 8, 9) else 4
    X, H = design(i, n_levels)
    Xs = sigma_n(i)
    if i < 8:
        t = A12Truth(101 + i, d=2)
        y = np.array([t.value(x[None, :], h) for x, h in zip(X, H[:, 0], strict=True)])
        y = y + 0.01 * rng.standard_normal(len(y))
        data, zp = ti.pad(X, H, y)
        return data, zp, Xs, np.asarray(t.truth(Xs)), f"A12 seed {101 + i} p={t.p:.2f}", n_levels
    data, zp = ti.pad(X, H, np.zeros(len(X)))
    smp = inference._Sampler(data, zp, zp, CFG, SCALES, 2, False)
    st = smp.draw_prior(jax.random.key(500 + i))
    t = smp.layout.unpack(st["theta"])
    T = dict(
        sigma_mu=float(np.exp(t["log_sigma_mu"])), ell_mu=np.exp(np.asarray(t["log_ell_mu"])),
        beta0=float(rng.normal()), c0=np.asarray(t["c0"]), c1=np.asarray(t["c1"]),
        sigma_delta=np.exp(np.asarray(t["log_sigma_delta"])), ell_x=np.exp(np.asarray(t["log_ell_x"])),
        ell_h=np.exp(np.asarray(t["log_ell_h"])), p0=np.exp(np.asarray(t["log_p0"])),
        m_s=float(t["m_s"]), b_s=np.asarray(t["b_s"]),
    )  # fmt: skip
    zeta_row = np.asarray(st["zeta"])[np.asarray(smp.aux.row_site)][: len(X)]
    y, mu_true = ti.simulate(rng, T, X, H, Xs, zeta_row=zeta_row)
    data, zp = ti.pad(X, H, y)
    return data, zp, Xs, mu_true, f"prior draw p={T['p0'][0]:.2f} sd_noise={np.exp(0.5 * T['m_s']):.3f}", n_levels


def run_mcmc(data, zp, Xs, key):
    """Full MCMC with doubling settings until rhat < 1.05 for log p0 and the predictive mean at every x."""
    t_total = 0.0
    for wu in (500, 1000, 2000):
        n0 = inference.n_compiled()
        t0 = time.perf_counter()
        post = inference.fit(key, data, zp, zp, CFG, SCALES, 2, wu, wu, 4)
        jax.block_until_ready(post.params)
        dt = time.perf_counter() - t0
        t_total += dt
        mean, var = quadrature.mu_moments(post, data, CFG, Xs)
        S = mean.shape[0]
        m_chain = mean.reshape(4, S // 4, -1)
        rh_m = max(dg.rhat(m_chain[:, :, j]) for j in range(m_chain.shape[2]))
        rh_p = dg.rhat(np.asarray(post.theta["log_p0"])[..., 0])
        rh_all = max(v[0] for v in post.diagnostics.values())
        if max(rh_m, rh_p) < 1.05 or wu == 2000:
            return post, mean, var, dict(
                warmup=wu, samples=wu, seconds_last=dt, seconds_total=t_total, rhat_mean=rh_m,
                rhat_logp0=rh_p, rhat_max_all=rh_all, converged=bool(max(rh_m, rh_p) < 1.05),
                compiled_new=inference.n_compiled() > n0,
            )  # fmt: skip


def p_share(logp, m, v, n_bins=10):
    """Variance of E[m_s | p] as a share of total = E[v_s] + Var(m_s), per x. Draws binned on log p0 into
    n_bins quantile bins; the bin means carry sampling noise var_b / n_b, which is subtracted (clipped at 0).
    Returns (share_raw, share_corrected, total)."""
    S = len(logp)
    order = np.argsort(logp)
    bins = np.array_split(order, n_bins)
    mbar = m.mean(axis=0)
    raw, corr = np.zeros(m.shape[1]), np.zeros(m.shape[1])
    for b in bins:
        w = len(b) / S
        mb = m[b].mean(axis=0)
        raw += w * (mb - mbar) ** 2
        corr += w * ((mb - mbar) ** 2 - m[b].var(axis=0, ddof=1) / len(b))
    total = v.mean(axis=0) + m.var(axis=0)
    return raw / total, np.maximum(corr, 0.0) / total, total


def run_dataset(i: int, out: Path):
    data, zp, Xs, f0, label, n_levels = make_dataset(i)
    key = jax.random.key(i)
    post, m_mc, v_mc, info = run_mcmc(data, zp, Xs, key)
    w_eq = np.full(len(m_mc), 1.0 / len(m_mc))
    med_mc, sig_mc, *_ = quadrature.summarize(w_eq, m_mc, v_mc)
    logp = np.asarray(post.theta["log_p0"]).reshape(-1)
    raw, corr, total = p_share(logp, m_mc, v_mc)
    res = dict(id=i, label=label, n_levels=n_levels, n=int(data.n), mcmc=info)
    res["p_share_raw"], res["p_share_corr"] = raw.tolist(), corr.tolist()
    # p-quadrature: a first call (compile + run), then the timed warm call
    kw = dict(key=key, data=data, z=zp, bounds=zp, cfg=CFG, scales=SCALES, n_controls=2)
    t0 = time.perf_counter()
    q = quadrature.fit_quadrature(**kw)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    q = quadrature.fit_quadrature(**kw)
    t_warm = time.perf_counter() - t0
    m_q, v_q = quadrature.mu_moments(q, data, CFG, Xs)
    med_q, sig_q, *_ = quadrature.summarize(q.weights, m_q, v_q)
    q40 = quadrature.fit_quadrature(**kw, n_grid2=40)
    m40, v40 = quadrature.mu_moments(q40, data, CFG, Xs)
    med40, sig40, *_ = quadrature.summarize(q40.weights, m40, v40)
    res["quad"] = dict(
        seconds_first=t_first, seconds_warm=t_warm, n_extensions=q.pass1["n_extensions"],
        converged_frac=float(np.mean(q.converged)), log_p_range=[float(q.log_p[0, 0]), float(q.log_p[-1, 0])],
        weight_max=float(q.weights.max()), edge_weight=float(max(q.weights[0], q.weights[-1])),
        sigma_change_40=float(np.max(np.abs(sig40 / sig_q - 1))), sigma_change_40_median=float(np.median(np.abs(sig40 / sig_q - 1))),
        median_change_40_over_sigma=float(np.max(np.abs(med40 - med_q) / sig_q)),
    )  # fmt: skip
    res["x"] = dict(
        f0=f0.tolist(), m_mcmc=med_mc.tolist(), s_mcmc=sig_mc.tolist(), m_quad=med_q.tolist(), s_quad=sig_q.tolist()
    )
    res["posterior_logp"] = dict(
        mcmc_sd=float(logp.std()), mcmc_mean=float(logp.mean()),
        quad_mean=float(np.sum(q.weights * q.log_p[:, 0])),
        quad_sd=float(np.sqrt(np.sum(q.weights * (q.log_p[:, 0] - np.sum(q.weights * q.log_p[:, 0])) ** 2))),
    )  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ds_{i}.json").write_text(json.dumps(res, indent=1))
    print(
        f"[{i}] {label}: n={res['n']} mcmc {info['seconds_total']:.0f}s (rhat p {info['rhat_logp0']:.3f} "
        f"mean {info['rhat_mean']:.3f}, {info['warmup']} it) quad {t_warm:.1f}s ext {q.pass1['n_extensions']} "
        f"med|dm|/s {np.median(np.abs(med_q - med_mc) / sig_mc):.3f}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", type=int, nargs="+", default=list(range(12)))
    a = ap.parse_args()
    for i in a.ids:
        run_dataset(i, Path(a.out))


if __name__ == "__main__":
    main()
