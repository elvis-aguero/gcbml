"""Gates G0-G7 (spec Section 3, Step 3): diagnostics and calibration checks of a fit.

Every gate returns GateResult(name, status, stats): status is "pass", "fail" or "not testable" (never
"pass" by default when a test cannot be run). Thresholds are the spec's [assumption] values and are
keyword arguments.

g0_order(log_p0_draws (S, k), p_x_draws=None (S, s, k), sigma_pi_draws=None (S, k), prior_sd=1.0)
    P(p_j0 > 0.5) >= 0.9 AND posterior sd of log p_j0 <= 0.5 * prior_sd, for every component j; when
    P(sigma_pi_j > 0.05) >= 0.9, also P(p_j(x) > 0.5) >= 0.9 at every x of Sigma_N. "Not testable" if the
    order varies and p_x_draws is None. stats: p_gt_half (k,), sd_log_p (k,), varying (k,), p_x_min (k,).
g1_level_holdout(z_B, inside95_B, testable=True)
    Inputs from the cross-fitted hold-out (stacking.py): z_B (n,) standardised held-out scores on half B,
    inside95_B (n,) bool, whether the observed value lies in the 95% predictive interval.
    Three checks, all must hold [thresholds: assumption]:
      * coverage: the exact two-sided binomial test of n_inside ~ Bin(n, 0.95) has p >= alpha (0.05);
        scipy.stats.binomtest. stats also holds the Clopper-Pearson limits.
      * no sign bias: |mean z| <= 2 se, se = sample sd / sqrt(n).
      * C_LOO near 1: C_LOO = (1/n) sum z_i^2, the cross-validation criterion of Bachoc (1301.4320, eq 6),
        C_LOO = (1/n) sum_i (y_i - yhat_{i,-i})^2 / (sigma^2 c_{i,-i}^2), where z_i is the standardised
        left-out residual. The paper shows (App. C, Prop C.1) that E C_LOO = 1 when the covariance is
        right, with var C_LOO given by eq 27 (a trace; bounded below by (2/n)(lmin/lmax)^2). For
        independent z_i, var C_LOO = 2/n: the acceptance band |C_LOO - 1| <= 2 sqrt(2/n) is an ASSUMPTION
        built on that case, and is too tight for correlated z.
    The hold-out and the coverage test implement the finest-mesh predictive check of Oliver et al.
    (1311.0828, eq 17: <q_h> = E[q] - C_0 h^p - e_{h,N}; the observation must be a plausible draw from it).
g2_block_loo(params, data, z, cfg, blocks, w=None) leave out whole runs. Closed-form block LOO of a GP with
    fixed parameters (block_loo below), per draw; draws are pooled by moments (below); the left-out outputs
    are whitened with the Cholesky factor of the pooled covariance, e = L^{-1}(z_b - mean_b). Pass if the
    whitened scores look N(0, 1): |mean e| <= 2 / sqrt(N), the number of |e| > 1.96 within the exact
    binomial limits (as G1), and sum_b M_b, M_b = e_b^T e_b, within the two-sided chi-square(N) limits.
    The multivariate U statistic of Overstall & Woods (1506.04489, eq 10) is u_statistic(E) =
    |I_k + E^T E|^{-1}, E the standardised prediction errors of n_0 validation points of a k-output
    emulator. The paper derives its reference distribution for the matrix-t predictive of an emulator with
    UNKNOWN scale matrix: E U = prod_s (k + delta - s) / (k + delta - s + n_0). Here k = 1 and the scale
    of each draw is given, which is the delta -> infinity limit: delta (1 - U) / (n_0 U) -> chi2_{n_0} / n_0
    (the paper's F(n_0, delta) result for k = 1), i.e. the Mahalanobis distance M_b against chi-square.
    That limit is what the gate tests [assumption]; u_statistic is provided and tested against the paper.
g3_noise(groups: list of arrays of Lambda(y) of runs at the same site, s2_mean per group)
    T = sum_g sum_r (x_gr - xbar_g)^2 / s2_g ~ chi-square(sum_g (n_g - 1)) for fixed s2 and Gaussian noise;
    pass if the TWO-SIDED p = 2 min(cdf, sf) > 0.05 [assumption: two-sided]; "not testable" without
    replicates (no group with n_g >= 2). s2_mean is the posterior mean of the group's s^2, used as a plug-in.
g4_pre_asymptotic(m_full (s,), sigma_epi_full (s,), m_without_coarsest (s,) or None, sigma_epi_without (s,))
    z = dm / sqrt(max(sigma_w^2 - sigma_f^2, (0.1 sigma_f)^2)); "fail" if > 10% of Sigma_N has |z| > 2.5,
    with the advice "remove the coarsest level". None (nothing to remove) -> "not testable".
g5_monotone(median_curve (s,), direction, tol=0.0) no violation along a declared monotone coordinate (the
    curve is ordered along it); direction "increasing" | "decreasing" (or +1 | -1); None -> "not testable".
g6_shape(draws (S, s) physical, w) median over x of max(|q025 - (m - 1.96 sigma)|, |q975 - (m + 1.96 sigma)|)
    / sigma < 0.2, with m the median and sigma = (q84 - q16) / 2 as in predict.summarize (identity transform).
    A Gaussian has sigma = 0.9945 sd, so even exact Gaussian draws differ by 0.011 sigma.
g7_prior(log_prior_ratio_fn, draws, w, transform, scenarios, theta=None) halve and double each prior scale:
    log_prior_ratio_fn(scenario, theta) -> (S,) log pi_new(theta_s) - log pi_old(theta_s) for each scenario
    (the caller builds it from its priors; theta are the draws it needs). Importance weights
    w' ~ w exp(ratio); ESS = 1 / sum w'^2. A scenario with ESS < 400 cannot be judged (refit). For the
    others: pass if the median moves < 0.5 sigma_epi and sigma_epi changes < 20% at every x; draws are in
    Lambda units and summarised with predict.summarize (physical scale). Verdict: any judged scenario that
    violates -> "fail" (stats["verdict"] = "prior-dominated"); else any scenario with low ESS ->
    status "not testable" with stats["verdict"] = "refit needed" (GateResult has no fourth status); else pass.
    The unit of a "scale" is the caller's: scenarios may be any hashable labels.

block_loo(params, data, z, cfg, block) -> (mean (m,), cov (m, m)) is the closed form of the module's G2.
    With Q the precision of z with beta integrated (Gaussian beta: Q = (K + A B A^T)^{-1}; flat beta:
    Q = K^{-1} - K^{-1} A G^{-1} A^T K^{-1}, G = A^T K^{-1} A, the limit of the Gaussian case as B grows) and
    r = z - rho0 (- A b0), the left-out block b has mean z_b - Q_bb^{-1} (Q r)_b and covariance Q_bb^{-1}
    (Dubrule 1983; Bachoc 1301.4320 Prop 3.1 for one row). The block must be whole runs, so that the noise
    of the left-out rows is independent of the kept rows. Q's columns come from triangular solves; the (m, m)
    block Q_bb is solved against the identity by its Cholesky factor (no inverse of K).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve
from scipy import stats as sps

from gcbml import linalg, model, predict


class GateResult(NamedTuple):
    name: str
    status: str  # "pass" | "fail" | "not testable"
    stats: dict


def _status(ok: bool) -> str:
    return "pass" if ok else "fail"


# ----------------------------------------------------------------------------------------------
# G0
# ----------------------------------------------------------------------------------------------


def g0_order(
    log_p0_draws,
    p_x_draws=None,
    sigma_pi_draws=None,
    prior_sd: float = 1.0,
    *,
    p_min: float = 0.5,
    prob: float = 0.9,
    sd_frac: float = 0.5,
    sigma_pi_min: float = 0.05,
) -> GateResult:
    lp = np.asarray(log_p0_draws, dtype=float)
    if lp.ndim == 1:
        lp = lp[:, None]
    p_gt = np.mean(np.exp(lp) > p_min, axis=0)
    sd = np.std(lp, axis=0, ddof=1)
    k = lp.shape[1]
    ok = (p_gt >= prob) & (sd <= sd_frac * prior_sd)
    varying = np.zeros(k, dtype=bool)
    p_x_min = np.full(k, np.nan)
    if sigma_pi_draws is not None:
        sp = np.asarray(sigma_pi_draws, dtype=float).reshape(lp.shape[0], -1)
        varying = np.mean(sp > sigma_pi_min, axis=0) >= prob
    st = dict(p_gt_half=p_gt, sd_log_p=sd, varying=varying, p_x_min=p_x_min)
    if varying.any():
        if p_x_draws is None:
            return GateResult(
                "G0", "not testable", dict(st, reason="order varies with x but p_x_draws is None")
            )
        px = np.asarray(p_x_draws, dtype=float)
        p_x_min = np.min(np.mean(px > p_min, axis=0), axis=0)  # (k,): worst x
        st["p_x_min"] = p_x_min
        ok = ok & (~varying | (p_x_min >= prob))
    return GateResult("G0", _status(bool(np.all(ok))), st)


# ----------------------------------------------------------------------------------------------
# G1
# ----------------------------------------------------------------------------------------------


def g1_level_holdout(
    z_B,
    inside95_B,
    testable: bool = True,
    *,
    alpha: float = 0.05,
    bias_se: float = 2.0,
    c_loo_se: float = 2.0,
) -> GateResult:
    z = np.asarray(z_B, dtype=float).reshape(-1)
    inside = np.asarray(inside95_B, dtype=bool).reshape(-1)
    n = z.size
    if not testable or n < 2:
        return GateResult(
            "G1", "not testable", dict(n=n, reason="hold-out not testable" if not testable else "n < 2")
        )
    k = int(inside.sum())
    bt = sps.binomtest(k, n, 0.95)
    ci = bt.proportion_ci(confidence_level=1.0 - alpha, method="exact")
    mean_z = float(z.mean())
    se = float(z.std(ddof=1) / np.sqrt(n))
    c_loo = float(np.mean(z**2))
    band = c_loo_se * np.sqrt(2.0 / n)
    ok_cov = bt.pvalue >= alpha
    ok_bias = abs(mean_z) <= bias_se * se
    ok_c = abs(c_loo - 1.0) <= band
    st = dict(
        n=n,
        n_inside=k,
        coverage=k / n,
        coverage_pvalue=float(bt.pvalue),
        coverage_limits=(float(ci.low), float(ci.high)),
        mean_z=mean_z,
        se_mean_z=se,
        c_loo=c_loo,
        c_loo_band=float(band),
        ok_coverage=bool(ok_cov),
        ok_bias=bool(ok_bias),
        ok_c_loo=bool(ok_c),
    )
    return GateResult("G1", _status(bool(ok_cov and ok_bias and ok_c)), st)


# ----------------------------------------------------------------------------------------------
# G2
# ----------------------------------------------------------------------------------------------


def block_loo(params, data, z, cfg, block):
    """Closed-form leave-one-block-out predictive (mean (m,), cov (m, m)) of the outputs z[block]."""
    S = model._setup(params, data, cfg)
    z = jnp.asarray(z, dtype=float)
    block = jnp.asarray(block, dtype=int)
    n, m = z.shape[0], block.shape[0]
    E = jnp.zeros((n, m)).at[block, jnp.arange(m)].set(1.0)
    if cfg.beta_prior is None:
        F, KiA, Fg = model._flat_pieces(S)
        r = jnp.where(S.mask, z - S.rho0, 0.0)
        Qr = linalg.solve(F, r) - KiA @ linalg.solve(Fg, KiA.T @ r)
        QE = linalg.solve(F, E) - KiA @ linalg.solve(Fg, KiA[block].T)
    else:
        b0, bsd = model._beta_prior(cfg)
        F = linalg.factor(S.K + (S.A * bsd**2) @ S.A.T, S.mask)
        r = jnp.where(S.mask, z - S.rho0 - S.A @ b0, 0.0)
        Qr = linalg.solve(F, r)
        QE = linalg.solve(F, E)
    Qbb = QE[block]
    Qbb = 0.5 * (Qbb + Qbb.T)
    Lq = jnp.linalg.cholesky(Qbb)
    mean = z[block] - cho_solve((Lq, True), Qr[block])
    cov = cho_solve((Lq, True), jnp.eye(m))
    return mean, 0.5 * (cov + cov.T)


def u_statistic(E) -> float:
    """U = |I_k + E^T E|^{-1} (Overstall & Woods 1506.04489, eq 10); E is (n_0, k)."""
    E = np.asarray(E, dtype=float)
    if E.ndim == 1:
        E = E[:, None]
    sign, logdet = np.linalg.slogdet(np.eye(E.shape[1]) + E.T @ E)
    return float(np.exp(-logdet))


def g2_block_loo(
    params,
    data,
    z,
    cfg,
    blocks,
    w=None,
    *,
    alpha: float = 0.05,
    min_outputs: int = 10,
    z_crit: float = 1.96,
    bias_se: float = 2.0,
) -> GateResult:
    n_out = int(sum(len(b) for b in blocks))
    if len(blocks) == 0 or n_out < min_outputs:
        return GateResult(
            "G2", "not testable", dict(n_blocks=len(blocks), n_outputs=n_out, min_outputs=min_outputs)
        )
    z = jnp.asarray(z, dtype=float)
    S = z.shape[0]
    w = (
        jnp.full(S, 1.0 / S)
        if w is None
        else jnp.asarray(w, dtype=float) / jnp.sum(jnp.asarray(w, dtype=float))
    )
    z_obs = jnp.sum(w[:, None] * z, axis=0)  # equal across draws except for imputed censored rows
    whitened, marginal, maha = [], [], []
    per_draw = jax.jit(jax.vmap(lambda p, zz, b: block_loo(p, data, zz, cfg, b), in_axes=(0, 0, None)))
    for b in blocks:
        b = np.asarray(b, dtype=int)
        ms, Cs = per_draw(params, z, b)
        mbar = jnp.sum(w[:, None] * ms, axis=0)
        d = ms - mbar[None, :]
        C = jnp.sum(w[:, None, None] * Cs, axis=0) + jnp.einsum("s,si,sj->ij", w, d, d)
        C = 0.5 * (C + C.T)
        res = z_obs[b] - mbar
        L = jnp.linalg.cholesky(C)
        e = jax.scipy.linalg.solve_triangular(L, res, lower=True)
        whitened.append(np.asarray(e))
        marginal.append(np.asarray(res / jnp.sqrt(jnp.diag(C))))
        maha.append(float(e @ e))
    e_all = np.concatenate(whitened)
    N = e_all.size
    n_in = int(np.sum(np.abs(e_all) < z_crit))
    bt = sps.binomtest(n_in, N, 2.0 * sps.norm.cdf(z_crit) - 1.0)
    tot = float(np.sum(maha))
    p_chi = float(min(1.0, 2.0 * min(sps.chi2.cdf(tot, N), sps.chi2.sf(tot, N))))
    mean_e, se = float(e_all.mean()), 1.0 / np.sqrt(N)
    ok_bias, ok_cov, ok_chi = abs(mean_e) <= bias_se * se, bt.pvalue >= alpha, p_chi >= alpha
    st = dict(
        n_blocks=len(blocks),
        n_outputs=N,
        mean_e=mean_e,
        se_mean_e=se,
        n_inside=n_in,
        coverage_pvalue=float(bt.pvalue),
        mahalanobis=maha,
        mahalanobis_total=tot,
        chi2_pvalue=p_chi,
        z_whitened=e_all,
        z_marginal=np.concatenate(marginal),
        ok_bias=bool(ok_bias),
        ok_coverage=bool(ok_cov),
        ok_chi2=bool(ok_chi),
    )
    return GateResult("G2", _status(bool(ok_bias and ok_cov and ok_chi)), st)


# ----------------------------------------------------------------------------------------------
# G3
# ----------------------------------------------------------------------------------------------


def g3_noise(groups, s2_mean, *, alpha: float = 0.05) -> GateResult:
    T, dof = 0.0, 0
    for g, s2 in zip(groups, s2_mean):
        x = np.asarray(g, dtype=float).reshape(-1)
        if x.size < 2:
            continue
        T += float(np.sum((x - x.mean()) ** 2) / float(s2))
        dof += x.size - 1
    if dof == 0:
        return GateResult("G3", "not testable", dict(reason="no replicates", dof=0))
    p = float(min(1.0, 2.0 * min(sps.chi2.cdf(T, dof), sps.chi2.sf(T, dof))))
    return GateResult("G3", _status(p > alpha), dict(statistic=T, dof=dof, p=p))


# ----------------------------------------------------------------------------------------------
# G4, G5, G6
# ----------------------------------------------------------------------------------------------


def g4_pre_asymptotic(
    m_full,
    sigma_epi_full,
    m_without_coarsest,
    sigma_epi_without=None,
    *,
    z_max: float = 2.5,
    frac_max: float = 0.1,
    floor: float = 0.1,
) -> GateResult:
    """z = dm / sqrt(max(sigma_w^2 - sigma_f^2, (floor sigma_f)^2)) at every x of Sigma_N (spec Step 3, G4).

    Under the model, adding data changes a posterior mean by a quantity of variance sigma_w^2 - sigma_f^2
    (w: without the coarsest level, f: full). Fail if more than ``frac_max`` of the points have |z| > z_max.
    ``sigma_epi_without=None`` is taken as sigma_f (only the floor sets the scale).
    """
    if m_without_coarsest is None:
        return GateResult("G4", "not testable", dict(reason="no level can be removed"))
    sf = np.asarray(sigma_epi_full, float)
    sw = sf if sigma_epi_without is None else np.asarray(sigma_epi_without, float)
    dm = np.asarray(m_without_coarsest, float) - np.asarray(m_full, float)
    z = dm / np.sqrt(np.maximum(sw**2 - sf**2, (floor * sf) ** 2))
    frac = float(np.mean(np.abs(z) > z_max))
    ok = frac <= frac_max
    st = dict(
        z=z, max_abs_z=float(np.max(np.abs(z))), frac_exceed=frac, n_violations=int(np.sum(np.abs(z) > z_max))
    )
    if not ok:
        st["advice"] = "remove the coarsest level"
    return GateResult("G4", _status(ok), st)


def g5_monotone(median_curve, direction=None, tol: float = 0.0) -> GateResult:
    if direction is None:
        return GateResult("G5", "not testable", dict(reason="no monotone coordinate declared"))
    sign = {"increasing": 1.0, "decreasing": -1.0, 1: 1.0, -1: -1.0}[direction]
    d = sign * np.diff(np.asarray(median_curve, dtype=float))
    viol = d < -tol
    return GateResult(
        "G5", _status(not viol.any()), dict(n_violations=int(viol.sum()), worst=float(min(d.min(), 0.0)))
    )


def g6_shape(draws, w, *, tol: float = 0.2) -> GateResult:
    """Median over the points of max(|q025 - (m - 1.96 s)|, |q975 - (m + 1.96 s)|) / s must be below ``tol``.

    A warning in the report, not a blocking gate (spec): s is a quantile half-width and P1 does not assume a
    Gaussian. stats: rel_diff (s,) per point, median_rel_diff, rel_diff_lo, rel_diff_hi.
    """
    from gcbml import transforms

    s = predict.summarize(
        jnp.asarray(draws, dtype=float), jnp.asarray(w, dtype=float), transforms.get("identity")
    )
    m, sig = np.asarray(s["m"]), np.asarray(s["sigma"])
    if not (np.all(np.isfinite(sig)) and np.all(sig > 0)):
        return GateResult("G6", "not testable", dict(reason="zero or non-finite spread"))
    lo = np.abs(np.asarray(s["q025"]) - (m - 1.96 * sig)) / sig
    hi = np.abs(np.asarray(s["q975"]) - (m + 1.96 * sig)) / sig
    rel = np.maximum(lo, hi)
    med = float(np.median(rel))
    return GateResult(
        "G6",
        _status(med < tol),
        dict(rel_diff=rel, median_rel_diff=med, rel_diff_lo=lo, rel_diff_hi=hi, m=m, sigma=sig),
    )


# ----------------------------------------------------------------------------------------------
# G7
# ----------------------------------------------------------------------------------------------


def g7_prior(
    log_prior_ratio_fn,
    draws,
    w,
    transform,
    scenarios,
    theta=None,
    *,
    ess_min: float = 400.0,
    shift_max: float = 0.5,
    change_max: float = 0.2,
) -> GateResult:
    draws = jnp.asarray(draws, dtype=float)
    w = jnp.asarray(w, dtype=float)
    w = w / jnp.sum(w)
    base = predict.summarize(draws, w, transform)
    m0, s0 = np.asarray(base["m"]), np.asarray(base["sigma"])
    ess, shift, change, judged_bad, low = {}, {}, {}, [], []
    for sc in scenarios:
        lr = jnp.asarray(log_prior_ratio_fn(sc, theta), dtype=float)
        lw = jnp.log(jnp.maximum(w, 1e-300)) + lr
        wn = jnp.exp(lw - jax.scipy.special.logsumexp(lw))
        e = float(1.0 / jnp.sum(wn**2))
        ess[sc] = e
        if not e >= ess_min:
            low.append(sc)
            continue
        new = predict.summarize(draws, wn, transform)
        shift[sc] = (np.asarray(new["m"]) - m0) / s0
        change[sc] = np.asarray(new["sigma"]) / s0 - 1.0
        if np.any(np.abs(shift[sc]) >= shift_max) or np.any(np.abs(change[sc]) >= change_max):
            judged_bad.append(sc)
    st = dict(ess=ess, m_shift=shift, sigma_change=change, low_ess=low, violating=judged_bad)
    if judged_bad:
        return GateResult("G7", "fail", dict(st, verdict="prior-dominated"))
    if low:
        return GateResult("G7", "not testable", dict(st, verdict="refit needed"))
    return GateResult("G7", "pass", dict(st, verdict="pass"))
