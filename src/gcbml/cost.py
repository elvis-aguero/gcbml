"""Cost model (spec Section 2.6), learned from the costs the oracle reports.

For run r with unit controls u_r (n_controls,), integer level vector l_r (k,) (0 = coarsest) and an
optional oracle quote q_r (core-hours; None -> no quote):

    log2 c_r = kappa0 + sum_j g_j(l_rj) + a * has_q_r * (log2 q_r - qbar) + omega(u_r, l_r) + eta_r
    g_j(l) = sum_{m<l} Delta_{j,m},  Delta_{j,0} ~ N(gamma_j, s_gamma_j^2),  Delta_{j,m+1} = Delta_{j,m} +
    eps_jm,
    eps_jm ~ N(0, s_delta_j^2)
    omega ~ GP(0, sigma_w^2 ard_matern52 over (u, l / l_scale)),   eta_r ~ N(0, s_eta^2)

Delta_{j,m} is the log2 cost step of refining component j from level m to m + 1; it follows a random walk over
the levels, so the step may grow or shrink (spec 2.6). The walk is a Gaussian process in the level with
    E g_j(l) = gamma_j l,
    Cov(g_j(l), g_j(l')) = s_gamma_j^2 l l' + s_delta_j^2 [ (m-1) m (2m-1)/6 + |l - l'| (m-1) m / 2 ],  m =
    min(l, l'),
(from g(l) = l Delta_0 + sum_{i<l-1} (l-1-i) eps_i and sum_{a=1}^{m-1} a (a + d) with d = |l - l'|), defined
for any
level, probed or not. kappa0 and a are Gaussian too, so kappa0, Delta, a and omega are all integrated out
exactly.
A run stopped at its cap has a right-censored cost: log2 c_r >= log2 cap_r (Tobit; handled by data
augmentation, like model.censored_sweep).

Priors (problem-specific scales are required inputs, CostPrior): kappa0 ~ N(k0_mean, k0_sd^2); the walk as
above
(gamma_mean, gamma_sd = s_gamma, s_delta); a ~ N(1, 0.5^2) (a quote is informative but not trusted);
(sigma_w, ell_w) PC prior with sigma0 = 1 (log2 units), ell0 = 0.1; s_eta: exponential with P(s_eta > 1) =
0.05.

Inference: given (sigma_w, ell_w, s_eta) and the imputed censored values, log2 c is Gaussian with mean
A b0 + sum_j gamma_j l_j and covariance K_t = sigma_w^2 ard_matern52((u, l / l_scale)) + s_eta^2 I + A B A^T +
K_walk,
A = [1, has_q (log2q - qbar)], b0 = (k0_mean, 1), B = diag(k0_sd^2, 0.5^2); K_walk is the walk covariance
above.
The omega kernel has one length scale per input (d_u + k of them), so theta = (log sigma_w, log s_eta,
log ell_1..ell_{d_u+k}) has 2 + d_u + k entries; the s_eta prior is the exponential with rate -log 0.05; the
(sigma_w, ell_w) prior is priors.pc_matern_logpdf (applied per ARD length scale, a heuristic, spec 2.5).
Slice-sample theta (mcmc.slice), Gibbs the censored values (truncated normal full conditionals); every solve
goes
through gcbml.linalg. Censored rows hold log2 cap in ``log2c``. Predictions use the imputed log2c of each
posterior draw. ``coef_posterior`` (an addition to the stub) gives (kappa0, Delta steps, a) | y, hyper per
draw.

CostData: NamedTuple(U (n_pad, n_controls) unit, L (n_pad, k) float levels, log2c (n_pad,),
                     censored (n_pad,) bool, log2q (n_pad,) (0 where no quote), has_q (n_pad,) bool,
                     mask (n_pad,) bool)
CostPrior: dataclass(k0_mean, k0_sd, gamma_mean (k,), gamma_sd (k,), s_delta (k,) [all required], l_scale=4.0)
fit_cost(key, data, prior, n_warmup, n_samples, n_chains=4) -> CostPosterior (draws of hyperparameters and
    imputed log2c; diagnostics as in mcmc.diagnostics.summary)
predict_log2(post, U_new, L_new, log2q_new, has_q_new) -> (mean (S, m), var (S, m))
    Predictive of log2 c for new runs per posterior draw (includes eta).
expected_cost(mean, var, w) -> (m,)       E[c] = weighted average of 2^mean * exp(0.5 (ln 2)^2 var).
cost_cap(mean, var, w, q=0.95) -> (m,)     quantile q of the pooled predictive of c (mixture over draws;
    solve the mixture CDF by bisection in log2 space).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import log_ndtr, ndtr, ndtri

from gcbml import linalg
from gcbml.kernels import ard_matern
from gcbml.mcmc.chains import run_chains
from gcbml.mcmc.diagnostics import summary
from gcbml.mcmc.slice import slice_step
from gcbml.model import _lower_truncated_normal
from gcbml.priors import pc_matern_logpdf

A_PRIOR_MEAN = 1.0
A_PRIOR_SD = 0.5
SIGMA0_W = 1.0
ELL0_W = 0.1
S_ETA_RATE = -math.log(0.05)  # P(s_eta > 1) = 0.05 for an exponential prior
NU = 2.5
LN2 = math.log(2.0)


class CostData(NamedTuple):
    U: Any
    L: Any
    log2c: Any
    censored: Any
    log2q: Any
    has_q: Any
    mask: Any


@dataclass(frozen=True)
class CostPrior:
    k0_mean: float
    k0_sd: float
    gamma_mean: tuple[float, ...]
    gamma_sd: tuple[float, ...]  # s_gamma
    s_delta: float | tuple[float, ...]  # sd of the random-walk increment of the per-level step, log2 units
    l_scale: float = 4.0  # scale of the levels as GP inputs (a numerical scale, not a prior)


class CostPosterior(NamedTuple):
    hyper: dict  # sigma_w (S,), s_eta (S,), ell (S, d_u + k); draws pooled over chains
    log2c: Any  # (S, n_pad) imputed log2 costs (observed rows unchanged)
    diagnostics: dict
    data: CostData
    prior: CostPrior


def _clean(data: CostData) -> CostData:
    """Zero every padded row (garbage, even NaN, in padded rows must not enter anything)."""
    mask = jnp.asarray(data.mask, dtype=bool)
    m1 = mask[:, None]
    has_q = jnp.asarray(data.has_q, dtype=bool) & mask
    return CostData(
        U=jnp.where(m1, jnp.asarray(data.U, dtype=float), 0.0),
        L=jnp.where(m1, jnp.asarray(data.L, dtype=float), 0.0),
        log2c=jnp.where(mask, jnp.asarray(data.log2c, dtype=float), 0.0),
        censored=jnp.asarray(data.censored, dtype=bool) & mask,
        log2q=jnp.where(has_q, jnp.asarray(data.log2q, dtype=float), 0.0),
        has_q=has_q,
        mask=mask,
    )


def _qbar(data: CostData):
    """Mean log2 quote over the real runs that have one (0 if there is none); data must be clean."""
    n = jnp.sum(data.has_q)
    return jnp.where(n > 0, jnp.sum(data.log2q) / jnp.maximum(n, 1), 0.0)


def _design(log2q, has_q, qbar):
    """A = [1, has_q (log2q - qbar)], (n, 2)."""
    hq = jnp.asarray(has_q, dtype=float)
    q = hq * (jnp.asarray(log2q, dtype=float) - qbar)
    return jnp.stack([jnp.ones_like(q), q], axis=1)


def _beta_prior(prior: CostPrior):
    """Prior mean and sd of (kappa0, a)."""
    return jnp.asarray([prior.k0_mean, A_PRIOR_MEAN], dtype=float), jnp.asarray(
        [prior.k0_sd, A_PRIOR_SD], dtype=float
    )


def _walk_scales(prior: CostPrior):
    k = len(prior.gamma_mean)
    sg = jnp.asarray(prior.gamma_sd, dtype=float)
    sd = jnp.broadcast_to(jnp.asarray(prior.s_delta, dtype=float), (k,))
    return jnp.asarray(prior.gamma_mean, dtype=float), sg, sd


def _walk_mean(L, prior: CostPrior):
    """E sum_j g_j(l_j) = sum_j gamma_j l_j, (n,)."""
    gamma, _, _ = _walk_scales(prior)
    return jnp.asarray(L, dtype=float) @ gamma


def _walk_cov(La, Lb, prior: CostPrior):
    """Cov(sum_j g_j(La_j), sum_j g_j(Lb_j)) with broadcasting of La (..., k) against Lb (..., k).

    Per component, with m = min(l, l'), d = |l - l'|:
    s_gamma^2 l l' + s_delta^2 [(m-1) m (2m-1)/6 + d (m-1) m / 2].
    """
    _, sg, sd = _walk_scales(prior)
    La, Lb = jnp.asarray(La, dtype=float), jnp.asarray(Lb, dtype=float)
    m, d = jnp.minimum(La, Lb), jnp.abs(La - Lb)
    walk = (m - 1.0) * m * (2.0 * m - 1.0) / 6.0 + d * (m - 1.0) * m / 2.0
    return jnp.sum(sg**2 * La * Lb + sd**2 * walk, axis=-1)


def _inputs(U, L, prior: CostPrior):
    return jnp.concatenate([jnp.asarray(U, dtype=float), jnp.asarray(L, dtype=float) / prior.l_scale], axis=1)


class _Ctx(NamedTuple):
    data: CostData  # clean
    X: Any
    A: Any
    b0: Any
    bsd: Any
    qbar: Any
    m0: Any  # prior mean of log2 c at the real runs: A b0 + sum_j gamma_j l_j
    Kw: Any  # walk covariance between the real runs


def _context(data: CostData, prior: CostPrior) -> _Ctx:
    d = _clean(data)
    qbar = _qbar(d)
    A = jnp.where(d.mask[:, None], _design(d.log2q, d.has_q, qbar), 0.0)
    b0, bsd = _beta_prior(prior)
    m0 = jnp.where(d.mask, A @ b0 + _walk_mean(d.L, prior), 0.0)
    Kw = _walk_cov(d.L[:, None, :], d.L[None, :, :], prior)
    return _Ctx(d, _inputs(d.U, d.L, prior), A, b0, bsd, qbar, m0, Kw)


def _unpack(theta):
    return jnp.exp(theta[0]), jnp.exp(theta[1]), jnp.exp(theta[2:])


def _total_cov(ctx: _Ctx, sigma_w, s_eta, ell):
    """K_t = sigma_w^2 ard + s_eta^2 I + A B A^T + K_walk on the real block (beta, Delta integrated)."""
    mask = ctx.data.mask
    K = sigma_w**2 * ard_matern(ctx.X, ctx.X, ell, NU) + (ctx.A * ctx.bsd**2) @ ctx.A.T + ctx.Kw
    K = K + s_eta**2 * jnp.eye(K.shape[0])
    return jnp.where(mask[:, None] & mask[None, :], K, 0.0)


def _log_prior(theta):
    sigma_w, s_eta, _ = _unpack(theta)
    lp = pc_matern_logpdf(theta[0], theta[2:], SIGMA0_W, ELL0_W)
    # s_eta ~ Exponential(rate); density of log s_eta = log rate - rate s + log s
    return lp + math.log(S_ETA_RATE) - S_ETA_RATE * s_eta + theta[1]


def _log_post(theta, y, ctx: _Ctx):
    sigma_w, s_eta, ell = _unpack(theta)
    F = linalg.factor(_total_cov(ctx, sigma_w, s_eta, ell), ctx.data.mask)
    r = jnp.where(ctx.data.mask, y - ctx.m0, 0.0)
    val = linalg.gaussian_logpdf(F, r, ctx.data.mask) + _log_prior(theta)
    return jnp.where(jnp.isfinite(val), val, -jnp.inf)


def _censored_sweep(key, theta, y, ctx: _Ctx):
    """One Gibbs sweep over the censored rows: Gaussian full conditional truncated to y_i >= log2 cap_i.

    With Q the precision of the beta- and omega-integrated Gaussian, mean_i = y_i - (Q r)_i / Q_ii and
    var_i = 1 / Q_ii (same algebra as model.censored_sweep); Q e_i comes from two triangular solves.
    """
    sigma_w, s_eta, ell = _unpack(theta)
    mask = ctx.data.mask
    n = y.shape[0]
    F = linalg.factor(_total_cov(ctx, sigma_w, s_eta, ell), mask)
    bounds = ctx.data.log2c
    r = jnp.where(mask, y - ctx.m0, 0.0)
    Qr = linalg.solve(F, r)
    do = ctx.data.censored & mask

    def step(carry, inp):
        y, Qr = carry
        i, k, flag = inp

        def draw(_):
            col = linalg.solve(F, jnp.zeros(n).at[i].set(1.0))
            qii = col[i]
            sd = 1.0 / jnp.sqrt(qii)
            m = y[i] - Qr[i] / qii
            x = _lower_truncated_normal(k, (bounds[i] - m) / sd)
            yi = m + sd * x
            return y.at[i].set(yi), Qr + col * (yi - y[i])

        return lax.cond(flag, draw, lambda _: (y, Qr), None), None

    (y_new, _), _ = lax.scan(step, (y, Qr), (jnp.arange(n), jax.random.split(key, n), do))
    return y_new


def fit_cost(
    key, data: CostData, prior: CostPrior, n_warmup: int, n_samples: int, n_chains: int = 4
) -> CostPosterior:
    """Slice-sample theta given the imputed costs, Gibbs-impute the censored costs given theta."""
    ctx = _context(data, prior)
    mask, cens = ctx.data.mask, ctx.data.censored
    n_pad = mask.shape[0]
    p = ctx.X.shape[1]
    has_cens = bool(np.any(np.asarray(cens)))
    y0 = jnp.where(cens, ctx.data.log2c + 0.5, ctx.data.log2c)

    k_init, k_run = jax.random.split(key)
    base = jnp.concatenate([jnp.log(jnp.asarray([0.5, 0.3])), jnp.full(p, math.log(0.5))])
    theta0 = base + 0.3 * jax.random.normal(k_init, (n_chains, 2 + p))
    init = {"theta": theta0, "y": jnp.broadcast_to(y0, (n_chains, n_pad))}
    init_widths = {"theta": jnp.ones(2 + p), "y": jnp.ones(n_pad)}

    def make_step(widths):
        w = widths["theta"]

        def step(k, state):
            k1, k2 = jax.random.split(k)
            y = state["y"]
            theta, n_evals = slice_step(k1, state["theta"], lambda t: _log_post(t, y, ctx), w)
            if has_cens:
                y = _censored_sweep(k2, theta, y, ctx)
            return {"theta": theta, "y": y}, {"n_evals": n_evals}

        return step

    res = run_chains(k_run, init, make_step, init_widths, n_warmup, n_samples, n_chains)
    th = np.asarray(res.samples["theta"])  # (chains, n, 2 + p)
    diag_in = {"sigma_w": np.exp(th[..., 0]), "s_eta": np.exp(th[..., 1])}
    for j in range(p):
        diag_in[f"ell_{j}"] = np.exp(th[..., 2 + j])
    diagnostics = summary(diag_in)
    diagnostics["n_evals"] = res.info["n_evals"]
    flat = th.reshape(-1, 2 + p)
    hyper = {
        "sigma_w": jnp.asarray(np.exp(flat[:, 0])),
        "s_eta": jnp.asarray(np.exp(flat[:, 1])),
        "ell": jnp.asarray(np.exp(flat[:, 2:])),
    }
    log2c = jnp.asarray(np.asarray(res.samples["y"]).reshape(-1, n_pad))
    return CostPosterior(hyper, log2c, diagnostics, ctx.data, prior)


def coef_posterior(post: CostPosterior, max_level: int = 0):
    """Posterior of (kappa0, Delta, a) given y and the hyperparameters, per draw.

    Delta holds the log2 steps Delta_{j,m}, m = 0..M-1, for every component j, with M = max(max_level, highest
    probed level), so a level above the probed ones can be asked for (its steps are then the walk's
    extrapolation). Returns (mean (S, 2 + k M), cov (S, 2 + k M, 2 + k M)) in the order
    (kappa0, Delta_{0,0..M-1}, Delta_{1,0..M-1}, ..., a); Gaussian, with omega and eta integrated out:
    mean = b0 + B A^T K_t^{-1} (y - m0), cov = B - B A^T K_t^{-1} A B,
    A = [1, Phi, quote], Phi_{r,(j,m)} = 1[m < l_rj].
    """
    ctx = _context(post.data, post.prior)
    k = ctx.data.L.shape[1]
    real = np.asarray(ctx.data.mask)
    M = int(max(max_level, np.max(np.asarray(ctx.data.L)[real]) if real.any() else 0))
    gamma, sg, sd = _walk_scales(post.prior)
    m_idx = jnp.arange(M, dtype=float)
    # Phi: (n, k, M)
    Phi = (m_idx[None, None, :] < ctx.data.L[:, :, None]).astype(float)
    Phi = jnp.where(ctx.data.mask[:, None, None], Phi, 0.0).reshape(Phi.shape[0], k * M)
    Af = jnp.concatenate([ctx.A[:, :1], Phi, ctx.A[:, 1:]], axis=1)  # (n, 2 + k M)
    b0 = jnp.concatenate([ctx.b0[:1], jnp.repeat(gamma, M), ctx.b0[1:]])
    walk = sg[:, None, None] ** 2 + sd[:, None, None] ** 2 * jnp.minimum(
        m_idx[None, :, None], m_idx[None, None, :]
    )
    B = jnp.zeros((2 + k * M, 2 + k * M))
    B = B.at[0, 0].set(ctx.bsd[0] ** 2).at[-1, -1].set(ctx.bsd[1] ** 2)
    for j in range(k):
        B = B.at[1 + j * M : 1 + (j + 1) * M, 1 + j * M : 1 + (j + 1) * M].set(walk[j])
    BAt = B @ Af.T  # (q, n)

    def one(sigma_w, s_eta, ell, y):
        F = linalg.factor(_total_cov(ctx, sigma_w, s_eta, ell), ctx.data.mask)
        r = jnp.where(ctx.data.mask, y - ctx.m0, 0.0)
        V = jax.scipy.linalg.solve_triangular(F.L, BAt.T, lower=True)  # (n, q)
        mean = b0 + BAt @ linalg.solve(F, r)
        return mean, B - V.T @ V

    h = post.hyper
    return jax.vmap(one)(h["sigma_w"], h["s_eta"], h["ell"], post.log2c)


def predict_log2(post: CostPosterior, U_new, L_new, log2q_new, has_q_new):
    """Predictive of log2 c at new runs, per posterior draw: (mean (S, m), var (S, m)), eta included."""
    ctx = _context(post.data, post.prior)
    L_new = jnp.asarray(L_new, dtype=float)
    has_q_new = jnp.asarray(has_q_new, dtype=bool)
    log2q_new = jnp.where(has_q_new, jnp.asarray(log2q_new, dtype=float), 0.0)
    An = _design(log2q_new, has_q_new, ctx.qbar)
    Xn = _inputs(U_new, L_new, post.prior)
    mask = ctx.data.mask
    AB = ctx.A * ctx.bsd**2
    vb = jnp.sum(An * ctx.bsd**2 * An, axis=1) + _walk_cov(L_new, L_new, post.prior)
    Cw = _walk_cov(ctx.data.L[:, None, :], L_new[None, :, :], post.prior)
    mn0 = An @ ctx.b0 + _walk_mean(L_new, post.prior)

    def one(sigma_w, s_eta, ell, y):
        F = linalg.factor(_total_cov(ctx, sigma_w, s_eta, ell), mask)
        C = sigma_w**2 * ard_matern(ctx.X, Xn, ell, NU) + AB @ An.T + Cw
        C = jnp.where(mask[:, None], C, 0.0)
        v = sigma_w**2 + s_eta**2 + vb
        r = jnp.where(mask, y - ctx.m0, 0.0)
        mean, var = linalg.conditional(F, C, v, r)
        return mean + mn0, jnp.maximum(var, 0.0)

    h = post.hyper
    return jax.vmap(one)(h["sigma_w"], h["s_eta"], h["ell"], post.log2c)


def _norm_w(w, S):
    w = jnp.asarray(w, dtype=float)
    return w / jnp.sum(w)


def expected_cost(mean, var, w):
    """E[c] = sum_s w_s 2^{mean_s} exp(0.5 (ln 2)^2 var_s), with w normalised to sum to one."""
    mean, var = jnp.asarray(mean, dtype=float), jnp.asarray(var, dtype=float)
    w = _norm_w(w, mean.shape[0])
    return jnp.sum(w[:, None] * jnp.exp(LN2 * mean + 0.5 * LN2**2 * var), axis=0)


def expected_capped_cost(mean, var, w, cap):
    """E[min(c, cap)] under the pooled lognormal mixture of log2 c ~ N(mean_s, var_s), weights w_s.

    A run is stopped at its cap, so this is the cost a run is charged (spec 2.6). Per component, with
    X = 2^Y, m = ln2 mean, s = ln2 sqrt(var), k = cap:
        E[X; X < k] = exp(m + s^2/2) Phi((ln k - m - s^2)/s),   P(X >= k) = Phi((m - ln k)/s),
        E[min(X, k)] = E[X; X < k] + k P(X >= k).
    ``cap`` is a scalar or an array of shape (m,), in the units of c. Tends to expected_cost as cap -> inf.
    """
    mean, var = jnp.asarray(mean, dtype=float), jnp.asarray(var, dtype=float)
    w = _norm_w(w, mean.shape[0])
    cap = jnp.asarray(cap, dtype=float)
    m = LN2 * mean
    s = LN2 * jnp.sqrt(jnp.maximum(var, 1e-300))
    lk = jnp.log(cap)
    # E[X; X < k] = exp(log Phi(z1) + m + s^2/2): in log space, the product of an overflowing and an
    # underflowing factor
    part = jnp.exp(log_ndtr((lk - m - s**2) / s) + m + 0.5 * s**2) + cap * ndtr((m - lk) / s)
    return jnp.sum(w[:, None] * part, axis=0)


def prob_finish(mean, var, w, cap):
    """P(c <= cap) under the pooled mixture of log2 c ~ N(mean_s, var_s), weights w_s; ``cap`` in the units of
    c.

    The probability that a run stopped at ``cap`` finishes before it (spec 2.6): sum_s w_s Phi((log2 cap -
    mean_s) / sd_s).
    """
    mean, var = jnp.asarray(mean, dtype=float), jnp.asarray(var, dtype=float)
    w = _norm_w(w, mean.shape[0])
    sd = jnp.sqrt(jnp.maximum(var, 1e-300))
    return jnp.sum(w[:, None] * ndtr((jnp.log2(jnp.asarray(cap, dtype=float)) - mean) / sd), axis=0)


def cost_cap(mean, var, w, q: float = 0.95):
    """Quantile q of the mixture sum_s w_s N(mean_s, var_s) of log2 c, mapped to c = 2^x.

    The mixture quantile lies between the smallest and the largest component quantile, so that interval
    brackets the root; 200 bisection steps in log2 space (far beyond float64 resolution).
    """
    mean, var = jnp.asarray(mean, dtype=float), jnp.asarray(var, dtype=float)
    w = _norm_w(w, mean.shape[0])
    sd = jnp.sqrt(jnp.maximum(var, 1e-300))
    zq = ndtri(jnp.asarray(q, dtype=float))
    comp = mean + zq * sd
    lo, hi = jnp.min(comp, axis=0), jnp.max(comp, axis=0)

    def body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        cdf = jnp.sum(w[:, None] * ndtr((mid - mean) / sd), axis=0)
        below = cdf < q
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = lax.fori_loop(0, 200, body, (lo, hi))
    return jnp.exp2(0.5 * (lo + hi))
