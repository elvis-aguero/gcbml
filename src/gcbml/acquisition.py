"""Step 5 of the algorithm (spec Section 3): the next runs, by maximum uncertainty reduction per cost.

    a* = argmax_{a admissible} (H_n - E J_n(a)) / E c(a)          (MR-SUR, Stroh et al. 2007.13553 eq 11)
    H_n = sum_{x in Sigma_N} (sigma_epi^2(x) / eps^2(x) - 1)_+     (hinge, mode "hinge"; zero iff P1 holds)
    H_n = beta^{-1} log sum_x exp(beta sigma_epi^2 / eps^2)         (mode "softmax", beta = 20; P2 criterion)

sigma_epi(x) is the PHYSICAL half-width (q84 - q16)/2 of the pooled posterior of f(x) = Lambda^{-1}(mu(x))
(spec 2.8): the mixture over posterior draws (and structures, with their stacking weights) of the
Gaussian conditionals of mu(x) at h = 0, its quantiles found in Lambda units (bisection on the mixture
CDF) and mapped through Lambda^{-1} (monotone; for a decreasing Lambda, q16 and q84 swap).

A candidate run a = (u, h): its outputs are the rows Xa (m, d) (m > 1 under module S1: one row per v),
all at the same hbar (k,). Order of new rows: v1 supports a shared order only (P columns constant);
new rows take P[0] of each draw. [assumption; varying order needs pi(x) at new x: later]

Fantasies (spec Step 5): for one candidate and posterior draws s = 1..S (pooled, weights w_s):
  1. pick a draw s ~ w; draw the fantasy outputs ya ~ joint predictive of the m outputs under draw s
     (model.joint_new: mean and covariance including noise and the within-run correlation);
  2. for every draw s', condition its GP on (data + ya) exactly: linalg.append on its factor;
  3. reweight: w'_s' proportional to w_s' * p(ya | data, draw s') (the joint predictive density);
     if the effective sample size (sum w')^2 / sum w'^2 < 50 [assumption, spec], keep w (conservative);
  4. recompute sigma_epi on Sigma_N with the conditioned draws and w', then H.
  E J_n(a) = average of H over the fantasies; gain = max(H_n - E J_n, 0) (negative = MC noise, clipped).
  Fantasies: start at 16, double until the MC standard error of the gain of the best candidate is below
  10% of the gap to the second best, or 256 is reached.
Stacking weights over structures are held fixed during fantasies [assumption, spec].

Budget (spec Step 5): a candidate is admissible only if cap(a) <= C_rem = budget - spent - sum caps of
pending runs. Pending runs enter as variance-only conditioning (append their rows to every draw's factor
with their predictive mean as the value: the posterior variance does not depend on the value).

Batches: greedy. After choosing a, add it to the pending set (variance-only), recompute, choose the next,
until q runs are chosen or nothing admissible is left.

Implemented in W4-B (see the notes at the end of the docstring).

model.joint_new(params, data, z, cfg, Xn, Hn, Pn, same_run=True) -> (mean (m,), cov (m, m))
    Joint predictive of m NEW outputs of one new run in Lambda units, given the data, beta integrated,
    including the noise (noise_var_new (m,) argument) and, if cfg.within_run, their within-run correlation.
    Plus the cross-covariance needed to append them to the factor (return a small NamedTuple).

Interfaces in this module:
Candidate: NamedTuple(Xa (m, d) unit, hbar (k,), levels (k,) int, cost_mean, cost_cap, noise_var (m,) per
    draw-free estimate or (S, m))
StructurePosterior: NamedTuple(params: ModelParams with leading axis S, z (S, n_pad), cfg: ModelConfig,
    transform: transforms.Transform, weight: float)
sigma_epi_physical(structures, data, Xs) -> (s,)
H_value(sigma_epi, eps, mode="hinge", beta=20.0) -> scalar
expected_gain(key, structures, data, cand, Xs, eps, n_fantasy, mode) -> (gain, mc_se)
select_batch(key, structures, data, candidates, Xs, eps, q, budget_remaining, pending=(), mode="hinge",
             max_draws=64) -> (chosen indices, table of (gain, cost_mean, ratio, admissible) per candidate)
    max_draws: subsample this many pooled draws (by weight, systematic resampling) for speed; document it.

How it is computed (W4-B) and where it differs from the stubs:
  * One joint Gaussian per draw. For every draw, the posterior (given the data, beta integrated) of
    u = [mu(Xs); the outputs of every candidate and pending run] is built once (model._new_blocks and
    model._posterior_joint; the blocks are those of model.joint_new). Everything afterwards is small linear
    algebra on (U, U) arrays, vmapped over draws: no step refactors the n x n data matrix. Observing a
    candidate's rows is the Gaussian conditioning of u: covariance minus G C^{-1} G^T, mean plus
    G C^{-1} (ya - mean). This equals extending the draw's factor with linalg.append and reading predict_mu
    on the enlarged data (tests/test_acquisition.py checks it, for flat and Gaussian beta).
  * Draws: all (expected_gain, sigma_epi_physical) or max_draws of them (select_batch), thinned by
    systematic resampling at the fixed offset 1/2, so the call is deterministic. A structure that receives
    no draw drops out of that call.
  * sigma_epi of the pool: all structures with one Lambda: Newton/bisection on the mixture CDF in Lambda
    units, then Lambda^{-1} (spec 2.8). Structures with different Lambda: bisection on the physical-scale
    mixture CDF. A Gaussian on the Lambda scale that puts mass outside the domain of Lambda^{-1} (reciprocal
    with mu near 0) has no finite sigma_epi; such draws give inf/NaN and are not guarded.
  * The 16/84 half-width of a Gaussian is 0.9945 sd, not sd: sigma_epi of one Gaussian draw with the
    identity is 0.9945 times its predictive sd (the stub text equates them).
  * Fantasies: the draw and the output come from the pooled weights; reweighting is within each structure
    (structure weights fixed, as the spec says); a fantasy drawn in the units of one structure is mapped to
    the others through Lambda^{-1} (Jacobians cancel within a structure). ESS is that of the pooled weights;
    if ESS < ess_min the weights are kept (ess_min: keyword added to expected_gain and select_batch).
    Default 10 [assumption, reviewer 2026-10-04]: the spec's 50 nearly disables reweighting at max_draws = 64
    (it falls back whenever the effective draws drop by more than 22%), and it falls back exactly for the
    probes that are most informative about the order p, i.e. it undervalues unprobed finer levels (A12).
    The A12 synthetic test decides the final value.
  * Fantasies per candidate: 16 for every admissible one, then doubled for the best two only, until the s.e.
    of the best ratio is below 10% of its gap to the second best or 256 is reached (spec: "doubled until").
  * Variance-only property: with one draw and the identity, sigma_epi does not depend on the fantasy value, so
    the expected gain equals the deterministic gain exactly (tests). With a log or reciprocal transform, or
    several draws, the value moves the means and the property does not hold.
  * Added to the stubs: expected_gain_detail, GainDetail, GainTable (select_batch returns the table of the
    first greedy step as a GainTable of arrays, indexed by field name), ess_min, n_fantasy_start/max.
    Each candidate can be chosen once per batch (list replicates as separate candidates). A batch ends early
    when no admissible candidate has a positive gain.
  * Candidate.noise_var: (m,), or (S_k, m) with S_k the number of draws of EACH structure (per-draw noise).
    New rows take the order P[0] of each draw, hbar = Candidate.hbar for all m rows, and form a run of their
    own (their noise is independent of the data's). Candidate.levels is not used here.
"""

from __future__ import annotations

import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp, ndtr
from scipy.special import ndtri as _ndtri

from gcbml import linalg, model
from gcbml.data import PaddedData
from gcbml.kernels import ard_matern

_P_LO, _P_HI = 0.16, 0.84
_PS = np.array([_P_LO, _P_HI])
_Z84 = float(_ndtri(_P_HI))  # the 16/84 quantile half-width of a Gaussian is _Z84 * sd (0.9945 sd)
_BISECT = 48  # bracket width * 2^-48 ~ 4e-15 of the bracket


class Candidate(NamedTuple):
    Xa: Any
    hbar: Any
    levels: Any
    cost_mean: float
    cost_cap: float
    noise_var: Any


class StructurePosterior(NamedTuple):
    params: Any
    z: Any
    cfg: Any
    transform: Any
    weight: float


class GainDetail(NamedTuple):
    """expected_gain plus what is needed to judge it: fantasies used and the share that kept the weights."""

    gain: float
    mc_se: float
    n_fantasy: int
    fallback_frac: float


class GainTable(NamedTuple):
    """Per candidate (arrays of length n_candidates), at the first greedy step of select_batch."""

    gain: Any
    mc_se: Any
    cost_mean: Any
    ratio: Any
    admissible: Any
    n_fantasy: Any
    fallback_frac: Any


# ----------------------------------------------------------------------------------------------
# sigma_epi: quantile half-width of the pooled posterior of f(x), and H
# ----------------------------------------------------------------------------------------------


def _same_units(segs) -> bool:
    return len({(s[2].name, s[3]) for s in segs}) == 1


def _z_quantiles(ps, m, sd, W):
    """Quantiles ps (2,) of the mixture sum_d W_d N(m_d, sd_d^2) per column; m, sd (D, s) -> (2, s).

    The p-quantile of a mixture lies between the smallest and the largest of the components' p-quantiles
    (the mixture CDF is the weighted mean of the component CDFs): that is the bracket. Safeguarded Newton
    on the mixture CDF from the weighted mean of the component quantiles, bisection when a Newton step
    leaves the bracket; it stops when the step is below 1e-13 of the bracket scale (a few iterations).
    """
    zp = jax.scipy.special.ndtri(ps)
    qd = m[:, None, :] + sd[:, None, :] * zp[None, :, None]  # (D, 2, s) component quantiles
    lo, hi = jnp.min(qd, axis=0), jnp.max(qd, axis=0)
    x0 = jnp.sum(W[:, None, None] * qd, axis=0)
    scale = 1e-13 * (jnp.abs(lo) + jnp.abs(hi) + (hi - lo))

    def step(st):
        it, x, lo, hi, _ = st
        u = (x[None] - m[:, None, :]) / sd[:, None, :]
        F = jnp.sum(W[:, None, None] * ndtr(u), axis=0)
        f = jnp.sum(W[:, None, None] * jnp.exp(-0.5 * u * u) / sd[:, None, :], axis=0) / jnp.sqrt(
            2.0 * jnp.pi
        )
        lo = jnp.where(F < ps[:, None], x, lo)
        hi = jnp.where(F < ps[:, None], hi, x)
        newton = (F - ps[:, None]) / jnp.maximum(f, 1e-300)
        done = jnp.abs(newton) <= scale  # the Newton step is below the tolerance: x is the root
        xn = x - newton
        xn = jnp.where((xn > lo) & (xn < hi), xn, 0.5 * (lo + hi))
        xn = jnp.where(done, x, xn)
        return it + 1, xn, lo, hi, jnp.where(done, 0.0, jnp.abs(xn - x))

    def cont(st):
        it, _, _, _, dx = st
        return (it < 100) & jnp.any(dx > 0.0)

    init = (0, x0, lo, hi, jnp.full_like(x0, jnp.inf))
    return jax.lax.while_loop(cont, step, init)[1]


def _y_quantiles(ps, m, sd, W, segs):
    """Quantiles of the physical mixture over structures with different Lambda (bisection on y)."""
    s = m.shape[1]
    lo = jnp.full((2, s), jnp.inf)
    hi = jnp.full((2, s), -jnp.inf)
    for a, b, tf, _ in segs:
        for edge in (m[a:b] - 8.0 * sd[a:b], m[a:b] + 8.0 * sd[a:b]):
            y = tf.inverse(edge)
            ok = jnp.isfinite(y) & tf.domain_ok(y)
            lo = jnp.minimum(lo, jnp.min(jnp.where(ok, y, jnp.inf), axis=0))
            hi = jnp.maximum(hi, jnp.max(jnp.where(ok, y, -jnp.inf), axis=0))

    def cdf(y):
        tot = 0.0
        for a, b, tf, inc in segs:
            ok = tf.domain_ok(y)
            u = (tf.forward(jnp.where(ok, y, 1.0))[None] - m[a:b, None, :]) / sd[a:b, None, :]
            c = jnp.where(ok[None], ndtr(u) if inc else ndtr(-u), 0.0)
            tot = tot + jnp.sum(W[a:b, None, None] * c, axis=0)
        return tot

    def body(_, st):
        lo, hi = st
        mid = 0.5 * (lo + hi)
        below = cdf(mid) < ps[:, None]
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(0, _BISECT, body, (lo, hi))
    return 0.5 * (lo + hi)


def _halfwidth(m, sd, W, segs):
    """(q84 - q16) / 2 of the pooled physical posterior, per column. m, sd (D, s); W (D,) sums to 1."""
    ps = jnp.array([_P_LO, _P_HI])
    sd = jnp.maximum(sd, 1e-12)
    if _same_units(segs):
        tf = segs[0][2]
        q = tf.inverse(
            _z_quantiles(ps, m, sd, W)
        )  # monotone: the 16/84 quantiles map to the 16/84 (or 84/16)
        return 0.5 * jnp.abs(q[1] - q[0])
    q = _y_quantiles(ps, m, sd, W, segs)
    return 0.5 * (q[1] - q[0])


def _draw_sigma(m, sd, segs):
    """Own half-width of each draw (a Gaussian in its Lambda units), physical scale: (D, s)."""
    out = []
    for a, b, tf, _ in segs:
        hi, lo = tf.inverse(m[a:b] + _Z84 * sd[a:b]), tf.inverse(m[a:b] - _Z84 * sd[a:b])
        out.append(0.5 * jnp.abs(hi - lo))
    return jnp.concatenate(out, axis=0)


def H_value(sigma_epi, eps, mode: str = "hinge", beta: float = 20.0):
    """H_n of the module docstring from sigma_epi (s,) and eps (scalar or (s,))."""
    r = (jnp.asarray(sigma_epi, dtype=float) / jnp.asarray(eps, dtype=float)) ** 2
    if mode == "hinge":
        return jnp.sum(jnp.maximum(r - 1.0, 0.0))
    if mode == "softmax":
        return logsumexp(beta * r) / beta
    raise ValueError(f"mode must be 'hinge' or 'softmax', got {mode!r}")


# ----------------------------------------------------------------------------------------------
# Pooled draws
# ----------------------------------------------------------------------------------------------


def _n_draws(st: StructurePosterior) -> int:
    return int(jnp.shape(st.params.sigma_mu)[0])


def _select_draws(structures, max_draws):
    """[(structure index, draw indices (n_k,), pooled weights (n_k,))] summing to 1.

    Pooled weight of draw j of structure k is weight_k / S_k (weights normalised over structures). If the
    pool has more than max_draws draws, it is thinned by systematic resampling at the fixed offset 1/2
    (deterministic: the same call gives the same draws, and the counts, hence the compiled shapes, change
    only when the weights change); every kept draw then has weight 1/max_draws.
    """
    sizes = [_n_draws(st) for st in structures]
    sw = np.array([float(st.weight) for st in structures])
    sw = sw / sw.sum()
    W = np.concatenate([np.full(n, w / n) for n, w in zip(sizes, sw, strict=True)])
    owner = np.concatenate([np.full(n, k) for k, n in enumerate(sizes)])
    local = np.concatenate([np.arange(n) for n in sizes])
    if max_draws is None or W.size <= max_draws:
        pick, Wp = np.arange(W.size), W
    else:
        cum = np.cumsum(W)
        cum[-1] = 1.0
        pick = np.minimum(np.searchsorted(cum, (np.arange(max_draws) + 0.5) / max_draws), W.size - 1)
        Wp = np.full(max_draws, 1.0 / max_draws)
    out = []
    for k in range(len(structures)):
        sel = owner[pick] == k
        if sel.any():
            out.append((k, local[pick][sel], Wp[sel]))
    return out


class _Pool(NamedTuple):
    mean: tuple  # per structure (S_k, U): posterior mean of u = [mu(Xs); new output rows of all blocks]
    cov: tuple  # per structure (S_k, U, U)
    w: tuple  # per structure (S_k,): weights within the structure, sum 1
    sw: Any  # (K,) structure weights
    segs: tuple  # per structure (start, stop, transform, increasing) into the concatenated draws
    s: int  # number of Sigma_N points: the first s entries of u
    rows: list  # per candidate block: the indices of its rows in u


@functools.partial(jax.jit, static_argnames=("cfg",))
def _build_struct(params, z, dX, dH, drun, dmask, Xs, Xn, Hn, grp, nv, cfg):
    """Posterior (given the data, beta integrated) of u = [mu(Xs); new outputs], per draw."""
    s = Xs.shape[0]

    def one(p, zz, nvr):
        data = PaddedData(X=dX, H=dH, y=zz, censored=jnp.zeros_like(dmask), run=drun, mask=dmask)
        S = model._setup(p, data, cfg)
        Pn = jnp.broadcast_to(p.P[0], Hn.shape)  # v1: a shared order, the first row of P (real rows first)
        nb = model._new_blocks(p, data, cfg, S, Xn, Hn, Pn, nvr, grp)
        Ms = model.mean_basis(Xs, cfg.mean_basis)
        Cmu = jnp.where(
            S.mask[:, None], S.rho1[:, None] * p.sigma_mu**2 * ard_matern(dX, Xs, p.ell_mu, cfg.nu_x), 0.0
        )
        Kmm = p.sigma_mu**2 * ard_matern(Xs, Xs, p.ell_mu, cfg.nu_x)
        Kmz = p.sigma_mu**2 * ard_matern(Xs, Xn, p.ell_mu, cfg.nu_x) * nb.rho1_new[None, :]
        offset = jnp.concatenate([jnp.zeros(s), nb.rho0_new])
        Au = jnp.concatenate([Ms, nb.A_new])
        Cdu = jnp.concatenate([Cmu, nb.cross], axis=1)
        Kuu = jnp.block([[Kmm, Kmz], [Kmz.T, nb.k_new]])
        return model._posterior_joint(p, data, zz, cfg, S, offset, Au, Cdu, Kuu)

    return jax.vmap(one)(params, z, nv)


def _draw_noise(noise_var, take, m):
    """Noise variance of a candidate's m rows per kept draw: (m,) shared, or (S_k, m) per draw."""
    nv = jnp.asarray(noise_var, dtype=float)
    if nv.ndim == 1:
        return jnp.broadcast_to(nv, (take.shape[0], m))
    return nv[take]


def _build_pool(structures, data, Xs, cands, max_draws=None) -> _Pool:
    """Posterior joint of mu(Xs) and the outputs of every candidate block, for the (thinned) draws."""
    Xs = jnp.asarray(Xs, dtype=float)
    s = Xs.shape[0]
    Xn = jnp.asarray(np.concatenate([np.atleast_2d(np.asarray(c.Xa, dtype=float)) for c in cands]))
    ms = [np.atleast_2d(np.asarray(c.Xa)).shape[0] for c in cands]
    Hn = jnp.asarray(
        np.concatenate(
            [
                np.tile(np.asarray(c.hbar, dtype=float)[None, :], (m, 1))
                for c, m in zip(cands, ms, strict=True)
            ]
        )
    )
    grp = jnp.asarray(np.concatenate([np.full(m, i) for i, m in enumerate(ms)]))
    starts = np.concatenate([[0], np.cumsum(ms)])
    dX, dH = jnp.asarray(data.X, dtype=float), jnp.asarray(data.H, dtype=float)
    drun, dmask = jnp.asarray(data.run), jnp.asarray(data.mask, dtype=bool)
    means, covs, ws, sws, segs = [], [], [], [], []
    pos = 0
    for k, idx, Wk in _select_draws(structures, max_draws):
        st = structures[k]
        take = jnp.asarray(idx)
        params = jax.tree_util.tree_map(lambda a, take=take: jnp.asarray(a)[take], st.params)
        z = jnp.asarray(st.z)[take]
        nv = jnp.concatenate(
            [_draw_noise(c.noise_var, take, m) for c, m in zip(cands, ms, strict=True)], axis=1
        )
        mean, cov = _build_struct(params, z, dX, dH, drun, dmask, Xs, Xn, Hn, grp, nv, st.cfg)
        means.append(mean)
        covs.append(cov)
        ws.append(jnp.asarray(Wk / Wk.sum()))
        sws.append(Wk.sum())
        segs.append((pos, pos + len(idx), st.transform, bool(st.cfg.increasing)))
        pos += len(idx)
    rows = [s + np.arange(starts[i], starts[i + 1]) for i in range(len(cands))]
    return _Pool(tuple(means), tuple(covs), tuple(ws), jnp.asarray(sws), tuple(segs), s, rows)


def _pool_W(pool: _Pool):
    return jnp.concatenate([pool.sw[k] * w for k, w in enumerate(pool.w)])


def _pool_sigma(pool: _Pool):
    """sigma_epi (s,) of the pooled current posterior."""
    s = pool.s
    m = jnp.concatenate([a[:, :s] for a in pool.mean])
    v = jnp.concatenate([jnp.diagonal(c, axis1=1, axis2=2)[:, :s] for c in pool.cov])
    return _halfwidth(m, jnp.sqrt(jnp.maximum(v, 0.0)), _pool_W(pool), pool.segs)


@jax.jit
def _cond_cov_one(cov, idx):
    def one(c):
        CaI = c[:, idx]
        F = linalg.factor(CaI[idx], jnp.ones(idx.shape[0], dtype=bool))
        return c - CaI @ linalg.solve(F, CaI.T)

    return jax.vmap(one)(cov)


def _condition_variance_only(pool: _Pool, idx) -> _Pool:
    """Condition every draw on observing rows idx WITHOUT the values: means stay, covariances shrink.

    Exact for a Gaussian posterior (the posterior covariance does not depend on the observed value): this is
    the same as appending the rows to every draw's factor with their predictive mean as the value.
    """
    idx = jnp.asarray(idx)
    return pool._replace(cov=tuple(_cond_cov_one(c, idx) for c in pool.cov))


@functools.partial(jax.jit, static_argnames=("s",))
def _cond_values_one(mean, cov, idx, ya, s):
    def one(mu, c):
        CaI = c[:s][:, idx]
        F = linalg.factor(c[idx][:, idx], jnp.ones(idx.shape[0], dtype=bool))
        A = linalg.solve(F, CaI.T).T
        return mu[:s] + A @ (ya - mu[idx]), jnp.diagonal(c)[:s] - jnp.sum(A * CaI, axis=1)

    return jax.vmap(one)(mean, cov)


def _condition_values(pool: _Pool, block: int, ya):
    """Posterior (mean, var), each (D, s), of mu(Xs) per draw after observing block's outputs = ya."""
    idx = jnp.asarray(pool.rows[block])
    out = [
        _cond_values_one(m, c, idx, jnp.asarray(ya, dtype=float), pool.s)
        for m, c in zip(pool.mean, pool.cov, strict=True)
    ]
    return jnp.concatenate([o[0] for o in out]), jnp.concatenate([o[1] for o in out])


# ----------------------------------------------------------------------------------------------
# Fantasies
# ----------------------------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("s", "mode", "segs"))
def _fantasy_gains(keys, mean_t, cov_t, w_t, sw, idx, eps, ess_min, s, mode, segs):
    """H_n - H(fantasy f) for each key, and whether the weights were kept (ESS < ess_min), per fantasy."""
    K, m = len(mean_t), idx.shape[0]
    mmu = jnp.concatenate([a[:, :s] for a in mean_t])
    varmu = jnp.concatenate([jnp.diagonal(c, axis1=1, axis2=2)[:, :s] for c in cov_t])
    mI = jnp.concatenate([a[:, idx] for a in mean_t])
    CI = jnp.concatenate([c[:, idx][:, :, idx] for c in cov_t])
    G = jnp.concatenate([c[:, :s][:, :, idx] for c in cov_t])
    F = jax.vmap(lambda C: linalg.factor(C, jnp.ones(m, dtype=bool)))(CI)
    A = jax.vmap(lambda f, g: linalg.solve(f, g.T).T)(F, G)  # (D, s, m): G CI^{-1}
    var_post = jnp.maximum(varmu - jnp.sum(A * G, axis=-1), 0.0)
    logdet = jax.vmap(linalg.logdet)(F)
    W = jnp.concatenate([sw[k] * w_t[k] for k in range(K)])
    logw0 = jnp.log(jnp.concatenate(list(w_t)))
    D = W.shape[0]
    H_n = H_value(_halfwidth(mmu, jnp.sqrt(varmu), W, segs), eps, mode)
    same = _same_units(segs)

    def one(key):
        ks = jax.random.split(key, K + 2)
        kc = jax.random.categorical(ks[0], jnp.log(sw))
        e = jax.random.normal(ks[1], (m,))
        src = []
        for k in range(K):
            a = segs[k][0] + jax.random.categorical(ks[2 + k], jnp.log(w_t[k]))
            src.append(mI[a] + F.L[a] @ e)
        src = jnp.stack(src)  # (K, m): the fantasy in the units of the structure it was drawn from
        if same:
            ya = jnp.broadcast_to(src[kc], (D, m))
        else:
            yphys = jnp.stack([segs[k][2].inverse(src[k]) for k in range(K)])[kc]
            ya = jnp.concatenate(
                [jnp.broadcast_to(segs[k][2].forward(yphys), (segs[k][1] - segs[k][0], m)) for k in range(K)]
            )
        resid = ya - mI
        quad = jnp.sum(resid * jax.vmap(linalg.solve)(F, resid), axis=-1)
        logp = -0.5 * quad - 0.5 * logdet - 0.5 * m * jnp.log(2.0 * jnp.pi)
        mean_new = mmu + jnp.einsum("dsm,dm->ds", A, resid)
        logw = logw0 + logp
        parts = []
        for k in range(K):
            a, b = segs[k][0], segs[k][1]
            parts.append(sw[k] * jnp.exp(logw[a:b] - logsumexp(logw[a:b])))
        W_rw = jnp.concatenate(parts)
        ess = 1.0 / jnp.sum(W_rw**2)
        use = jnp.all(jnp.isfinite(W_rw)) & (ess >= ess_min)
        W_use = jnp.where(use, W_rw, W)
        H_f = H_value(_halfwidth(mean_new, jnp.sqrt(var_post), W_use, segs), eps, mode)
        return H_n - H_f, ~use

    return jax.vmap(one)(keys)


def _gain_samples(key, pool: _Pool, block: int, eps, n, mode, ess_min):
    keys = jax.random.split(key, n)
    return _fantasy_gains(
        keys,
        pool.mean,
        pool.cov,
        pool.w,
        pool.sw,
        jnp.asarray(pool.rows[block]),
        jnp.asarray(eps, dtype=float),
        jnp.asarray(ess_min, dtype=float),
        s=pool.s,
        mode=mode,
        segs=pool.segs,
    )


def _summarise(g, fell):
    g, fell = np.asarray(g), np.asarray(fell)
    n = g.size
    se = g.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return GainDetail(max(float(g.mean()), 0.0), float(se), n, float(fell.mean()))


def expected_gain_detail(
    key,
    structures,
    data,
    cand: Candidate,
    Xs,
    eps,
    n_fantasy: int,
    mode: str = "hinge",
    ess_min: float = 10.0,
) -> GainDetail:
    """expected_gain with the number of fantasies and the share that kept the weights (ESS < ess_min)."""
    pool = _build_pool(structures, data, Xs, [cand])
    g, fell = _gain_samples(key, pool, 0, eps, n_fantasy, mode, ess_min)
    return _summarise(g, fell)


def expected_gain(
    key,
    structures,
    data,
    cand: Candidate,
    Xs,
    eps,
    n_fantasy: int,
    mode: str = "hinge",
    ess_min: float = 10.0,
):
    """Expected gain of one candidate by n_fantasy fantasies: (gain >= 0, Monte Carlo s.e. of its mean)."""
    d = expected_gain_detail(key, structures, data, cand, Xs, eps, n_fantasy, mode, ess_min)
    return d.gain, d.mc_se


# ----------------------------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------------------------


def sigma_epi_physical(structures, data, Xs):
    """sigma_epi (s,): half-width (q84 - q16)/2 of the pooled posterior of f(x) at the points Xs (spec 2.8).

    Pools all draws of all structures (weights weight_k / S_k); see the module docstring.
    """
    Xs = jnp.asarray(Xs, dtype=float)
    dX, dH = jnp.asarray(data.X, dtype=float), jnp.asarray(data.H, dtype=float)
    drun, dmask = jnp.asarray(data.run), jnp.asarray(data.mask, dtype=bool)
    ms, vs, ws, segs, pos = [], [], [], [], 0
    sw = np.array([float(st.weight) for st in structures])
    sw = sw / sw.sum()
    for st, wk in zip(structures, sw, strict=True):
        m, v = _predict_struct(st.params, jnp.asarray(st.z), dX, dH, drun, dmask, Xs, st.cfg)
        n = m.shape[0]
        ms.append(m)
        vs.append(v)
        ws.append(jnp.full(n, wk / n))
        segs.append((pos, pos + n, st.transform, bool(st.cfg.increasing)))
        pos += n
    return _halfwidth(jnp.concatenate(ms), jnp.sqrt(jnp.concatenate(vs)), jnp.concatenate(ws), tuple(segs))


@functools.partial(jax.jit, static_argnames=("cfg",))
def _predict_struct(params, z, dX, dH, drun, dmask, Xs, cfg):
    def one(p, zz):
        data = PaddedData(X=dX, H=dH, y=zz, censored=jnp.zeros_like(dmask), run=drun, mask=dmask)
        return model.predict_mu(p, data, zz, cfg, Xs)

    return jax.vmap(one)(params, z)


def _n_threads() -> int:
    return max(1, min(len(os.sched_getaffinity(0)), 8))


def _evaluate(key, pool, blocks, costs, eps, mode, ess_min, n0, nmax):
    """Gains of the candidate blocks: n0 fantasies each, then doubling for the two best until the MC s.e. of
    the best ratio is below 10% of its gap to the second best, or nmax fantasies are reached."""
    samples = {}
    counter = [0]

    def add_many(jobs):
        """Run (candidate, n fantasies) jobs; keys are assigned in order, so the result does not depend on
        the threads (XLA releases the GIL: elementwise CPU kernels are single-threaded, so threads give
        up to one core per candidate)."""
        keyed = []
        for i, n in jobs:
            keyed.append((i, n, jax.random.fold_in(key, counter[0])))
            counter[0] += 1

        def run(job):
            i, n, k = job
            g, f = _gain_samples(k, pool, blocks[i], eps, n, mode, ess_min)
            return i, np.asarray(g), np.asarray(f)

        with ThreadPoolExecutor(max_workers=_n_threads()) as ex:
            for i, g, f in ex.map(run, keyed):
                old = samples.get(i, ([], []))
                samples[i] = (old[0] + [g], old[1] + [f])

    def detail(i):
        return _summarise(np.concatenate(samples[i][0]), np.concatenate(samples[i][1]))

    add_many([(i, n0) for i in blocks])
    for _ in range(64):
        if len(blocks) < 2:
            break
        det = {i: detail(i) for i in blocks}
        order = sorted(blocks, key=lambda i: -det[i].gain / costs[i])
        b, c = order[0], order[1]
        gap = det[b].gain / costs[b] - det[c].gain / costs[c]
        if det[b].mc_se / costs[b] <= 0.1 * gap or det[b].n_fantasy >= nmax:
            break
        add_many([(i, min(det[i].n_fantasy, nmax - det[i].n_fantasy)) for i in (b, c)])
    return {i: detail(i) for i in blocks}


def select_batch(
    key,
    structures,
    data,
    candidates,
    Xs,
    eps,
    q: int,
    budget_remaining: float,
    pending=(),
    mode: str = "hinge",
    max_draws: int = 64,
    ess_min: float = 10.0,
    n_fantasy_start: int = 16,
    n_fantasy_max: int = 256,
):
    """Greedy batch of up to q candidates by MR-SUR (module docstring). Returns (chosen indices, GainTable).

    The table is that of the first greedy step (all candidates, with the pending runs conditioned on).
    A candidate is chosen at most once per batch. The batch stops early when no admissible candidate has a
    positive gain (hinge: P1 already holds). Pending runs are Candidates already started: their caps are
    reserved and their rows condition every draw (variance only). Fantasies: n_fantasy_start for every
    admissible candidate, then doubled for the two best only (the ones the stopping rule compares).
    """
    cands, pend = list(candidates), list(pending)
    n = len(cands)
    if n == 0:
        z = np.zeros(0)
        return [], GainTable(z, z, z, z, z.astype(bool), z.astype(int), z)
    for c in cands:
        if not c.cost_mean > 0:
            raise ValueError("every candidate needs cost_mean > 0")
    pool = _build_pool(structures, data, Xs, cands + pend, max_draws)
    if pend:
        pool = _condition_variance_only(pool, np.concatenate(pool.rows[n:]))
    c_rem = float(budget_remaining) - sum(float(p.cost_cap) for p in pend)
    caps = np.array([float(c.cost_cap) for c in cands])
    costs = np.array([float(c.cost_mean) for c in cands])
    available = np.ones(n, dtype=bool)
    chosen, table = [], None
    for step in range(q):
        adm = available & (caps <= c_rem)
        blocks = [int(i) for i in np.flatnonzero(adm)]
        det = _evaluate(
            jax.random.fold_in(key, step), pool, {i: i for i in blocks}, costs, eps, mode, ess_min,
            n_fantasy_start, n_fantasy_max,
        )  # fmt: skip
        gain = np.array([det[i].gain if i in det else 0.0 for i in range(n)])
        if table is None:
            table = GainTable(
                gain,
                np.array([det[i].mc_se if i in det else 0.0 for i in range(n)]),
                costs,
                gain / costs,
                adm,
                np.array([det[i].n_fantasy if i in det else 0 for i in range(n)]),
                np.array([det[i].fallback_frac if i in det else 0.0 for i in range(n)]),
            )
        ratio = np.where(adm & (gain > 0), gain / costs, -np.inf)
        if not np.isfinite(ratio.max(initial=-np.inf)):
            break
        best = int(np.argmax(ratio))
        chosen.append(best)
        available[best] = False
        c_rem -= caps[best]
        pool = _condition_variance_only(pool, pool.rows[best])
    return chosen, table


@functools.partial(jax.jit, static_argnames=("s", "mode", "segs"))
def _variance_only_H(mean_t, cov_t, w_t, sw, idx_mat, eps, s, mode, segs):
    """H after observing the rows idx (variance only, weights fixed), for each row of idx_mat (C, m)."""
    mmu = jnp.concatenate([a[:, :s] for a in mean_t])
    W = jnp.concatenate([sw[k] * w_t[k] for k in range(len(mean_t))])

    def var_after(c, idx):
        G = c[:s][:, idx]
        F = linalg.factor(c[idx][:, idx], jnp.ones(idx.shape[0], dtype=bool))
        return jnp.maximum(jnp.diagonal(c)[:s] - jnp.sum(linalg.solve(F, G.T).T * G, axis=1), 0.0)

    def one(idx):
        var = jnp.concatenate([jax.vmap(lambda c: var_after(c, idx))(c) for c in cov_t])
        return H_value(_halfwidth(mmu, jnp.sqrt(var), W, segs), eps, mode)

    return jax.vmap(one)(idx_mat)
