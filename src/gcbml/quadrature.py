"""p-quadrature: an inference mode that replaces the MCMC by a grid over the order(s) p0 and an optimum over
  the
other settings (project design, not a published method; the pieces are the model of spec 2.2-2.5).

Let p = p0_j (j = 1..k) and phi = every other setting (theta without log p0, the noise-field hyperparameters
(log sigma_z, log ell_z) and the latent noise field zeta). mu, delta and beta are integrated out exactly by
model.log_marginal, as in inference.py.

Definition (brief of the speed branch, items 2 and 4):
  (1) Grid. 20 values of log p0, evenly spaced over the central 99% of its prior N(log_p_mean, log_p_sd^2),
      i.e. log_p_mean +- z_{0.995} log_p_sd. k = 1: a line. k = 2: the tensor grid (20 x 20, visited in a
      snake order so that neighbours are consecutive). k > 2: ValueError.
  (2) For every grid value p_k, phi_hat_k = argmax_phi [log p(D | p_k, phi) + log pi(phi)] by L-BFGS on the
      unconstrained phi with JAX gradients (scipy.optimize L-BFGS-B without bounds), started at the optimum
      of the previous grid point (first point: the prior median of every setting, zeta = 0).
      Non-centred latent field: phi holds eps with zeta = L(sigma_z, ell_z) eps and eps ~ N(0, I), so that the
      optimum over (sigma_z, ell_z, zeta) is a bounded problem [assumption: the brief says "include zeta in
      phi"; the centred form has an unbounded density as sigma_z -> 0]. Varying order and censored outputs
      are not supported (NotImplementedError).
  (3) Weight w_k proportional to exp(log p(D | p_k, phi_hat_k) + log pi(p_k) + log pi(phi_hat_k)) *
    Delta(log p)
      (the prior terms are the densities of the unconstrained variables, as in inference.py; Delta is
      constant on the equally spaced grid; weights sum to 1).
  Reviewer amendment: two passes. Pass 2 puts n_grid nodes evenly over the central 99.9% of the pass-1 weight
  (widened by one pass-1 spacing each side; pass 1 is extended by 10 nodes if >99% of its weight is on an edge
  node); the final weights come from pass 2 only. See fit_quadrature.
  (4) Output: QuadPosterior, with the fields of inference.Posterior (params, z, theta, diagnostics, n_evals)
      having leading axes (1, K), plus the weights. inference.flatten works on it unchanged. A
        prediction is the
      weighted mixture of the K exact Gaussian predictions of model.predict_mu (mu_moments,
        mixture_quantiles).
      to_equal_weight turns it into an inference.Posterior of equal-weight draws for consumers that
        ignore weights.

Baselines of the benchmark:
  (d) fit_plugin: phi_hat at ONE given log p0 (no grid), weight 1.
  (f) Brownian: ModelConfig(h_kernel="lb", gamma_fixed=0.5) (Tuo-Wu-Yu; kernels.lifted_brownian at gamma =
    0.5),
      with either fit_plugin / fit_quadrature (plug-in or p-quadrature) or inference.fit (full MCMC).

fit_quadrature(key, data, z, bounds, cfg, scales, n_controls, n_grid=20, grid=None, mass=0.99,
               maxiter=1000) -> QuadPosterior
    key and bounds are unused (the signature follows inference.fit); ``grid`` (K, k) replaces the
      default grid.
fit_plugin(key, data, z, bounds, cfg, scales, n_controls, log_p, maxiter=1000) -> QuadPosterior (K = 1)
log_p_grid(scales, k, n_grid=20, mass=0.99) -> (K, k)
mu_moments(post, data, cfg, Xs) -> (mean (K, s), var (K, s))      per node, Lambda units
mixture_quantiles(w, mean, var, qs) -> (len(qs), s)                exact quantiles of sum_k w_k N(mean_k,
  var_k)
"""

from __future__ import annotations

import collections
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

from gcbml import inference, linalg, model
from gcbml.priors import normal_logpdf

N_GRID = 20
CENTRAL_MASS = 0.99
MAX_COMPILED = 4


class QuadPosterior(NamedTuple):
    """Weighted nodes. params, z, theta have leading axes (1, K) like inference.Posterior (n_chains = 1)."""

    params: Any
    z: Any
    theta: dict
    diagnostics: dict
    n_evals: Any
    weights: Any  # (K,) sums to 1
    log_p: Any  # (K, k) grid of log p0
    log_post: Any  # (K,) log p(D | p_k, phi_hat_k) + log pi(p_k) + log pi(phi_hat_k)
    converged: Any  # (K,) L-BFGS reported success
    pass1: Any = None  # dict: pass-1 grid, weights, n_extensions


# ----------------------------------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------------------------------


def log_p_grid(scales, k: int, n_grid: int = N_GRID, mass: float = CENTRAL_MASS) -> np.ndarray:
    """(K, k) grid of log p0: n_grid equally spaced values over the central ``mass`` of N(log_p_mean,
    log_p_sd^2)."""
    if k > 2:
        raise ValueError(f"p-quadrature supports k <= 2 resolution components, got k = {k}")
    z = float(ndtri(0.5 + 0.5 * mass))
    axis = scales.log_p_mean + scales.log_p_sd * np.linspace(-z, z, n_grid)
    if k == 1:
        return axis[:, None]
    rows = []
    for i, a in enumerate(axis):
        for b in axis if i % 2 == 0 else axis[::-1]:  # snake order
            rows.append((a, b))
    return np.asarray(rows)


# ----------------------------------------------------------------------------------------------
# the objective: -(log p(D | p, phi) + log pi(p) + log pi(phi))
# ----------------------------------------------------------------------------------------------


def _prior_median_theta(sampler) -> tuple[dict, np.ndarray]:
    """(theta dict, hz) at the prior median of every setting (start of the first grid node), log p0
    excluded."""
    sc, k, d = sampler.sc, sampler.k, sampler.d
    ln2 = math.log(2.0)

    def pc(sigma0, shape):
        l1 = -math.log(inference.ALPHA) * math.sqrt(sc.ell0)
        l2 = -math.log(inference.ALPHA) / sigma0
        return math.log(ln2 / l2), np.full(shape, -2.0 * math.log(ln2 / l1))

    t = {}
    t["log_sigma_mu"], t["log_ell_mu"] = pc(sc.S_mu, (d,))
    t["c0"], t["c1"] = np.zeros(k), np.zeros(k)
    ls, le = pc(sc.S_delta, (k, d))
    t["log_sigma_delta"], t["log_ell_x"] = np.full(k, ls), le
    t["log_ell_h"] = np.full(k, math.log(0.5))
    t["logit_gamma"] = np.zeros(k)
    t["m_s"], t["b_s"] = 2.0 * math.log(sc.S_noise), np.zeros(k)
    t["log_ell_v"] = pc(1.0, ())[1]
    lsz, lez = pc(1.0, (sampler.n_hz - 1,))
    return t, np.concatenate([[lsz], lez])


class _Problem:
    """Layout of phi = (theta without log_p0, hz, eps) and the compiled value-and-gradient of -log
    posterior."""

    def __init__(self, sampler):
        self.s = sampler
        self.free = [(n, sh) for n, sh in sampler.layout.fields if n != "log_p0"]
        self.n_theta = sum(int(np.prod(sh)) for _, sh in self.free)
        self.n_hz = sampler.n_hz
        self.n_sites = sampler.n_sites
        self.size = self.n_theta + self.n_hz + self.n_sites

    def unpack(self, u, logp):
        t, off = {"log_p0": logp}, 0
        for name, shape in self.free:
            m = int(np.prod(shape))
            t[name] = u[off : off + m].reshape(shape)
            off += m
        hz = u[off : off + self.n_hz]
        eps = u[off + self.n_hz :]
        return t, hz, eps

    def initial(self) -> np.ndarray:
        t, hz = _prior_median_theta(self.s)
        parts = [np.reshape(t[name], -1) for name, _ in self.free]
        return np.concatenate(parts + [hz, np.zeros(self.n_sites)]).astype(float)

    def neg_log_post(self, u, logp, aux):
        smp = self.s.with_aux(aux)
        t, hz, eps = self.unpack(u, logp)
        L = linalg.factor(smp.zeta_cov(hz), aux.site_mask).L
        zeta = L @ eps
        lp = (
            smp.log_prior_theta(t, hz)
            + normal_logpdf(eps, 0.0, 1.0)
            + smp.loglik(t, zeta, smp._no_pi(), aux.z0)
        )
        return -lp


_VG: collections.OrderedDict = collections.OrderedDict()


def _value_and_grad(prob: _Problem):
    sampler = prob.s
    key = (inference._template_key(sampler), tuple(a.shape for a in sampler.aux))
    hit = _VG.get(key)
    if hit is None:
        hit = jax.jit(jax.value_and_grad(prob.neg_log_post, argnums=0))
        _VG[key] = hit
        while len(_VG) > MAX_COMPILED:
            _VG.popitem(last=False)
    _VG.move_to_end(key)
    return hit


def clear_compiled() -> None:
    _VG.clear()


# ----------------------------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------------------------


def _snake(axes: list[np.ndarray]) -> np.ndarray:
    if len(axes) == 1:
        return axes[0][:, None]
    rows = []
    for i, a in enumerate(axes[0]):
        for b in axes[1] if i % 2 == 0 else axes[1][::-1]:
            rows.append((a, b))
    return np.asarray(rows)


def _solve_grid(prob, vg, aux, grid, u, maxiter):
    """L-BFGS at every node in order, each started at the previous optimum (the first at ``u``)."""
    us, vals, ok, n_evals = [], [], [], 0
    for lp in grid:
        lpj = jnp.asarray(lp)

        def fun(v, lpj=lpj):
            f, g = vg(jnp.asarray(v), lpj, aux)
            f, g = float(f), np.asarray(g)
            if not (np.isfinite(f) and np.all(np.isfinite(g))):
                return 1e12, np.zeros_like(v)
            return f, g

        res = minimize(
            fun,
            u,
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=maxiter, maxcor=20, ftol=1e-13, gtol=1e-7),
        )
        u = res.x
        us.append(res.x)
        vals.append(-float(res.fun))
        ok.append(bool(res.success))
        n_evals += int(res.nfev)
    return us, np.asarray(vals), ok, n_evals


def _softmax(vals):
    w = np.exp(vals - np.max(vals))
    return w / w.sum()  # Delta(log p) is constant on an equally spaced grid


def fit_quadrature(
    key,
    data,
    z,
    bounds,
    cfg,
    scales,
    n_controls: int,
    n_grid: int = N_GRID,
    grid=None,
    mass: float = CENTRAL_MASS,
    maxiter: int = 1000,
    two_pass: bool = True,
    n_grid2: int | None = None,
    mass2: float = 0.999,
    n_extend: int = 10,
    max_extensions: int = 5,
) -> QuadPosterior:
    """p-quadrature posterior (module docstring). ``grid`` (K, k) of log p0 replaces the default grid.

    Two passes (reviewer amendment). Pass 1: the prior grid. If more than 99% of the pass-1 weight sits on
    one edge node (k = 1), the grid is extended by ``n_extend`` nodes beyond that edge (same spacing, at most
    ``max_extensions`` times). Pass 2: ``n_grid2`` (default n_grid) nodes evenly spaced over the range that
    holds the central ``mass2`` of the pass-1 weight (per axis, from the marginal weights), widened by one
    pass-1 spacing on each side; the output weights come from pass 2 only. ``two_pass=False`` or an explicit
    ``grid`` gives a single pass. The pass-1 summary is in ``post.pass1`` (grid, weights, n_extensions).
    """
    del key, bounds
    if bool(np.any(np.asarray(data.censored, dtype=bool) & np.asarray(data.mask, dtype=bool))):
        raise NotImplementedError("p-quadrature does not support censored outputs")
    sampler = inference._Sampler(data, z, z, cfg, scales, n_controls, False)
    k = sampler.k
    explicit = grid is not None
    grid = (
        log_p_grid(scales, k, n_grid, mass) if not explicit else np.asarray(grid, dtype=float).reshape(-1, k)
    )
    prob = _Problem(sampler)
    vg = _value_and_grad(prob)
    aux = sampler.aux
    u0 = prob.initial()
    us, vals, ok, n_evals = _solve_grid(prob, vg, aux, grid, u0, maxiter)
    info = {"n_extensions": 0, "pass1_grid": grid, "pass1_weights": _softmax(vals)}
    if two_pass and not explicit:
        if k == 1:
            for _ in range(max_extensions):
                w1 = _softmax(vals)
                if w1[0] > 0.99:
                    dx = grid[1, 0] - grid[0, 0]
                    new = grid[0, 0] - dx * np.arange(n_extend, 0, -1)
                    u_s = us[0]
                    nu, nv, nok, ne = _solve_grid(prob, vg, aux, new[::-1, None], u_s, maxiter)
                    us, vals, ok = nu[::-1] + us, np.concatenate([nv[::-1], vals]), nok[::-1] + ok
                    grid = np.concatenate([new[:, None], grid])
                elif w1[-1] > 0.99:
                    dx = grid[-1, 0] - grid[-2, 0]
                    new = grid[-1, 0] + dx * np.arange(1, n_extend + 1)
                    nu, nv, nok, ne = _solve_grid(prob, vg, aux, new[:, None], us[-1], maxiter)
                    us, vals, ok = us + nu, np.concatenate([vals, nv]), ok + nok
                    grid = np.concatenate([grid, new[:, None]])
                else:
                    break
                n_evals += ne
                info["n_extensions"] += 1
            info["pass1_grid"], info["pass1_weights"] = grid, _softmax(vals)
        w1 = _softmax(vals)
        dx = [float(np.min(np.diff(np.unique(grid[:, j])))) for j in range(k)]
        axes = []
        for j in range(k):
            ax = np.unique(grid[:, j])
            marg = np.array([w1[np.isclose(grid[:, j], a)].sum() for a in ax])
            cdf = np.cumsum(marg)
            lo = ax[np.searchsorted(cdf, 0.5 * (1.0 - mass2))]
            hi = ax[min(np.searchsorted(cdf, 1.0 - 0.5 * (1.0 - mass2)), len(ax) - 1)]
            axes.append(np.linspace(lo - dx[j], hi + dx[j], n_grid2 or n_grid))
        grid2 = _snake(axes)
        j0 = int(np.argmin(np.sum((np.asarray(grid) - grid2[0]) ** 2, axis=1)))  # nearest pass-1 optimum
        n_evals1 = n_evals
        us, vals, ok, ne = _solve_grid(prob, vg, aux, grid2, us[j0], maxiter)
        n_evals = n_evals1 + ne
        grid = grid2
    w = _softmax(vals)
    plist, tlist = [], []
    for v, lp in zip(us, grid, strict=True):
        t, hz, eps = prob.unpack(jnp.asarray(v), jnp.asarray(lp))
        zeta = linalg.factor(sampler.zeta_cov(hz), aux.site_mask).L @ eps
        plist.append(sampler.params(t, zeta, sampler._no_pi()))
        tl = dict(t)
        tl["log_sigma_z"], tl["log_ell_z"] = hz[0], hz[1:]
        tlist.append(tl)
    stack = lambda *a: jnp.stack(a)[None]  # noqa: E731
    params = jax.tree_util.tree_map(stack, *plist)
    theta = {n: np.asarray(jnp.stack([t[n] for t in tlist])[None]) for n in tlist[0]}
    zs = jnp.broadcast_to(jnp.asarray(aux.z0, dtype=float), (1, len(us), aux.z0.shape[0]))
    return QuadPosterior(params, zs, theta, {}, n_evals, w, grid, vals, np.asarray(ok), info)


def fit_plugin(
    key, data, z, bounds, cfg, scales, n_controls: int, log_p, maxiter: int = 1000
) -> QuadPosterior:
    """Baseline (d): phi_hat at one given log p0 (shape (k,) or scalar), no grid, weight 1."""
    k = int(np.asarray(data.H).shape[1])
    grid = np.asarray(log_p, dtype=float).reshape(1, k)
    return fit_quadrature(key, data, z, bounds, cfg, scales, n_controls, grid=grid, maxiter=maxiter)


def flatten(post: QuadPosterior):
    """(params with leading axis K, z (K, n_pad), weights (K,)); the first two are inference.flatten's
    output."""
    p, z = inference.flatten(post)
    return p, z, np.asarray(post.weights)


def to_equal_weight(post: QuadPosterior, n: int = 256) -> inference.Posterior:
    """inference.Posterior of n equal-weight draws, by deterministic systematic resampling of the nodes.

    Node k appears round(n w_k) times, up to the usual systematic-resampling slack; for consumers that treat
    every draw as equally likely (campaign.py pooling). The exact weighted prediction is mu_moments +
    mixture_quantiles; this resampling is only a bridge.
    """
    cdf = np.cumsum(np.asarray(post.weights))
    idx = np.searchsorted(cdf, (np.arange(n) + 0.5) / n * cdf[-1])
    idx = np.minimum(idx, len(cdf) - 1)
    take = jnp.asarray(idx)
    params = jax.tree_util.tree_map(lambda a: a[:, take], post.params)
    theta = {name: v[:, idx] for name, v in post.theta.items()}
    return inference.Posterior(params, post.z[:, take], theta, {}, post.n_evals)


# ----------------------------------------------------------------------------------------------
# prediction
# ----------------------------------------------------------------------------------------------


def mu_moments(post, data, cfg, Xs, n_batch: int = 32):
    """(mean (S, s), var (S, s)) of mu(Xs) given each draw/node (model.predict_mu), Lambda units."""
    params, zz = inference.flatten(post)
    Xs = jnp.asarray(Xs, dtype=float)

    def one(args):
        p, z = args
        return model.predict_mu(p, data, z, cfg, Xs)

    mean, var = jax.lax.map(one, (params, zz), batch_size=n_batch)
    return np.asarray(mean), np.maximum(np.asarray(var), 0.0)


def mixture_quantiles(w, mean, var, qs, n_iter: int = 80) -> np.ndarray:
    """Exact quantiles of sum_k w_k N(mean_k, var_k) per column, by bisection on the mixture CDF.
    (len(qs), s)."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    mean, sd = np.asarray(mean, dtype=float), np.sqrt(np.maximum(np.asarray(var, dtype=float), 1e-300))
    lo = (mean - 12.0 * sd).min(axis=0)
    hi = (mean + 12.0 * sd).max(axis=0)
    out = []
    for q in qs:
        a, b = lo.copy(), hi.copy()
        for _ in range(n_iter):
            c = 0.5 * (a + b)
            F = (w[:, None] * ndtr((c[None, :] - mean) / sd)).sum(axis=0)
            above = F >= q
            b = np.where(above, c, b)
            a = np.where(above, a, c)
        out.append(0.5 * (a + b))
    return np.asarray(out)


def summarize(w, mean, var):
    """(median, sigma = (q84 - q16) / 2, q025, q975) of the mixture, identity transform."""
    q = mixture_quantiles(w, mean, var, [0.5, 0.16, 0.84, 0.025, 0.975])
    return q[0], 0.5 * (q[2] - q[1]), q[3], q[4]


def decompose(w, mean, var):
    """Total variance E[v] + Var(m) of the mixture, and its two parts, per column."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    m_bar = (w[:, None] * mean).sum(axis=0)
    e_v = (w[:, None] * var).sum(axis=0)
    v_m = (w[:, None] * (mean - m_bar) ** 2).sum(axis=0)
    return e_v + v_m, e_v, v_m
