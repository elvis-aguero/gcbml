"""Posterior sampling for one candidate structure (spec Section 2.5): the Gibbs sampler of Yi-h.

Given (mu, delta, beta) integrated out (model.log_marginal), the remaining unknowns are:

  global block theta (unconstrained vector, slice-sampled coordinate-wise, mcmc.slice):
      log sigma_mu, log ell_mu (d), c0 (k), c1 (k), log p0 (k),
      log sigma_delta (k), log ell_x (k, d), and log ell_h (k) [twy2] or logit gamma (k) [lb],
      m_s, b_s (k), log ell_v [only if within_run]
  latent noise field zeta (n_sites)          elliptical slice (mcmc.elliptical), prior noise.zeta_cov
  its hyperparameters (log sigma_z, log ell_z (d_u + k))   surrogate-data slice (mcmc.surrogate)
  varying order (only if cfg_varying_order and h_kernel == "twy2", spec 2.3):
      log p_j(x) = log p0_j + pi_j(x), pi_j a latent GP on the distinct unit x of the real rows,
      elliptical slice for pi_j, surrogate-data slice for (log sigma_pi_j, log ell_pi_j);
      PC prior with P(sigma_pi > 0.3) = 0.05 and P(ell < 0.1) = 0.05 (spec 2.5 table)
  censored outputs: model.censored_sweep

Priors: priors.PriorScales; c0, c1 ~ N(0, S_c^2); sigma_mu, ell_mu PC prior with sigma0 = S_mu;
sigma_delta, ell_x PC prior with sigma0 = S_delta; log ell_h ~ N(log 0.5, 1) on hbar units [assumption,
document]; logit gamma ~ logit-uniform; log p0 ~ N(log_p_mean, log_p_sd^2); ell_v PC-type as ell.

One Gibbs iteration = slice sweep over theta; ESS step for zeta; surrogate step for its hyperparameters;
(same two for each pi_j); one censored sweep. The state is a pytree; mcmc.chains.run_chains runs 4 chains
(vmap), warm-up adapts the slice widths. Starting points: drawn from the prior, then 200 iterations of
slice updates on theta only (state the rule).

Implementation (W3-A). Block structure of one Gibbs iteration, in this order:
  1. theta: one coordinate-wise slice sweep (mcmc.slice) of log p(z | theta, zeta, pi) + log prior;
  2. zeta: one elliptical slice step (mcmc.elliptical) with prior chol(noise.zeta_cov);
  3. (log sigma_z, log ell_z): one surrogate-data slice sweep (mcmc.surrogate) that moves zeta with them;
  4. varying order only, for each component j: elliptical step for pi_j, then surrogate-data sweep for
     (log sigma_pi_j, log ell_pi_j);
  5. one censored sweep (model.censored_sweep), skipped at trace time when no row is censored.
The sampler state is a dict {theta, hz, zeta, z[, pi, hpi]}; hz = (log sigma_z, log ell_z), hpi[j] =
(log sigma_pi_j, log ell_pi_j). Slice widths adapt in warm-up (mcmc.chains.adapt_widths).

Starting rule [assumption]: per chain, 16 candidate states are drawn from the prior (theta, hz, zeta, pi),
the first one with a finite log density is kept (not the best one: the starts stay over-dispersed), and
then N_INIT_SWEEPS = 30 slice sweeps update theta only (zeta, pi and z fixed, widths 1).

Further assumptions: (a) surrogate noise for zeta (Murray & Adams Section 3.2): site i with n_i rows has
the Fisher variance 2 / n_i of log s^2 and S_ii = 1 / max(n_i / 2 - 1 / sigma_z^2, 0.25 / sigma_z^2)
(eq 12 with the positivity threshold), a valid choice for any S > 0; (b) surrogate noise for pi_j is the
constant 0.25 (sd 0.5 in log p); (c) the latent pi_j lives on the distinct rows of the unit coordinates X
(all d columns, v included); (d) varying_order=True is ignored for h_kernel == "lb" (spec 2.3: LB has one
shared order); (e) the PC prior of ell_v uses the ell factor of pc_matern_logpdf with ell0 = scales.ell0;
(f) theta in the Posterior also holds log_sigma_z, log_ell_z (and log_sigma_pi, log_ell_pi).
fit(key, data, z, bounds, cfg, scales, n_controls, n_warmup, n_samples, n_chains=4, varying_order=False)
        -> Posterior
    data: PaddedData; z: (n_pad,) Lambda(y) with censored rows set to their bound initially;
    bounds: (n_pad,) Lambda(y) of censored rows (unused elsewhere).
Posterior: NamedTuple with
    params: ModelParams with leading axes (n_chains, n_samples)   (constrained, ready for model.*)
    z: (n_chains, n_samples, n_pad) imputed outputs
    theta: dict name -> (n_chains, n_samples, ...) unconstrained draws (for diagnostics)
    diagnostics: mcmc.diagnostics.summary over every scalar of theta and log sigma_z
    n_evals: total log-likelihood evaluations
flatten(posterior) -> (ModelParams with one leading axis S, z (S, n_pad))     pools chains for prediction.
"""

from __future__ import annotations

import collections
import copy
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from gcbml import linalg, noise
from gcbml.data import PaddedData
from gcbml.kernels import DeltaParams, ard_matern
from gcbml.mcmc import diagnostics as dg
from gcbml.mcmc.chains import chain_map, make_phase, run_chains
from gcbml.mcmc.elliptical import ess_step
from gcbml.mcmc.slice import slice_step
from gcbml.mcmc.surrogate import surrogate_slice_step
from gcbml.model import ModelConfig, ModelParams, censored_sweep, log_marginal
from gcbml.priors import (
    PriorScales,
    logit_uniform_logpdf,
    lognormal_logpdf,
    normal_logpdf,
    pc_matern_logpdf,
)

N_INIT_SWEEPS = 30  # theta-only slice sweeps of the starting rule (200 in the first version: 7x the cost)
N_CANDIDATES = 16  # prior draws tried per chain for a finite starting density
PI_SIGMA0, PI_ELL0 = 0.3, 0.1  # P(sigma_pi > 0.3) = 0.05, P(ell_pi < 0.1) = 0.05 (spec 2.5)
PI_SURROGATE_VAR = 0.25  # surrogate-data noise variance of pi_j (assumption b)
ALPHA = 0.05  # PC prior tail probability


class Posterior(NamedTuple):
    params: Any
    z: Any
    theta: dict
    diagnostics: dict
    n_evals: Any


# ----------------------------------------------------------------------------------------------
# prior pieces
# ----------------------------------------------------------------------------------------------


def _pc_ell_logpdf(log_ell, ell0):
    """The length-scale factor of pc_matern_logpdf (the call with no length scale is the sigma factor)."""
    zero = jnp.zeros(())
    return pc_matern_logpdf(zero, log_ell, 1.0, ell0) - pc_matern_logpdf(zero, jnp.zeros((0,)), 1.0, ell0)


def _pc_sample(key, sigma0, ell0, ell_shape):
    """(log sigma, log ell) from the PC prior: sigma ~ Exp(l2), ell^(-1/2) ~ Exp(l1) (Thm 2.6, d = 1)."""
    k1, k2 = jax.random.split(key)
    l1 = -math.log(ALPHA) * math.sqrt(ell0)
    l2 = -math.log(ALPHA) / sigma0
    log_sigma = jnp.log(jax.random.exponential(k1, dtype=jnp.float64) / l2)
    log_ell = -2.0 * jnp.log(jax.random.exponential(k2, ell_shape, dtype=jnp.float64) / l1)
    return log_sigma, log_ell


class _Layout:
    """Names, shapes and offsets of the global vector theta (module docstring)."""

    def __init__(self, d: int, k: int, cfg: ModelConfig):
        fields = [
            ("log_sigma_mu", ()),
            ("log_ell_mu", (d,)),
            ("c0", (k,)),
            ("c1", (k,)),
            ("log_p0", (k,)),
            ("log_sigma_delta", (k,)),
            ("log_ell_x", (k, d)),
        ]
        fields.append(("log_ell_h", (k,)) if cfg.h_kernel == "twy2" else ("logit_gamma", (k,)))
        fields += [("m_s", ()), ("b_s", (k,))]
        if cfg.within_run:
            fields.append(("log_ell_v", ()))
        self.fields = fields
        self.size = sum(int(np.prod(s)) for _, s in fields)

    def unpack(self, vec):
        out, off = {}, 0
        for name, shape in self.fields:
            m = int(np.prod(shape))
            out[name] = vec[..., off : off + m].reshape(vec.shape[:-1] + shape)
            off += m
        return out


# ----------------------------------------------------------------------------------------------
# the sampler
# ----------------------------------------------------------------------------------------------


class _Aux(NamedTuple):
    """Everything that depends on the data values (arrays only, so that it can be a traced argument)."""

    X: jnp.ndarray
    H: jnp.ndarray
    y: jnp.ndarray
    censored: jnp.ndarray
    run: jnp.ndarray
    mask: jnp.ndarray
    z0: jnp.ndarray
    bounds: jnp.ndarray
    sites: jnp.ndarray
    row_site: jnp.ndarray
    site_mask: jnp.ndarray
    cnt: jnp.ndarray  # (n_sites_pad,) real rows per site
    xs: jnp.ndarray  # (nx_pad, d) distinct unit coordinates (varying order only)
    x_row: jnp.ndarray
    x_mask: jnp.ndarray

    @property
    def data(self) -> PaddedData:
        return PaddedData(X=self.X, H=self.H, y=self.y, censored=self.censored, run=self.run, mask=self.mask)


class _Sampler:
    """Gibbs sampler of one structure; see the module docstring. All methods are pure and jittable."""

    def __init__(
        self,
        data,
        z,
        bounds,
        cfg: ModelConfig,
        scales: PriorScales,
        n_controls: int,
        varying_order: bool,
        weight=1.0,
    ):
        self.cfg, self.sc, self.nu = cfg, scales, n_controls
        self.weight = float(weight)
        X = np.asarray(data.X, dtype=float)
        mask = np.asarray(data.mask, dtype=bool)
        self.n_pad, self.d = X.shape
        self.k = np.asarray(data.H).shape[1]
        self.varying = bool(varying_order) and cfg.h_kernel == "twy2"
        self.any_censored = bool(np.any(np.asarray(data.censored, dtype=bool) & mask))
        self.layout = _Layout(self.d, self.k, cfg)
        sites, row_site, site_mask = noise.build_sites(data, n_controls)
        cnt = np.bincount(row_site[mask], minlength=sites.shape[0]).astype(float)
        if self.varying:
            xs, x_row, x_mask = noise.unique_rows(X, mask)
        else:
            xs, x_row, x_mask = np.zeros((1, self.d)), np.zeros(self.n_pad, dtype=np.int64), np.ones(1, bool)
        self.aux = _Aux(
            jnp.asarray(X),
            jnp.asarray(data.H, dtype=float),
            jnp.asarray(data.y, dtype=float),
            jnp.asarray(data.censored, dtype=bool),
            jnp.asarray(data.run),
            jnp.asarray(mask),
            jnp.asarray(z, dtype=float),
            jnp.asarray(bounds, dtype=float),
            jnp.asarray(sites),
            jnp.asarray(row_site),
            jnp.asarray(site_mask),
            jnp.asarray(cnt),
            jnp.asarray(xs),
            jnp.asarray(x_row),
            jnp.asarray(x_mask),
        )
        self.n_sites = sites.shape[0]
        self.n_xs = xs.shape[0]
        self.n_hz = 1 + n_controls + self.k

    def with_aux(self, aux: _Aux) -> _Sampler:
        """Same static structure, other data values (the arrays may be traced)."""
        other = copy.copy(self)
        other.aux = aux
        return other

    # --- parameters and densities ----------------------------------------------------------

    def params(self, t: dict, zeta, pi) -> ModelParams:
        a = self.aux
        logp = jnp.broadcast_to(t["log_p0"], (self.n_pad, self.k))
        if self.varying:
            logp = logp + pi[:, a.x_row].T
        twy2 = self.cfg.h_kernel == "twy2"
        ell_h = jnp.exp(t["log_ell_h"]) if twy2 else jnp.ones(self.k)
        gamma = jnp.full(self.k, 0.5) if twy2 else jax.nn.sigmoid(t["logit_gamma"])
        delta = DeltaParams(jnp.exp(t["log_sigma_delta"]), jnp.exp(t["log_ell_x"]), ell_h, gamma)
        nv = noise.noise_var(t["m_s"], t["b_s"], zeta, a.sites, a.row_site, a.mask)
        ell_v = jnp.exp(t["log_ell_v"]) if self.cfg.within_run else jnp.ones(())
        return ModelParams(
            jnp.exp(t["log_sigma_mu"]),
            jnp.exp(t["log_ell_mu"]),
            t["c0"],
            t["c1"],
            jnp.exp(logp),
            delta,
            nv,
            ell_v,
        )

    def loglik(self, t, zeta, pi, z):
        if self.weight == 0.0:
            return jnp.zeros(())
        lm = log_marginal(self.params(t, zeta, pi), self.aux.data, z, self.cfg)
        return self.weight * lm

    def log_prior_theta(self, t, hz):
        sc = self.sc
        lp = pc_matern_logpdf(t["log_sigma_mu"], t["log_ell_mu"], sc.S_mu, sc.ell0)
        lp += normal_logpdf(t["c0"], 0.0, sc.S_c) + normal_logpdf(t["c1"], 0.0, sc.S_c)
        lp += normal_logpdf(t["log_p0"], sc.log_p_mean, sc.log_p_sd)
        lp += jnp.sum(
            jax.vmap(lambda s, e: pc_matern_logpdf(s, e, sc.S_delta, sc.ell0))(
                t["log_sigma_delta"], t["log_ell_x"]
            )
        )
        if self.cfg.h_kernel == "twy2":
            lp += lognormal_logpdf(t["log_ell_h"], math.log(0.5), 1.0)  # assumption (module docs)
        else:
            lp += logit_uniform_logpdf(t["logit_gamma"])
        lp += noise.log_prior(t["m_s"], t["b_s"], hz[0], hz[1:], sc)
        if self.cfg.within_run:
            lp += _pc_ell_logpdf(t["log_ell_v"], sc.ell0)
        return lp

    def log_prior_hz(self, hz):
        # the (sigma_z, ell_z) part of noise.log_prior; the m_s, b_s part is a constant here
        return noise.log_prior(0.0, jnp.zeros(self.k), hz[0], hz[1:], self.sc) - noise.log_prior(
            0.0, jnp.zeros(self.k), 0.0, jnp.zeros(self.n_hz - 1), self.sc
        )

    def log_prior_hpi(self, h):
        return pc_matern_logpdf(h[0], h[1:], PI_SIGMA0, PI_ELL0)

    def zeta_cov(self, hz):
        return noise.zeta_cov(hz[0], hz[1:], self.aux.sites, self.aux.site_mask)

    def pi_cov(self, h):
        xs, xm = self.aux.xs, self.aux.x_mask
        K = jnp.exp(2.0 * h[0]) * ard_matern(xs, xs, jnp.exp(h[1:]), 2.5)
        return jnp.where(xm[:, None] & xm[None, :], K, 0.0) + jnp.diag(jnp.where(xm, 0.0, 1.0))

    def zeta_surrogate_var(self, hz):
        s2 = jnp.exp(2.0 * hz[0])
        prec = self.aux.cnt / 2.0 - 1.0 / s2
        return 1.0 / jnp.maximum(prec, 0.25 / s2)

    def _no_pi(self):
        return jnp.zeros((self.k, self.n_xs))

    # --- prior draws and starting rule -----------------------------------------------------

    def draw_prior(self, key):
        sc, k, d = self.sc, self.k, self.d
        ks = jax.random.split(key, 16)
        t = {}
        t["log_sigma_mu"], t["log_ell_mu"] = _pc_sample(ks[0], sc.S_mu, sc.ell0, (d,))
        t["c0"] = sc.S_c * jax.random.normal(ks[1], (k,))
        t["c1"] = sc.S_c * jax.random.normal(ks[2], (k,))
        t["log_p0"] = sc.log_p_mean + sc.log_p_sd * jax.random.normal(ks[3], (k,))
        ls, le = jax.vmap(lambda kk: _pc_sample(kk, sc.S_delta, sc.ell0, (d,)))(jax.random.split(ks[4], k))
        t["log_sigma_delta"], t["log_ell_x"] = ls, le
        if self.cfg.h_kernel == "twy2":
            t["log_ell_h"] = math.log(0.5) + jax.random.normal(ks[5], (k,))
        else:
            u = jax.random.uniform(ks[5], (k,), minval=1e-12, maxval=1.0 - 1e-12)
            t["logit_gamma"] = jnp.log(u) - jnp.log1p(-u)
        t["m_s"] = 2.0 * math.log(sc.S_noise) + math.log(10.0) * jax.random.normal(ks[6])
        t["b_s"] = sc.b_s_sd * jax.random.normal(ks[7], (k,))
        if self.cfg.within_run:
            t["log_ell_v"] = _pc_sample(ks[8], 1.0, sc.ell0, (1,))[1][0]
        theta = jnp.concatenate([jnp.reshape(t[name], (-1,)) for name, _ in self.layout.fields])
        lsz, lez = _pc_sample(ks[9], 1.0, sc.ell0, (self.n_hz - 1,))
        hz = jnp.concatenate([lsz[None], lez])
        zeta = linalg.factor(self.zeta_cov(hz), self.aux.site_mask).L @ jax.random.normal(
            ks[10], (self.n_sites,)
        )
        st = {"theta": theta, "hz": hz, "zeta": zeta, "z": self.aux.z0}
        if self.varying:
            hp = jax.vmap(lambda kk: _pc_sample(kk, PI_SIGMA0, PI_ELL0, (self.d,)))(
                jax.random.split(ks[11], k)
            )
            hpi = jnp.concatenate([hp[0][:, None], hp[1]], axis=1)
            Ls = jax.vmap(lambda h: linalg.factor(self.pi_cov(h), self.aux.x_mask).L)(hpi)
            eps = jax.random.normal(ks[12], (k, self.n_xs))
            st["hpi"] = hpi
            st["pi"] = jnp.einsum("kij,kj->ki", Ls, eps)
        return st

    def log_density(self, st):
        """Joint unnormalised log density of a state (used to reject non-finite starts)."""
        t = self.layout.unpack(st["theta"])
        pi = st["pi"] if self.varying else self._no_pi()
        lp = self.log_prior_theta(t, st["hz"]) + self.loglik(t, st["zeta"], pi, st["z"])
        return jnp.where(jnp.isnan(lp), -jnp.inf, lp)

    def init_state(self, key):
        """Starting rule (module docstring): first finite prior draw, then theta-only sweeps."""
        k_c, k_s = jax.random.split(key)
        cands = jax.vmap(self.draw_prior)(jax.random.split(k_c, N_CANDIDATES))
        ok = jax.vmap(lambda s: jnp.isfinite(self.log_density(s)))(cands)
        pick = jnp.argmax(ok)
        st = jax.tree_util.tree_map(lambda a: a[pick], cands)
        pi = st["pi"] if self.varying else self._no_pi()

        def logdens(vec):
            return self._theta_logdensity(vec, st["hz"], st["zeta"], pi, st["z"])

        def sweep(vec, k):
            vec, n = slice_step(k, vec, logdens, 1.0)
            return vec, n

        theta, n = lax.scan(sweep, st["theta"], jax.random.split(k_s, N_INIT_SWEEPS))
        return {**st, "theta": theta}, jnp.sum(n) + N_CANDIDATES

    def init_chains(self, key, n_chains):
        st, n = jax.jit(jax.vmap(self.init_state))(jax.random.split(key, n_chains))
        self.init_evals = int(jnp.sum(n))
        return st

    def init_widths(self):
        w = {"theta": jnp.ones(self.layout.size), "hz": jnp.ones(self.n_hz), "zeta": jnp.ones(self.n_sites)}
        w["z"] = jnp.ones(self.n_pad)
        if self.varying:
            w["hpi"] = jnp.ones((self.k, 1 + self.d))
            w["pi"] = jnp.ones((self.k, self.n_xs))
        return w

    def _whitened_hz_move(self, key, hz, zeta, width, loglik):
        """Slice sweep over hz = (log sigma_z, log ell_z) with eps = L(hz)^{-1} zeta fixed (zeta = L eps).

        Target in (hz, eps): p(hz) N(eps; 0, I) p(data | L(hz) eps); the eps prior does not depend on hz, so
        each coordinate is a plain slice update on log p(hz) + loglik(L(hz) eps). Complements the surrogate
        move: it is efficient when the data constrain zeta weakly (few replicates), where the centred and the
        surrogate parametrisations mix slowly.
        """
        sm = self.aux.site_mask
        # the jittered factor of linalg (as the elliptical move): a plain Cholesky fails for the large ell_z
        # that the PC prior favours, which would truncate the prior
        L0 = linalg.factor(self.zeta_cov(hz), sm).L
        eps = jax.scipy.linalg.solve_triangular(L0, zeta, lower=True)
        ok0 = jnp.all(jnp.isfinite(eps))

        def logdens(h):
            L = linalg.factor(self.zeta_cov(h), sm).L
            lp = self.log_prior_hz(h) + loglik(L @ eps)
            return jnp.where(jnp.isfinite(lp), lp, -jnp.inf)

        h_new, n = slice_step(key, hz, logdens, width)
        h_new = jnp.where(ok0, h_new, hz)
        L1 = linalg.factor(self.zeta_cov(h_new), sm).L
        return h_new, jnp.where(ok0, L1 @ eps, zeta), n

    # --- one Gibbs iteration ---------------------------------------------------------------

    def _theta_logdensity(self, vec, hz, zeta, pi, z):
        t = self.layout.unpack(vec)
        lp = self.log_prior_theta(t, hz) + self.loglik(t, zeta, pi, z)
        return jnp.where(jnp.isnan(lp), -jnp.inf, lp)

    def make_step(self, widths):
        a = self.aux

        def step(key, st):
            ks = jax.random.split(key, 8)
            theta, hz, zeta, z = st["theta"], st["hz"], st["zeta"], st["z"]
            pi = st["pi"] if self.varying else self._no_pi()
            hpi = st["hpi"] if self.varying else None
            # 1. global block
            theta, n1 = slice_step(
                ks[0], theta, lambda v: self._theta_logdensity(v, hz, zeta, pi, z), widths["theta"]
            )
            t = self.layout.unpack(theta)
            # 2. latent noise field
            L = linalg.factor(self.zeta_cov(hz), a.site_mask).L
            zeta, n2 = ess_step(ks[1], zeta, L, lambda f: self.loglik(t, f, pi, z))
            # 3. its hyperparameters, jointly with zeta
            hz, zeta, n3 = surrogate_slice_step(
                ks[2],
                hz,
                zeta,
                self.zeta_cov,
                lambda f: self.loglik(t, f, pi, z),
                self.log_prior_hz,
                self.zeta_surrogate_var,
                widths["hz"],
            )
            # 3b. non-centred move: (log sigma_z, log ell_z) at fixed eps = L(hz)^{-1} zeta
            hz, zeta, n3b = self._whitened_hz_move(
                ks[5], hz, zeta, widths["hz"], lambda f: self.loglik(t, f, pi, z)
            )
            n_evals = n1 + n2 + n3 + n3b
            out = {"theta": theta, "hz": hz, "zeta": zeta}
            # 4. varying order
            if self.varying:
                kj = jax.random.split(ks[3], 2 * self.k)
                for j in range(self.k):

                    def ll_pi(f, j=j, pi=pi):
                        return self.loglik(t, zeta, pi.at[j].set(f), z)

                    Lj = linalg.factor(self.pi_cov(hpi[j]), a.x_mask).L
                    pj, nj = ess_step(kj[2 * j], pi[j], Lj, ll_pi)
                    hj, pj, nh = surrogate_slice_step(
                        kj[2 * j + 1],
                        hpi[j],
                        pj,
                        self.pi_cov,
                        ll_pi,
                        self.log_prior_hpi,
                        lambda h: jnp.full(self.n_xs, PI_SURROGATE_VAR),
                        widths["hpi"][j],
                    )
                    pi, hpi = pi.at[j].set(pj), hpi.at[j].set(hj)
                    n_evals = n_evals + nj + nh
                out["pi"], out["hpi"] = pi, hpi
            # 5. censored outputs
            if self.any_censored:
                z = censored_sweep(ks[4], self.params(t, zeta, pi), a.data, z, self.cfg, a.bounds)
            out["z"] = z
            return out, {"n_evals": n_evals.astype(jnp.int64)}

        return step

    # --- output ----------------------------------------------------------------------------

    def constrain(self, st):
        """State (no batch axes) -> ModelParams."""
        t = self.layout.unpack(st["theta"])
        pi = st["pi"] if self.varying else self._no_pi()
        return self.params(t, st["zeta"], pi)


MAX_COMPILED = 4  # compiled programs kept alive (LRU); a campaign grows n_pad through ~10 buckets
_COMPILED: collections.OrderedDict = collections.OrderedDict()


def _make_runner(template: _Sampler, n_warmup: int, n_samples: int, n_chains: int):
    """Starting rule, warm-up (width adaptation), sampling, constrained parameters.

    Every stage is a compiled program mapped over the chains by mcmc.chains (chain_map: one single-chain
    program per chain, run in parallel threads). The data values enter as arguments, so a second dataset
    of the same shapes reuses the compiled programs (the template only fixes the static structure).
    """

    def init_one(key, aux):
        smp = template.with_aux(aux)
        st, n = smp.init_state(key)
        return st, n, jnp.isfinite(smp.log_density(st))

    init_all = chain_map(init_one, n_chains, 1)
    phase = make_phase(
        lambda w, aux: template.with_aux(aux).make_step(w), max(n_warmup, n_samples, 1), n_chains
    )
    constrain_all = chain_map(
        lambda samples, aux: jax.vmap(template.with_aux(aux).constrain)(samples), n_chains, 1
    )

    def run(aux, key):
        k_init, k_run = jax.random.split(key)
        init, n_init, finite = init_all(jax.random.split(k_init, n_chains), aux)
        res = run_chains(
            k_run,
            init,
            None,
            template.init_widths(),
            n_warmup,
            n_samples,
            n_chains,
            phase=phase,
            extra=(aux,),
        )
        params = constrain_all(res.samples, aux)
        return res, params, jnp.sum(n_init), finite

    return run


def _template_key(sampler: _Sampler):
    return (sampler.cfg, sampler.sc, sampler.nu, sampler.varying, sampler.weight, sampler.any_censored)


def _runner_for(sampler: _Sampler, n_warmup: int, n_samples: int, n_chains: int):
    """The compiled program for this static structure and these shapes, from a bounded LRU.

    The key is (structure, array shapes, chain settings). Programs of the least recently used keys are
    dropped once more than MAX_COMPILED exist, which releases their executables (a long campaign that
    compiled every size kept all of them and ran out of code memory).
    """
    shapes = tuple(a.shape for a in sampler.aux)
    key = (_template_key(sampler), shapes, n_warmup, n_samples, n_chains)
    hit = _COMPILED.get(key)
    if hit is None:
        hit = _make_runner(sampler, n_warmup, n_samples, n_chains)
        _COMPILED[key] = hit
        while len(_COMPILED) > MAX_COMPILED:
            _COMPILED.popitem(last=False)
    _COMPILED.move_to_end(key)
    return hit


def n_compiled() -> int:
    return len(_COMPILED)


def clear_compiled() -> None:
    _COMPILED.clear()


def compiled_entry_for_test():
    """The most recently used compiled program (tests: weak references to see that evicted ones are freed)."""
    return next(reversed(_COMPILED.values()))


def _fit_impl(
    key,
    data,
    z,
    bounds,
    cfg,
    scales,
    n_controls,
    n_warmup,
    n_samples,
    n_chains=4,
    varying_order=False,
    likelihood_weight=1.0,
):
    """fit() with a likelihood weight (0 turns the data off: the chain then targets the prior; tests only)."""
    sampler = _Sampler(data, z, bounds, cfg, scales, n_controls, varying_order, likelihood_weight)
    res, params, n_init, finite = _runner_for(sampler, int(n_warmup), int(n_samples), int(n_chains))(
        sampler.aux, key
    )
    if sampler.weight != 0.0 and not bool(jnp.all(finite)):
        raise RuntimeError("no prior draw with a finite posterior density was found for some chain")
    s = res.samples
    theta = sampler.layout.unpack(s["theta"])
    theta["log_sigma_z"], theta["log_ell_z"] = s["hz"][..., 0], s["hz"][..., 1:]
    if sampler.varying:
        theta["log_sigma_pi"], theta["log_ell_pi"] = s["hpi"][..., 0], s["hpi"][..., 1:]
    theta = {name: np.asarray(v) for name, v in theta.items()}
    scalars = {}
    for name, v in theta.items():
        flat = v.reshape(v.shape[:2] + (-1,))
        for i in range(flat.shape[-1]):
            scalars[name if flat.shape[-1] == 1 and v.ndim == 2 else f"{name}[{i}]"] = flat[:, :, i]
    n_evals = int(n_init) + int(np.sum(res.info["n_evals"]))
    if res.warmup_info is not None:
        n_evals += int(np.sum(res.warmup_info["n_evals"]))
    return Posterior(params, s["z"], theta, dg.summary(scalars), n_evals)


def fit(
    key,
    data,
    z,
    bounds,
    cfg,
    scales,
    n_controls: int,
    n_warmup: int,
    n_samples: int,
    n_chains: int = 4,
    varying_order: bool = False,
) -> Posterior:
    return _fit_impl(
        key, data, z, bounds, cfg, scales, n_controls, n_warmup, n_samples, n_chains, varying_order
    )


def flatten(posterior: Posterior):
    def pool(a):
        return a.reshape((-1,) + a.shape[2:])

    return jax.tree_util.tree_map(pool, posterior.params), pool(posterior.z)
