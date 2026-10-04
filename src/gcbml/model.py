"""Yi-h model in Lambda units (spec Sections 2.2-2.3), with mu, delta and beta integrated out exactly.

For output i (row of the padded data) with unit coordinates x_i, scaled resolution hbar_i (k,) and
per-row orders p_i (k,):

    z_i = Lambda(y_i) = rho0_i + rho1_i * mu(x_i) + delta(x_i, hbar_i) + e_i
    rho0_i = sum_j c0_j hbar_ij^{p_ij},   rho1_i = 1 + sum_j c1_j hbar_ij^{p_ij}
    mu(x)  = m(x)^T beta + g(x),   g ~ GP(0, sigma_mu^2 ard_matern(., ., ell_mu, nu_x))
    delta  ~ GP(0, kernels.delta_cov(...))   (additive over the k components)
    e      ~ N(0, S),  S = D R D,  D = diag(sqrt(noise_var)),  R_il = matern(|v_i - v_l| / ell_v, 1.5)
             if rows i and l belong to the same run (S1 within-run correlation), else 0;
             R = I unless cfg.within_run.

m(x) is the mean basis: "constant" -> [1]; "linear" -> [1, x_1, ..., x_d]. Let A = diag(rho1) M (n, q).

Given all parameters except beta, z is Gaussian with mean rho0 + A beta and covariance

    K = diag(rho1) K_g diag(rho1) + K_delta + S.

beta prior:
  * Gaussian N(b0, diag(bsd^2)): integrate it in: mean rho0 + A b0, covariance K + A diag(bsd^2) A^T.
  * flat (cfg.beta_prior is None): the restricted likelihood (beta integrated over a flat prior):
        log p(z) = -0.5 r^T (K^{-1} - K^{-1} A (A^T K^{-1} A)^{-1} A^T K^{-1}) r - 0.5 log|K|
                   - 0.5 log|A^T K^{-1} A| - 0.5 (n - q) log 2pi,      r = z - rho0.
    A must have full column rank (spec 2.5). A^T K^{-1} A is a small (q, q) matrix: factor it with
    linalg.factor (mask of ones) as well. Never form an explicit inverse.

Implemented in W2-A. All functions below are pure jax (jit, and vmap over parameter draws), use
gcbml.linalg for every solve, and respect the padding mask: padded rows must not affect any result.

mean_basis(X, kind) -> M (n, q)
rho(params, H, P) -> (rho0 (n,), rho1 (n,))        with hbar^p := 0 where hbar = 0.
covariance(params, data, cfg) -> K (n_pad, n_pad)   the covariance above, without the beta term.
log_marginal(params, data, z, cfg) -> scalar       flat or Gaussian beta (cfg.beta_prior).
    z: (n_pad,) transformed outputs (censored rows hold their current imputed values).
    If the factor is non-finite (jitter ladder failed), return -inf.
predict_mu(params, data, z, cfg, Xs) -> (mean (s,), var (s,))
    Posterior of mu(x*) = m(x*)^T beta + g(x*) at unit points Xs (s, d), given z, with beta integrated
    (flat: universal-kriging formulas, including the variance from estimating beta; Gaussian: the joint
    Gaussian conditional). This is the h = 0 prediction.
predict_level(params, data, z, cfg, Xs, Hs, Ps, noise_var_s=None) -> (mean, var)
    Posterior of the noise-free level value f(x*, h*) in Lambda units, rho0* + rho1* mu(x*) + delta(x*, h*).
    If noise_var_s is given, add it to var (predictive density of a new run; used for hold-out scores).
censored_sweep(key, params, data, z, cfg, bounds) -> z_new
    One Gibbs sweep over the censored rows (data.censored & data.mask), in index order. Each censored z_i
    is drawn from its Gaussian full conditional given all other z (beta integrated as above), truncated to
    the side allowed by its bound: for an increasing Lambda the true value is >= bounds_i; for a decreasing
    Lambda (reciprocal) it is <= bounds_i (cfg.increasing). Full conditional of row i, with Q the inverse of
    the (beta-integrated) covariance: mean_i = z_i - (Q r)_i / Q_ii, var_i = 1 / Q_ii. Keep Q r current after
    each draw. Obtain Q_ii from triangular solves, not from an explicit inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri

from gcbml import linalg
from gcbml.kernels import DeltaParams, ard_matern, delta_cov, matern


@dataclass(frozen=True)
class ModelConfig:
    """Static (trace-time) choices of one candidate structure (spec 2.7: M = (k_h family, Lambda, levels))."""

    h_kernel: str = "twy2"  # "twy2" or "lb"
    nu_x: float = 2.5
    nu_h: float = 1.5
    mean_basis: str = "constant"  # "constant" or "linear"
    beta_prior: tuple[tuple[float, ...], tuple[float, ...]] | None = None  # (b0, bsd); None = flat
    within_run: bool = False  # S1: noise correlated within a run through ell_v over the v coordinates
    v_index: tuple[int, ...] = ()  # columns of X that are output coordinates v (S1)
    increasing: bool = True  # Lambda increasing (identity, log) or decreasing (reciprocal)


class ModelParams(NamedTuple):
    """Constrained parameters of one posterior draw. k resolution components, d inputs, n_pad rows."""

    sigma_mu: jnp.ndarray  # ()
    ell_mu: jnp.ndarray  # (d,)
    c0: jnp.ndarray  # (k,)
    c1: jnp.ndarray  # (k,)
    P: jnp.ndarray  # (n_pad, k) per-row order p_j(x_i); constant columns if the order is shared
    delta: DeltaParams
    noise_var: jnp.ndarray  # (n_pad,) s^2(x_i, h_i), from gcbml.noise
    ell_v: jnp.ndarray  # () within-run correlation length over v (ignored unless cfg.within_run)


def _hpow(h, p):
    """h**p with exactly 0 (and a finite gradient) at h = 0."""
    pos = h > 0
    return jnp.where(pos, jnp.where(pos, h, 1.0) ** p, 0.0)


def mean_basis(X, kind: str):
    """Mean basis M (n, q): "constant" -> [1]; "linear" -> [1, x_1, ..., x_d]."""
    X = jnp.asarray(X, dtype=float)
    ones = jnp.ones((X.shape[0], 1))
    if kind == "constant":
        return ones
    if kind == "linear":
        return jnp.concatenate([ones, X], axis=1)
    raise ValueError(f"mean_basis must be 'constant' or 'linear', got {kind!r}")


def rho(params: ModelParams, H, P):
    """(rho0, rho1) of rows with scaled resolutions H (n, k) and orders P (n, k); hbar^p := 0 at hbar = 0."""
    hp = _hpow(jnp.asarray(H, dtype=float), P)
    return hp @ params.c0, 1.0 + hp @ params.c1


def _noise_cov(params: ModelParams, data, cfg: ModelConfig):
    """S = D R D, with R = I unless cfg.within_run (Matern 3/2 in |v_i - v_l| / ell_v inside a run)."""
    nv = params.noise_var
    if not cfg.within_run:
        return jnp.diag(nv)
    if not cfg.v_index:
        raise ValueError("within_run=True needs cfg.v_index (the columns of X that are output coordinates)")
    V = jnp.asarray(data.X, dtype=float)[:, jnp.asarray(cfg.v_index)]
    r2 = jnp.maximum(jnp.sum((V[:, None, :] - V[None, :, :]) ** 2, axis=-1), 0.0)
    R = matern(jnp.sqrt(r2) / params.ell_v, 1.5)
    run = jnp.asarray(data.run)
    same = run[:, None] == run[None, :]
    sd = jnp.sqrt(nv)
    off = jnp.where(same, sd[:, None] * sd[None, :] * R, 0.0)
    eye = jnp.eye(nv.shape[0], dtype=bool)
    return jnp.where(eye, jnp.diag(nv), off)  # exact diagonal (also when nv < 0, which factor() rejects)


class _Setup(NamedTuple):
    mask: jnp.ndarray
    rho0: jnp.ndarray  # (n,) zero on padded rows
    rho1: jnp.ndarray  # (n,) zero on padded rows
    A: jnp.ndarray  # (n, q) diag(rho1) M, zero on padded rows
    K: jnp.ndarray  # (n, n) covariance without the beta term, zero outside the real block


def _setup(params: ModelParams, data, cfg: ModelConfig) -> _Setup:
    mask = jnp.asarray(data.mask, dtype=bool)
    X = jnp.asarray(data.X, dtype=float)
    H = jnp.asarray(data.H, dtype=float)
    rho0, rho1 = rho(params, H, params.P)
    rho0, rho1 = jnp.where(mask, rho0, 0.0), jnp.where(mask, rho1, 0.0)
    A = jnp.where(mask[:, None], rho1[:, None] * mean_basis(X, cfg.mean_basis), 0.0)
    Kg = params.sigma_mu**2 * ard_matern(X, X, params.ell_mu, cfg.nu_x)
    Kd = delta_cov(X, H, X, H, params.P, params.P, params.delta, cfg.h_kernel, cfg.nu_x, cfg.nu_h)
    K = rho1[:, None] * Kg * rho1[None, :] + Kd + _noise_cov(params, data, cfg)
    K = jnp.where(mask[:, None] & mask[None, :], K, 0.0)
    return _Setup(mask, rho0, rho1, A, K)


def covariance(params: ModelParams, data, cfg: ModelConfig):
    """K = diag(rho1) K_g diag(rho1) + K_delta + S on the real block; padded rows and columns are zero.

    (linalg.factor adds the identity on the padded block.)
    """
    return _setup(params, data, cfg).K


def _beta_prior(cfg: ModelConfig):
    b0, bsd = cfg.beta_prior
    return jnp.asarray(b0, dtype=float), jnp.asarray(bsd, dtype=float)


def _flat_pieces(S: _Setup):
    """Factor of K, K^{-1} A and the factor of the small matrix G = A^T K^{-1} A (flat beta)."""
    F = linalg.factor(S.K, S.mask)
    KiA = linalg.solve(F, S.A)
    G = S.A.T @ KiA
    Fg = linalg.factor(0.5 * (G + G.T), jnp.ones(G.shape[0], dtype=bool))
    return F, KiA, Fg


def log_marginal(params: ModelParams, data, z, cfg: ModelConfig):
    """log p(z | params) with mu, delta and beta integrated out; -inf if the covariance cannot be factored.

    Gaussian beta: log N(z; rho0 + A b0, K + A diag(bsd^2) A^T). Flat beta: the restricted likelihood of the
    module docstring. With q coefficients all of prior sd bsd, flat = Gaussian + q log(bsd) + (q/2) log(2 pi)
    in the limit bsd -> infinity (log|K + A B A^T| = log|B| + log|B^{-1} + A^T K^{-1} A|, and the Gaussian
    carries n/2 log 2 pi against (n - q)/2 log 2 pi); the gap shrinks like 1/bsd^2 (tests/test_model.py).
    """
    S = _setup(params, data, cfg)
    z = jnp.asarray(z, dtype=float)
    n = jnp.sum(S.mask)
    if cfg.beta_prior is None:
        r = jnp.where(S.mask, z - S.rho0, 0.0)
        F, KiA, Fg = _flat_pieces(S)
        Kir = linalg.solve(F, r)
        Atr = S.A.T @ Kir
        q = S.A.shape[1]
        val = (
            -0.5 * (r @ Kir - linalg.quad(Fg, Atr))
            - 0.5 * linalg.logdet(F)
            - 0.5 * linalg.logdet(Fg)
            - 0.5 * (n - q) * jnp.log(2.0 * jnp.pi)
        )
    else:
        b0, bsd = _beta_prior(cfg)
        Kt = S.K + (S.A * bsd**2) @ S.A.T
        r = jnp.where(S.mask, z - S.rho0 - S.A @ b0, 0.0)
        val = linalg.gaussian_logpdf(linalg.factor(Kt, S.mask), r, S.mask)
    return jnp.where(jnp.isfinite(val), val, -jnp.inf)


def _posterior(params, data, z, cfg, S: _Setup, As, C, v, offset):
    """Posterior of f_* = offset + As beta + u_*, with u_* the non-beta part (cov C with z, variance v).

    C (n, s) must be zero on padded rows. Flat beta: universal kriging (generalised least squares beta_hat
    plus the variance of estimating it). Gaussian beta: the joint Gaussian conditional.
    """
    z = jnp.asarray(z, dtype=float)
    if cfg.beta_prior is None:
        r = jnp.where(S.mask, z - S.rho0, 0.0)
        F, KiA, Fg = _flat_pieces(S)
        Kir = linalg.solve(F, r)
        bhat = linalg.solve(Fg, S.A.T @ Kir)
        KiC = linalg.solve(F, C)
        mean = offset + As @ bhat + C.T @ (Kir - KiA @ bhat)
        Rm = As.T - S.A.T @ KiC
        var = v - jnp.sum(C * KiC, axis=0) + jnp.sum(Rm * linalg.solve(Fg, Rm), axis=0)
    else:
        b0, bsd = _beta_prior(cfg)
        Kt = S.K + (S.A * bsd**2) @ S.A.T
        Ct = C + S.A @ ((bsd**2)[:, None] * As.T)
        vt = v + jnp.sum(As * bsd**2 * As, axis=1)
        r = jnp.where(S.mask, z - S.rho0 - S.A @ b0, 0.0)
        mean, var = linalg.conditional(linalg.factor(Kt, S.mask), Ct, vt, r)
        mean = mean + offset + As @ b0
    return mean, jnp.maximum(var, 0.0)


def predict_mu(params: ModelParams, data, z, cfg: ModelConfig, Xs):
    """Posterior mean and variance of mu(x*) = m(x*)^T beta + g(x*) at unit points Xs (s, d)."""
    S = _setup(params, data, cfg)
    X = jnp.asarray(data.X, dtype=float)
    Xs = jnp.asarray(Xs, dtype=float)
    Ms = mean_basis(Xs, cfg.mean_basis)
    Kxs = params.sigma_mu**2 * ard_matern(X, Xs, params.ell_mu, cfg.nu_x)
    C = jnp.where(S.mask[:, None], S.rho1[:, None] * Kxs, 0.0)
    v = jnp.full(Xs.shape[0], params.sigma_mu**2)
    return _posterior(params, data, z, cfg, S, Ms, C, v, jnp.zeros(Xs.shape[0]))


def predict_level(params: ModelParams, data, z, cfg: ModelConfig, Xs, Hs, Ps, noise_var_s=None):
    """Posterior of the noise-free level f(x*, h*) = rho0* + rho1* mu(x*) + delta(x*, h*); see module docs."""
    S = _setup(params, data, cfg)
    X = jnp.asarray(data.X, dtype=float)
    H = jnp.asarray(data.H, dtype=float)
    Xs, Hs, Ps = (jnp.asarray(a, dtype=float) for a in (Xs, Hs, Ps))
    rho0s, rho1s = rho(params, Hs, Ps)
    Ms = mean_basis(Xs, cfg.mean_basis)
    Kxs = params.sigma_mu**2 * ard_matern(X, Xs, params.ell_mu, cfg.nu_x)
    Kds = delta_cov(X, H, Xs, Hs, params.P, Ps, params.delta, cfg.h_kernel, cfg.nu_x, cfg.nu_h)
    C = jnp.where(S.mask[:, None], S.rho1[:, None] * Kxs * rho1s[None, :] + Kds, 0.0)
    Kss = delta_cov(Xs, Hs, Xs, Hs, Ps, Ps, params.delta, cfg.h_kernel, cfg.nu_x, cfg.nu_h)
    v = rho1s**2 * params.sigma_mu**2 + jnp.diag(Kss)
    mean, var = _posterior(params, data, z, cfg, S, rho1s[:, None] * Ms, C, v, rho0s)
    if noise_var_s is not None:
        var = var + jnp.asarray(noise_var_s, dtype=float)
    return mean, var


def _lower_truncated_normal(key, a):
    """Standard normal draw conditioned on x >= a, by inverse-CDF on the stable tail (a is a scalar)."""
    u = jax.random.uniform(key, (), minval=jnp.finfo(float).tiny, maxval=1.0)
    # a <= 0: x = Phi^{-1}(Phi(a) + u (1 - Phi(a))); a > 0: x = -Phi^{-1}(u Phi(-a)) (no cancellation)
    lo = ndtri(ndtr(a) + u * ndtr(-a))
    mid = -ndtri(u * ndtr(-a))
    # beyond a = 30, Phi(-a) is ~1e-197 and the tail is Rayleigh to relative O(1/a^2): x = sqrt(a^2 - 2 log u)
    tail = jnp.sqrt(a * a - 2.0 * jnp.log(u))
    x = jnp.where(a > 30.0, tail, jnp.where(a > 0.0, mid, lo))
    return jnp.maximum(x, a)


def censored_sweep(key, params: ModelParams, data, z, cfg: ModelConfig, bounds):
    """One Gibbs sweep over the censored rows, in index order; see the module docstring.

    The full conditional of row i uses the (beta-integrated) precision Q only through its column i, which
    is obtained by triangular solves (flat beta: Q = K^{-1} - K^{-1}A G^{-1} A^T K^{-1}, so
    Q e_i = K^{-1} e_i - (K^{-1}A) G^{-1} (K^{-1}A)_i^T). Q r is kept current after each draw.
    """
    S = _setup(params, data, cfg)
    z = jnp.asarray(z, dtype=float)
    bounds = jnp.asarray(bounds, dtype=float)
    n = z.shape[0]
    if cfg.beta_prior is None:
        F, KiA, Fg = _flat_pieces(S)
        r = jnp.where(S.mask, z - S.rho0, 0.0)
        Qr = linalg.solve(F, r) - KiA @ linalg.solve(Fg, KiA.T @ r)

        def qcol(i):
            u = linalg.solve(F, jnp.zeros(n).at[i].set(1.0))
            return u - KiA @ linalg.solve(Fg, KiA[i])
    else:
        b0, bsd = _beta_prior(cfg)
        F = linalg.factor(S.K + (S.A * bsd**2) @ S.A.T, S.mask)
        r = jnp.where(S.mask, z - S.rho0 - S.A @ b0, 0.0)
        Qr = linalg.solve(F, r)

        def qcol(i):
            return linalg.solve(F, jnp.zeros(n).at[i].set(1.0))

    sgn = 1.0 if cfg.increasing else -1.0
    do = jnp.asarray(data.censored, dtype=bool) & S.mask

    def step(carry, inp):
        z, Qr = carry
        i, k, flag = inp

        def draw(_):
            col = qcol(i)
            qii = col[i]
            sd = 1.0 / jnp.sqrt(qii)
            m = z[i] - Qr[i] / qii
            a = sgn * (bounds[i] - m) / sd  # allowed region is sgn * (x - bound) >= 0
            x = _lower_truncated_normal(k, a)
            zi = m + sgn * sd * x
            return z.at[i].set(zi), Qr + col * (zi - z[i])

        return jax.lax.cond(flag, draw, lambda _: (z, Qr), None), None

    (z_new, _), _ = jax.lax.scan(step, (z, Qr), (jnp.arange(n), jax.random.split(key, n), do))
    return z_new
