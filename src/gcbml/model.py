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

TODO(W2-A): implement the functions below. All are pure jax (jit, and vmap over parameter draws), use
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

import jax.numpy as jnp

from gcbml.kernels import DeltaParams


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


def mean_basis(X, kind: str):
    raise NotImplementedError("W2-A")


def rho(params: ModelParams, H, P):
    raise NotImplementedError("W2-A")


def covariance(params: ModelParams, data, cfg: ModelConfig):
    raise NotImplementedError("W2-A")


def log_marginal(params: ModelParams, data, z, cfg: ModelConfig):
    raise NotImplementedError("W2-A")


def predict_mu(params: ModelParams, data, z, cfg: ModelConfig, Xs):
    raise NotImplementedError("W2-A")


def predict_level(params: ModelParams, data, z, cfg: ModelConfig, Xs, Hs, Ps, noise_var_s=None):
    raise NotImplementedError("W2-A")


def censored_sweep(key, params: ModelParams, data, z, cfg: ModelConfig, bounds):
    raise NotImplementedError("W2-A")
