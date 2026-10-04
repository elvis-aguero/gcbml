"""Prediction on the physical scale (spec Section 2.8).

Inputs are posterior draws in Lambda units, pooled over candidate structures with stacking weights.
Every reported spread is a quantile half-width, sigma = (q84 - q16) / 2, and every centre is a median,
because Lambda^{-1} of a Gaussian need not have finite moments (e.g. 1/rate). Draw weights w (S,) sum to 1.

TODO(W2-A): implement. NumPy or jax; these run outside the sampler.

weighted_quantile(values (S, s), w (S,), q) -> (s,)
    Quantile q of a weighted sample per column. Linear interpolation of the weighted empirical CDF
    (state the convention in the docstring); exact for equal weights against numpy.quantile(method=...)
    with the matching method.
sample_gaussian_draws(key, mean (S, s), var (S, s), n_rep) -> (S * n_rep, s), w_rep
    For each posterior draw, n_rep samples from its Gaussian conditional (e.g. of mu at Sigma_N).
    Returns the samples and the weights repeated (w / n_rep each).
summarize(z_draws, w, transform) -> dict(m=..., sigma=..., q025=..., q975=..., q16=..., q84=...)
    Back-transform with transform.inverse, then weighted quantiles. sigma_epi = summarize(mu draws).sigma.
sigma_env(c0 (S, k), c1 (S, k), P (S, s, k), sigma_delta (S, k), kx_diag (S, s, k), mu (S, s), Hbar (s, k), w)
    Spec 2.8, in Lambda units:
      sigma_env^2(x, h) = sum_S w_S sum_j hbar_j^{2 p_j} ((c0_j + c1_j mu)^2 + sigma_delta_j^2 kx_jj(x)).
    kx_diag is the x-kernel diagonal of each component (1 for a correlation kernel). Returns (s,) sd.
sigma_fid(f_level_draws (S, s), f0_draws (S, s), w, transform)
    sqrt(E[(Lambda^{-1}(f_level) - Lambda^{-1}(f0))^2]) over weighted draws, physical scale (spec 2.8).
sigma_know(f_level_draws, w, transform)   quantile half-width of Lambda^{-1}(f_level) (spec 1, table).
s0(m_y (s,), noise_sd0_draws (S, s), w, transform, key, n_rep)
    Half-width (q84 - q16)/2 of Lambda^{-1}(Lambda(m_y) + e), e ~ N(0, s^2(x, 0)), over the draws of s.
sigma_tot(mu_draws (S, s), noise_sd0_draws (S, s), w, transform, key, n_rep)
    Quantile half-width of Lambda^{-1}(mu + e) at h = 0 (the only definition; not a quadrature sum).
"""

from __future__ import annotations


def weighted_quantile(values, w, q):
    raise NotImplementedError("W2-A")


def sample_gaussian_draws(key, mean, var, n_rep):
    raise NotImplementedError("W2-A")


def summarize(z_draws, w, transform):
    raise NotImplementedError("W2-A")


def sigma_env(c0, c1, P, sigma_delta, kx_diag, mu, Hbar, w):
    raise NotImplementedError("W2-A")


def sigma_fid(f_level_draws, f0_draws, w, transform):
    raise NotImplementedError("W2-A")


def sigma_know(f_level_draws, w, transform):
    raise NotImplementedError("W2-A")


def s0(m_y, noise_sd0_draws, w, transform, key, n_rep):
    raise NotImplementedError("W2-A")


def sigma_tot(mu_draws, noise_sd0_draws, w, transform, key, n_rep):
    raise NotImplementedError("W2-A")
