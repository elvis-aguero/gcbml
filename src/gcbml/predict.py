"""Prediction on the physical scale (spec Section 2.8).

Inputs are posterior draws in Lambda units, pooled over candidate structures with stacking weights.
Every reported spread is a quantile half-width, sigma = (q84 - q16) / 2, and every centre is a median,
because Lambda^{-1} of a Gaussian need not have finite moments (e.g. 1/rate). Draw weights w (S,) sum to 1.

Implemented in W2-A with jax.numpy (pure, jittable).

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

import jax
import jax.numpy as jnp


def _wq_column(x, w, q):
    """Quantile q of one weighted column: knots at p_i = cumsum(w)_i - w_i / 2 over the sorted draws."""
    order = jnp.argsort(jnp.where(w > 0, x, jnp.inf))  # zero-weight draws go last and are ignored
    xs, ws = x[order], w[order]
    p = jnp.cumsum(ws) - 0.5 * ws
    last = xs[jnp.maximum(jnp.sum(ws > 0) - 1, 0)]
    xs = jnp.where(ws > 0, xs, last)
    return jnp.interp(q, p, xs)


def weighted_quantile(values, w, q):
    """Quantile q in [0, 1] of a weighted sample, per column of ``values`` (S, s); returns (s,).

    Convention (Hazen): sort the draws, put draw i at the probability p_i = sum_{l<=i} w_l - w_i / 2 of the
    normalised weights, interpolate linearly between the knots and hold the end values outside [p_1, p_S].
    For equal weights p_i = (i - 1/2) / S, which is ``numpy.quantile(..., method="hazen")`` exactly.
    Zero-weight draws are ignored. A draw of weight w_i behaves as one atom at its centre. The quantile of
    a sample where each draw is repeated in proportion to its weight agrees with it exactly at the atom
    centres p_i, and to within the spacing of neighbouring draws elsewhere (it is flat inside an atom).
    """
    values = jnp.asarray(values, dtype=float)
    w = jnp.asarray(w, dtype=float)
    w = w / jnp.sum(w)
    return jax.vmap(lambda col: _wq_column(col, w, q), in_axes=1)(values)


def sample_gaussian_draws(key, mean, var, n_rep, w=None):
    """n_rep samples N(mean_S, var_S) for each of the S posterior draws (columns independent).

    Returns (samples (S * n_rep, s), w_rep (S * n_rep,)). Row k * n_rep + r is replicate r of draw k and has
    weight w_k / n_rep (``w`` defaults to 1 / S). ``w`` is an addition to the stub signature: the stub
    returns "the weights repeated" but had no weights to repeat.
    """
    mean = jnp.asarray(mean, dtype=float)
    var = jnp.asarray(var, dtype=float)
    S, s = mean.shape
    w = jnp.full(S, 1.0 / S) if w is None else jnp.asarray(w, dtype=float)
    eps = jax.random.normal(key, (S, n_rep, s))
    out = mean[:, None, :] + jnp.sqrt(jnp.maximum(var, 0.0))[:, None, :] * eps
    return out.reshape(S * n_rep, s), jnp.repeat(w / n_rep, n_rep)


def _halfwidth(y, w):
    return 0.5 * (weighted_quantile(y, w, 0.84) - weighted_quantile(y, w, 0.16))


def summarize(z_draws, w, transform):
    """Back-transform draws (S, s) with transform.inverse; median, quantile half-width and quantiles."""
    y = transform.inverse(jnp.asarray(z_draws, dtype=float))
    q = {name: weighted_quantile(y, w, p) for name, p in _LEVELS}
    return dict(
        m=q["m"],
        sigma=0.5 * (q["q84"] - q["q16"]),
        q025=q["q025"],
        q975=q["q975"],
        q16=q["q16"],
        q84=q["q84"],
    )


_LEVELS = (("m", 0.5), ("q025", 0.025), ("q975", 0.975), ("q16", 0.16), ("q84", 0.84))


def sigma_env(c0, c1, P, sigma_delta, kx_diag, mu, Hbar, w):
    """Fidelity envelope in Lambda units (spec 2.8): sqrt of the weighted draw average of

        sum_j hbar_j^{2 p_j} ((c0_j + c1_j mu)^2 + sigma_delta_j^2 kx_jj(x)),

    with hbar^{2p} := 0 at hbar = 0. Shapes: c0, c1, sigma_delta (S, k); P, kx_diag (S, s, k); mu (S, s);
    Hbar (s, k); w (S,). Returns (s,).
    """
    c0, c1, P, sd, kx, mu, Hb, w = (
        jnp.asarray(a, dtype=float) for a in (c0, c1, P, sigma_delta, kx_diag, mu, Hbar, w)
    )
    pos = Hb[None] > 0
    hp = jnp.where(pos, jnp.where(pos, Hb[None], 1.0) ** (2.0 * P), 0.0)
    term = (c0[:, None, :] + c1[:, None, :] * mu[:, :, None]) ** 2 + sd[:, None, :] ** 2 * kx
    per_draw = jnp.sum(hp * term, axis=-1)  # (S, s)
    return jnp.sqrt(jnp.sum(w[:, None] * per_draw, axis=0) / jnp.sum(w))


def sigma_fid(f_level_draws, f0_draws, w, transform):
    """sqrt(E_w[(Lambda^{-1}(f_level) - Lambda^{-1}(f0))^2]) per column (spec 2.8)."""
    w = jnp.asarray(w, dtype=float)
    d = transform.inverse(jnp.asarray(f_level_draws, dtype=float)) - transform.inverse(
        jnp.asarray(f0_draws, dtype=float)
    )
    return jnp.sqrt(jnp.sum(w[:, None] * d**2, axis=0) / jnp.sum(w))


def sigma_know(f_level_draws, w, transform):
    """Quantile half-width (q84 - q16) / 2 of Lambda^{-1}(f_level) per column."""
    return _halfwidth(transform.inverse(jnp.asarray(f_level_draws, dtype=float)), w)


def s0(m_y, noise_sd0_draws, w, transform, key, n_rep):
    """Half-width of Lambda^{-1}(Lambda(m_y) + e), e ~ N(0, s^2(x, 0)), over the draws of s (spec 2.8)."""
    sd = jnp.asarray(noise_sd0_draws, dtype=float)
    center = jnp.broadcast_to(transform.forward(jnp.asarray(m_y, dtype=float)), sd.shape)
    z, wr = sample_gaussian_draws(key, center, sd**2, n_rep, w)
    return _halfwidth(transform.inverse(z), wr)


def sigma_tot(mu_draws, noise_sd0_draws, w, transform, key, n_rep):
    """Half-width of Lambda^{-1}(mu + e) at h = 0, e ~ N(0, s^2(x, 0)) (spec 2.8; the only definition)."""
    sd = jnp.asarray(noise_sd0_draws, dtype=float)
    z, wr = sample_gaussian_draws(key, jnp.asarray(mu_draws, dtype=float), sd**2, n_rep, w)
    return _halfwidth(transform.inverse(z), wr)
