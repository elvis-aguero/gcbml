"""Slice sampling of covariance hyperparameters with surrogate data (Murray & Adams 2010, arXiv 1006.0868).

Used for the hyperparameters theta of a latent Gaussian field f ~ N(0, Sigma_theta) (in gcbml: the
latent log-noise-variance field zeta and the varying-order field pi). Sampling theta with f fixed
mixes badly when the data are strong (Section 2, Fig. 1); the surrogate-data method reparametrises f
through auxiliary Gaussian "surrogate data" g ~ N(f, S_theta) (eq 7) so that theta and f move together.

surrogate_slice_step(key, theta, f, prior_cov, loglik, log_prior_theta, surrogate_var, width)
        -> (theta_new, f_new, n_evals)
    theta: (q,) unconstrained hyperparameters; f: (n,) latent field.
    prior_cov(theta) -> (n, n) covariance Sigma_theta (zero mean).
    loglik(f) -> scalar log-likelihood of the data given f.
    log_prior_theta(theta) -> scalar.
    surrogate_var(theta) -> (n,) diagonal of S_theta (Section 3.2 explains how to choose it: match
        the site posterior (eq 12), (S)_ii = (1/v_i - 1/(Sigma)_ii)^{-1}, thresholded at positive
        values; for a Gaussian likelihood this is the observation noise).
    width: (q,) slice widths for theta.

One call is a sweep over the q coordinates of theta in fixed order. Each coordinate update is one run
of Algorithm 4 of the paper, with the other coordinates held fixed:
    1. draw surrogate data g ~ N(f, S_theta)                                      (Alg. 4, step 3)
    2. eta = L_R^{-1} (f - m), where R, m are the posterior covariance and mean of f given g
       (eq 9) and L_R L_R^T = R                                                   (step 4, eq 10)
    3. randomly centred bracket of width width[j]: v ~ U(0, width[j]),
       [theta_j - v, theta_j - v + width[j]]                                      (step 5)
    4. level log y = log U + log L(f) + log N(g; 0, Sigma + S) + log p(theta)      (steps 6-7)
    5. propose theta_j' uniformly in the bracket, set f' = L_R(theta') eta + m(theta'), accept if
       log L(f') + log N(g; 0, Sigma' + S') + log p(theta') > log y (eq 11), else shrink the bracket
       toward theta_j and repeat                                                   (steps 8-16)
The paper's Algorithm 4 has no stepping out (it only suggests it in Section 3.1), and neither does this
function: width[j] is the initial bracket, and the shrinkage makes too-wide brackets harmless. A too
small width only limits the move size, never validity.
Shrinkage is bounded (MAX_SHRINK = 100 proposals per coordinate); if the bound is hit the coordinate
keeps its current (theta_j, f), which keeps invariance.

Numerics. No explicit inverse: with B = I + S^{-1/2} Sigma S^{-1/2} and L_B L_B^T = B (all
eigenvalues of B are >= 1, so it is well conditioned), V = L_B^{-1} S^{-1/2} Sigma gives
R = Sigma - V^T V, m = V^T L_B^{-1} S^{-1/2} g, and Sigma + S = S^{1/2} B S^{1/2} gives
log N(g; 0, Sigma + S) = -1/2 |L_B^{-1} S^{-1/2} g|^2 - 1/2 sum log S - sum log diag L_B + const
(the same algebra as Rasmussen & Williams Alg. 3.1). R is positive semi-definite only up to round-off,
so L_R is the Cholesky factor of R + 1e-12 mean(diag Sigma) I; this perturbs only directions whose
variance is below that level. A non-finite factorisation gives target density 0 (the point is
rejected).
n_evals counts the calls of loglik (the current state once per coordinate, plus one per proposal),
which is also the number of covariance factorisations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_factor, solve_triangular

MAX_SHRINK = 100
R_JITTER = 1e-12


def _chol(K):
    """Lower Cholesky factor (NaN-filled on failure). The single place to swap for gcbml.linalg.factor."""
    return jnp.tril(cho_factor(K, lower=True)[0])


def _tri_solve(L, b):
    """L^{-1} b by a triangular solve (never an explicit inverse)."""
    return solve_triangular(L, b, lower=True)


def _surrogate_posterior(sigma, s_diag, g):
    """Return (m, L_R, log N(g; 0, Sigma + S) up to the constant -n/2 log 2 pi), eqs 8-10."""
    n = sigma.shape[0]
    s_inv_half = 1.0 / jnp.sqrt(s_diag)
    w = s_inv_half[:, None] * sigma
    b = jnp.eye(n, dtype=sigma.dtype) + w * s_inv_half[None, :]
    lb = _chol(b)
    v = _tri_solve(lb, w)
    t = _tri_solve(lb, s_inv_half * g)
    m = v.T @ t
    r = sigma - v.T @ v
    r = r + R_JITTER * jnp.mean(jnp.diag(sigma)) * jnp.eye(n, dtype=sigma.dtype)
    lr = _chol(r)
    log_g = -0.5 * jnp.sum(t**2) - 0.5 * jnp.sum(jnp.log(s_diag)) - jnp.sum(jnp.log(jnp.diag(lb)))
    return m, lr, log_g


def surrogate_slice_step(key, theta, f, prior_cov, loglik, log_prior_theta, surrogate_var, width):
    theta = jnp.asarray(theta, dtype=jnp.float64)
    f = jnp.asarray(f, dtype=jnp.float64)
    width = jnp.broadcast_to(jnp.asarray(width, dtype=jnp.float64), theta.shape)
    keys = jax.random.split(key, theta.shape[0])

    def coordinate(j, carry):
        theta, f, n_evals = carry
        k_g, k_v, k_u, k_loop = jax.random.split(keys[j], 4)
        # step 3: surrogate data
        s0 = surrogate_var(theta)
        g = f + jnp.sqrt(s0) * jax.random.normal(k_g, f.shape, dtype=jnp.float64)
        # step 4: whitened variate at the current hyperparameters
        m0, lr0, log_g0 = _surrogate_posterior(prior_cov(theta), s0, g)
        eta = _tri_solve(lr0, f - m0)
        # steps 5-7: bracket and level
        w = width[j]
        v = jax.random.uniform(k_v, dtype=jnp.float64) * w
        lo0, hi0 = theta[j] - v, theta[j] - v + w
        logy = jnp.log(jax.random.uniform(k_u, dtype=jnp.float64))
        logy = logy + loglik(f) + log_g0 + log_prior_theta(theta)

        def target(th_j):
            th = theta.at[j].set(th_j)
            m, lr, log_g = _surrogate_posterior(prior_cov(th), surrogate_var(th), g)
            fp = lr @ eta + m
            lp = loglik(fp) + log_g + log_prior_theta(th)
            return jnp.where(jnp.isnan(lp), -jnp.inf, lp), fp

        def cond(s):
            _, _, _, _, _, ok, it, _ = s
            return (~ok) & (it < MAX_SHRINK)

        def body(s):
            lo, hi, _, _, _, _, it, key = s
            key, sub = jax.random.split(key)
            th_j = lo + jax.random.uniform(sub, dtype=jnp.float64) * (hi - lo)
            lp, fp = target(th_j)
            ok = lp > logy
            lo = jnp.where(~ok & (th_j < theta[j]), th_j, lo)  # steps 12-15
            hi = jnp.where(~ok & (th_j >= theta[j]), th_j, hi)
            return lo, hi, th_j, fp, lp, ok, it + 1, key

        init = (lo0, hi0, theta[j], f, logy, jnp.array(False), jnp.int32(0), k_loop)
        _, _, th_j, fp, _, ok, it, _ = lax.while_loop(cond, body, init)
        theta = jnp.where(ok, theta.at[j].set(th_j), theta)
        f = jnp.where(ok, fp, f)
        return theta, f, n_evals + 1 + it

    theta, f, n_evals = lax.fori_loop(0, theta.shape[0], coordinate, (theta, f, jnp.int32(0)))
    return theta, f, n_evals
