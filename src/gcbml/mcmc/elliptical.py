"""Elliptical slice sampling (Murray, Adams & MacKay 2010, arXiv 1001.0175).

Implements the algorithm of Fig. 2 (steps 1-10).

ess_step(key, f, prior_chol, loglik) -> (f_new, n_evals)
    f: (n,) current state with prior N(0, L L^T), prior_chol = L (lower, (n, n)).
    loglik: callable f -> scalar. Draw nu = L @ z, z ~ N(0, I) (step 1); level
    log y = loglik(f) + log U (step 2); angle theta ~ U[0, 2 pi), bracket [theta - 2 pi, theta]
    (step 3); propose f' = f cos(theta) + nu sin(theta) (step 4); while loglik(f') <= log y, shrink
    the bracket toward 0 (step 8: theta < 0 sets the lower end, else the upper end) and draw a new
    theta uniformly in it (steps 9-10).

    The shrinkage loop is a bounded lax.while_loop: at most MAX_SHRINK = 200 proposals in total. If the
    bound is hit, f is returned unchanged (the identity move keeps the target invariant) and
    ess_step_info reports it through its third output; ess_step keeps the two-value interface of the
    specification. n_evals = 1 (loglik(f), step 2) + the number of proposals.

    The prior must be zero-mean with a covariance fixed during the update (paper Section 2): callers
    sample mean levels as separate variables (Section 2.5).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

MAX_SHRINK = 200


def ess_step_info(key, f, prior_chol, loglik):
    """As ess_step, plus a boolean that is True when the shrinkage bound was hit."""
    f = jnp.asarray(f, dtype=jnp.float64)
    k_nu, k_u, k_t, k_loop = jax.random.split(key, 4)
    nu = prior_chol @ jax.random.normal(k_nu, f.shape, dtype=jnp.float64)  # step 1
    logy = loglik(f) + jnp.log(jax.random.uniform(k_u, dtype=jnp.float64))  # step 2
    theta0 = jax.random.uniform(k_t, minval=0.0, maxval=2.0 * jnp.pi, dtype=jnp.float64)  # step 3

    def propose(theta):
        fp = f * jnp.cos(theta) + nu * jnp.sin(theta)  # step 4
        return fp, loglik(fp)

    fp0, lp0 = propose(theta0)

    def cond(s):
        _, _, _, _, _, lp, it, _ = s
        return (lp <= logy) & (it < MAX_SHRINK - 1)

    def body(s):
        theta, lo, hi, _, _, _, it, key = s
        lo = jnp.where(theta < 0.0, theta, lo)  # step 8
        hi = jnp.where(theta < 0.0, hi, theta)
        key, sub = jax.random.split(key)
        theta = jax.random.uniform(sub, minval=lo, maxval=hi, dtype=jnp.float64)  # step 9
        fp, lp = propose(theta)
        return theta, lo, hi, fp, theta, lp, it + 1, key

    init = (theta0, theta0 - 2.0 * jnp.pi, theta0, fp0, theta0, lp0, jnp.int32(0), k_loop)
    _, _, _, fp, _, lp, it, _ = lax.while_loop(cond, body, init)
    hit = lp <= logy
    return jnp.where(hit, f, fp), 1 + 1 + it, hit


def ess_step(key, f, prior_chol, loglik):
    f_new, n_evals, _ = ess_step_info(key, f, prior_chol, loglik)
    return f_new, n_evals
