"""Weights of the candidate structures (spec Section 2.7).

A structure M = (h-kernel family, transform Lambda, set of levels used). Each is fitted WITHOUT the finest
level; its held-out finest-level runs are scored by their log predictive density ON THE PHYSICAL SCALE
(the Jacobian of Lambda included, so structures with different transforms compare fairly):

    score_M(i) = log( sum_s w_s N(Lambda_M(y_i); mean_s,i, var_s,i + s^2_s,i) ) + log|Lambda_M'(y_i)|

with (mean, var) from model.predict_level of draw s (the noise variance of the held-out row added), pooled
over the draws of M (w_s = 1 / S). Weights maximise the stacking objective of Yao et al. (1704.02030: eq 2.2
with S the log score, which the paper displays, without a number, after eq 2.3) with a Dirichlet(alpha)
penalty, alpha = 2 [assumption, spec]:

    max_w  sum_i log sum_M w_M exp(score_M(i)) + (alpha - 1) sum_M log w_M,   w on the simplex.

The paper's eq 2.2 averages over n; the sum here only rescales the likelihood against the penalty, i.e. it
fixes alpha as a count of pseudo-observations. [assumption: the sum, not the mean]

Cross-fitting (spec 2.7): the held-out SITES (distinct u) are split at random into two halves; weights
from half A are used to calibrate (gate G1) on half B and vice versa. Each half needs >= 6 sites
[assumption]; otherwise the weights are equal and G1 is "not testable". With only 2 levels left after the
hold-out, p is not identifiable: equal weights, flagged (the caller knows the number of levels left; the
functions here only see scores).

holdout(data, levels_of_rows, finest) -> (train PaddedData, heldout row indices)
    The training data are ``data`` with the finest-level rows masked out (same arrays and padding, so one
    compiled program serves every structure). Censored finest-level rows are left out of both sets
    [assumption]: a lower bound has no density to score.
split_sites(key, X_heldout_u, min_sites=6) -> (idx_A, idx_B) or None when too few sites
    Row indices into X_heldout_u. The smaller half has floor(n_sites / 2) sites.
log_scores(structure_posteriors, train, heldout_rows) -> (M, n_heldout)
    structure_posteriors: list of acquisition.StructurePosterior fitted on ``train``. The order p and the
    noise variance of a held-out row are read from params.P[row] and params.noise_var[row] of each draw
    [assumption: the fit defines them on the held-out rows; true for a shared order].
stack_weights(scores (M, n), alpha=2.0) -> (M,)
    The problem is strictly concave in w for alpha > 1 and has an interior maximum. Its stationarity
    condition, sum_i p_Mi(w) + alpha - 1 = lambda w_M with p_Mi the posterior probability that held-out
    row i belongs to structure M, gives the EM fixed point (the MAP of a mixture with Dirichlet prior):
        w_M <- (sum_i p_Mi(w) + alpha - 1) / (n + M (alpha - 1)),
    which increases the objective at every step. Started at equal weights. (The stub suggested a softmax
    parametrisation with scipy.optimize; the fixed point needs no step size and reaches the KKT conditions
    to round-off, which tests/test_stacking.py checks.) Returns a NumPy array: the solve is not jittable.
cross_fitted_weights(key, scores, site_of_row, alpha=2.0) -> CrossFit(w_A, w_B, idx_A, idx_B, testable)
    w_A is stacked on the columns idx_A and w_B on idx_B. Not testable: equal weights, empty idx_A, idx_B.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from gcbml import model


class CrossFit(NamedTuple):
    w_A: Any
    w_B: Any
    idx_A: Any
    idx_B: Any
    testable: bool


def holdout(data, levels_of_rows, finest: int):
    """Mask the finest-level rows out of ``data``; return (train, held-out row indices)."""
    levels = np.asarray(levels_of_rows)
    mask = np.asarray(data.mask, dtype=bool)
    cens = np.asarray(data.censored, dtype=bool)
    is_finest = mask & (levels == finest)
    rows = np.flatnonzero(is_finest & ~cens)
    train = dataclasses.replace(data, mask=mask & ~is_finest)
    return train, rows


def _split_site_ids(key, site_ids, min_sites):
    """Row indices (idx_A, idx_B) of a random split of the distinct sites in half, or None."""
    site_ids = np.asarray(site_ids)
    _, inv = np.unique(site_ids, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    n_sites = int(inv.max()) + 1 if inv.size else 0
    n_a = n_sites // 2
    if n_a < min_sites:
        return None
    perm = np.asarray(jax.random.permutation(key, n_sites))
    in_a = np.zeros(n_sites, dtype=bool)
    in_a[perm[:n_a]] = True
    return np.flatnonzero(in_a[inv]), np.flatnonzero(~in_a[inv])


def split_sites(key, X_heldout_u, min_sites: int = 6):
    """Split held-out rows by distinct u into two halves of sites; None if a half has < min_sites."""
    X = np.asarray(X_heldout_u)
    if X.ndim == 1:
        X = X[:, None]
    return _split_site_ids(key, X, min_sites)


def log_scores(structure_posteriors, train, heldout_rows):
    """Score matrix (M, n_heldout): pooled-mixture log density of the held-out rows on the physical scale."""
    rows = jnp.asarray(np.asarray(heldout_rows, dtype=int))
    X = jnp.asarray(train.X, dtype=float)[rows]
    H = jnp.asarray(train.H, dtype=float)[rows]
    y = jnp.asarray(train.y, dtype=float)[rows]
    out = []
    for sp in structure_posteriors:
        tf = sp.transform
        zh, log_jac = tf.forward(y), tf.log_abs_jac(y)

        def one(args, cfg=sp.cfg):
            params, z = args
            return model.predict_level(params, train, z, cfg, X, H, params.P[rows], params.noise_var[rows])

        mean, var = jax.jit(lambda p, z, one=one: jax.lax.map(one, (p, z)))(sp.params, sp.z)  # (S, n)
        lp = -0.5 * (jnp.log(2.0 * jnp.pi * var) + (zh[None, :] - mean) ** 2 / var)
        lp = jnp.where(jnp.isfinite(lp), lp, -jnp.inf)
        out.append(jax.scipy.special.logsumexp(lp, axis=0) - jnp.log(lp.shape[0]) + log_jac)
    return jnp.stack(out)


def stack_weights(scores, alpha: float = 2.0, max_iter: int = 200_000, tol: float = 1e-15):
    """Dirichlet-penalised log-score stacking weights (see the module docstring); returns (M,)."""
    s = np.asarray(scores, dtype=float)
    M, n = s.shape
    if M == 1:
        return np.ones(1)
    w = np.full(M, 1.0 / M)
    denom = n + M * (alpha - 1.0)
    for _ in range(max_iter):
        lw = np.log(np.maximum(w, 1e-300))[:, None] + s
        p = np.exp(lw - logsumexp(lw, axis=0)[None, :])
        w_new = (p.sum(axis=1) + (alpha - 1.0)) / denom
        done = np.max(np.abs(w_new - w)) < tol
        w = w_new
        if done:
            break
    return w / w.sum()


def cross_fitted_weights(key, scores, site_of_row, alpha: float = 2.0) -> CrossFit:
    """Weights from each half of the held-out sites (spec 2.7); equal weights, not testable, if < 6 sites."""
    scores = np.asarray(scores, dtype=float)
    M = scores.shape[0]
    halves = _split_site_ids(key, site_of_row, 6)
    if halves is None:
        eq = np.full(M, 1.0 / M)
        empty = np.zeros(0, dtype=int)
        return CrossFit(eq, eq.copy(), empty, empty.copy(), False)
    idx_a, idx_b = halves
    w_a, w_b = stack_weights(scores[:, idx_a], alpha), stack_weights(scores[:, idx_b], alpha)
    return CrossFit(w_a, w_b, idx_a, idx_b, True)
