"""Dense linear algebra behind a solver interface (exact now; a sparse solver can be added later).

All arrays are padded to a bucket size (gcbml._config.bucket). A boolean ``mask`` marks real rows.
Padded rows/columns are decoupled: the masked covariance is

    K_m = where(mask_i & mask_j, K, 0) + diag(where(mask, 0, 1))

so the padded block is the identity, contributes 0 to log det, and right-hand sides must be zero
on padded rows. Every function is jit- and vmap-compatible (vmap over posterior draws/fantasies).

factor(K, mask) -> Factor
    Cholesky of K_m. Jitter ladder: try jitter = 0; if the factor has a non-finite entry, retry
    with jitter = 10^{-12}, 10^{-10}, ..., 10^{-4} times mean(diag(K) over masked rows),
    inside lax.while_loop.
    The returned Factor records the jitter used (callers report it). If 10^{-4} still fails, the factor
    is non-finite; callers treat that as log-likelihood = -inf.
solve(F, b)            K_m^{-1} b, with b of shape (n,) or (n, r); two triangular solves (no inverse).
logdet(F)              log det K_m = 2 sum log diag L.
quad(F, b)             b^T K_m^{-1} b.
gaussian_logpdf(F, r, mask)   log N(r; 0, K) over masked rows only: -0.5 quad - 0.5 logdet - 0.5 n log 2pi.
append(F, K_cross, K_new, mask_new) -> Factor
    Block Cholesky update after appending q rows: K_cross (n, q) between old and new rows, K_new (q, q).
    Equals factor() of the enlarged matrix to round-off (used for fantasies; spec Step 5).
conditional(F, K_cross, K_new_diag, r) -> (mean, var)
    GP conditional at new points: mean = K_cross^T K_m^{-1} r,
    var = K_new_diag - colsum(V^2), V = L^{-1} K_cross.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


class Factor(NamedTuple):
    L: jnp.ndarray  # (n, n) lower Cholesky factor of the masked, jittered matrix
    jitter: jnp.ndarray  # scalar actually added to the diagonal


_LADDER = jnp.array([0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4])


def _mask_matrix(K, mask):
    """K_m = where(mask_i & mask_j, K, 0) + diag(where(mask, 0, 1)); padded entries (even NaN) are dropped."""
    both = mask[:, None] & mask[None, :]
    return jnp.where(both, K, 0.0) + jnp.diag(jnp.where(mask, 0.0, 1.0))


def _cholesky_ladder_impl(Km, mask, scale):
    """Cholesky of Km + jitter * scale * diag(mask), with jitter walked up the ladder until it is finite.

    Returns (L, jitter * scale). If the last rung fails too, L is non-finite.
    """
    dmask = jnp.where(mask, 1.0, 0.0)

    def attempt(i):
        return jnp.linalg.cholesky(Km + _LADDER[i] * scale * jnp.diag(dmask))

    def cond(state):
        i, L = state
        return (~jnp.all(jnp.isfinite(L))) & (i < _LADDER.shape[0] - 1)

    def body(state):
        i, _ = state
        return i + 1, attempt(i + 1)

    i, L = jax.lax.while_loop(cond, body, (0, attempt(0)))
    return L, _LADDER[i] * scale


@jax.custom_jvp
def _cholesky_ladder(Km, mask, scale):
    """_cholesky_ladder_impl with a derivative: a lax.while_loop cannot be reverse-differentiated.

    The tangent is that of jnp.linalg.cholesky at the jittered matrix that the ladder settled on, with the
    jitter held constant (its dependence on K is of the order of the jitter, <= 1e-4 of the mean variance and
    0 whenever the plain factor exists). The primal output is unchanged, so the samplers are not affected.
    """
    return _cholesky_ladder_impl(Km, mask, scale)


@_cholesky_ladder.defjvp
def _cholesky_ladder_jvp(primals, tangents):
    Km, mask, scale = primals
    dKm = tangents[0]
    L, jit = _cholesky_ladder_impl(Km, mask, scale)
    Kj = Km + jit * jnp.diag(jnp.where(mask, 1.0, 0.0))
    _, dL = jax.jvp(jnp.linalg.cholesky, (Kj,), (dKm,))
    return (L, jit), (dL, jnp.zeros_like(jit))


def factor(K: jnp.ndarray, mask: jnp.ndarray) -> Factor:
    """Cholesky of the masked matrix with the jitter ladder (see the module docstring)."""
    K = jnp.asarray(K, dtype=float)
    mask = jnp.asarray(mask, dtype=bool)
    Km = _mask_matrix(K, mask)
    scale = jnp.sum(jnp.where(mask, jnp.diag(K), 0.0)) / jnp.maximum(jnp.sum(mask), 1)
    L, jitter = _cholesky_ladder(Km, mask, scale)
    return Factor(L, jitter)


def _lsolve(L, b):
    return solve_triangular(L, b, lower=True)


def solve(F: Factor, b: jnp.ndarray) -> jnp.ndarray:
    """K_m^{-1} b by two triangular solves; b is (n,) or (n, r) and zero on padded rows."""
    return solve_triangular(F.L, _lsolve(F.L, b), lower=True, trans=1)


def logdet(F: Factor) -> jnp.ndarray:
    return 2.0 * jnp.sum(jnp.log(jnp.diag(F.L)))


def quad(F: Factor, b: jnp.ndarray) -> jnp.ndarray:
    w = _lsolve(F.L, b)
    return jnp.sum(w * w)


def gaussian_logpdf(F: Factor, r: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """log N(r; 0, K) over the masked rows; values of r in padded rows are ignored."""
    mask = jnp.asarray(mask, dtype=bool)
    r0 = jnp.where(mask, r, 0.0)
    n = jnp.sum(mask)
    return -0.5 * quad(F, r0) - 0.5 * logdet(F) - 0.5 * n * jnp.log(2.0 * jnp.pi)


def append(F: Factor, K_cross: jnp.ndarray, K_new: jnp.ndarray, mask_new: jnp.ndarray) -> Factor:
    """Block Cholesky update for q appended rows.

    K_cross must be zero on the padded rows of the existing factor (the factor does not store its mask).
    Columns/rows of padded new rows are dropped here. The jitter already in F is added to the real new
    diagonal entries so the result matches factor() of the enlarged matrix; if the Schur complement is
    not positive definite the ladder adds more, and the returned jitter is the total.
    """
    mask_new = jnp.asarray(mask_new, dtype=bool)
    C = jnp.where(mask_new[None, :], jnp.asarray(K_cross, dtype=float), 0.0)
    V = _lsolve(F.L, C)  # (n, q)
    Kn = _mask_matrix(jnp.asarray(K_new, dtype=float), mask_new)
    S = Kn - V.T @ V + F.jitter * jnp.diag(jnp.where(mask_new, 1.0, 0.0))
    scale = jnp.sum(jnp.where(mask_new, jnp.diag(K_new), 0.0)) / jnp.maximum(jnp.sum(mask_new), 1)
    Ln, extra = _cholesky_ladder(S, mask_new, scale)
    n, q = F.L.shape[0], Ln.shape[0]
    top = jnp.concatenate([F.L, jnp.zeros((n, q))], axis=1)
    bottom = jnp.concatenate([V.T, Ln], axis=1)
    return Factor(jnp.concatenate([top, bottom], axis=0), F.jitter + extra)


def conditional(F: Factor, K_cross: jnp.ndarray, K_new_diag: jnp.ndarray, r: jnp.ndarray):
    """GP conditional mean and variance at new points (K_cross and r zero on padded rows)."""
    V = _lsolve(F.L, K_cross)
    mean = V.T @ _lsolve(F.L, r)
    var = K_new_diag - jnp.sum(V * V, axis=0)
    return mean, var
