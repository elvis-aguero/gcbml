"""Dense linear algebra behind a solver interface (exact now; a sparse solver can be added later).

All arrays are padded to a bucket size (gcbml._config.bucket). A boolean ``mask`` marks real rows.
Padded rows/columns are decoupled: the masked covariance is

    K_m = where(mask_i & mask_j, K, 0) + diag(where(mask, 0, 1))

so the padded block is the identity, contributes 0 to log det, and right-hand sides must be zero
on padded rows. Every function is jit- and vmap-compatible (vmap over posterior draws/fantasies).

TODO(W1-A): implement.

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

import jax.numpy as jnp


class Factor(NamedTuple):
    L: jnp.ndarray  # (n, n) lower Cholesky factor of the masked, jittered matrix
    jitter: jnp.ndarray  # scalar actually added to the diagonal


def factor(K: jnp.ndarray, mask: jnp.ndarray) -> Factor:
    raise NotImplementedError("W1-A")


def solve(F: Factor, b: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError("W1-A")


def logdet(F: Factor) -> jnp.ndarray:
    raise NotImplementedError("W1-A")


def quad(F: Factor, b: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError("W1-A")


def gaussian_logpdf(F: Factor, r: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError("W1-A")


def append(F: Factor, K_cross: jnp.ndarray, K_new: jnp.ndarray, mask_new: jnp.ndarray) -> Factor:
    raise NotImplementedError("W1-A")


def conditional(F: Factor, K_cross: jnp.ndarray, K_new_diag: jnp.ndarray, r: jnp.ndarray):
    raise NotImplementedError("W1-A")
