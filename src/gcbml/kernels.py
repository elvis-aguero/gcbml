"""Covariance functions (spec Section 2.3; guide Sections 3 and 6).

All functions are pure jax.numpy, jit- and vmap-compatible, and return dense matrices.
Inputs are unit-scaled x in [0,1]^d and scaled resolutions hbar = h / h_c > 0 (hbar = 0 is allowed
and must give zero variance for the error kernels).

TODO(W1-A): implement every function below exactly as documented.

matern(r, nu)
    Matérn correlation of a scaled distance r >= 0, for nu in {0.5, 1.5, 2.5}:
      nu=0.5: exp(-r);  nu=1.5: (1 + sqrt3 r) exp(-sqrt3 r);  nu=2.5: (1 + sqrt5 r + 5 r^2/3) exp(-sqrt5 r).
    Raise ValueError for any other nu (at trace time).

ard_matern(X1, X2, ell, nu)
    (n1, n2) matrix of matern(r) with r_ij = sqrt(sum_k ((X1_ik - X2_jk) / ell_k)^2). Must be exactly
    symmetric when X1 is X2 (compute r from squared differences, clip at 0 before sqrt).

twy2(h1, h2, p1, p2, ell_h, nu)
    TWY2 kernel in h (Tuo, Wu & Yu 2014; Bect et al. 2103.14559 Prop. 3 with L = 2p), with an order
    that may differ per row (spec 2.3, "order that can vary with x"):
        k(i, j) = h1_i^{p1_i} * h2_j^{p2_j} * matern(|h1_i - h2_j| / ell_h, nu)
    h1: (n1,), h2: (n2,), p1: (n1,), p2: (n2,). With p1 = p2 = p constant this is (h h')^p c(h - h').
    Use where(h > 0, h**p, 0) so hbar = 0 gives exactly 0.

lifted_brownian(h1, h2, p, gamma, a=1.0)
    Boutelet & Sung 2503.23158 eq 4 with one fidelity component (m = 1) and l = 2p:
        k = 0.5 * a * ( h1^{2p} + h2^{2p} - | h1^{p/gamma} - h2^{p/gamma} |^{2 gamma} )
    gamma in (0, 1). With gamma = 0.5 it equals a * min(h1, h2)^{2p} (B&S Section 2.2). Variance a h^{2p}.
    One shared order p (no per-row order: spec 2.3).

delta_cov(X1, H1, X2, H2, P1, P2, comps, h_kernel, nu_x, nu_h)
    Additive multi-component error covariance (spec 2.3):
        K = sum_j sigma_j^2 * ard_matern(X1, X2, ell_x[j], nu_x) * k_h_j(H1[:, j], H2[:, j])
    H: (n, k) hbar; P: (n, k) per-row order (constant columns when the order does not vary).
    ``comps`` is a DeltaParams; ``h_kernel`` is "twy2" or "lb".
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class DeltaParams(NamedTuple):
    """Constrained parameters of the k error components. Shapes: sigma (k,), ell_x (k, d),
    ell_h (k,) (TWY2 only), gamma (k,) (LB only)."""

    sigma: jnp.ndarray
    ell_x: jnp.ndarray
    ell_h: jnp.ndarray
    gamma: jnp.ndarray


def matern(r: jnp.ndarray, nu: float) -> jnp.ndarray:
    raise NotImplementedError("W1-A")


def ard_matern(X1: jnp.ndarray, X2: jnp.ndarray, ell: jnp.ndarray, nu: float) -> jnp.ndarray:
    raise NotImplementedError("W1-A")


def twy2(h1, h2, p1, p2, ell_h, nu: float = 1.5) -> jnp.ndarray:
    raise NotImplementedError("W1-A")


def lifted_brownian(h1, h2, p, gamma, a=1.0) -> jnp.ndarray:
    raise NotImplementedError("W1-A")


def delta_cov(
    X1, H1, X2, H2, P1, P2, comps: DeltaParams, h_kernel: str, nu_x: float = 2.5, nu_h: float = 1.5
) -> jnp.ndarray:
    raise NotImplementedError("W1-A")
