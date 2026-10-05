"""Covariance functions (spec Section 2.3; guide Sections 3 and 6).

All functions are pure jax.numpy, jit- and vmap-compatible, and return dense matrices.
Inputs are unit-scaled x in [0,1]^d and scaled resolutions hbar = h / h_c > 0 (hbar = 0 is allowed
and must give zero variance for the error kernels).

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

import math
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
    """Matern correlation of a scaled distance r >= 0 for nu in {0.5, 1.5, 2.5}."""
    r = jnp.asarray(r, dtype=float)
    if nu == 0.5:
        return jnp.exp(-r)
    if nu == 1.5:
        a = math.sqrt(3.0) * r
        return (1.0 + a) * jnp.exp(-a)
    if nu == 2.5:
        a = math.sqrt(5.0) * r
        return (1.0 + a + 5.0 * r**2 / 3.0) * jnp.exp(-a)
    raise ValueError(f"nu must be 0.5, 1.5 or 2.5, got {nu!r}")


def ard_matern(X1: jnp.ndarray, X2: jnp.ndarray, ell: jnp.ndarray, nu: float) -> jnp.ndarray:
    """(n1, n2) ARD Matern correlation matrix; exactly symmetric when X1 is X2."""
    # Scale the coordinates first and add the d squared differences one dimension at a time: no (n1, n2, d)
    # intermediate and no per-pair division (3x faster on one CPU core,
    # n = 240, d = 2).
    A, B = X1 / ell, X2 / ell
    r2 = 0.0
    for k in range(A.shape[1]):
        r2 = r2 + (A[:, k][:, None] - B[:, k][None, :]) ** 2
    r2 = jnp.maximum(r2, 0.0)
    if X1 is X2:  # fused multiply-adds can break exact symmetry of r2: average it with its transpose
        r2 = 0.5 * (r2 + r2.T)
    pos = r2 > 0  # sqrt has an infinite slope at 0: keep the gradient finite (0) on the diagonal
    return matern(jnp.where(pos, jnp.sqrt(jnp.where(pos, r2, 1.0)), 0.0), nu)


def _hpow(h, p):
    """h**p with exactly 0 (and a finite gradient) at h = 0."""
    pos = h > 0
    return jnp.where(pos, jnp.where(pos, h, 1.0) ** p, 0.0)


def twy2(h1, h2, p1, p2, ell_h, nu: float = 1.5) -> jnp.ndarray:
    """k(i, j) = h1_i^{p1_i} h2_j^{p2_j} matern(|h1_i - h2_j| / ell_h, nu), exactly 0 where hbar = 0."""
    h1, h2 = jnp.asarray(h1, dtype=float), jnp.asarray(h2, dtype=float)
    b1, b2 = _hpow(h1, p1), _hpow(h2, p2)
    d = jnp.abs(h1[:, None] - h2[None, :]) / ell_h
    return b1[:, None] * b2[None, :] * matern(d, nu)


def lifted_brownian(h1, h2, p, gamma, a=1.0) -> jnp.ndarray:
    """0.5 a (h1^{2p} + h2^{2p} - |h1^{p/gamma} - h2^{p/gamma}|^{2 gamma}), shared order p."""
    h1, h2 = jnp.asarray(h1, dtype=float), jnp.asarray(h2, dtype=float)
    d = jnp.abs(_hpow(h1, p / gamma)[:, None] - _hpow(h2, p / gamma)[None, :])
    pos = d > 0
    dpow = jnp.where(pos, jnp.where(pos, d, 1.0) ** (2.0 * gamma), 0.0)
    k = 0.5 * a * (_hpow(h1, 2.0 * p)[:, None] + _hpow(h2, 2.0 * p)[None, :] - dpow)
    # exact zero in any row/column with hbar = 0 (the formula cancels only to round-off)
    return jnp.where((h1 > 0)[:, None] & (h2 > 0)[None, :], k, 0.0)


def delta_cov(
    X1, H1, X2, H2, P1, P2, comps: DeltaParams, h_kernel: str, nu_x: float = 2.5, nu_h: float = 1.5
) -> jnp.ndarray:
    """Additive multi-component error covariance sum_j sigma_j^2 k_x,j k_h,j (spec 2.3).

    For ``h_kernel == "lb"`` the order is shared, so it is read from row 0 of column j of P1 (and P2 is
    ignored): put real rows first and keep the column constant.
    """
    if h_kernel not in ("twy2", "lb"):
        raise ValueError(f"h_kernel must be 'twy2' or 'lb', got {h_kernel!r}")
    K = 0.0
    for j in range(comps.sigma.shape[0]):
        kx = ard_matern(X1, X2, comps.ell_x[j], nu_x)
        if h_kernel == "twy2":
            kh = twy2(H1[:, j], H2[:, j], P1[:, j], P2[:, j], comps.ell_h[j], nu_h)
        else:
            kh = lifted_brownian(H1[:, j], H2[:, j], P1[0, j], comps.gamma[j])
        K = K + comps.sigma[j] ** 2 * kx * kh
    return K
