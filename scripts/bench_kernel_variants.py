"""Micro-benchmark of ard_matern variants on one core (XLA CPU), n = 240, d = 2."""

import time

import jax
import jax.numpy as jnp
import numpy as np

import gcbml  # noqa: F401
from gcbml.kernels import ard_matern, matern

n, d = 240, 2
X = jnp.asarray(np.random.default_rng(0).random((n, d)))
ell = jnp.asarray([0.3, 0.5])


def v_prescale(X1, X2, ell, nu):
    A, B = X1 / ell, X2 / ell
    diff = A[:, None, :] - B[None, :, :]
    r2 = jnp.maximum(jnp.sum(diff**2, axis=-1), 0.0)
    return matern(jnp.sqrt(r2), nu)


def v_sum_dims(X1, X2, ell, nu):
    A, B = X1 / ell, X2 / ell
    r2 = 0.0
    for k in range(A.shape[1]):
        r2 = r2 + (A[:, k][:, None] - B[:, k][None, :]) ** 2
    return matern(jnp.sqrt(jnp.maximum(r2, 0.0)), nu)


def v_m52(X1, X2, ell, nu):
    A, B = X1 / ell, X2 / ell
    r2 = 0.0
    for k in range(A.shape[1]):
        r2 = r2 + (A[:, k][:, None] - B[:, k][None, :]) ** 2
    a = jnp.sqrt(5.0 * jnp.maximum(r2, 0.0))
    return (1.0 + a + 5.0 * r2 / 3.0) * jnp.exp(-a)


def v_sym_avg(X1, X2, ell, nu):
    K = v_m52(X1, X2, ell, nu)
    return 0.5 * (K + K.T)


def v_sym_r2(X1, X2, ell, nu):
    A, B = X1 / ell, X2 / ell
    r2 = 0.0
    for k in range(A.shape[1]):
        r2 = r2 + (A[:, k][:, None] - B[:, k][None, :]) ** 2
    r2 = jnp.maximum(r2, 0.0)
    r2 = 0.5 * (r2 + r2.T)
    a = jnp.sqrt(5.0 * r2)
    return (1.0 + a + 5.0 * r2 / 3.0) * jnp.exp(-a)


def v_tril(X1, X2, ell, nu):
    K = v_m52(X1, X2, ell, nu)
    return jnp.tril(K) + jnp.tril(K, -1).T


ref = ard_matern(X, X, ell, 2.5)
for name, f in [
    ("current", ard_matern),
    ("prescale", v_prescale),
    ("sum_dims", v_sum_dims),
    ("m52 fused", v_m52),
    ("avg", v_sym_avg),
    ("sym r2", v_sym_r2),
    ("tril", v_tril),
]:
    g = jax.jit(lambda e, f=f: f(X, X, e, 2.5))
    jax.block_until_ready(g(ell))
    t = time.perf_counter()
    for _ in range(200):
        jax.block_until_ready(g(ell))
    out = g(ell)
    print(
        f"{name:10s} {(time.perf_counter() - t) / 200 * 1e3:6.3f} ms  max|diff| {float(jnp.max(jnp.abs(out - ref))):.2e} symmetric {bool(jnp.all(out == out.T))}"
    )
