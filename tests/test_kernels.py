import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gcbml  # noqa: F401
from gcbml import kernels
from gcbml.kernels import DeltaParams


def min_eig_ratio(K):
    w = np.linalg.eigvalsh(np.asarray(K))
    return w[0] / w[-1]


@pytest.mark.parametrize("r", [0.0, 0.3, 1.0, 2.5])
def test_matern_closed_forms(r):
    s3, s5 = math.sqrt(3), math.sqrt(5)
    np.testing.assert_allclose(kernels.matern(jnp.asarray(r), 0.5), math.exp(-r), rtol=1e-14)
    np.testing.assert_allclose(
        kernels.matern(jnp.asarray(r), 1.5), (1 + s3 * r) * math.exp(-s3 * r), rtol=1e-14
    )
    np.testing.assert_allclose(
        kernels.matern(jnp.asarray(r), 2.5), (1 + s5 * r + 5 * r**2 / 3) * math.exp(-s5 * r), rtol=1e-14
    )


def test_matern_hand_values():
    # computed by hand: nu=1.5 at r=1 -> (1+1.7320508075688772) e^{-1.7320508075688772}
    np.testing.assert_allclose(kernels.matern(jnp.asarray(1.0), 1.5), 0.4833577245965, rtol=1e-10)
    np.testing.assert_allclose(kernels.matern(jnp.asarray(1.0), 0.5), 0.36787944117144233, rtol=1e-14)


@pytest.mark.parametrize("nu", [1.0, 2.0, 3.5, 0])
def test_matern_bad_nu_raises(nu):
    with pytest.raises(ValueError, match="nu"):
        kernels.matern(jnp.asarray(1.0), nu)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_ard_matern_symmetric_psd_and_matches_loop(nu):
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(50, 4))
    ell = np.array([0.3, 0.8, 1.5, 0.5])
    K = kernels.ard_matern(jnp.asarray(X), jnp.asarray(X), jnp.asarray(ell), nu)
    assert np.array_equal(np.asarray(K), np.asarray(K).T)
    assert min_eig_ratio(K) >= -1e-10
    np.testing.assert_array_equal(np.diag(np.asarray(K)), 1.0)
    ref = np.empty((50, 50))
    for i in range(50):
        for j in range(50):
            r = math.sqrt(sum(((X[i, k] - X[j, k]) / ell[k]) ** 2 for k in range(4)))
            ref[i, j] = float(kernels.matern(jnp.asarray(r), nu))
    np.testing.assert_allclose(np.asarray(K), ref, atol=1e-13)


def test_ard_matern_rectangular_shape():
    X1, X2 = np.random.default_rng(2).uniform(size=(5, 3)), np.random.default_rng(3).uniform(size=(7, 3))
    assert kernels.ard_matern(jnp.asarray(X1), jnp.asarray(X2), jnp.ones(3), 1.5).shape == (5, 7)


def test_twy2_psd_with_per_row_orders():
    rng = np.random.default_rng(4)
    h = rng.uniform(0.02, 1.0, size=40)
    p = rng.uniform(0.3, 3.0, size=40)
    K = kernels.twy2(jnp.asarray(h), jnp.asarray(h), jnp.asarray(p), jnp.asarray(p), 0.4, 1.5)
    assert min_eig_ratio(K) >= -1e-10


def test_twy2_zero_in_row_with_hbar_zero():
    h1 = jnp.array([0.0, 0.5, 1.0])
    h2 = jnp.array([0.3, 0.0, 0.7, 1.0])
    p1, p2 = jnp.array([1.0, 2.0, 0.5]), jnp.array([1.0, 1.5, 2.0, 0.7])
    K = np.asarray(kernels.twy2(h1, h2, p1, p2, 0.5, 1.5))
    assert np.all(K[0, :] == 0.0)
    assert np.all(K[:, 1] == 0.0)
    assert np.all(K[1:, [0, 2, 3]] != 0.0)


def test_twy2_constant_order_equals_formula():
    rng = np.random.default_rng(5)
    h = rng.uniform(0.05, 1.0, 9)
    p, ell, s3 = 1.7, 0.35, math.sqrt(3)
    K = np.asarray(kernels.twy2(jnp.asarray(h), jnp.asarray(h), jnp.full(9, p), jnp.full(9, p), ell, 1.5))
    ref = np.array(
        [
            [(a * b) ** p * (1 + s3 * abs(a - b) / ell) * math.exp(-s3 * abs(a - b) / ell) for b in h]
            for a in h
        ]
    )
    np.testing.assert_allclose(K, ref, rtol=1e-13)


def test_lb_gamma_half_equals_brownian_min_power():
    rng = np.random.default_rng(6)
    h1, h2 = rng.uniform(0.0, 1.0, 12), rng.uniform(0.0, 1.0, 9)
    for p in (0.5, 1.0, 2.3):
        K = kernels.lifted_brownian(jnp.asarray(h1), jnp.asarray(h2), p, 0.5)
        ref = np.minimum.outer(h1, h2) ** (2 * p)
        np.testing.assert_allclose(np.asarray(K), ref, atol=1e-12)


def test_lb_matches_paper_eq4_loop_with_scale():
    h = np.array([0.1, 0.4, 0.9])
    p, g, a = 1.3, 0.3, 2.5
    K = np.asarray(kernels.lifted_brownian(jnp.asarray(h), jnp.asarray(h), p, g, a))
    for i in range(3):
        for j in range(3):
            ref = (
                0.5
                * a
                * (h[i] ** (2 * p) + h[j] ** (2 * p) - abs(h[i] ** (p / g) - h[j] ** (p / g)) ** (2 * g))
            )
            assert abs(K[i, j] - ref) < 1e-13


@pytest.mark.parametrize("gamma", [0.1, 0.3, 0.7, 0.9])
def test_lb_psd(gamma):
    h = np.random.default_rng(7).uniform(0.02, 1.0, 40)
    K = kernels.lifted_brownian(jnp.asarray(h), jnp.asarray(h), 1.5, gamma)
    assert min_eig_ratio(K) >= -1e-10


def test_lb_variance_is_a_h_to_2p():
    h = jnp.array([0.0, 0.2, 0.7, 1.0])
    K = kernels.lifted_brownian(h, h, 1.8, 0.35, 3.0)
    np.testing.assert_allclose(np.diag(np.asarray(K)), 3.0 * np.asarray(h) ** 3.6, atol=1e-14)


def _comps(k, d, seed=8):
    r = np.random.default_rng(seed)
    return DeltaParams(
        sigma=jnp.asarray(r.uniform(0.5, 2.0, k)),
        ell_x=jnp.asarray(r.uniform(0.3, 1.5, (k, d))),
        ell_h=jnp.asarray(r.uniform(0.2, 0.8, k)),
        gamma=jnp.asarray(r.uniform(0.2, 0.8, k)),
    )


@pytest.mark.parametrize("h_kernel", ["twy2", "lb"])
def test_delta_cov_equals_explicit_sum(h_kernel):
    r = np.random.default_rng(9)
    n1, n2, d, k = 6, 5, 3, 2
    X1, X2 = r.uniform(size=(n1, d)), r.uniform(size=(n2, d))
    H1, H2 = r.uniform(0.05, 1, (n1, k)), r.uniform(0.05, 1, (n2, k))
    P1, P2 = np.tile([1.2, 0.8], (n1, 1)), np.tile([1.2, 0.8], (n2, 1))
    c = _comps(k, d)
    K = np.asarray(kernels.delta_cov(*map(jnp.asarray, (X1, H1, X2, H2, P1, P2)), c, h_kernel))
    ref = np.zeros((n1, n2))
    for j in range(k):
        for a in range(n1):
            for b in range(n2):
                rr = math.sqrt(sum(((X1[a, m] - X2[b, m]) / float(c.ell_x[j, m])) ** 2 for m in range(d)))
                s5 = math.sqrt(5)
                kx = (1 + s5 * rr + 5 * rr**2 / 3) * math.exp(-s5 * rr)
                h1, h2, p = H1[a, j], H2[b, j], P1[0, j]
                if h_kernel == "twy2":
                    q = abs(h1 - h2) / float(c.ell_h[j])
                    kh = h1**p * h2**p * (1 + math.sqrt(3) * q) * math.exp(-math.sqrt(3) * q)
                else:
                    g = float(c.gamma[j])
                    kh = 0.5 * (h1 ** (2 * p) + h2 ** (2 * p) - abs(h1 ** (p / g) - h2 ** (p / g)) ** (2 * g))
                ref[a, b] += float(c.sigma[j]) ** 2 * kx * kh
    np.testing.assert_allclose(K, ref, rtol=1e-12)


@pytest.mark.parametrize("h_kernel", ["twy2", "lb"])
def test_delta_cov_error_vanishes_only_when_every_component_vanishes(h_kernel):
    d, k = 2, 2
    X = jnp.array([[0.2, 0.3], [0.7, 0.1], [0.5, 0.5]])
    H = jnp.array([[0.0, 0.5], [0.0, 0.0], [0.4, 0.0]])
    P = jnp.ones((3, k))
    K = np.asarray(kernels.delta_cov(X, H, X, H, P, P, _comps(k, d), h_kernel))
    v = np.diag(K)
    assert v[0] > 0  # first component 0, second > 0
    assert v[2] > 0  # first > 0, second 0
    assert v[1] == 0.0  # every component 0
    assert np.all(K[1, :] == 0.0)
    assert min_eig_ratio(K) >= -1e-10


@pytest.mark.parametrize("h_kernel", ["twy2", "lb"])
def test_delta_cov_psd_on_random_points(h_kernel):
    r = np.random.default_rng(10)
    n, d, k = 30, 4, 2
    X, H = r.uniform(size=(n, d)), r.uniform(0.02, 1, (n, k))
    P = np.tile([1.0, 2.0], (n, 1))
    K = kernels.delta_cov(*map(jnp.asarray, (X, H, X, H, P, P)), _comps(k, d), h_kernel)
    assert min_eig_ratio(K) >= -1e-10


def test_kernels_jit_and_vmap_over_parameter_batch():
    r = np.random.default_rng(11)
    X, h = jnp.asarray(r.uniform(size=(8, 3))), jnp.asarray(r.uniform(0.1, 1, 8))
    p = jnp.full(8, 1.3)
    ells = jnp.asarray(r.uniform(0.3, 1.5, (4, 3)))
    got = jax.jit(jax.vmap(lambda e: kernels.ard_matern(X, X, e, 2.5)))(ells)
    for b in range(4):
        np.testing.assert_allclose(got[b], kernels.ard_matern(X, X, ells[b], 2.5), atol=1e-14)
    lh = jnp.array([0.2, 0.5, 0.9])
    got = jax.jit(jax.vmap(lambda e: kernels.twy2(h, h, p, p, e)))(lh)
    for b in range(3):
        np.testing.assert_allclose(got[b], kernels.twy2(h, h, p, p, lh[b]), atol=1e-14)
    gs = jnp.array([0.2, 0.5, 0.8])
    got = jax.jit(jax.vmap(lambda g: kernels.lifted_brownian(h, h, 1.3, g)))(gs)
    for b in range(3):
        np.testing.assert_allclose(got[b], kernels.lifted_brownian(h, h, 1.3, gs[b]), atol=1e-14)
    H = jnp.stack([h, h], axis=1)
    P = jnp.ones((8, 2))
    batch = DeltaParams(*[jnp.stack([x, 1.1 * x, 0.9 * x]) for x in _comps(2, 3)])
    for hk in ("twy2", "lb"):
        f = jax.jit(jax.vmap(lambda c, hk=hk: kernels.delta_cov(X, H, X, H, P, P, c, hk)))
        got = f(batch)
        assert got.shape == (3, 8, 8)
        one = kernels.delta_cov(X, H, X, H, P, P, DeltaParams(*[x[1] for x in batch]), hk)
        np.testing.assert_allclose(got[1], one, atol=1e-13)


def test_kernel_gradients_finite_at_hbar_zero():
    def f(ell_h):
        h = jnp.array([0.0, 0.5])
        return jnp.sum(kernels.twy2(h, h, jnp.ones(2), jnp.ones(2), ell_h))

    def g(gamma):
        h = jnp.array([0.0, 0.5])
        return jnp.sum(kernels.lifted_brownian(h, h, 1.0, gamma))

    assert np.isfinite(float(jax.grad(f)(0.3)))
    assert np.isfinite(float(jax.grad(g)(0.3)))
