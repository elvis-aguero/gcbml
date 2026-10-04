import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import linalg as sla
from scipy import stats

import gcbml  # noqa: F401
from gcbml import linalg


def spd(n, seed):
    r = np.random.default_rng(seed)
    A = r.normal(size=(n, n))
    return A @ A.T / n + 0.5 * np.eye(n)


def rel(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(b))


@pytest.mark.parametrize("n", [50, 200])
def test_factor_solve_logdet_quad_logpdf_match_dense(n):
    K = spd(n, n)
    r = np.random.default_rng(1)
    b, B = r.normal(size=n), r.normal(size=(n, 3))
    mask = jnp.ones(n, dtype=bool)
    F = linalg.factor(jnp.asarray(K), mask)
    assert float(F.jitter) == 0.0
    assert rel(np.asarray(F.L) @ np.asarray(F.L).T, K) < 1e-12
    assert rel(linalg.solve(F, jnp.asarray(b)), np.linalg.solve(K, b)) < 1e-10
    assert rel(linalg.solve(F, jnp.asarray(B)), np.linalg.solve(K, B)) < 1e-10
    assert abs(float(linalg.logdet(F)) - np.linalg.slogdet(K)[1]) < 1e-10 * abs(np.linalg.slogdet(K)[1])
    assert abs(float(linalg.quad(F, jnp.asarray(b))) - b @ np.linalg.solve(K, b)) < 1e-10 * (
        b @ np.linalg.solve(K, b)
    )
    ref = stats.multivariate_normal(np.zeros(n), K).logpdf(b)
    got = float(linalg.gaussian_logpdf(F, jnp.asarray(b), mask))
    assert abs(got - ref) < 1e-10 * abs(ref)


def _padded(n, N, seed):
    r = np.random.default_rng(seed)
    K = spd(n, seed)
    mask = np.zeros(N, dtype=bool)
    idx = np.sort(r.choice(N, n, replace=False))
    mask[idx] = True
    Kp = 100.0 * r.normal(size=(N, N))  # garbage everywhere, including cross terms with real rows
    Kp[np.ix_(idx, idx)] = K
    return K, Kp, mask, idx, r


def test_padded_masked_equals_unpadded_with_garbage_in_padding():
    n, N = 50, 64
    K, Kp, mask, idx, r = _padded(n, N, 3)
    y = r.normal(size=n)
    yp = 1e3 * r.normal(size=N)  # garbage residual in padded rows
    yp[idx] = y
    bz = np.zeros(N)
    bz[idx] = y  # solve/quad need zero rhs on padded rows
    F0 = linalg.factor(jnp.asarray(K), jnp.ones(n, dtype=bool))
    Fp = linalg.factor(jnp.asarray(Kp), jnp.asarray(mask))
    assert abs(float(linalg.logdet(Fp)) - float(linalg.logdet(F0))) < 1e-10
    assert abs(float(linalg.quad(Fp, jnp.asarray(bz))) - float(linalg.quad(F0, jnp.asarray(y)))) < 1e-10
    sp = np.asarray(linalg.solve(Fp, jnp.asarray(bz)))
    np.testing.assert_allclose(sp[idx], np.linalg.solve(K, y), rtol=1e-10)
    assert np.all(sp[~mask] == 0.0)
    lp = float(linalg.gaussian_logpdf(Fp, jnp.asarray(yp), jnp.asarray(mask)))
    l0 = float(linalg.gaussian_logpdf(F0, jnp.asarray(y), jnp.ones(n, dtype=bool)))
    assert abs(lp - l0) < 1e-10 * abs(l0)


def test_padding_with_nan_garbage_is_ignored():
    K, Kp, mask, idx, r = _padded(30, 40, 4)
    Kp[~mask, :] = np.nan
    Kp[:, ~mask] = np.nan
    y = np.full(40, np.nan)
    y[idx] = r.normal(size=30)
    Fp = linalg.factor(jnp.asarray(Kp), jnp.asarray(mask))
    got = float(linalg.gaussian_logpdf(Fp, jnp.asarray(y), jnp.asarray(mask)))
    assert abs(got - stats.multivariate_normal(np.zeros(30), K).logpdf(y[idx])) < 1e-10


def test_jitter_ladder_rank_deficient_psd_gets_finite_factor():
    n = 10
    K = np.ones((n, n))  # rank 1: Cholesky pivot is exactly 0 at jitter 0
    F = linalg.factor(jnp.asarray(K), jnp.ones(n, dtype=bool))
    assert np.all(np.isfinite(np.asarray(F.L)))
    assert float(F.jitter) > 0.0
    L = np.asarray(F.L)
    np.testing.assert_allclose(L @ L.T, K + float(F.jitter) * np.eye(n), atol=1e-12)
    assert float(F.jitter) <= 1e-4 * 1.0 + 1e-18


def test_jitter_scales_with_mean_diagonal():
    n = 8
    F1 = linalg.factor(jnp.asarray(np.ones((n, n))), jnp.ones(n, dtype=bool))
    F2 = linalg.factor(jnp.asarray(1e6 * np.ones((n, n))), jnp.ones(n, dtype=bool))
    np.testing.assert_allclose(float(F2.jitter), 1e6 * float(F1.jitter), rtol=1e-12)


def test_negative_eigenvalue_gives_nonfinite_factor():
    r = np.random.default_rng(5)
    Q, _ = np.linalg.qr(r.normal(size=(12, 12)))
    w = np.linspace(1, 3, 12)
    w[4] = -1.0
    K = Q @ np.diag(w) @ Q.T
    F = linalg.factor(jnp.asarray((K + K.T) / 2), jnp.ones(12, dtype=bool))
    assert not np.all(np.isfinite(np.asarray(F.L)))
    assert not np.isfinite(float(linalg.logdet(F)))


def test_append_equals_factor_of_enlarged_matrix():
    n, q = 40, 5
    K = spd(n + q, 6)
    F = linalg.factor(jnp.asarray(K[:n, :n]), jnp.ones(n, dtype=bool))
    Fa = linalg.append(F, jnp.asarray(K[:n, n:]), jnp.asarray(K[n:, n:]), jnp.ones(q, dtype=bool))
    Ff = linalg.factor(jnp.asarray(K), jnp.ones(n + q, dtype=bool))
    assert Fa.L.shape == (n + q, n + q)
    assert rel(Fa.L, Ff.L) < 1e-12
    assert float(Fa.jitter) == 0.0
    assert abs(float(linalg.logdet(Fa)) - float(linalg.logdet(Ff))) < 1e-10


def test_append_with_padded_old_rows_and_masked_new_rows():
    n, N, q = 25, 32, 4
    K, Kp, mask, idx, r = _padded(n, N, 7)
    Kall = spd(n + 3, 8)  # 3 real new rows, 1 padded new row
    Kp[np.ix_(idx, idx)] = Kall[:n, :n]
    cross = np.zeros((N, q))
    cross[idx, :3] = Kall[:n, n:]
    cross[idx, 3] = 55.0  # garbage cross column of the padded new row
    Knew = 77.0 * r.normal(size=(q, q))
    Knew[:3, :3] = Kall[n:, n:]
    mnew = np.array([True, True, True, False])
    F = linalg.factor(jnp.asarray(Kp), jnp.asarray(mask))
    Fa = linalg.append(F, jnp.asarray(cross), jnp.asarray(Knew), jnp.asarray(mnew))
    big = np.zeros((N + q, N + q))
    big[:N, :N] = Kp
    big[:N, N:] = cross
    big[N:, :N] = cross.T
    big[N:, N:] = Knew
    ref = linalg.factor(jnp.asarray(big), jnp.asarray(np.concatenate([mask, mnew])))
    assert rel(Fa.L, ref.L) < 1e-11
    assert (
        abs(
            float(linalg.logdet(Fa))
            - float(linalg.logdet(linalg.factor(jnp.asarray(Kall), jnp.ones(n + 3, bool))))
        )
        < 1e-10
    )


def test_conditional_equals_dense_formula():
    n, m = 60, 7
    Kfull = spd(n + m, 9)
    K, Kc, Kss = Kfull[:n, :n], Kfull[:n, n:], Kfull[n:, n:]
    r = np.random.default_rng(10).normal(size=n)
    F = linalg.factor(jnp.asarray(K), jnp.ones(n, dtype=bool))
    mean, var = linalg.conditional(F, jnp.asarray(Kc), jnp.asarray(np.diag(Kss)), jnp.asarray(r))
    S = sla.solve(K, np.column_stack([r, Kc]), assume_a="pos")
    np.testing.assert_allclose(np.asarray(mean), Kc.T @ S[:, 0], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(var), np.diag(Kss) - np.einsum("ij,ij->j", Kc, S[:, 1:]), rtol=1e-9)
    assert np.all(np.asarray(var) > 0)


def test_jit_and_vmap_over_batch_of_matrices():
    n = 30
    Ks = np.stack([spd(n, s) for s in range(4)])
    Ks[2] = np.ones((n, n))  # one rank-deficient member: its ladder runs longer than the others
    bs = np.random.default_rng(11).normal(size=(4, n))
    mask = jnp.ones(n, dtype=bool)
    Fb = jax.jit(jax.vmap(linalg.factor, in_axes=(0, None)))(jnp.asarray(Ks), mask)
    assert Fb.L.shape == (4, n, n)
    assert float(Fb.jitter[0]) == 0.0
    assert float(Fb.jitter[2]) > 0.0
    q = jax.jit(jax.vmap(linalg.quad))(Fb, jnp.asarray(bs))
    ld = jax.jit(jax.vmap(linalg.logdet))(Fb)
    for i in (0, 1, 3):
        assert abs(float(q[i]) - bs[i] @ np.linalg.solve(Ks[i], bs[i])) < 1e-10 * float(q[i])
        assert abs(float(ld[i]) - np.linalg.slogdet(Ks[i])[1]) < 1e-10 * abs(float(ld[i]))
    assert np.all(np.isfinite(np.asarray(Fb.L[2])))
    lp = jax.vmap(linalg.gaussian_logpdf, in_axes=(0, 0, None))(Fb, jnp.asarray(bs), mask)
    assert lp.shape == (4,)


def test_vmap_over_residual_draws_with_shared_factor():
    n = 20
    K = spd(n, 12)
    mask = jnp.ones(n, dtype=bool)
    F = linalg.factor(jnp.asarray(K), mask)
    R = np.random.default_rng(13).normal(size=(5, n))
    lp = jax.jit(jax.vmap(linalg.gaussian_logpdf, in_axes=(None, 0, None)))(F, jnp.asarray(R), mask)
    np.testing.assert_allclose(
        np.asarray(lp), stats.multivariate_normal(np.zeros(n), K).logpdf(R), rtol=1e-10
    )
