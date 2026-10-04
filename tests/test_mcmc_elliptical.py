"""Elliptical slice sampling (Murray, Adams & MacKay 2010, arXiv 1001.0175, Fig. 2)."""

import jax
import jax.numpy as jnp
import numpy as np
from _mcmc_helpers import assert_mean_close, batch_se
from scipy.stats import norm

import gcbml  # noqa: F401  (float64)
from gcbml.mcmc.elliptical import ess_step, ess_step_info


def sq_exp(x, ell=1.0):
    d = x[:, None] - x[None, :]
    return np.exp(-0.5 * (d / ell) ** 2)


def run_chain(chol, loglik, n, f0=None, seed=0, burn=200):
    f0 = jnp.zeros(chol.shape[0]) if f0 is None else f0

    def body(f, k):
        f, n_evals = ess_step(k, f, chol, loglik)
        return f, (f, n_evals)

    keys = jax.random.split(jax.random.key(seed), n + burn)
    _, (fs, ne) = jax.lax.scan(body, f0, keys)
    return np.asarray(fs[burn:]), np.asarray(ne[burn:])


def test_gaussian_likelihood_matches_conjugate_posterior():
    n, s = 5, 0.5
    x = np.linspace(0.0, 2.0, n)
    K = sq_exp(x, 1.0) + 1e-10 * np.eye(n)
    y = np.array([0.3, -0.4, 0.9, 0.7, -0.2])
    # reference: textbook conjugate update (Rasmussen & Williams eqs 2.19, 2.23 in the f-space form)
    A = K + s**2 * np.eye(n)
    mean = K @ np.linalg.solve(A, y)
    cov = K - K @ np.linalg.solve(A, K)
    chol = jnp.asarray(np.linalg.cholesky(K))
    yj = jnp.asarray(y)
    loglik = lambda f: -0.5 * jnp.sum((yj - f) ** 2) / s**2  # noqa: E731
    fs, _ = run_chain(chol, loglik, 20000, seed=1)
    assert_mean_close(fs, mean)
    centred = fs - mean
    prods = (centred[:, :, None] * centred[:, None, :]).reshape(len(fs), -1)
    assert_mean_close(prods, cov.reshape(-1))


def test_probit_likelihood_matches_quadrature():
    K = np.array([[1.0, 0.6], [0.6, 1.0]])
    y = np.array([1.0, -1.0])
    g = np.linspace(-7.0, 7.0, 701)
    f1, f2 = np.meshgrid(g, g, indexing="ij")
    Kinv = np.linalg.inv(K)  # reference-side only
    quad = Kinv[0, 0] * f1**2 + 2 * Kinv[0, 1] * f1 * f2 + Kinv[1, 1] * f2**2
    post = np.exp(-0.5 * quad) * norm.cdf(y[0] * f1) * norm.cdf(y[1] * f2)
    post /= post.sum()
    mean = np.array([(post * f1).sum(), (post * f2).sum()])
    second = np.array([(post * f1**2).sum(), (post * f1 * f2).sum()])
    chol = jnp.asarray(np.linalg.cholesky(K))
    yj = jnp.asarray(y)
    loglik = lambda f: jnp.sum(jax.scipy.stats.norm.logcdf(yj * f))  # noqa: E731
    fs, _ = run_chain(chol, loglik, 20000, seed=2)
    assert_mean_close(fs, mean)
    assert_mean_close(np.stack([fs[:, 0] ** 2, fs[:, 0] * fs[:, 1]], axis=1), second)


def test_prior_only_gives_prior_moments_and_n_evals_is_one_plus_proposals():
    K = np.array([[2.0, 1.0], [1.0, 1.5]])
    chol = jnp.asarray(np.linalg.cholesky(K))
    fs, ne = run_chain(chol, lambda f: 0.0 * f[0], 15000, seed=3)
    # a constant likelihood always accepts the first proposal: 1 (current state) + 1 (proposal)
    assert np.all(ne == 2)
    prods = (fs[:, :, None] * fs[:, None, :]).reshape(len(fs), -1)
    assert_mean_close(prods, K.reshape(-1))
    assert_mean_close(fs, np.zeros(2))
    assert np.all(batch_se(fs) > 0)


def test_hitting_the_bound_returns_f_unchanged_and_flags_it():
    f = jnp.array([0.3, -0.2])
    chol = jnp.eye(2)
    f_new, n_evals, hit = ess_step_info(jax.random.key(0), f, chol, lambda f: -jnp.inf)
    np.testing.assert_array_equal(f_new, f)
    assert bool(hit)
    assert int(n_evals) == 1 + 200
    _, _, hit = ess_step_info(jax.random.key(0), f, chol, lambda f: 0.0 * f[0])
    assert not bool(hit)


def test_jit_and_vmap_work():
    chol = jnp.eye(3)
    loglik = lambda f: -0.5 * jnp.sum(f**2)  # noqa: E731
    keys = jax.random.split(jax.random.key(1), 8)
    f = jnp.ones((8, 3))
    out, ne = jax.jit(jax.vmap(lambda k, f: ess_step(k, f, chol, loglik)))(keys, f)
    assert out.shape == (8, 3) and ne.shape == (8,)
    assert np.all(np.isfinite(out))
