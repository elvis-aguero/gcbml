"""Surrogate-data slice sampling (Murray & Adams 2010, arXiv 1006.0868) against a grid posterior.

Model: f ~ N(0, sigma^2 C_ell), y = f + N(0, s^2 I). Integrating f out gives y ~ N(0, sigma^2 C_ell + s^2 I),
so the posterior of theta = (log sigma, log ell) is computed on a 2-D grid, an independent reference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import gcbml  # noqa: F401  (float64)
from gcbml.mcmc.elliptical import ess_step
from gcbml.mcmc.surrogate import surrogate_slice_step

N_PTS = 20
JITTER = 1e-8
PRIOR_MEAN = np.array([0.0, np.log(0.3)])
PRIOR_SD = np.array([1.0, 1.0])


def cov_np(theta, x):
    sig, ell = np.exp(theta[0]), np.exp(theta[1])
    d = x[:, None] - x[None, :]
    return sig**2 * np.exp(-0.5 * (d / ell) ** 2) + JITTER * np.eye(len(x))


def make_data(s, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, N_PTS))
    f = np.linalg.cholesky(cov_np(np.array([0.0, np.log(0.3)]), x)) @ rng.standard_normal(N_PTS)
    return x, f + s * rng.standard_normal(N_PTS)


def grid_posterior(x, y, s, n=140):
    a = np.linspace(-2.5, 2.5, n)
    b = np.linspace(-4.0, 1.5, n)
    logp = np.empty((n, n))
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            A = cov_np(np.array([ai, bj]), x) + s**2 * np.eye(len(x))
            L = np.linalg.cholesky(A)
            z = np.linalg.solve(L, y)
            logp[i, j] = -0.5 * z @ z - np.log(np.diag(L)).sum()
    logp += (
        -0.5 * ((a[:, None] - PRIOR_MEAN[0]) / PRIOR_SD[0]) ** 2
        - 0.5 * ((b[None, :] - PRIOR_MEAN[1]) / PRIOR_SD[1]) ** 2
    )
    p = np.exp(logp - logp.max())
    p /= p.sum()
    pa, pb = p.sum(axis=1), p.sum(axis=0)
    mean = np.array([(pa * a).sum(), (pb * b).sum()])
    var = np.array([(pa * (a - mean[0]) ** 2).sum(), (pb * (b - mean[1]) ** 2).sum()])
    return mean, var


def make_model(x, y, s):
    xj, yj = jnp.asarray(x), jnp.asarray(y)

    def prior_cov(theta):
        d = xj[:, None] - xj[None, :]
        return jnp.exp(2 * theta[0]) * jnp.exp(-0.5 * (d / jnp.exp(theta[1])) ** 2) + JITTER * jnp.eye(N_PTS)

    loglik = lambda f: -0.5 * jnp.sum((yj - f) ** 2) / s**2  # noqa: E731
    log_prior = lambda th: -0.5 * jnp.sum(((th - PRIOR_MEAN) / PRIOR_SD) ** 2)  # noqa: E731
    surr = lambda th: s**2 * jnp.ones(N_PTS)  # noqa: E731  (site posterior of a Gaussian likelihood)
    return prior_cov, loglik, log_prior, surr


def run_chain(x, y, s, n_iter, key, n_ess=5, width=1.0):
    prior_cov, loglik, log_prior, surr = make_model(x, y, s)
    w = jnp.full(2, width)

    def body(state, k):
        theta, f = state
        k1, k2 = jax.random.split(k)
        theta, f, _ = surrogate_slice_step(k1, theta, f, prior_cov, loglik, log_prior, surr, w)
        chol = jnp.linalg.cholesky(prior_cov(theta))
        for kk in jax.random.split(k2, n_ess):
            f, _ = ess_step(kk, f, chol, loglik)
        return (theta, f), theta

    _, thetas = jax.lax.scan(body, (jnp.asarray(PRIOR_MEAN), jnp.asarray(y)), jax.random.split(key, n_iter))
    return thetas


def grid_sample(x, y, s, n_draws, seed, n=140):
    """Exact-ish draws of (theta, f) from the joint posterior: theta from the grid, f | theta, y Gaussian."""
    a = np.linspace(-2.5, 2.5, n)
    b = np.linspace(-4.0, 1.5, n)
    logp = np.empty((n, n))
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            A = cov_np(np.array([ai, bj]), x) + s**2 * np.eye(len(x))
            L = np.linalg.cholesky(A)
            z = np.linalg.solve(L, y)
            logp[i, j] = -0.5 * z @ z - np.log(np.diag(L)).sum()
    logp += -0.5 * (
        ((a[:, None] - PRIOR_MEAN[0]) / PRIOR_SD[0]) ** 2 + ((b[None, :] - PRIOR_MEAN[1]) / PRIOR_SD[1]) ** 2
    )
    p = np.exp(logp - logp.max())
    p /= p.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(n * n, size=n_draws, p=p.ravel())
    th = np.stack(
        [
            a[idx // n] + (rng.random(n_draws) - 0.5) * (a[1] - a[0]),
            b[idx % n] + (rng.random(n_draws) - 0.5) * (b[1] - b[0]),
        ],
        axis=1,
    )
    f = np.empty((n_draws, len(x)))
    for k in range(n_draws):
        f[k] = f_conditional(th[k], x, y, s, rng)[2]
    return a, b, p, th, f


def f_conditional(theta, x, y, s, rng=None):
    """Mean, Cholesky factor of the covariance, and (if rng) a draw of f | theta, y."""
    Sg = cov_np(theta, x)
    A = Sg + s**2 * np.eye(len(x))
    mu = Sg @ np.linalg.solve(A, y)
    C = Sg - Sg @ np.linalg.solve(A, Sg)
    C = 0.5 * (C + C.T) + 1e-12 * np.eye(len(x))
    Lc = np.linalg.cholesky(C)
    return mu, Lc, (None if rng is None else mu + Lc @ rng.standard_normal(len(x)))


@pytest.mark.parametrize("s,seed", [(0.3, 1), (0.02, 2)])
def test_one_step_leaves_joint_posterior_invariant(s, seed):
    """Start from exact posterior draws of (theta, f); one step must keep theta ~ grid marginal and f | theta."""
    n_draws = 1500
    x, y = make_data(s, seed=seed)
    a, b, p, th0, f0 = grid_sample(x, y, s, n_draws, seed=seed)
    prior_cov, loglik, log_prior, surr = make_model(x, y, s)
    w = jnp.ones(2)
    keys = jax.random.split(jax.random.key(100 + seed), n_draws)
    step = jax.vmap(lambda k, t, f: surrogate_slice_step(k, t, f, prior_cov, loglik, log_prior, surr, w))
    th1, f1, _ = jax.jit(step)(keys, jnp.asarray(th0), jnp.asarray(f0))
    th1, f1 = np.asarray(th1), np.asarray(f1)
    assert np.all(th1 != th0)
    for c, (g, marg) in enumerate([(a, p.sum(axis=1)), (b, p.sum(axis=0))]):
        cdf = lambda v, g=g, marg=marg: np.interp(v, g + (g[1] - g[0]) / 2, np.cumsum(marg))  # noqa: E731
        assert stats.kstest(th1[:, c], cdf).pvalue > 0.01
    # given theta', f' must be N(mu, C): standardise the first coordinate and the whole residual norm
    z0, q = [], []
    for k in range(n_draws):
        mu, Lc, _ = f_conditional(th1[k], x, y, s)
        r = np.linalg.solve(Lc, f1[k] - mu)
        z0.append(r[0])
        q.append(r @ r)
    assert stats.kstest(np.array(z0), "norm").pvalue > 0.01
    assert stats.kstest(np.array(q), stats.chi2(N_PTS).cdf).pvalue > 0.01


@pytest.mark.slow
@pytest.mark.parametrize("s,seed", [(0.3, 3), (0.02, 4)])
def test_hyperparameter_posterior_matches_grid(s, seed):
    x, y = make_data(s, seed=seed)
    mean, var = grid_posterior(x, y, s)
    n_chains, n_iter, burn = 8, 6000, 300
    keys = jax.random.split(jax.random.key(20 + seed), n_chains)
    th = np.asarray(jax.jit(jax.vmap(lambda k: run_chain(x, y, s, n_iter, k)))(keys))[
        :, burn:
    ]  # (chains, n, 2)
    for est, truth in [(th.mean(axis=1), mean), (((th - mean) ** 2).mean(axis=1), var)]:
        se = est.std(axis=0, ddof=1) / np.sqrt(n_chains)  # between-chain MC standard error
        assert np.all(np.abs(est.mean(axis=0) - truth) < 4 * se), (est.mean(axis=0), truth, se)


def test_n_evals_counts_loglik_calls_and_jit_works():
    x, y = make_data(0.3)
    calls = []
    prior_cov, loglik0, log_prior, surr = make_model(x, y, 0.3)

    def loglik(f):
        jax.debug.callback(lambda: calls.append(1))
        return loglik0(f)

    step = jax.jit(
        lambda k, th, f: surrogate_slice_step(k, th, f, prior_cov, loglik, log_prior, surr, jnp.ones(2))
    )
    th, f, ne = step(jax.random.key(0), jnp.asarray(PRIOR_MEAN), jnp.asarray(y))
    jax.block_until_ready(f)
    jax.effects_barrier()
    assert int(ne) == len(calls)
    assert th.shape == (2,) and f.shape == (N_PTS,)
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(th))
