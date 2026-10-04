"""Slice sampler (Neal 2003, arXiv physics/0009028): moments against closed forms and exact draws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _mcmc_helpers import assert_mean_close, batch_se
from scipy import stats

import gcbml  # noqa: F401  (float64)
from gcbml.mcmc.slice import slice_step


def run_chain(ld, x0, widths, n, seed=0, burn=200, max_steps_out=32):
    def body(x, k):
        x, n_evals = slice_step(k, x, ld, widths, max_steps_out)
        return x, (x, n_evals)

    keys = jax.random.split(jax.random.key(seed), n + burn)
    _, (xs, ne) = jax.lax.scan(body, jnp.asarray(x0, dtype=jnp.float64), keys)
    return np.asarray(xs[burn:]), np.asarray(ne[burn:])


def test_1d_gaussian_mean_and_variance():
    ld = lambda x: -0.5 * ((x[0] - 1.0) / 2.0) ** 2  # noqa: E731
    xs, _ = run_chain(ld, [0.0], jnp.array([3.0]), 10000, seed=1)
    assert_mean_close(xs[:, 0], 1.0)
    assert_mean_close((xs[:, 0] - 1.0) ** 2, 4.0)


def test_5d_correlated_gaussian_moments():
    d, rho = 5, 0.9
    cov = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
    prec = jnp.asarray(np.linalg.inv(cov))  # reference-side inverse only: this is the target's definition
    ld = lambda x: -0.5 * x @ prec @ x  # noqa: E731
    xs, _ = run_chain(ld, np.zeros(d), jnp.ones(d), 12000, seed=2)
    assert_mean_close(xs, np.zeros(d))
    prods = xs[:, :, None] * xs[:, None, :]
    assert_mean_close(prods.reshape(len(xs), -1), cov.reshape(-1))


def test_target_with_minus_inf_outside_support_gamma():
    ld = lambda x: jnp.where(x[0] > 0, jnp.log(jnp.maximum(x[0], 1e-300)) - x[0], -jnp.inf)  # noqa: E731
    xs, _ = run_chain(ld, [1.0], jnp.array([2.0]), 10000, seed=3)
    assert np.all(xs > 0)
    assert_mean_close(xs[:, 0], 2.0)
    assert_mean_close((xs[:, 0] - 2.0) ** 2, 2.0)


def test_banana_against_exact_draws():
    s1, b = 1.5, 0.5

    def ld(x):
        return -0.5 * (x[0] / s1) ** 2 - 0.5 * (x[1] - b * (x[0] ** 2 - s1**2)) ** 2

    def stats_of(x):
        return np.stack([x[:, 0], x[:, 1], x[:, 0] ** 2, x[:, 1] ** 2, x[:, 0] * x[:, 1]], axis=1)

    xs, _ = run_chain(ld, [0.0, 0.0], jnp.array([3.0, 3.0]), 15000, seed=4)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200000, 2))
    exact = np.stack([s1 * z[:, 0], z[:, 1] + b * (s1**2 * z[:, 0] ** 2 - s1**2)], axis=1)
    se = np.sqrt(batch_se(stats_of(xs)) ** 2 + stats_of(exact).var(axis=0) / len(exact))
    diff = np.abs(stats_of(xs).mean(axis=0) - stats_of(exact).mean(axis=0))
    assert np.all(diff < 4 * se), (diff, se)


@pytest.mark.parametrize("kind", ["gaussian", "gamma"])
@pytest.mark.parametrize("width,m", [(3.0, 32), (0.05, 3)])
def test_one_step_leaves_target_invariant_ks(kind, width, m):
    n = 2000
    if kind == "gaussian":
        ld = lambda x: -0.5 * ((x[0] - 1.0) / 2.0) ** 2  # noqa: E731
        x0 = 1.0 + 2.0 * jax.random.normal(jax.random.key(10), (n, 1))
        cdf = stats.norm(1.0, 2.0).cdf
    else:
        ld = lambda x: jnp.where(x[0] > 0, jnp.log(jnp.maximum(x[0], 1e-300)) - x[0], -jnp.inf)  # noqa: E731
        x0 = jax.random.gamma(jax.random.key(10), 2.0, (n, 1))
        cdf = stats.gamma(2.0).cdf
    w = jnp.array([width])
    keys = jax.random.split(jax.random.key(11), n)
    x1, _ = jax.jit(jax.vmap(lambda k, x: slice_step(k, x, ld, w, m)))(keys, x0)
    assert stats.kstest(np.asarray(x1[:, 0]), cdf).pvalue > 0.01
    assert not np.allclose(x1, x0)  # the step moves


def test_n_evals_counts_every_logdensity_call():
    calls = []

    def ld(x):
        jax.debug.callback(lambda: calls.append(1))
        return -0.5 * jnp.sum(x**2)

    x, n_evals = slice_step(jax.random.key(5), jnp.array([0.3, -0.2, 1.0]), ld, jnp.ones(3))
    jax.block_until_ready(x)
    jax.effects_barrier()
    assert int(n_evals) == len(calls)
    assert int(n_evals) >= 4  # the initial level plus at least one proposal per coordinate


def test_jit_works_and_matches_eager():
    ld = lambda x: -0.5 * jnp.sum(x**2)  # noqa: E731
    w = jnp.ones(2)
    k, x = jax.random.key(6), jnp.array([0.5, -0.5])
    f = jax.jit(lambda k, x: slice_step(k, x, ld, w))
    xa, na = f(k, x)
    xb, nb = slice_step(k, x, ld, w)
    np.testing.assert_allclose(xa, xb)
    assert int(na) == int(nb)
