"""Running chains in parallel (spec Section 2.5): vmap over chains, scan over iterations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gcbml  # noqa: F401  (float64)
from gcbml.mcmc import diagnostics as dg
from gcbml.mcmc.chains import adapt_widths, run_chains
from gcbml.mcmc.slice import slice_step


def gaussian_make_step(widths):
    ld = lambda x: -0.5 * jnp.sum(((x - 1.0) / jnp.array([2.0, 0.5])) ** 2)  # noqa: E731

    def step(key, x):
        x, n_evals = slice_step(key, x, ld, widths)
        return x, {"n_evals": n_evals}

    return step


def test_four_chains_in_parallel_return_expected_shapes_and_moments():
    init = 1.0 + jax.random.normal(jax.random.key(0), (4, 2))
    res = run_chains(jax.random.key(1), init, gaussian_make_step, jnp.ones(2), 300, 1500)
    assert res.samples.shape == (4, 1500, 2)
    assert not np.allclose(res.samples[0], res.samples[1])  # chains are independent
    x = np.asarray(res.samples)
    np.testing.assert_allclose(x.mean(axis=(0, 1)), [1.0, 1.0], atol=0.2)
    np.testing.assert_allclose(x.std(axis=(0, 1)), [2.0, 0.5], rtol=0.1)
    assert dg.rhat(x[:, :, 0]) < 1.02
    assert res.info["n_evals"].shape == (4,)
    assert np.all(np.asarray(res.info["n_evals"]) > 1500)


def ar1_make_step(widths):
    def step(key, x):
        return 0.9 * x + jax.random.normal(key, x.shape), {}

    return step


def test_same_key_gives_identical_results_and_other_key_differs():
    init = jnp.zeros((4, 2))
    a = run_chains(jax.random.key(7), init, ar1_make_step, jnp.ones(2), 50, 100)
    b = run_chains(jax.random.key(7), init, ar1_make_step, jnp.ones(2), 50, 100)
    c = run_chains(jax.random.key(8), init, ar1_make_step, jnp.ones(2), 50, 100)
    np.testing.assert_array_equal(a.samples, b.samples)
    np.testing.assert_array_equal(a.widths, b.widths)
    assert not np.allclose(a.samples, c.samples)


def counter_make_step(widths):
    def step(key, x):
        return x + 1.0, {"n": jnp.int32(1)}

    return step


def test_warmup_draws_are_discarded_and_thinning_keeps_every_thin_th_state():
    init = jnp.zeros((3,))  # three chains, scalar state
    res = run_chains(jax.random.key(0), init, counter_make_step, jnp.ones(()), 7, 5, n_chains=3)
    np.testing.assert_array_equal(res.samples, np.tile(np.arange(8.0, 13.0), (3, 1)))
    assert np.all(np.asarray(res.info["n"]) == 5)
    assert np.all(np.asarray(res.warmup_info["n"]) == 7)
    res = run_chains(jax.random.key(0), init, counter_make_step, jnp.ones(()), 7, 5, n_chains=3, thin=3)
    np.testing.assert_array_equal(res.samples, np.tile(7.0 + 3.0 * np.arange(1, 6), (3, 1)))
    assert np.all(np.asarray(res.info["n"]) == 15)  # every step costs, kept or not


def test_pytree_state_is_supported():
    def make_step(widths):
        def step(key, s):
            return {"a": s["a"] + 1.0, "b": s["b"] * 2.0}, {}

        return step

    init = {"a": jnp.zeros((4, 2)), "b": jnp.ones((4,))}
    res = run_chains(jax.random.key(0), init, make_step, {"a": jnp.ones(2), "b": jnp.ones(())}, 2, 3)
    assert res.samples["a"].shape == (4, 3, 2) and res.samples["b"].shape == (4, 3)
    np.testing.assert_array_equal(res.samples["b"][0], [8.0, 16.0, 32.0])


def test_adapt_widths_is_twice_pooled_sd_of_second_half_floored():
    rng = np.random.default_rng(0)
    first = 100.0 * rng.standard_normal((3, 50, 3))  # first half: must be ignored
    second = rng.standard_normal((3, 50, 3)) * np.array([1.0, 0.25, 0.0]) + np.array([0.0, 5.0, 2.0])
    w = adapt_widths(jnp.asarray(np.concatenate([first, second], axis=1)))
    expected = 2.0 * second.reshape(-1, 3).std(axis=0, ddof=1)
    expected = np.maximum(expected, 1e-3)
    np.testing.assert_allclose(w, expected, rtol=1e-12)
    assert float(w[2]) == 1e-3  # constant coordinate hits the floor


def test_adapt_widths_pools_over_chains_including_their_offsets():
    x = np.zeros((2, 10, 1))
    x[1] += 3.0  # two constant chains at 0 and 3: pooled sd is not zero
    w = adapt_widths(jnp.asarray(x))
    np.testing.assert_allclose(w, 2.0 * np.concatenate([x[0, 5:], x[1, 5:]]).std(axis=0, ddof=1), rtol=1e-12)


def test_run_chains_uses_adapted_widths_after_warmup():
    def make_step(widths):
        def step(key, x):  # an exact N(0, 3^2) draw, independent of the state; reports the widths it got
            return 3.0 * jax.random.normal(key, x.shape), {"w": widths}

        return step

    res = run_chains(jax.random.key(0), jnp.zeros((4, 1)), make_step, jnp.array([100.0]), 400, 10)
    np.testing.assert_allclose(
        res.warmup_info["w"][:, 0] / 400, 100.0
    )  # warm-up starts from the given widths
    np.testing.assert_allclose(res.info["w"] / 10, np.broadcast_to(res.widths, (4, 1)))  # sampling: adapted
    assert res.widths[0] == pytest.approx(6.0, rel=0.1)


def test_warmup_and_sampling_share_one_compilation():
    traced = []

    def make_step(widths):
        traced.append(1)  # runs once per trace

        def step(key, x):
            return x + jax.random.normal(key, x.shape) * widths, {}

        return step

    res = run_chains(jax.random.key(0), jnp.zeros((4, 2)), make_step, jnp.ones(2), 30, 50)
    assert len(traced) == 1  # both phases: one program (widths are traced arguments)
    assert res.samples.shape == (4, 50, 2)


def test_chains_run_on_several_devices_when_available():
    import gcbml.mcmc.chains as ch

    f = ch.map_chains(lambda x: x * 2.0, 4)
    np.testing.assert_array_equal(
        jax.jit(f)(jnp.arange(8.0).reshape(4, 2)), 2.0 * np.arange(8.0).reshape(4, 2)
    )
    assert (ch.chain_mesh(4) is None) == (len(jax.devices()) == 1)


def test_run_chains_with_zero_warmup_keeps_initial_widths():
    res = run_chains(
        jax.random.key(0), jnp.zeros((2, 2)), gaussian_make_step, jnp.array([0.7, 0.9]), 0, 5, n_chains=2
    )
    np.testing.assert_array_equal(res.widths, [0.7, 0.9])


@pytest.mark.slow
def test_run_chains_and_diagnostics_converge_on_correlated_gaussian():
    d, rho = 5, 0.9
    cov = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
    prec = jnp.asarray(np.linalg.inv(cov))  # reference-side only: defines the target density

    def make_step(widths):
        def step(key, x):
            x, n_evals = slice_step(key, x, lambda z: -0.5 * z @ prec @ z, widths)
            return x, {"n_evals": n_evals}

        return step

    init = 3.0 * jax.random.normal(jax.random.key(0), (4, d))  # over-dispersed starts
    res = run_chains(jax.random.key(1), init, make_step, jnp.ones(d), 1000, 8000)
    x = np.asarray(res.samples)
    summ = dg.summary({f"x{i}": x[:, :, i] for i in range(d)})
    assert dg.converged(summ), summ
    np.testing.assert_allclose(x.reshape(-1, d).mean(axis=0), 0.0, atol=0.15)
    np.testing.assert_allclose(np.cov(x.reshape(-1, d).T), cov, atol=0.1)
