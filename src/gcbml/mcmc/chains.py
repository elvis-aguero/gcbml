"""Running several chains (spec Section 2.5: 4 chains).

run_chains(key, init, make_step, init_widths, n_warmup, n_samples, n_chains=4, thin=1) -> ChainResult
    init: pytree with a leading chain axis (n_chains, ...) (checked).
    make_step(widths) -> step is a closure factory; step(key, state) -> (state, info), where info is a
    dict of scalars (e.g. n_evals, flags). Chains run in parallel with jax.vmap, iterations with
    lax.scan.
    Phase 1 (warm-up): n_warmup iterations with step = make_step(init_widths); the draws are discarded
    for inference but used to adapt the widths. Phase 2: widths = adapt_widths(warm-up draws), then
    n_samples * thin iterations with step = make_step(widths); every thin-th state is kept. With
    n_warmup < 2 there is nothing to adapt from, so init_widths is kept.
    ChainResult.samples: pytree with leading axes (n_chains, n_samples, ...).
    ChainResult.info: info summed over all sampling-phase iterations (thinned ones included, because
    they cost evaluations too), per chain: each leaf has shape (n_chains,).
    ChainResult.widths: the widths used in the sampling phase (reusable for a later call).
    ChainResult.warmup_info: as info, for the warm-up phase (cost accounting). The last two fields
    are additions to the specification's two-field result.
    Keys: one key per chain from jax.random.split(key, n_chains); the same key gives identical output.
adapt_widths(warmup_samples) -> widths
    Slice widths for the sampling phase: 2 x the per-coordinate sd (ddof = 1) of the draws in the
    second half of the warm-up, pooled over chains (all chains' draws in one sample, so the spread
    between chain means counts), floored at 1e-3. warmup_samples is a pytree with leaves of shape
    (n_chains, n_warmup, ...); the result has the structure of the pytree with leaf shapes (...).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

WIDTH_FLOOR = 1e-3


class ChainResult(NamedTuple):
    samples: Any
    info: dict
    widths: Any = None
    warmup_info: Any = None


def adapt_widths(warmup_samples):
    def one(a):
        n = a.shape[1]
        second = a[:, n // 2 :]
        pooled = second.reshape((-1,) + a.shape[2:])
        return jnp.maximum(2.0 * jnp.std(pooled, axis=0, ddof=1), WIDTH_FLOOR)

    return jax.tree_util.tree_map(one, warmup_samples)


def chain_mesh(n_chains: int):
    """Mesh over the largest number of local devices that divides n_chains, or None for a single device."""
    n_dev = len(jax.devices())
    use = max(d for d in range(1, min(n_dev, n_chains) + 1) if n_chains % d == 0)
    if use == 1:
        return None
    return jax.sharding.Mesh(np.array(jax.devices()[:use]), ("chain",))


def map_chains(fn, n_chains: int, n_args: int = 1):
    """vmap ``fn`` over a leading chain axis of all its (n_args) arguments, one block of chains per device.

    On several devices (gcbml._config creates one CPU device per available core) the chains run
    independently in parallel, each device with its own control flow: a vmapped while_loop would
    otherwise run every chain until the slowest one finishes, on one thread. With one device this is
    jax.vmap.
    """
    mesh = chain_mesh(n_chains)
    vf = jax.vmap(fn)
    if mesh is None:
        return vf
    spec = jax.sharding.PartitionSpec("chain")
    return jax.shard_map(vf, mesh=mesh, in_specs=(spec,) * n_args, out_specs=spec, check_vma=False)


def _phase(make_step, buf_size: int, n_chains: int):
    """One jitted program that serves warm-up and sampling.

    Arguments: widths (traced, replicated), keys (n_chains,), state, n_keep and thin (traced int32).
    It runs n_keep * thin steps and keeps every thin-th state in a buffer of buf_size rows (static), so
    the two phases differ only in argument values and share one compilation.
    """

    def one_chain(w, key, s0, n_keep, thin):
        step = make_step(w)
        _, info0 = jax.eval_shape(step, key, s0)
        acc0 = jax.tree_util.tree_map(lambda sd: jnp.zeros(sd.shape, sd.dtype), info0)
        buf0 = jax.tree_util.tree_map(lambda x: jnp.zeros((buf_size,) + x.shape, x.dtype), s0)

        def kept(i, carry):
            s, acc, buf = carry
            ki = jax.random.fold_in(key, i)

            def inner(j, c):
                s, acc = c
                s, info = step(jax.random.fold_in(ki, j), s)
                return s, jax.tree_util.tree_map(jnp.add, acc, info)

            s, acc = lax.fori_loop(0, thin, inner, (s, acc))
            buf = jax.tree_util.tree_map(lambda b, x: b.at[i].set(x), buf, s)
            return s, acc, buf

        return lax.fori_loop(0, n_keep, kept, (s0, acc0, buf0))

    def run(w, keys, state, n_keep, thin):
        f = jax.vmap(lambda k, s: one_chain(w, k, s, n_keep, thin))
        mesh = chain_mesh(n_chains)
        if mesh is not None:
            spec = jax.sharding.PartitionSpec("chain")
            f = jax.shard_map(f, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False)
        return f(keys, state)

    return jax.jit(run)


def run_chains(
    key, init, make_step, init_widths, n_warmup: int, n_samples: int, n_chains: int = 4, thin: int = 1
) -> ChainResult:
    sizes = {leaf.shape[0] for leaf in jax.tree_util.tree_leaves(init)}
    if sizes != {n_chains}:
        raise ValueError(f"every leaf of init needs a leading chain axis of size {n_chains}, got {sizes}")
    k_warm, k_samp = jax.random.split(key)
    warm_keys = jax.random.split(k_warm, n_chains)
    samp_keys = jax.random.split(k_samp, n_chains)
    phase = _phase(make_step, max(n_warmup, n_samples, 1), n_chains)

    mesh = chain_mesh(n_chains)

    def place(tree, spec):
        if mesh is None:
            return tree
        sh = jax.sharding.NamedSharding(mesh, spec)
        return jax.tree_util.tree_map(lambda a: jax.device_put(a, sh), tree)

    def go(w, keys, state, n_keep, th):
        # identical placement in both phases, so that the second call reuses the first compilation
        w = place(w, jax.sharding.PartitionSpec())
        keys, state = (
            place(keys, jax.sharding.PartitionSpec("chain")),
            place(state, jax.sharding.PartitionSpec("chain")),
        )
        sf, acc, buf = phase(w, keys, state, jnp.int32(n_keep), jnp.int32(th))
        kept = jax.tree_util.tree_map(lambda b: b[:, :n_keep], buf)
        return sf, kept, acc

    state, widths, warm_info = init, init_widths, None
    if n_warmup > 0:
        state, warm, warm_info = go(init_widths, warm_keys, init, n_warmup, 1)
        if n_warmup >= 2:
            widths = adapt_widths(warm)
    _, samples, info = go(widths, samp_keys, state, n_samples, thin)
    return ChainResult(samples, info, widths, warm_info)
