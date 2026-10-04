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


def _run_phase(step, keys, state, n_steps: int, thin: int):
    """Vmap over chains, scan over iterations. Returns (final state, kept states, summed info)."""
    n_keep = n_steps // thin

    def one_chain(key, s0):
        def kept_block(carry, k):
            s, acc = carry

            def inner(c, kk):
                s, acc = c
                s, info = step(kk, s)
                return (s, jax.tree_util.tree_map(jnp.add, acc, info)), None

            (s, acc), _ = lax.scan(inner, (s, acc), jax.random.split(k, thin))
            return (s, acc), s

        _, info0 = jax.eval_shape(step, key, s0)
        acc0 = jax.tree_util.tree_map(lambda sd: jnp.zeros(sd.shape, sd.dtype), info0)
        (sf, acc), kept = lax.scan(kept_block, (s0, acc0), jax.random.split(key, n_keep))
        return sf, kept, acc

    return jax.jit(jax.vmap(one_chain))(keys, state)


def run_chains(
    key, init, make_step, init_widths, n_warmup: int, n_samples: int, n_chains: int = 4, thin: int = 1
) -> ChainResult:
    sizes = {leaf.shape[0] for leaf in jax.tree_util.tree_leaves(init)}
    if sizes != {n_chains}:
        raise ValueError(f"every leaf of init needs a leading chain axis of size {n_chains}, got {sizes}")
    k_warm, k_samp = jax.random.split(key)
    warm_keys = jax.random.split(k_warm, n_chains)
    samp_keys = jax.random.split(k_samp, n_chains)

    state, widths, warm_info = init, init_widths, None
    if n_warmup > 0:
        state, warm, warm_info = _run_phase(make_step(init_widths), warm_keys, init, n_warmup, 1)
        if n_warmup >= 2:
            widths = adapt_widths(warm)
    _, samples, info = _run_phase(make_step(widths), samp_keys, state, n_samples * thin, thin)
    return ChainResult(samples, info, widths, warm_info)
