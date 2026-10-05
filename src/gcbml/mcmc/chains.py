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

import contextlib
import ctypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor
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


_BLAS_LOCK = threading.Lock()
_BLAS_DEPTH = 0
_BLAS_SAVED: list = []


def _openblas_handles():
    """ctypes handles of the OpenBLAS libraries loaded in this process (Linux; empty elsewhere)."""
    try:
        with open("/proc/self/maps") as fh:
            paths = sorted({line.split()[-1] for line in fh if "openblas" in line and ".so" in line})
    except OSError:
        return []
    return [ctypes.CDLL(p) for p in paths]


@contextlib.contextmanager
def single_threaded_blas():
    """Run the body with every loaded OpenBLAS limited to one thread, then restore the previous count.

    jaxlib's CPU Cholesky and triangular solves call the OpenBLAS that ships with scipy, which has its own
    thread pool (one thread per core). Several chains running at once each wake that pool, the pools
    fight for the cores and spin, and a program that takes 5 ms alone takes 50 ms or more (measured at
    n_pad 368: 4 concurrent chains 7-20x slower than one chain). One BLAS thread per chain removes it.
    The setting is process-wide while the body runs (hence the reference count) and is undone on exit,
    so a user's own scipy/numpy code keeps its threads. This works after JAX and NumPy have started, so
    it needs no environment variable before ``import gcbml``. If no OpenBLAS is found this does nothing.
    """
    global _BLAS_DEPTH
    with _BLAS_LOCK:
        if _BLAS_DEPTH == 0:
            _BLAS_SAVED.clear()
            for lib in _openblas_handles():
                for pre in ("scipy_openblas", "openblas"):
                    get, put = (
                        getattr(lib, f"{pre}_get_num_threads", None),
                        getattr(lib, f"{pre}_set_num_threads", None),
                    )
                    if get is not None and put is not None:
                        get.restype = ctypes.c_int
                        put.argtypes = [ctypes.c_int]
                        _BLAS_SAVED.append((put, get()))
                        put(1)
                        break
        _BLAS_DEPTH += 1
    try:
        yield
    finally:
        with _BLAS_LOCK:
            _BLAS_DEPTH -= 1
            if _BLAS_DEPTH == 0:
                for put, n in _BLAS_SAVED:
                    put(n)


def chain_scheme() -> str:
    """How chains are parallelised: env GCBML_CHAIN_SCHEME = threads (default), shard or vmap."""
    scheme = os.environ.get("GCBML_CHAIN_SCHEME", "threads")
    if scheme not in ("threads", "shard", "vmap"):
        raise ValueError(f"GCBML_CHAIN_SCHEME must be threads, shard or vmap, got {scheme!r}")
    return scheme


def _is_traced(tree) -> bool:
    return any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(tree))


def chain_mesh(n_chains: int):
    """Mesh over the largest number of local devices that divides n_chains, or None for a single device."""
    n_shared = len(jax.devices())
    use = max(d for d in range(1, min(n_shared, n_chains) + 1) if n_chains % d == 0)
    if use == 1:
        return None
    return jax.sharding.Mesh(np.array(jax.devices()[:use]), ("chain",))


def chain_map(fn, n_chains: int, n_batched: int):
    """Return f(*batched, *shared): fn applied to every chain, run in parallel.

    The first ``n_batched`` arguments carry a leading chain axis of size n_chains, the rest are shared
    by all chains. Outputs are stacked on a leading chain axis, as jax.vmap would.

    Scheme threads (default): ``fn`` is compiled once as a single-chain program (the compiled program is
    kept, keyed by argument shapes) and the chains are launched from separate Python threads. XLA releases
    the GIL while a program runs, so the programs overlap on the CPU cores, and every chain has its own
    while_loops. Scheme vmap: one vmapped program, on one device. Scheme shard: that program with one
    block of chains per CPU device (jax.shard_map; needs GCBML_HOST_DEVICES). A vmapped program runs on
    one thread, and its while_loops run every chain as long as the slowest one needs; this is why vmap is
    not the default. Inside an outer jit (traced arguments) every scheme is jax.vmap.
    """
    scheme = chain_scheme()
    n_shared = lambda args: len(args) - n_batched  # noqa: E731

    def vmapped(*args):
        return jax.vmap(fn, in_axes=(0,) * n_batched + (None,) * n_shared(args))(*args)

    if scheme != "threads":

        def run(*args):
            f = vmapped
            mesh = chain_mesh(n_chains) if scheme == "shard" else None
            if mesh is not None:
                P = jax.sharding.PartitionSpec
                specs = (P("chain"),) * n_batched + (P(),) * n_shared(args)
                f = jax.shard_map(vmapped, mesh=mesh, in_specs=specs, out_specs=P("chain"), check_vma=False)
            return f(*args)

        return jax.jit(run)

    compiled: dict = {}

    def run_threads(*args):
        if _is_traced(args):
            return vmapped(*args)
        batched, shared = args[:n_batched], args[n_batched:]
        pick = lambda i: jax.tree_util.tree_map(lambda a: a[i], batched)  # noqa: E731
        leaves, treedef = jax.tree_util.tree_flatten(args)
        sig = (treedef, tuple((jnp.shape(leaf), jnp.result_type(leaf)) for leaf in leaves))
        if sig not in compiled:
            compiled[sig] = jax.jit(fn).lower(*pick(0), *shared).compile()
        exe = compiled[sig]
        with single_threaded_blas(), ThreadPoolExecutor(max_workers=n_chains) as pool:
            outs = list(pool.map(lambda i: jax.block_until_ready(exe(*pick(i), *shared)), range(n_chains)))
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *outs)

    return run_threads


def make_phase(make_step, buf_size: int, n_chains: int):
    """One phase for all chains: phase(w, keys, state, n_keep, thin, *extra) -> (states, buffers, info).

    make_step(w, *extra) -> step. It runs n_keep * thin steps per chain and keeps every thin-th state in a
    buffer of buf_size rows (static), so warm-up and sampling differ only in argument values and share one
    compilation. Arguments w, n_keep, thin and extra are traced and shared by the chains. keys and state
    have a leading chain axis. The result has a leading chain axis.
    """

    def one_chain(key, s0, w, n_keep, thin, *extra):
        step = make_step(w, *extra)
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

    mapped = chain_map(one_chain, n_chains, 2)

    def phase(w, keys, state, n_keep, thin, *extra):
        return mapped(keys, state, w, jnp.int32(n_keep), jnp.int32(thin), *extra)

    return phase


def run_chains(
    key,
    init,
    make_step,
    init_widths,
    n_warmup: int,
    n_samples: int,
    n_chains: int = 4,
    thin: int = 1,
    *,
    phase=None,
    extra=(),
) -> ChainResult:
    """See the module docstring. ``phase`` (from make_phase, with make_step(w, *extra)) and ``extra`` let a
    caller keep the compiled program between calls; without them make_step(w) is compiled for this call."""
    sizes = {leaf.shape[0] for leaf in jax.tree_util.tree_leaves(init)}
    if sizes != {n_chains}:
        raise ValueError(f"every leaf of init needs a leading chain axis of size {n_chains}, got {sizes}")
    k_warm, k_samp = jax.random.split(key)
    warm_keys = jax.random.split(k_warm, n_chains)
    samp_keys = jax.random.split(k_samp, n_chains)
    if phase is None:
        phase = make_phase(make_step, max(n_warmup, n_samples, 1), n_chains)

    def go(w, keys, state, n_keep, th):
        sf, acc, buf = phase(w, keys, state, n_keep, th, *extra)
        kept = jax.tree_util.tree_map(lambda b: b[:, :n_keep], buf)
        return sf, kept, acc

    state, widths, warm_info = init, init_widths, None
    if n_warmup > 0:
        state, warm, warm_info = go(init_widths, warm_keys, init, n_warmup, 1)
        if n_warmup >= 2:
            widths = adapt_widths(warm)
    _, samples, info = go(widths, samp_keys, state, n_samples, thin)
    return ChainResult(samples, info, widths, warm_info)
