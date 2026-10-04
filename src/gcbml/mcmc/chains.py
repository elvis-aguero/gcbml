"""Running several chains (spec Section 2.5: 4 chains).

TODO(W1-B): implement.

run_chains(key, init, step, n_warmup, n_samples, n_chains=4, thin=1) -> ChainResult
    init: pytree with a leading chain axis (n_chains, ...).
    step(key, state) -> (state, info) where info is a dict of scalars (e.g. n_evals, flags).
    Chains run in parallel with jax.vmap; iterations with lax.scan. Warm-up draws are discarded.
    ChainResult.samples: pytree with leading axes (n_chains, n_samples, ...); .info: summed info.
adapt_widths(samples_warmup) -> widths
    Slice widths for the post-warm-up phase: 2 x the per-coordinate posterior sd over the second
    half of the warm-up, pooled over chains, floored at 1e-3 (state the rule in the docstring).
    run_chains uses it between warm-up and sampling when step accepts a ``widths`` argument through
    a closure factory: make_step(widths) -> step.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class ChainResult(NamedTuple):
    samples: Any
    info: dict


def run_chains(
    key, init, make_step, init_widths, n_warmup: int, n_samples: int, n_chains: int = 4, thin: int = 1
) -> ChainResult:
    raise NotImplementedError("W1-B")


def adapt_widths(warmup_samples):
    raise NotImplementedError("W1-B")
