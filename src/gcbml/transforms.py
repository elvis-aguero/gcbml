"""Output transforms Lambda (spec Section 2.2; guide Section 5, "What Lambda does").

Lambda is a monotone map that sets the units in which the grid error is written (e.g. log for
relative errors). Likelihoods compared across transforms must be densities of the RAW output y,
so every transform provides log|dLambda/dy| (spec Section 2.7: scores on the physical scale).

TODO(W1-A): implement. Each transform is a frozen dataclass with jnp-compatible methods:
    forward(y) -> z,  inverse(z) -> y,  log_abs_jac(y) -> log|dz/dy|,  domain_ok(y) -> bool array
Required transforms: "identity", "log" (y > 0), "reciprocal" (y > 0; z = 1/y, decreasing is allowed:
monotone maps keep medians). ``get(name)`` returns the transform by name and raises KeyError otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Transform:
    name: str
    forward: Callable[[jnp.ndarray], jnp.ndarray]
    inverse: Callable[[jnp.ndarray], jnp.ndarray]
    log_abs_jac: Callable[[jnp.ndarray], jnp.ndarray]
    domain_ok: Callable[[jnp.ndarray], jnp.ndarray]


def get(name: str) -> Transform:
    raise NotImplementedError("W1-A")
