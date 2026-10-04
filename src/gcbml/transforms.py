"""Output transforms Lambda (spec Section 2.2; guide Section 5, "What Lambda does").

Lambda is a monotone map that sets the units in which the grid error is written (e.g. log for
relative errors). Likelihoods compared across transforms must be densities of the RAW output y,
so every transform provides log|dLambda/dy| (spec Section 2.7: scores on the physical scale).

Each transform is a frozen dataclass with jnp-compatible methods:
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


_TRANSFORMS = {
    "identity": Transform(
        name="identity",
        forward=lambda y: jnp.asarray(y),
        inverse=lambda z: jnp.asarray(z),
        log_abs_jac=lambda y: jnp.zeros_like(jnp.asarray(y, dtype=float)),
        domain_ok=lambda y: jnp.isfinite(jnp.asarray(y)),
    ),
    # z = log y, dz/dy = 1/y
    "log": Transform(
        name="log",
        forward=lambda y: jnp.log(y),
        inverse=lambda z: jnp.exp(z),
        log_abs_jac=lambda y: -jnp.log(y),
        domain_ok=lambda y: jnp.asarray(y) > 0,
    ),
    # z = 1/y, dz/dy = -1/y^2
    "reciprocal": Transform(
        name="reciprocal",
        forward=lambda y: 1.0 / jnp.asarray(y),
        inverse=lambda z: 1.0 / jnp.asarray(z),
        log_abs_jac=lambda y: -2.0 * jnp.log(y),
        domain_ok=lambda y: jnp.asarray(y) > 0,
    ),
}


def get(name: str) -> Transform:
    """Return the transform called ``name``; raise KeyError (listing the valid names) otherwise."""
    try:
        return _TRANSFORMS[name]
    except KeyError:
        raise KeyError(f"unknown transform {name!r}; valid names: {sorted(_TRANSFORMS)}") from None
