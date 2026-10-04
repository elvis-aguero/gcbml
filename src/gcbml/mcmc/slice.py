"""Univariate slice sampling, coordinate-wise (Neal 2003, arXiv physics/0009028, Section 4).

slice_step(key, x, logdensity, widths, max_steps_out=32) -> (x_new, n_evals)
    One sweep over the coordinates of x (shape (m,)) in a fixed order. For coordinate i: draw the
    level log y = log f(x) - e with e ~ Exponential(1) (Section 4); step out from a randomly placed
    interval of width widths[i] with the procedure of Fig. 3; then shrink with the procedure of
    Fig. 5 until a point inside the slice is drawn.

    max_steps_out is the integer m of Fig. 3: the interval has at most m * widths[i] in total. Fig. 3
    splits the m - 1 possible steps at random between the two sides (J = floor(m V) to the left,
    K = m - 1 - J to the right), so the limit is on the *total* number of steps, not per side.
    The random split is part of the proof of invariance, so the limit does not bias the sampler.

    logdensity may return -inf (outside the support): such points are never in the slice, so they
    only end the stepping out or shrink the interval. The shrinkage loop is bounded
    (MAX_SHRINK = 100 draws per coordinate). If the bound is hit, the coordinate keeps its current
    value, which keeps invariance (the identity move is always valid); slice_step_info returns the
    number of such failures. slice_step keeps the two-value interface of the specification.

    n_evals (int) is the number of logdensity calls in the sweep: one for the starting point, plus
    those of the end-point tests in Fig. 3 and one per shrinkage draw (Fig. 5). The value of
    logdensity at the accepted point is carried to the next coordinate, so it is not recomputed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

MAX_SHRINK = 100


def _step_out_side(logdensity_1d, edge, j, w, logy, sign):
    """Fig. 3: move ``edge`` outwards (sign = -1 left, +1 right) by w while j > 0 and the edge is inside the slice."""
    has = j > 0
    inside0 = lax.cond(has, lambda: logdensity_1d(edge) > logy, lambda: jnp.array(False))

    def cond(s):
        _, j, inside, _ = s
        return inside & (j > 0)

    def body(s):
        edge, j, _, n = s
        edge = edge + sign * w
        j = j - 1
        more = j > 0
        inside = lax.cond(more, lambda: logdensity_1d(edge) > logy, lambda: jnp.array(False))
        return edge, j, inside, n + more.astype(jnp.int32)

    edge, _, _, n = lax.while_loop(cond, body, (edge, j, inside0, has.astype(jnp.int32)))
    return edge, n


def _update_coordinate(key, x0i, logp0, logdensity_1d, w, m):
    """One univariate update (Figs. 3 and 5). Returns (x_new_i, logp_new, n_evals, failed)."""
    k_e, k_u, k_v, k_s = jax.random.split(key, 4)
    logy = logp0 - jax.random.exponential(k_e, dtype=jnp.float64)
    u = jax.random.uniform(k_u, dtype=jnp.float64)
    v = jax.random.uniform(k_v, dtype=jnp.float64)
    left = x0i - w * u
    right = left + w
    j = jnp.floor(m * v).astype(jnp.int32)
    k = (m - 1) - j
    left, n_l = _step_out_side(logdensity_1d, left, j, w, logy, -1.0)
    right, n_r = _step_out_side(logdensity_1d, right, k, w, logy, +1.0)

    def cond(s):
        _, _, _, _, accepted, it, _, _ = s
        return (~accepted) & (it < MAX_SHRINK)

    def body(s):
        lb, rb, _, _, _, it, n, key = s
        key, sub = jax.random.split(key)
        x1 = lb + jax.random.uniform(sub, dtype=jnp.float64) * (rb - lb)
        f1 = logdensity_1d(x1)
        accepted = f1 > logy
        lb = jnp.where(~accepted & (x1 < x0i), x1, lb)
        rb = jnp.where(~accepted & (x1 >= x0i), x1, rb)
        return lb, rb, x1, f1, accepted, it + 1, n + 1, key

    init = (left, right, x0i, logp0, jnp.array(False), jnp.int32(0), jnp.int32(0), k_s)
    _, _, x1, f1, accepted, _, n_s, _ = lax.while_loop(cond, body, init)
    x_new = jnp.where(accepted, x1, x0i)
    logp_new = jnp.where(accepted, f1, logp0)
    return x_new, logp_new, n_l + n_r + n_s, ~accepted


def slice_step_info(key, x, logdensity, widths, max_steps_out: int = 32):
    """As slice_step, and also return the number of coordinates whose shrinkage bound was hit."""
    x = jnp.asarray(x, dtype=jnp.float64)
    widths = jnp.broadcast_to(jnp.asarray(widths, dtype=jnp.float64), x.shape)
    keys = jax.random.split(key, x.shape[0])

    def body(i, carry):
        x, logp, n, n_fail = carry

        def ld1(v):
            return logdensity(x.at[i].set(v))

        xi, logp, ni, failed = _update_coordinate(keys[i], x[i], logp, ld1, widths[i], max_steps_out)
        return x.at[i].set(xi), logp, n + ni, n_fail + failed.astype(jnp.int32)

    logp0 = logdensity(x)
    x, _, n, n_fail = lax.fori_loop(0, x.shape[0], body, (x, logp0, jnp.int32(1), jnp.int32(0)))
    return x, n, n_fail


def slice_step(key, x, logdensity, widths, max_steps_out: int = 32):
    x_new, n_evals, _ = slice_step_info(key, x, logdensity, widths, max_steps_out)
    return x_new, n_evals
