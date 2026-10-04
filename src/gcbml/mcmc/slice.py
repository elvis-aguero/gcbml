"""Univariate slice sampling, coordinate-wise (Neal 2003, arXiv physics/0009028).

TODO(W1-B): implement, after reading Neal's Section 4 (stepping out: Fig. 3; shrinkage: Fig. 5).

slice_step(key, x, logdensity, widths, max_steps_out=32) -> (x_new, n_evals)
    One sweep over the coordinates of x (shape (m,)) in a fixed order. For coordinate i: draw the
    level log y = logdensity(x) - Exponential(1); step out with width widths[i] (at most max_steps_out
    steps per side, randomly split as in Fig. 3); then shrink until a point is accepted.
    Must leave the target invariant. logdensity may return -inf (outside the support): such points
    are never accepted and only shrink the interval. Implement with lax.fori_loop / lax.while_loop
    so it jits; bound the shrinkage loop (e.g. 100 iterations) and, if the bound is hit, keep the
    current value for that coordinate (this keeps invariance) and count it in a returned flag.
    Return n_evals (int) = total logdensity calls in the sweep.
"""

from __future__ import annotations


def slice_step(key, x, logdensity, widths, max_steps_out: int = 32):
    raise NotImplementedError("W1-B")
