"""Elliptical slice sampling (Murray, Adams & MacKay 2010, arXiv 1001.0175).

TODO(W1-B): implement Fig. 2 of the paper.

ess_step(key, f, prior_chol, loglik) -> (f_new, n_evals)
    f: (n,) current state with prior N(0, L L^T), prior_chol = L (lower, (n, n)).
    loglik: callable f -> scalar. Draw nu = L @ z, z ~ N(0, I); level log y = loglik(f) + log U;
    angle theta ~ U[0, 2pi), bracket [theta - 2pi, theta]; propose f' = f cos(theta) + nu sin(theta);
    shrink the bracket toward 0 until loglik(f') > log y. Use lax.while_loop with a bound (e.g. 200);
    if the bound is hit return f unchanged (invariance kept) and set a returned flag.
    The prior must be zero-mean with a covariance fixed during the update (paper Section 2): callers
    sample mean levels as separate variables.
"""

from __future__ import annotations


def ess_step(key, f, prior_chol, loglik):
    raise NotImplementedError("W1-B")
