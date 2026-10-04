"""Convergence diagnostics (Vehtari, Gelman, Simpson, Carpenter & Bürkner 2021, arXiv 1903.08008).

TODO(W1-B): implement from the paper (read Sections 3 and 4 and Appendix A for the exact formulas),
in NumPy (diagnostics run outside jit). Input draws: array (n_chains, n_draws).

split_rhat(x)        classic split-R-hat on the given draws.
rank_normalize(x)    normal scores of fractional ranks over all chains, (r - 3/8) / (S + 1/4);
                     verify the formula in the paper.
rhat(x)              max(rank-normalised split-R-hat, folded rank-normalised split-R-hat) (Section 4.1).
ess(x)               ESS from split chains with the Geyer initial monotone sequence (Section 3.2).
bulk_ess(x)          ess(rank_normalize(x)).
tail_ess(x)          min of ess of the indicators I(x <= q05) and I(x <= q95) (Section 4.3).
summary(samples)     dict name -> (rhat, bulk_ess, tail_ess) for a dict of (n_chains, n_draws) arrays.
converged(summary, rhat_max=1.01, ess_min=400) -> bool   (spec 2.5: R-hat < 1.01, bulk- and tail-ESS > 400).

Tests compare against ArviZ (a dev dependency), which implements the same paper.
"""

from __future__ import annotations


def split_rhat(x):
    raise NotImplementedError("W1-B")


def rank_normalize(x):
    raise NotImplementedError("W1-B")


def rhat(x):
    raise NotImplementedError("W1-B")


def ess(x):
    raise NotImplementedError("W1-B")


def bulk_ess(x):
    raise NotImplementedError("W1-B")


def tail_ess(x):
    raise NotImplementedError("W1-B")


def summary(samples: dict) -> dict:
    raise NotImplementedError("W1-B")


def converged(summ: dict, rhat_max: float = 1.01, ess_min: float = 400.0) -> bool:
    raise NotImplementedError("W1-B")
