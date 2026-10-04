"""Weights of the candidate structures (spec Section 2.7).

A structure M = (h-kernel family, transform Lambda, set of levels used). Each is fitted WITHOUT the finest
level; its held-out finest-level runs are scored by their log predictive density ON THE PHYSICAL SCALE
(the Jacobian of Lambda included, so structures with different transforms compare fairly):

    score_M(i) = log( sum_s w_s N(Lambda_M(y_i); mean_s,i, var_s,i + s^2_s,i) ) + log|Lambda_M'(y_i)|

with (mean, var) from model.predict_level of draw s (the noise variance of the held-out row added), pooled
over the draws of M. Weights maximise the stacking objective of Yao et al. (1704.02030, eq 2.2, log score)
with a Dirichlet(alpha) penalty, alpha = 2 [assumption, spec]:

    max_w  sum_i log sum_M w_M exp(score_M(i)) + (alpha - 1) sum_M log w_M,   w on the simplex.

Cross-fitting (spec 2.7): the held-out SITES (distinct u) are split at random into two halves; weights
from half A are used to calibrate (gate G1) on half B and vice versa. Each half needs >= 6 sites
[assumption]; otherwise the weights are equal and G1 is "not testable". With only 2 levels left after the
hold-out, p is not identifiable: equal weights, flagged.

TODO(W5-A): implement.

holdout(data, levels_of_rows, finest) -> (train PaddedData, heldout row indices)
split_sites(key, X_heldout_u, min_sites=6) -> (idx_A, idx_B) or None when too few sites
log_scores(structure_posteriors, train, heldout_rows) -> (M, n_heldout)
    structure_posteriors: list of acquisition.StructurePosterior fitted on ``train``.
stack_weights(scores (M, n), alpha=2.0) -> (M,)
    Solve the concave problem (e.g. a softmax parametrisation with scipy.optimize, started at equal
    weights); check the optimality (KKT) conditions in a test.
cross_fitted_weights(key, scores, site_of_row, alpha=2.0) -> CrossFit(w_A, w_B, idx_A, idx_B, testable)
"""

from __future__ import annotations

from typing import Any, NamedTuple


class CrossFit(NamedTuple):
    w_A: Any
    w_B: Any
    idx_A: Any
    idx_B: Any
    testable: bool


def holdout(data, levels_of_rows, finest: int):
    raise NotImplementedError("W5-A")


def split_sites(key, X_heldout_u, min_sites: int = 6):
    raise NotImplementedError("W5-A")


def log_scores(structure_posteriors, train, heldout_rows):
    raise NotImplementedError("W5-A")


def stack_weights(scores, alpha: float = 2.0):
    raise NotImplementedError("W5-A")


def cross_fitted_weights(key, scores, site_of_row, alpha: float = 2.0) -> CrossFit:
    raise NotImplementedError("W5-A")
