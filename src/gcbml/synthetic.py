"""Synthetic truths with known answers, as Oracles (for tests and for the A12 prerequisite).

TODO(W6-B): implement.

A12Truth(seed, d=2): spec Section 3, Step 1, "the A12 test":
    f(x, h) = f0(x) + a(x) hbar^p, x in [0, 1]^2, f0 and a independent GP draws (Matérn 5/2, length 0.3,
    sd 1 and 0.5), p ~ U[0.7, 2.5]; hbar levels 1, 1/2, 1/4 probed, 1/8 and 1/16 candidates;
    noise sd 0.01; cost 2^{3 l} (l = level index) times lognormal(0, 0.1).
    Methods: problem() -> gcbml Problem; oracle() -> an Oracle (quote/submit/poll, immediate results);
    truth(x_unit) -> f0.
RichardsonToy(): the guide's Fig. T2 sequence at one condition (f0 = 1.4755, a = 0.7865, p = 1.5).
OscillatingTruth(): f(h) = 1 + 0.4 h^1.5 cos(2 pi h) (spec Fig. E caption), one condition.
a12_oracle_ranking(truth, campaign_state, candidates, n_refit_draws) -> ranking of the top-10 candidates
    by the value computed with full MCMC refits on fantasy outcomes (spec Step 1, "Oracle").
"""

from __future__ import annotations
