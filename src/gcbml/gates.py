"""Gates G0-G7 (spec Section 3, Step 3): diagnostics and calibration checks of a fit.

Every gate returns GateResult(name, status, stats): status is "pass", "fail" or "not testable" (never
"pass" by default when a test cannot be run). Thresholds are the spec's [assumption] values and are
keyword arguments. Read the spec table (docs/spec.md, Step 3) for the exact rules; read the cited papers
(arxiv MCP tools) for the statistics they name, and cite equation numbers only after checking them.

TODO(W5-A): implement.

g0_order(log_p0_draws (S, k), p_x_draws=None (S, s, k), sigma_pi_draws=None (S, k), prior_sd=1.0)
    P(p_j0 > 0.5) >= 0.9 AND posterior sd of log p_j0 <= 0.5 * prior_sd, for every component j; when
    P(sigma_pi_j > 0.05) >= 0.9, also P(p_j(x) > 0.5) >= 0.9 at every x of Sigma_N.
g1_level_holdout(z_B, inside95_B) coverage of the 95% intervals within exact binomial limits (two-sided,
    alpha 0.05), no sign bias (mean z within 2 se of 0), and C_LOO near 1 (Bachoc 1301.4320; Oliver
    1311.0828: check the definition and state the acceptance band). Inputs come from the cross-fitted
    hold-out (stacking.py); "not testable" when the hold-out is not testable.
g2_block_loo(...) leave out whole runs: z-scores of the left-out outputs ~ N(0, 1), and the U statistic of
    Overstall & Woods (1506.04489, check the equation). Closed-form block LOO for a GP with fixed parameters,
    per draw: with Q = K^{-1} (triangular solves, no explicit inverse of K), the left-out block b has
    mean y_b - Q_bb^{-1} (Q r)_b and covariance Q_bb^{-1}; pool over draws.
g3_noise(groups: list of arrays of Lambda(y) of runs at the same site, s2_mean per group)
    chi-square test of the replicate spread against the modelled s^2; pass if p > 0.05; "not testable"
    without replicates.
g4_pre_asymptotic(m_full (s,), sigma_epi_full (s,), m_without_coarsest (s,))
    the target moves less than sigma_epi at every x of Sigma_N; else "fail" with the advice
    "remove the coarsest level".
g5_monotone(median_curve (s,), direction) no violation along a declared monotone coordinate;
    "not testable" if none is declared.
g6_shape(draws (S, s) physical, w) |q025 - (m - 1.96 sigma)| and |q975 - (m + 1.96 sigma)| < 0.1 sigma.
g7_prior(log_prior_ratio_fn, draws, w, ...) halve and double each prior scale: importance weights of the
    posterior draws under the changed prior; if their ESS < 400, report "refit needed" (do not refit
    here); pass if m_y moves < 0.5 sigma_epi and sigma_epi changes < 20% at every x; else
    "prior-dominated".
"""

from __future__ import annotations

from typing import NamedTuple


class GateResult(NamedTuple):
    name: str
    status: str  # "pass" | "fail" | "not testable"
    stats: dict


def g0_order(*args, **kwargs) -> GateResult:
    raise NotImplementedError("W5-A")


def g1_level_holdout(*args, **kwargs) -> GateResult:
    raise NotImplementedError("W5-A")


def g2_block_loo(*args, **kwargs) -> GateResult:
    raise NotImplementedError("W5-A")


def g3_noise(*args, **kwargs) -> GateResult:
    raise NotImplementedError("W5-A")


def g4_pre_asymptotic(*args, **kwargs) -> GateResult:
    raise NotImplementedError("W5-A")


def g5_monotone(*args, **kwargs) -> GateResult:
    raise NotImplementedError("W5-A")


def g6_shape(*args, **kwargs) -> GateResult:
    raise NotImplementedError("W5-A")


def g7_prior(*args, **kwargs) -> GateResult:
    raise NotImplementedError("W5-A")
