"""Per-problem benchmark configuration: the unit of the QoI, the tolerance, prior scales and cost prior.

One table, one place (the eps column is mirrored in benchmarks/README.md and a test keeps them equal).

Units. The oracle returns y / M, with M the order of magnitude of the QoI over the region (one significant
digit, from the problem description). This is a choice of units, as a user nondimensionalises a code's
output; it makes the same prior scales meaningful for both output transforms. The relative tolerance is
unchanged by it. ``truth`` in these units is ``bp.truth(z) / M``.

Prior scales (gcbml's required user inputs, in Lambda units of the scaled QoI) and the cost prior are the
same for every deterministic problem; B4, whose runs are noisy, has a larger S_noise. The cost prior uses
the documented work exponent of the problem (level l costs 2^(gamma l) work units, the coarsest costs 1).
[proposal; see the report]

B3 is not in ``FEASIBLE``: it has an output coordinate v (n_controls < d), which Campaign v1 does not support.
B7 is the infeasible trap: its budget is set from its asymptotic level, not from a certificate.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from benchmarks import ALL_PROBLEMS
from benchmarks.base import BenchmarkProblem
from gcbml.campaign import CampaignSettings
from gcbml.cost import CostPrior
from gcbml.priors import PriorScales
from gcbml.problem import Problem, Tolerance

FEASIBLE = ("b1_poisson", "b2_upwind", "b4_sde", "b5_kink", "b6_heat")
TRAP = "b7_thin_layer"
KAPPAS = (1.5, 2.0, 4.0)
N0_PER_CONTROL = (8, 16, 32)
LEVELS = (2, 3, 4, 5, 6, 7)
SMOKE_REL_TOL = (
    0.5  # fast (smoke) runs use a loose tolerance so that a certificate exists after one short fit
)

# problem -> (M, relative tolerance eps, S_noise)
_TABLE = {
    "b1_poisson": (5.0, 0.01, 0.02),
    "b2_upwind": (1.0, 0.01, 0.02),
    "b4_sde": (0.3, 0.05, 0.1),
    "b5_kink": (1.0, 0.01, 0.02),
    "b6_heat": (0.5, 0.01, 0.02),
    "b7_thin_layer": (0.002, 0.01, 0.02),
}


@dataclass(frozen=True)
class Setup:
    name: str
    bp: BenchmarkProblem
    y_scale: float
    rel_tol: float
    scales: PriorScales
    cost_prior: CostPrior

    @property
    def n_controls(self) -> int:
        return self.bp.problem().inputs.n_controls

    def problem(self, budget: float = 1.0) -> Problem:
        """The gcbml Problem with this tolerance and ``budget`` (core-hours = work units)."""
        return dataclasses.replace(
            self.bp.problem(), tolerance=Tolerance(rel=self.rel_tol), budget=float(budget)
        )

    def truth(self, z: np.ndarray) -> np.ndarray:
        """Converged value at unit inputs z, in the units of the oracle."""
        return self.bp.truth(z) / self.y_scale


def get_setup(name: str, rel_tol: float | None = None) -> Setup:
    m, eps, s_noise = _TABLE[name]
    bp = ALL_PROBLEMS[name]()
    k = bp.problem().resolution.k
    return Setup(
        name=name,
        bp=bp,
        y_scale=m,
        rel_tol=eps if rel_tol is None else rel_tol,
        scales=PriorScales(S_mu=1.0, S_c=0.5, S_delta=0.5, S_noise=s_noise),
        cost_prior=CostPrior(
            k0_mean=0.0, k0_sd=1.0, gamma_mean=(bp.work_gamma / k,) * k, gamma_sd=(0.5,) * k
        ),
    )


def eps_table() -> dict[str, float]:
    return {n: v[1] for n, v in _TABLE.items()}


def default_settings(seed: int = 0) -> CampaignSettings:
    """gcbml with default settings (PROTOCOL Section 2: 'Campaign + run_campaign with default settings')."""
    return CampaignSettings(seed=seed)


def fast_settings(seed: int = 1) -> CampaignSettings:
    """Short chains for smoke runs and tests (the settings of tests/test_campaign.py FAST)."""
    return CampaignSettings(
        n_warmup=40,
        n_samples=40,
        n_chains=2,
        q=6,
        n_candidates_u=4,
        extra_levels=1,
        h_kernels=("twy2",),
        max_draws_acquisition=16,
        seed=seed,
    )
