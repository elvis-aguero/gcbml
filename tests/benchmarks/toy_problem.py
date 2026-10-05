"""A tiny benchmark problem with a known answer for the machinery tests (not one of B1-B7).

y = 1 + a + b^2 + h^2 / 2 on the unit square; truth 1 + a + b^2; work = N = 1/h (coarsest level N = 4
costs 1), no cost noise, so the cost of a design is a sum of powers of two that the tests compute by hand.
"""

import numpy as np

from benchmarks.base import BenchmarkProblem, geometric_levels, grid_cells, make_problem
from benchmarks.config import Setup
from gcbml.cost import CostPrior
from gcbml.priors import PriorScales
from gcbml.problem import ResolutionComponent


class Toy(BenchmarkProblem):
    name = "toy"
    expected_order = 2.0
    work_gamma = 1.0
    sigma_c = 0.0
    notes = "Quadratic bowl plus a h^2 grid error; exact truth; work N."

    def problem(self):
        return make_problem(
            ("a", "b"), (0.0, 0.0), (1.0, 1.0), 2, [ResolutionComponent("h", geometric_levels(1 / 4))]
        )

    def _solve(self, probe, seed):
        a, b = probe.u
        return np.array([1.0 + a + b**2 + 0.5 * probe.h[0] ** 2]), None

    def _work_raw(self, h):
        return float(grid_cells(h[0]))

    def truth(self, x_unit):
        z = np.atleast_2d(x_unit)
        return 1.0 + z[:, 0] + z[:, 1] ** 2


def toy_setup(rel_tol: float = 0.5) -> Setup:
    return Setup(
        name="toy",
        bp=Toy(),
        y_scale=1.0,
        rel_tol=rel_tol,
        scales=PriorScales(S_mu=1.0, S_c=0.5, S_delta=0.5, S_noise=0.02),
        cost_prior=CostPrior(0.0, 1.0, (1.0,), (0.5,)),
    )


def family_cost(n0: int, top: int) -> float:
    """Cost of the static design (n_l = max(3, n0 // 2^l) sites at level l, work 2^l), by hand."""
    return float(sum(min(n0, max(3, n0 // 2**lev)) * 2**lev for lev in range(top + 1)))
