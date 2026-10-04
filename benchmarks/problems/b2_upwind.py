r"""B2: steady 1-D advection-diffusion with first-order upwind, boundary layer at the outlet.

Equation: u' = eps u'' + s on (0,1), u(0) = 0, u(1) = g, with eps = 1/Pe. Solution: with
E(x) = expm1(x/eps) / expm1(1/eps),

    u(x) = s (x - E(x)) + g E(x).

Check: u' - eps u'' = s because the homogeneous part E solves E' = eps E'' and x solves x' = 1.
E is convex with E(0) = 0, E(1) = 1, so E(x) <= x and u > 0 for s, g > 0.

Scheme: backward (upwind) difference for u' and central for u'', solved as a banded system.
QoI: u(15/16), a point inside the boundary layer region for Pe >~ 16 and a node on every grid
(h = 1/(16 * 2^level)).

Pre-asymptotic behaviour: the discrete homogeneous solution is geometric with ratio r = 1 + h/eps
(derived by inserting r^i in the scheme), against exp(h/eps) for the exact one. The relative
difference is h/(2 eps) + ..., so the error is first order only once the cell Peclet number h/eps
is small. For Pe = 100 and h = 1/16 it is 6.25. For small Pe the first-order range starts at once.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from benchmarks.base import BenchmarkProblem, Timer, geometric_levels, grid_cells, make_problem
from gcbml.data import Probe, RunResult
from gcbml.problem import Problem, ResolutionComponent

X_Q = 15.0 / 16.0


def exact_u(x: float, pe: float, s: float, g: float) -> float:
    e = np.expm1(x * pe) / np.expm1(pe)
    return s * (x - e) + g * e


class UpwindBoundaryLayer(BenchmarkProblem):
    name = "b2_upwind"
    expected_order = 1.0
    cost_gamma_range = (0.0, 2.0)
    notes = (
        "Upwind advection-diffusion with an outlet boundary layer. Order 1 once the cell Peclet number "
        "h*Pe is small; pre-asymptotic on coarse grids at high Pe. Cost is tiny (tridiagonal solve): "
        "the measured growth is dominated by per-call overhead until level ~6."
    )

    def problem(self) -> Problem:
        return make_problem(
            ("peclet", "source", "outlet_value"),
            (2.0, 0.5, 0.5),
            (100.0, 2.0, 1.5),
            3,
            [ResolutionComponent("h", geometric_levels(1 / 16))],
        )

    def run(self, probe: Probe, seed: int) -> RunResult:
        pe, s, g = probe.u
        n = grid_cells(probe.h[0])
        with Timer() as t:
            u_q = _solve(pe, s, g, n)
        return self._result(probe, u_q, t.elapsed)

    def truth(self, x_unit: np.ndarray) -> np.ndarray:
        x = self.problem().inputs.from_unit(np.atleast_2d(x_unit))
        return np.array([exact_u(X_Q, *row) for row in x])


def _solve(pe: float, s: float, g: float, n: int) -> float:
    h, eps = 1.0 / n, 1.0 / pe
    m = n - 1
    lo, di, up = -1.0 / h - eps / h**2, 1.0 / h + 2 * eps / h**2, -eps / h**2
    ab = np.zeros((3, m))
    ab[0, 1:] = up
    ab[1, :] = di
    ab[2, :-1] = lo
    rhs = np.full(m, s)
    rhs[-1] -= up * g  # u_N = g enters the last equation; u_0 = 0 contributes nothing
    u = solve_banded((1, 1), ab, rhs)
    return float(u[int(round(X_Q * n)) - 1])
