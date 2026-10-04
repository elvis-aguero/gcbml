r"""B6: heat equation with two independent resolution components (k = 2): dx and dt.

Equation u_t = u_xx on (0,1), u(0,t) = u(1,t) = 0, u(x,0) = sin(pi x) + a sin(3 pi x) + b sin(5 pi x).
Inputs (T, a, b). QoI: u(1/2, T).

Truth. Each sine mode decays as exp(-(k pi)^2 t), so

    u(1/2, T) = e^{-pi^2 T} - a e^{-9 pi^2 T} + b e^{-25 pi^2 T},

because sin(k pi / 2) = 1, -1, 1 for k = 1, 3, 5.

Scheme. Central differences in space, M = 1/dx cells; backward Euler in time with n = 1/h_t steps of
size dt = T h_t (so h_t = dt / T is the resolution component: the step as a fraction of the horizon).
Initial data are sampled at the nodes, so each sine mode is a discrete eigenvector and the discrete
decay factor of mode k is (1 + dt mu_k)^(-n), mu_k = (2 - 2 cos(k pi dx)) / dx^2. Compared with
exp(-(k pi)^2 T): relative error in mu_k is -(k pi dx)^2 / 12 (order 2 in dx) and the backward Euler
error is (1/2) lambda^2 T dt (order 1 in dt). Both levels start at 4 (dx = 1/4, h_t = 1/4).
The error is the sum of the two contributions, so tests isolate each with the other one very fine.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from benchmarks.base import BenchmarkProblem, geometric_levels, grid_cells, make_problem
from gcbml.data import Probe
from gcbml.problem import Problem, ResolutionComponent


class HeatTwoResolutions(BenchmarkProblem):
    name = "b6_heat"
    expected_order = 1.0  # the lower of the two; per component see expected_orders
    n_resolution = 2
    work_gamma = 2.0
    notes = (
        "Heat equation, central differences in space (order 2 in dx) and backward Euler in time (order 1 "
        "in dt = T h_t). Two independent resolution components, h = (dx, h_t). QoI u(1/2, T)."
    )

    @property
    def expected_orders(self) -> tuple[float, ...]:
        return (2.0, 1.0)

    def problem(self) -> Problem:
        return make_problem(
            ("horizon", "mode3_weight", "mode5_weight"),
            (0.02, 0.0, 0.0),
            (0.1, 1.0, 1.0),
            3,
            [
                ResolutionComponent("dx", geometric_levels(1 / 4)),
                ResolutionComponent("dt_over_T", geometric_levels(1 / 4)),
            ],
        )

    def _solve(self, probe: Probe, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
        horizon, a, b = probe.u
        m, n = grid_cells(probe.h[0]), grid_cells(probe.h[1])
        y = _solve(horizon, a, b, m, n)
        return np.array([y]), None

    def _work_raw(self, h: tuple[float, ...]) -> float:
        """Work = M cells x n time steps (gamma 2 refining both components, 1 for each alone)."""
        return float(grid_cells(h[0]) * grid_cells(h[1]))

    def truth(self, x_unit: np.ndarray) -> np.ndarray:
        t, a, b = self.problem().inputs.from_unit(np.atleast_2d(x_unit)).T
        p2 = np.pi**2
        return np.exp(-p2 * t) - a * np.exp(-9 * p2 * t) + b * np.exp(-25 * p2 * t)


def _solve(horizon: float, a: float, b: float, m: int, n: int) -> float:
    dx, dt = 1.0 / m, horizon / n
    x = np.arange(1, m) * dx
    u = np.sin(np.pi * x) + a * np.sin(3 * np.pi * x) + b * np.sin(5 * np.pi * x)
    lap = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m - 1, m - 1), format="csc") / dx**2
    lu = spla.splu((sp.identity(m - 1, format="csc") + dt * lap).tocsc())
    for _ in range(n):
        u = lu.solve(u)
    return float(u[m // 2 - 1])
