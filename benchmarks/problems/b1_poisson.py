r"""B1: anisotropic Poisson problem on the unit square, 5-point finite differences.

Equation: -(u_xx + kappa u_yy) = f on (0,1)^2, with Dirichlet data taken from the manufactured solution

    u(x, y) = A exp(w x + (w/2) y).

Then u_xx = w^2 u and u_yy = (w^2/4) u, so f = -w^2 (1 + kappa/4) u. Inputs: amplitude A, rate w
(sets how sharply the solution grows, hence the size of the truncation error) and anisotropy kappa.

QoI: the integral of u over the square, evaluated from the grid solution with the composite
trapezoid rule (boundary values exact). Truth, from the product of two exponential integrals:

    Q = A (e^w - 1)/w * (e^{w/2} - 1)/(w/2).

Both the scheme (truncation error (h^2/12)(u_xxxx + kappa u_yyyy)) and the trapezoid rule are
second order, so Q_h - Q = O(h^2). Resolution: h = 1/N, N = 4 * 2^level.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from benchmarks.base import BenchmarkProblem, geometric_levels, grid_cells, make_problem
from gcbml.data import Probe
from gcbml.problem import Problem, ResolutionComponent


def exact_integral(amp: float, w: float) -> float:
    return amp * np.expm1(w) / w * np.expm1(w / 2.0) / (w / 2.0)


class Poisson2D(BenchmarkProblem):
    name = "b1_poisson"
    expected_order = 2.0
    work_gamma = 3.0
    notes = (
        "Anisotropic Poisson, manufactured exponential solution, 5-point stencil with a sparse LU solve. "
        "QoI is the trapezoid integral of the solution. Clean second order; cost per halving of h is "
        "about 2^3 (sparse direct solve)."
    )

    def problem(self) -> Problem:
        return make_problem(
            ("amplitude", "rate", "anisotropy"),
            (0.5, 1.0, 0.1),
            (2.0, 3.0, 1.0),
            3,
            [ResolutionComponent("h", geometric_levels(1 / 4))],
        )

    def _solve(self, probe: Probe, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
        amp, w, kap = probe.u
        n = grid_cells(probe.h[0])
        q = _solve(amp, w, kap, n)
        return np.array([q]), None

    def _work_raw(self, h: tuple[float, ...]) -> float:
        """Work = N^3: sparse LU of the 5-point matrix (n = N^2 unknowns, O(n^1.5) flops)."""
        return float(grid_cells(h[0]) ** 3)

    def truth(self, x_unit: np.ndarray) -> np.ndarray:
        x = self.problem().inputs.from_unit(np.atleast_2d(x_unit))
        return exact_integral(x[:, 0], x[:, 1])


def _solve(amp: float, w: float, kap: float, n: int) -> float:
    h = 1.0 / n
    s = np.arange(n + 1) * h
    ex, ey = np.exp(w * s), np.exp(0.5 * w * s)
    ue = amp * ex[:, None] * ey[None, :]  # ue[i, j] = u(x_i, y_j)
    m = n - 1
    t1 = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m)) / h**2
    eye = sp.identity(m)
    a = (sp.kron(t1, eye) + kap * sp.kron(eye, t1)).tocsc()  # unknown index i * m + j
    f = -(w**2) * (1.0 + 0.25 * kap) * ue[1:-1, 1:-1]
    rhs = f.copy()
    rhs[0, :] += ue[0, 1:-1] / h**2
    rhs[-1, :] += ue[-1, 1:-1] / h**2
    rhs[:, 0] += kap * ue[1:-1, 0] / h**2
    rhs[:, -1] += kap * ue[1:-1, -1] / h**2
    u = ue.copy()
    u[1:-1, 1:-1] = spla.spsolve(a, rhs.ravel()).reshape(m, m)
    wt = np.ones(n + 1)
    wt[[0, -1]] = 0.5
    return float(h * h * wt @ u @ wt)
