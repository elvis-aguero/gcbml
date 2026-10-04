r"""B7: a trap. A feature thinner than every affordable grid, so coarse levels look converged.

Problem: steady advection u' = S(x), u(0) = 0, with a narrow Gaussian source
S(x) = A exp(-(x - a)^2 / (2 w^2)), solved with first-order upwind (backward difference,
(u_i - u_{i-1})/h = S(x_i), a lower-bidiagonal system). QoI: the outlet value u(1). Inputs (a, A, w);
w is 0.3 to 0.9 thousandths of the domain.

Truth: u(1) = A w sqrt(pi/2) [erf((1-a)/(sqrt(2) w)) + erf(a/(sqrt(2) w))] (the Gaussian integral
over (0,1); the erf terms equal 2 to machine precision here).

The scheme is a right Riemann sum of S. While h is much larger than w the grid nodes miss the
Gaussian and the sum is about zero (or a stray A h exp(-d^2 / (2 w^2)) when a node falls within a
few w of a). Successive coarse levels all give almost the same near-zero value, so they look
converged, with an apparent error of almost zero, while the truth is A w sqrt(2 pi). The Riemann sum
of a smooth function converges spectrally (error about 2 exp(-2 pi^2 w^2 / h^2) for the trapezoid
sum) so it jumps to the truth when h reaches w. For h0 = 1/8 this happens between level 6 (h = 1/512) and
level 9 (h = 1/4096). A method that reports success from levels 0..5 is wrong.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erf

from benchmarks.base import BenchmarkProblem, geometric_levels, grid_cells, make_problem
from gcbml.data import Probe
from gcbml.problem import Problem, ResolutionComponent


class ThinLayerTrap(BenchmarkProblem):
    name = "b7_thin_layer"
    expected_order = None
    work_gamma = 1.0
    asymptotic_level = 8  # worst case over the region: every input is within 1% of truth from here on
    notes = (
        "TRAP. Upwind advection with a Gaussian source of width ~5e-4. Levels 0-5 miss the source and "
        "agree with each other near zero; the asymptotic (spectral) range begins at level ~7-9. "
        "The method must not claim success from the coarse levels."
    )

    def problem(self) -> Problem:
        return make_problem(
            ("position", "amplitude", "width"),
            (0.2, 0.5, 3e-4),
            (0.8, 2.0, 9e-4),
            3,
            [ResolutionComponent("h", geometric_levels(1 / 8))],
        )

    def _solve(self, probe: Probe, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
        a, amp, w = probe.u
        n = grid_cells(probe.h[0])
        h = 1.0 / n
        xs = np.arange(1, n + 1) * h
        src = amp * np.exp(-((xs - a) ** 2) / (2 * w * w))
        ab = np.zeros((2, n))
        ab[0, :] = 1.0 / h
        ab[1, :-1] = -1.0 / h
        u = solve_banded((1, 0), ab, src)
        y = float(u[-1])
        return np.array([y]), None

    def _work_raw(self, h: tuple[float, ...]) -> float:
        """Work = N: bidiagonal solve."""
        return float(grid_cells(h[0]))

    def truth(self, x_unit: np.ndarray) -> np.ndarray:
        a, amp, w = self.problem().inputs.from_unit(np.atleast_2d(x_unit)).T
        s2 = np.sqrt(2.0) * w
        return amp * w * np.sqrt(np.pi / 2.0) * (erf((1.0 - a) / s2) + erf(a / s2))
