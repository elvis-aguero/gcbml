r"""B5: oscillatory convergence, midpoint rule on a function with a jump at an input-dependent place.

Integrand f(x) = sin(w x) + j H(x - a) on (0,1) (H the Heaviside step), inputs (a, w, j) with
a the jump location. Truth:

    Q = (1 - cos w) / w + j (1 - a).

Scheme: composite midpoint rule with N = 1/h cells (h = 1/(4 * 2^level)). The smooth part has the
usual error (h^2/24) (f'(1) - f'(0)) + O(h^4).
The jump part: the sum of j h over midpoints m_i = (i + 1/2) h > a, i = 0..N-1, is j h (N - floor(a/h + 1/2)),
and j (1 - a) is the exact value, so the jump error is

    E_jump = Q_h - Q = j (a - h floor(a/h + 1/2)) = j h xi,   xi = a/h - floor(a/h + 1/2) in [-1/2, 1/2).

xi depends on the binary digits of a (h halves at each level, so a/h doubles): the error changes sign
and its magnitude is not monotone in h. It is O(h) only as an upper bound. The error never settles on
a rate, it only has the envelope j h / 2. This is the point of the problem.
"""

from __future__ import annotations

import numpy as np

from benchmarks.base import BenchmarkProblem, geometric_levels, grid_cells, make_problem
from gcbml.data import Probe
from gcbml.problem import Problem, ResolutionComponent


class MidpointOnJump(BenchmarkProblem):
    name = "b5_kink"
    expected_order = None
    work_gamma = 1.0
    notes = (
        "Midpoint rule on sin(w x) + j H(x - a). The error is j h xi(a/h), with xi in [-1/2, 1/2) set by "
        "the binary digits of a: it changes sign and is not monotone as h halves. No convergence order "
        "exists, only the O(h) envelope."
    )

    def problem(self) -> Problem:
        return make_problem(
            ("jump_location", "frequency", "jump_size"),
            (0.2, 1.0, 0.5),
            (0.8, 4.0, 2.0),
            3,
            [ResolutionComponent("h", geometric_levels(1 / 4))],
        )

    def _solve(self, probe: Probe, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
        a, w, j = probe.u
        n = grid_cells(probe.h[0])
        mid = (np.arange(n) + 0.5) / n
        q = float(np.mean(np.sin(w * mid) + j * (mid > a)))
        return np.array([q]), None

    def _work_raw(self, h: tuple[float, ...]) -> float:
        """Work = N midpoint evaluations."""
        return float(grid_cells(h[0]))

    def truth(self, x_unit: np.ndarray) -> np.ndarray:
        a, w, j = self.problem().inputs.from_unit(np.atleast_2d(x_unit)).T
        return (1.0 - np.cos(w)) / w + j * (1.0 - a)
