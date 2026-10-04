r"""B3: mixing-time analogue (module S1), 1-D diffusion of a segregated scalar.

Setup: c_t = D c_xx on (0,1) with no-flux walls. Initial state: c = 1 for x < L0, 0 otherwise
(segregated layers; mean L0). Degree of mixing: chi(t) = 1 - Var(t) / Var0, Var0 = L0 (1 - L0).

Closed form. The cosine series of the initial state is c0 = L0 + sum_n a_n cos(n pi x),
a_n = 2 sin(n pi L0) / (n pi). Each mode decays as exp(-D (n pi)^2 t), the mean square of
cos(n pi x) is 1/2, so

    Var(t) = sum_n (a_n^2 / 2) exp(-2 D (n pi)^2 t)
           = sum_n 2 sin^2(n pi L0) / (n pi)^2 * exp(-2 D (n pi)^2 t).

(At t = 0 Parseval gives Var0 = L0 (1 - L0); a test checks chi against a direct variance of the
series.) Var decreases monotonically in t, so the time to reach chi = v is the unique root of
chi(t) = v; truth() finds it with brentq
using 4000 modes (the terms are below 1e-300 for the smallest t involved).

Inputs: x = (D, L0, v). Controls u = (D, L0); one run returns the time to reach chi = v for
v in {0.5, 0.75, 0.9, 0.95} (the v coordinate, so n_controls = 2 < d = 3).

Scheme: cell-centred finite volumes (N = 1/h cells, second order in space, initial state
cell-averaged so the mean is conserved) and implicit Euler in time with dt = T_final * h
(N steps; first order in time, so the combined order is 1 with a small spatial part). chi uses the exact
Var0. The threshold time is found by linear interpolation of chi between steps (error O(dt^2)).
Outputs not reached by T_final = 0.2 are censored: y = T_final, a lower bound. truth() still
returns the converged time.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq

from benchmarks.base import BenchmarkProblem, Timer, geometric_levels, grid_cells, make_problem
from gcbml.data import Probe, RunResult
from gcbml.problem import Problem, ResolutionComponent

T_FINAL = 0.2
V_LEVELS = (0.5, 0.75, 0.9, 0.95)
N_MODES = 4000


def exact_chi(t: float, d: float, l0: float) -> float:
    n = np.arange(1, N_MODES + 1)
    w = 2.0 * np.sin(n * np.pi * l0) ** 2 / (n * np.pi) ** 2
    return 1.0 - float(w @ np.exp(-2.0 * d * (n * np.pi) ** 2 * t)) / (l0 * (1.0 - l0))


def exact_time(v: float, d: float, l0: float) -> float:
    return brentq(lambda t: exact_chi(t, d, l0) - v, 1e-9, 20.0 / d, xtol=1e-14, rtol=1e-13)


class MixingTime(BenchmarkProblem):
    name = "b3_mixing"
    expected_order = 1.0
    n_outputs = len(V_LEVELS)
    cost_gamma_range = (0.8, 3.0)
    t_final = T_FINAL
    notes = (
        "Time to reach mixing degree chi in {0.5, 0.75, 0.9, 0.95} for 1-D diffusion of two segregated "
        "layers (S1 outputs). Finite volumes + implicit Euler with dt tied to h; order 1. Slow mixing "
        "(small D) leaves the high thresholds unreached at T_final = 0.2: those outputs are censored."
    )

    def problem(self) -> Problem:
        return make_problem(
            ("diffusivity", "fill_fraction", "chi"),
            (0.5, 0.2, 0.5),
            (2.0, 0.5, 0.95),
            2,
            [ResolutionComponent("h", geometric_levels(1 / 8))],
        )

    def output_coords(self, u) -> np.ndarray:
        u = np.asarray(u, float)
        return np.column_stack([np.tile(u, (len(V_LEVELS), 1)), np.asarray(V_LEVELS)])

    def run(self, probe: Probe, seed: int) -> RunResult:
        d, l0 = probe.u
        n = grid_cells(probe.h[0])
        with Timer() as t:
            y, cens = _solve(d, l0, n)
        return self._result(probe, y, t.elapsed, cens)

    def truth(self, x_unit: np.ndarray) -> np.ndarray:
        x = self.problem().inputs.from_unit(np.atleast_2d(x_unit))
        return np.array([exact_time(v, d, l0) for d, l0, v in x])

    def valid_outputs(self, z: np.ndarray) -> np.ndarray:
        """Outputs safely inside the time window (not censored at any level near the truth)."""
        return self.truth_for_controls(z) < 0.8 * T_FINAL


def _solve(d: float, l0: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    h = 1.0 / n
    dt = T_FINAL * h
    edges = np.arange(n + 1) * h
    c = np.clip((l0 - edges[:-1]) / h, 0.0, 1.0)  # cell averages of the step
    lap = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="lil") / h**2
    lap[0, 0] = -1.0 / h**2
    lap[n - 1, n - 1] = -1.0 / h**2
    lu = spla.splu((sp.identity(n) - dt * d * lap.tocsc()).tocsc())
    var0 = l0 * (1.0 - l0)
    chi = np.empty(n + 1)
    chi[0] = 1.0 - h * np.sum((c - l0) ** 2) / var0
    for k in range(n):
        c = lu.solve(c)
        chi[k + 1] = 1.0 - h * np.sum((c - l0) ** 2) / var0
    y = np.full(len(V_LEVELS), T_FINAL)
    cens = np.ones(len(V_LEVELS), bool)
    for i, v in enumerate(V_LEVELS):
        idx = np.nonzero(chi >= v)[0]
        if idx.size:
            k = idx[0]
            y[i] = (k - 1 + (v - chi[k - 1]) / (chi[k] - chi[k - 1])) * dt if k > 0 else 0.0
            cens[i] = False
    return y, cens
