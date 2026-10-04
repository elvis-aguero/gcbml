r"""B4: Euler-Maruyama for an Ornstein-Uhlenbeck SDE, with Monte-Carlo noise (heteroscedastic).

SDE: dX = -theta X dt + sigma dW, X(0) = x0, horizon T = 1. QoI: E[X_T^2]. Inputs (theta, sigma, x0).

Truth. X_T is Gaussian with mean m = x0 e^{-theta T} and variance v = sigma^2 (1 - e^{-2 theta T}) / (2 theta)
(standard result: Ito isometry; see Gardiner, Handbook of Stochastic Methods; a test checks it
against the limit of the scheme), so

    E[X_T^2] = m^2 + v = x0^2 e^{-2 theta T} + sigma^2 (1 - e^{-2 theta T}) / (2 theta).

Scheme. X_{k+1} = (1 - theta h) X_k + sigma sqrt(h) Z_k, n = T/h steps (h = 1/(4 * 2^level)). It is
linear, so X_n is Gaussian too with m_h = x0 (1 - theta h)^n and v_h obeying v_{k+1} = (1 - theta h)^2 v_k
+ sigma^2 h, v_0 = 0. The exact expectation of the scheme is m_h^2 + v_h, which differs from the truth at
first order in h: weak order 1 (standard result, see Kloeden and Platen, Numerical Solution of SDEs;
a test measures it).
``discrete_expectation`` returns it, so the order is tested without Monte-Carlo noise.

A run averages X_T^2 over N = 2000 paths, so y = m_h^2 + v_h + noise. For a Gaussian X_T,
Var(X_T^2) = 2 v_h^2 + 4 m_h^2 v_h, hence the run-to-run sd is sqrt(Var(X_T^2) / N) (``run_sd``). This
depends strongly on (sigma, x0, theta): the problem is heteroscedastic. Replicates are different seeds.
"""

from __future__ import annotations

import numpy as np

from benchmarks.base import BenchmarkProblem, geometric_levels, grid_cells, make_problem
from gcbml.data import Probe
from gcbml.problem import Problem, ResolutionComponent

T_HORIZON = 1.0
N_PATHS = 2000


def exact_second_moment(theta: float, sigma: float, x0: float) -> float:
    e = np.exp(-2.0 * theta * T_HORIZON)
    return x0**2 * e + sigma**2 * (1.0 - e) / (2.0 * theta)


class OrnsteinUhlenbeckEM(BenchmarkProblem):
    name = "b4_sde"
    expected_order = 1.0
    work_gamma = 1.0
    notes = (
        "Euler-Maruyama for an OU process, QoI E[X_T^2] from 2000 paths. Weak order 1; the run-to-run sd "
        "is large and depends on the inputs (heteroscedastic); replicates are seeds. Cost is linear in "
        "the number of steps."
    )

    def __init__(self, n_paths: int = N_PATHS) -> None:
        self.n_paths = n_paths

    def problem(self) -> Problem:
        return make_problem(
            ("theta", "sigma", "x0"),
            (0.5, 0.2, 0.5),
            (3.0, 1.5, 2.0),
            3,
            [ResolutionComponent("h", geometric_levels(1 / 4))],
        )

    def _moments(self, x, h: float) -> tuple[float, float]:
        theta, sigma, x0 = x
        n = grid_cells(h / T_HORIZON)
        a = 1.0 - theta * h
        m, v = x0 * a**n, 0.0
        for _ in range(n):
            v = a * a * v + sigma**2 * h
        return m, v

    def discrete_expectation(self, x, h: float) -> float:
        """Exact E[X_T^2] of the Euler-Maruyama scheme (no sampling noise)."""
        m, v = self._moments(x, h)
        return m * m + v

    def run_sd(self, x, h: float) -> float:
        """Exact run-to-run standard deviation of y for ``n_paths`` paths."""
        m, v = self._moments(x, h)
        return float(np.sqrt((2.0 * v * v + 4.0 * m * m * v) / self.n_paths))

    def _solve(self, probe: Probe, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
        theta, sigma, x0 = probe.u
        h = probe.h[0]
        n = grid_cells(h / T_HORIZON)
        rng = np.random.default_rng(np.random.SeedSequence([seed, n]))
        x = np.full(self.n_paths, x0)
        amp = sigma * np.sqrt(h)
        for _ in range(n):
            x = (1.0 - theta * h) * x + amp * rng.standard_normal(self.n_paths)
        y = float(np.mean(x * x))
        return np.array([y]), None

    def _work_raw(self, h: tuple[float, ...]) -> float:
        """Work = n steps x a fixed number of paths."""
        return float(grid_cells(h[0]))

    def truth(self, x_unit: np.ndarray) -> np.ndarray:
        x = self.problem().inputs.from_unit(np.atleast_2d(x_unit))
        return np.array([exact_second_moment(*row) for row in x])
