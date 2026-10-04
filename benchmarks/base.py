"""Common interface of the benchmark problems.

A benchmark problem is a cheap NumPy/SciPy solver whose grid error is produced by a real
discretisation, together with the converged answer (h -> 0) known in closed form or certified.

Conventions shared by every problem:
- ``Probe.u`` holds the physical control values, ``Probe.h`` the physical resolution values (for
  example a cell size 1/N). The solver maps h to a grid by rounding 1/h to an integer.
- ``RunResult.cost`` is deterministic work, not CPU time: ``work(probe) * exp(sigma_c * N(0,1))``, with
  ``work`` the documented operation count of the discretisation normalised so that the coarsest level
  costs 1 work unit, and the normal draw seeded by (probe, seed). ``sigma_c`` is a class attribute
  (0.1 by default, 0 disables the noise). Measured CPU seconds (``time.process_time``) are available
  for information through ``measure_cpu``; they are dominated by call overhead for the 1-D problems.
- ``truth`` takes unit inputs of shape (n, d) (output coordinates v included for S1 problems) and
  returns the converged value for each row, shape (n,).
"""

from __future__ import annotations

import time
import zlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np

from gcbml.data import Probe, RunResult
from gcbml.problem import InputSpace, Problem, ResolutionComponent, ResolutionSpec, Tolerance

BUDGET_PLACEHOLDER = 1.0


def make_problem(
    names: Sequence[str],
    lower: Sequence[float],
    upper: Sequence[float],
    n_controls: int,
    components: Sequence[ResolutionComponent],
    rel_tol: float = 0.01,
) -> Problem:
    """Assemble a gcbml Problem with a relative tolerance and a placeholder budget."""
    inputs = InputSpace(
        tuple(names), tuple(float(t) for t in lower), tuple(float(t) for t in upper), n_controls
    )
    return Problem(
        inputs=inputs,
        resolution=ResolutionSpec(tuple(components)),
        tolerance=Tolerance(rel=rel_tol),
        budget=BUDGET_PLACEHOLDER,
    )


def geometric_levels(h0: float, n: int = 8, ratio: float = 2.0) -> tuple[float, ...]:
    """h0, h0/ratio, ... (n values, coarsest first)."""
    return tuple(h0 / ratio**i for i in range(n))


class Timer:
    """CPU-time stopwatch around the solve (``time.process_time``)."""

    def __enter__(self) -> Timer:
        self._t0 = time.process_time()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.process_time() - self._t0


def grid_cells(h: float) -> int:
    """Number of cells N = 1/h (h must be 1/integer up to rounding)."""
    n = int(round(1.0 / h))
    if n < 1 or abs(n * h - 1.0) > 1e-9:
        raise ValueError(f"h = {h} is not 1/integer")
    return n


class BenchmarkProblem(ABC):
    """One benchmark: a Problem description, a solver (``run``) and the converged answer (``truth``)."""

    name: str
    expected_order: float | None
    notes: str
    n_resolution: int = 1  # k, number of resolution components
    n_outputs: int = 1  # outputs per run (> 1 for S1 problems)
    work_gamma: float  # work grows as 2^(work_gamma * level) when every resolution component is refined
    sigma_c: float = 0.1  # sd of the log of the multiplicative cost noise; 0 disables it

    @property
    def expected_orders(self) -> tuple[float, ...] | None:
        """Order per resolution component (defaults to ``expected_order`` for k = 1)."""
        return None if self.expected_order is None else (self.expected_order,) * self.n_resolution

    @abstractmethod
    def problem(self) -> Problem: ...

    @abstractmethod
    def _solve(self, probe: Probe, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Run the discretisation; return the outputs and the censoring flags (or None)."""

    @abstractmethod
    def _work_raw(self, h: tuple[float, ...]) -> float:
        """Operation count (up to a constant) of one run at resolution h."""

    def work(self, probe: Probe) -> float:
        """Documented work of a run, in work units: the coarsest level of the problem costs 1."""
        h_c = tuple(c.h_c for c in self.problem().resolution.components)
        return self._work_raw(tuple(probe.h)) / self._work_raw(h_c)

    def cost(self, probe: Probe, seed: int) -> float:
        """work(probe) * exp(sigma_c * N(0,1)), the normal seeded by (probe, seed)."""
        w = self.work(probe)
        if self.sigma_c == 0:
            return w
        key = zlib.crc32(repr((probe.probe_id, probe.u, probe.h)).encode())
        z = np.random.default_rng(np.random.SeedSequence([seed, key])).standard_normal()
        return float(w * np.exp(self.sigma_c * z))

    def run(self, probe: Probe, seed: int) -> RunResult:
        y, cens = self._solve(probe, seed)
        return self._result(probe, y, self.cost(probe, seed), cens)

    def measure_cpu(self, probe: Probe, seed: int) -> float:
        """CPU seconds of the solve (information only; not the cost the budget is set against)."""
        with Timer() as t:
            self._solve(probe, seed)
        return t.elapsed

    @abstractmethod
    def truth(self, x_unit: np.ndarray) -> np.ndarray: ...

    # ---- helpers shared by problems and tests

    def output_coords(self, u: Sequence[float]) -> np.ndarray:
        """Physical coordinates (m, d) of the outputs of a run at controls u (u repeated, v appended)."""
        return np.asarray(u, float)[None, :]

    def truth_for_controls(self, z: np.ndarray) -> np.ndarray:
        """Converged values (m,) at the outputs of a run, from unit inputs z (only the controls are used)."""
        inp = self.problem().inputs
        u = inp.from_unit(np.asarray(z, float))[: inp.n_controls]
        return self.truth(inp.to_unit(self.output_coords(u)))

    def _result(
        self, probe: Probe, y: np.ndarray, cost: float, censored: np.ndarray | None = None
    ) -> RunResult:
        y = np.atleast_1d(np.asarray(y, float))
        cens = np.zeros(y.shape, bool) if censored is None else np.asarray(censored, bool)
        return RunResult(
            probe=probe,
            x=self.output_coords(probe.u),
            y=y,
            censored=cens,
            cost=float(cost),
            cost_censored=False,
        )


def values_at_levels(
    bp: BenchmarkProblem, z: np.ndarray, levels: Iterable[int | tuple[int, ...]], seed: int = 0
) -> np.ndarray:
    """Run ``bp`` at unit inputs z for each level (an int, or one int per resolution component).

    Returns the outputs, shape (n_levels, m).
    """
    p = bp.problem()
    u = tuple(float(t) for t in p.inputs.from_unit(np.asarray(z, float))[: p.inputs.n_controls])
    out = []
    for lev in levels:
        lev_t = (lev,) * p.resolution.k if isinstance(lev, int | np.integer) else tuple(lev)
        probe = Probe(f"l{lev_t}", u, tuple(float(t) for t in p.resolution.h_at(lev_t)))
        out.append(bp.run(probe, seed).y)
    return np.asarray(out)


def observed_orders(err: np.ndarray, ratio: float = 2.0) -> np.ndarray:
    """log_ratio(err_l / err_{l+1}) for errors of shape (L, m); returns (L-1, m)."""
    err = np.asarray(err, float)
    return np.log(err[:-1] / err[1:]) / np.log(ratio)


def self_convergence_orders(vals: np.ndarray, ratio: float = 2.0) -> np.ndarray:
    """Order from three successive levels, without the truth: log_r(|v_l - v_{l-1}| / |v_{l+1} - v_l|)."""
    d = np.abs(np.diff(np.asarray(vals, float)))
    return np.log(d[:-1] / d[1:]) / np.log(ratio)
