"""Problem definition (spec Section 1).

A problem is: an input space X with a region of interest Sigma; a resolution h with one or
more components; a tolerance on the epistemic uncertainty of the converged value; and a
compute budget. Nothing here is specific to any application.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class InputSpace:
    """Inputs x = (u, v), scaled to [0, 1] inside every kernel.

    ``u`` are the controls a run sets (the first ``n_controls`` coordinates). ``v`` are output
    coordinates that one run returns many values of (module S1, e.g. a threshold level). ``v``
    can be empty (``n_controls == len(names)``).
    """

    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    n_controls: int

    def __post_init__(self) -> None:
        if not (len(self.names) == len(self.lower) == len(self.upper)):
            raise ValueError("names, lower and upper must have the same length")
        if not 1 <= self.n_controls <= len(self.names):
            raise ValueError("n_controls must be in [1, d]")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("lower < upper is required for every input")

    @property
    def d(self) -> int:
        return len(self.names)

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        return (np.asarray(x, float) - lo) / (hi - lo)

    def from_unit(self, z: np.ndarray) -> np.ndarray:
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        return lo + np.asarray(z, float) * (hi - lo)


@dataclass(frozen=True)
class ResolutionComponent:
    """One resolution component h_j (a cell size, a time step, ...).

    ``values`` are the allowed values, coarsest first (strictly decreasing). Values finer than the
    last one are generated on demand by dividing by ``refine_ratio`` (finer levels are always
    allowed; the budget is the only limit). ``h_c = values[0]`` is fixed at Step 0 and never changes,
    so hbar = h / h_c keeps its meaning when a level is later dropped.
    """

    name: str
    values: tuple[float, ...]
    refine_ratio: float = 2.0

    def __post_init__(self) -> None:
        v = np.asarray(self.values, float)
        if v.size == 0 or np.any(v <= 0) or np.any(np.diff(v) >= 0):
            raise ValueError("values must be positive and strictly decreasing")
        if self.refine_ratio <= 1:
            raise ValueError("refine_ratio must be > 1")

    @property
    def h_c(self) -> float:
        return self.values[0]

    def level_value(self, level: int) -> float:
        """h at integer level (0 = coarsest). Levels beyond the listed values extend by refine_ratio."""
        if level < 0:
            raise ValueError("level must be >= 0")
        if level < len(self.values):
            return self.values[level]
        return self.values[-1] / self.refine_ratio ** (level - len(self.values) + 1)


@dataclass(frozen=True)
class ResolutionSpec:
    components: tuple[ResolutionComponent, ...]

    @property
    def k(self) -> int:
        return len(self.components)

    def hbar(self, h: np.ndarray) -> np.ndarray:
        """Scaled resolution hbar_j = h_j / h_{j,c}. h has shape (..., k)."""
        hc = np.array([c.h_c for c in self.components])
        return np.asarray(h, float) / hc

    def h_at(self, levels: tuple[int, ...]) -> np.ndarray:
        return np.array([c.level_value(lev) for c, lev in zip(self.components, levels)])


@dataclass(frozen=True)
class Tolerance:
    """Tolerance epsilon(x) on the physical scale: rel * m_y(x), or abs. Exactly one is set."""

    rel: float | None = None
    abs: float | None = None

    def __post_init__(self) -> None:
        if (self.rel is None) == (self.abs is None):
            raise ValueError("set exactly one of rel and abs")


@dataclass(frozen=True)
class Problem:
    """Spec Section 1. ``region_lower/upper`` define Sigma (defaults to all of X).

    ``sigma_n`` is the evaluation set Sigma_N: a scrambled Sobol set of 100 d points in Sigma,
    in unit coordinates of X. Every "max over Sigma" is computed on it.
    """

    inputs: InputSpace
    resolution: ResolutionSpec
    tolerance: Tolerance
    budget: float
    transforms: tuple[str, ...] = ("identity", "log")
    region_lower: tuple[float, ...] | None = None
    region_upper: tuple[float, ...] | None = None
    seed: int = 0
    _sigma_n: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        lo = np.asarray(self.region_lower if self.region_lower is not None else self.inputs.lower, float)
        hi = np.asarray(self.region_upper if self.region_upper is not None else self.inputs.upper, float)
        d = self.inputs.d
        m = int(np.ceil(np.log2(100 * d)))  # draw 2^m points (balanced), keep the first 100 d
        sob = qmc.Sobol(d, scramble=True, seed=self.seed).random_base2(m)[: 100 * d]
        object.__setattr__(self, "_sigma_n", self.inputs.to_unit(lo + sob * (hi - lo)))

    @property
    def sigma_n(self) -> np.ndarray:
        return self._sigma_n
