"""Oracle adapter: any BenchmarkProblem behind the gcbml Oracle protocol (quote / submit / poll).

Cost of a run: ``bp.cost(probe, run_seed)``, the deterministic work of the discretisation times a seeded
lognormal factor (benchmarks/base.py). A run whose cost would exceed its cap is stopped at the cap: the
result has cost = cap, cost_censored = True and no outputs (the generic one-output case of spec Step 5).
The solver is not run in that case.

Determinism: the seed of a run is a function of (oracle seed, probe_id) only, so the result of a probe does
not depend on the order of submission, and replicates (distinct probe ids at the same u and h) get distinct
noise (B4 draws its paths from this seed).

Quotes: ``quote`` returns None for every probe by default (no information beyond the cost model's own).
With ``quote_work=True`` it returns the documented work of the probe, a noise-free quote.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np

from benchmarks.base import BenchmarkProblem
from gcbml.data import Probe, RunResult


def run_seed(seed: int, probe_id: str) -> int:
    """Seed of one run: a function of the oracle seed and the probe id only."""
    key = zlib.crc32(probe_id.encode())
    return int(np.random.SeedSequence([int(seed), key]).generate_state(1, dtype=np.uint32)[0])


class BenchmarkOracle:
    """Results are immediate: ``submit`` computes them, ``poll`` hands them over."""

    def __init__(self, bp: BenchmarkProblem, seed: int = 0, quote_work: bool = False, y_scale: float = 1.0):
        self.bp = bp
        self.y_scale = float(y_scale)  # outputs are y / y_scale (a choice of units, benchmarks/config.py)
        self.seed = int(seed)
        self.quote_work = quote_work
        self._done: list[RunResult] = []
        self.n_submitted = 0
        self.n_capped = 0

    def quote(self, probes: Sequence[Probe]) -> list[float | None]:
        if not self.quote_work:
            return [None] * len(probes)
        return [float(self.bp.work(p)) for p in probes]

    def run(self, probe: Probe, cap: float = np.inf) -> RunResult:
        """One run with its cap (the same code path as submit)."""
        s = run_seed(self.seed, probe.probe_id)
        c = self.bp.cost(probe, s)
        if c > cap:
            self.n_capped += 1
            d = self.bp.output_coords(probe.u).shape[1]
            return RunResult(probe, np.zeros((0, d)), np.zeros(0), np.zeros(0, bool), float(cap), True)
        r = self.bp.run(probe, s)
        return RunResult(
            probe=probe,
            x=r.x,
            y=r.y / self.y_scale,
            censored=r.censored,
            cost=float(c),
            cost_censored=False,
            cost_hint=None if not self.quote_work else float(self.bp.work(probe)),
        )

    def submit(self, probes: Sequence[Probe], caps: Sequence[float]) -> None:
        for probe, cap in zip(probes, caps, strict=True):
            self.n_submitted += 1
            self._done.append(self.run(probe, float(cap)))

    def poll(self) -> list[RunResult]:
        out, self._done = self._done, []
        return out
