"""Probes, run results, and the dataset that enters the likelihood.

A probe a = (u, h, T) is one run (spec Section 1). It returns a vector of outputs y_a at the
coordinates O(a) (module S1: e.g. the QoI at every threshold v reached). Outputs that the run
did not reach are right-censored: y is then a lower bound. Costs are reported by the oracle;
a run stopped at its cost cap gives a right-censored cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gcbml._config import bucket


@dataclass(frozen=True)
class Probe:
    """One requested run. ``u``: controls (n_controls,); ``h``: resolution (k,); ``run_length``: T or None."""

    probe_id: str
    u: tuple[float, ...]
    h: tuple[float, ...]
    run_length: float | None = None


@dataclass(frozen=True)
class RunResult:
    """What the oracle returns for one probe.

    ``x``: (m, d) output coordinates in physical units (u repeated, plus v when S1 is used).
    ``y``: (m,) outputs. ``censored``: (m,) True where y is only a lower bound.
    ``cost``: core-hours spent. ``cost_censored``: True if the run was stopped at its cap.
    ``cost_hint``: the oracle's own quote before the run (optional; enters the cost model as an offset).
    ``replicate_of``: probe_id of the run this one replicates (same u and h), or None.
    """

    probe: Probe
    x: np.ndarray
    y: np.ndarray
    censored: np.ndarray
    cost: float
    cost_censored: bool = False
    cost_hint: float | None = None
    replicate_of: str | None = None


@dataclass(frozen=True)
class PaddedData:
    """Arrays for the likelihood, padded to a bucket size n_pad >= n. Padded rows have mask False.

    X: (n_pad, d) unit coordinates; H: (n_pad, k) hbar; y: (n_pad,) raw outputs (not transformed);
    censored: (n_pad,) bool; run: (n_pad,) int run index (-1 on padding); mask: (n_pad,) bool.
    """

    X: np.ndarray
    H: np.ndarray
    y: np.ndarray
    censored: np.ndarray
    run: np.ndarray
    mask: np.ndarray

    @property
    def n(self) -> int:
        return int(self.mask.sum())


class Dataset:
    """All run results so far. Append-only."""

    def __init__(self, results: list[RunResult] | None = None) -> None:
        self.results: list[RunResult] = list(results or [])

    def add(self, result: RunResult) -> None:
        self.results.append(result)

    def __len__(self) -> int:
        return len(self.results)

    def padded(self, to_unit, hbar, n_pad: int | None = None) -> PaddedData:
        """Stack all outputs. ``to_unit`` maps physical x to [0,1]^d; ``hbar`` maps h to h/h_c."""
        if not self.results:
            raise ValueError("empty dataset")
        X = np.vstack([to_unit(r.x) for r in self.results])
        H = np.vstack(
            [
                np.broadcast_to(hbar(np.asarray(r.probe.h, float)), (len(r.y), len(r.probe.h)))
                for r in self.results
            ]
        )
        y = np.concatenate([np.asarray(r.y, float) for r in self.results])
        cens = np.concatenate([np.asarray(r.censored, bool) for r in self.results])
        run = np.concatenate([np.full(len(r.y), i) for i, r in enumerate(self.results)])
        n = y.size
        n_pad = bucket(n) if n_pad is None else n_pad
        if n_pad < n:
            raise ValueError("n_pad < n")
        pad = n_pad - n
        # padded rows: copies of row 0 (finite, harmless) with mask False
        return PaddedData(
            X=np.vstack([X, np.repeat(X[:1], pad, 0)]),
            H=np.vstack([H, np.repeat(H[:1], pad, 0)]),
            y=np.concatenate([y, np.repeat(y[:1], pad)]),
            censored=np.concatenate([cens, np.zeros(pad, bool)]),
            run=np.concatenate([run, np.full(pad, -1)]),
            mask=np.concatenate([np.ones(n, bool), np.zeros(pad, bool)]),
        )
