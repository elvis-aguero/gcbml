"""The oracle: the user's simulator, behind an asynchronous interface.

gcbml never runs simulations. The oracle decides HOW to run a probe as cheaply as it can
(warm starts, checkpoint reuse, pausing and resuming are its business, not gcbml's). Whatever
it does shows up in the reported cost, which the cost model learns from.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from gcbml.data import Probe, RunResult


class Oracle(Protocol):
    def quote(self, probes: Sequence[Probe]) -> Sequence[float | None]:
        """Optional cost estimate per probe (core-hours), e.g. lower when a checkpoint can be reused.

        Return None for probes without a quote. Quotes enter the cost model as an offset covariate
        with a learned coefficient; they are never trusted blindly.
        """
        ...

    def submit(self, probes: Sequence[Probe], caps: Sequence[float]) -> None:
        """Start the runs. ``caps[i]`` is the cost (core-hours) at which run i must be stopped."""
        ...

    def poll(self) -> list[RunResult]:
        """Results of runs finished since the last poll (possibly empty)."""
        ...
