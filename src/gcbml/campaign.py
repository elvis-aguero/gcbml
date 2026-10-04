"""The algorithm of spec Section 3 as an ask/tell loop (Steps 0-6), with persistent state.

    c = Campaign(problem, scales, cost_prior, settings)
    probes = c.initial_design()                      # Step 1
    loop:  c.tell(results); probes = c.ask()         # Steps 2-5; [] when finished
    rep = c.report()                                 # the output of spec Section 3

Nothing here runs a simulation. run_campaign(campaign, oracle) is a convenience loop over an Oracle
(submit, poll) that respects the budget reservations.

Step 1, initial design [assumption; every number is a CampaignSettings field]:
  cheapest level 0: a scrambled Sobol set of n0 = 10 * n_controls points in Sigma (unit u);
  ladder: levels 1 and 2 use nested prefixes of that Sobol set of sizes max(3, n0 // 2^l);
  replicates: 2 extra runs at 3 sites (the first 3 Sobol points) at levels 0, 1 and 2;
  the A12 prerequisite is a test in tests/, not a runtime check.
Step 2: fit every structure (inference.fit) on all data; stacking weights (stacking.py) from the
  finest-level hold-out when testable, else equal weights. Structures = product of settings.h_kernels and
  problem.transforms (levels: all) [v1].
Step 3: gates G0-G4, G6, G7 (gates.py; G5 when settings.monotone is set). Repair (spec): at most 2 cycles:
  remove the coarsest level on a G4 fail; on other fails, label the result "uncalibrated" (never silently
  calibrated). Gate results go in the report.
Step 4: success if max over Sigma_N of sigma_epi / eps <= 1 and the gates pass -> ask() returns [];
  if the spent cost plus the cheapest admissible cap exceeds the budget -> P2 report, ask() returns [];
  else forecast (forecast.py): if infeasible switch the acquisition to mode "softmax" (P2) and record it;
  if p_exceed > 0.5 record a warning for the user.
Step 5: candidates = settings.n_candidates_u Sobol points in Sigma plus the existing sites, at every level
  from 0 to (finest level run so far + settings.extra_levels); cost of each from cost.fit_cost (refitted
  every ask) with quotes from the caller (ask(quotes=...)); acquisition.select_batch with q = settings.q.
State: everything needed to resume (results, spent cost, pending probes and their caps, RNG key, mode,
  history of reports) is saved to state_dir after every tell/ask as JSON + NPZ; Campaign.load(state_dir).

TODO(W6-B): implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CampaignSettings:
    n_warmup: int = 500
    n_samples: int = 500
    n_chains: int = 4
    q: int = 1
    n0_per_control: int = 10
    n_replicate_sites: int = 3
    n_replicates: int = 2
    n_candidates_u: int = 64
    extra_levels: int = 2
    h_kernels: tuple[str, ...] = ("twy2", "lb")
    varying_order: bool = False
    monotone: tuple[int, str] | None = None  # (coordinate index, "increasing" | "decreasing") for G5
    max_draws_acquisition: int = 64
    seed: int = 0


@dataclass
class Report:
    status: str  # "success" | "P2" | "uncalibrated" | "running"
    m_y: Any = None
    sigma_epi: Any = None
    s0: Any = None
    sigma_tot: Any = None
    sigma_env: Any = None
    sigma_fid: Any = None
    gates: list = field(default_factory=list)
    weights: Any = None
    spent: float = 0.0
    allocation: dict = field(default_factory=dict)  # level -> cost spent
    notes: list = field(default_factory=list)


class Campaign:
    def __init__(self, problem, scales, cost_prior, settings: CampaignSettings | None = None, state_dir=None):
        raise NotImplementedError("W6-B")

    def initial_design(self) -> list:
        raise NotImplementedError("W6-B")

    def tell(self, results) -> None:
        raise NotImplementedError("W6-B")

    def ask(self, quotes=None) -> list:
        raise NotImplementedError("W6-B")

    def report(self) -> Report:
        raise NotImplementedError("W6-B")

    @classmethod
    def load(cls, state_dir) -> Campaign:
        raise NotImplementedError("W6-B")


def run_campaign(campaign: Campaign, oracle, max_rounds: int = 1000) -> Report:
    raise NotImplementedError("W6-B")
