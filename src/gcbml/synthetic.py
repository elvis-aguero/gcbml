"""Synthetic truths with known answers, as Oracles (for tests and for the A12 prerequisite).

A12Truth(seed, d=2): spec Section 3, Step 1, "the A12 test":
    f(x, h) = f0(x) + a(x) hbar^p, x in [0, 1]^d, f0 and a independent GP draws (Matern 5/2, length 0.3,
    sd 1 and 0.5), p ~ U[0.7, 2.5]; hbar levels 1, 1/2, 1/4 probed, 1/8 and 1/16 candidates;
    noise sd 0.01; cost 2^{3 l} (l = level index) times lognormal(0, 0.1).
    Methods: problem() -> gcbml Problem; oracle() -> an Oracle (quote/submit/poll, immediate results);
    truth(x_unit) -> f0.
RichardsonToy(): the guide's Fig. T2 sequence at one condition (f0 = 1.4755, a = 0.7865, p = 1.5).
OscillatingTruth(): f(h) = 1 + 0.4 h^1.5 cos(2 pi h) (spec Fig. E caption), one condition.
a12_oracle_ranking(truth, campaign_state, candidates, n_refit_draws) -> ranking of the candidates
    by the value computed with full MCMC refits on fantasy outcomes (spec Step 1, "Oracle").

Implementation notes.
  * The GP draws are random Fourier features of a Matern 5/2 kernel (Bochner: the spectral density of a
    Matern nu kernel is a multivariate t with 2 nu degrees of freedom and scale 1/ell). With N_FEATURES
    features the draw is a smooth function that can be evaluated at any x, which a lazily sampled GP could
    not do consistently. Its covariance is the Matern 5/2 kernel up to an O(1/sqrt(N_FEATURES)) error
    (tests/test_synthetic.py checks it). [assumption: this is "a GP draw" of the stub]
  * hbar = h / h_c with h_c = 1 for every truth here, so the problem's resolution values ARE hbar.
  * Every random number of the oracle (noise, cost) is a function of (seed, probe_id), so the result of a
    probe does not depend on the order in which probes are submitted. Replicates need distinct probe ids.
  * A probe whose cost would exceed its cap is stopped at the cap: the result has cost = cap,
    cost_censored = True and no outputs (spec Step 5: in the generic case with one output a capped run
    gives only the cost datum).
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gcbml.data import Probe, RunResult
from gcbml.problem import InputSpace, Problem, ResolutionComponent, ResolutionSpec, Tolerance

N_FEATURES = 2048
NU = 2.5  # Matern smoothness of the A12 GP draws


def matern52(r, ell: float) -> np.ndarray:
    """Matern 5/2 correlation at distance r: (1 + sqrt5 r/l + 5 r^2/(3 l^2)) exp(-sqrt5 r/l)."""
    s = math.sqrt(5.0) * np.asarray(r, dtype=float) / ell
    return (1.0 + s + s**2 / 3.0) * np.exp(-s)


class MaternDraw:
    """One draw of a zero-mean GP with a Matern 5/2 kernel (variance sd^2, length ell) on R^d.

    f(x) = sd sqrt(2/M) sum_m cos(w_m . x + b_m), w_m = z_m sqrt(2 nu / chi2_m) / ell with z_m standard normal
    in R^d and chi2_m ~ chi-square(2 nu) (a multivariate t with 2 nu degrees of freedom, the spectral density
    of Matern nu), b_m uniform on [0, 2 pi). Then E f(x) f(x') = sd^2 k(|x - x'|).
    """

    def __init__(self, rng: np.random.Generator, d: int, sd: float, ell: float, n_features: int = N_FEATURES):
        chi2 = rng.chisquare(2.0 * NU, size=n_features)
        self.w = rng.standard_normal((n_features, d)) * (np.sqrt(2.0 * NU / chi2) / ell)[:, None]
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
        self.amp = sd * math.sqrt(2.0 / n_features)

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return self.amp * np.cos(x @ self.w.T + self.b).sum(axis=1)


def _level_of(h: float, h_c: float = 1.0, ratio: float = 2.0) -> int:
    return int(round(math.log(h_c / h, ratio)))


@dataclass
class SyntheticTruth:
    """Base class: a known f(x, hbar), a noise level, a cost law, and a Problem that describes it."""

    noise_sd: float = 0.01
    cost_sigma: float = 0.1
    cost_base: float = 2.0
    cost_gamma: float = 3.0
    cost_unit: float = 1.0  # cost of a level-0 run before the lognormal factor
    d: int = 1
    budget: float = 1000.0
    eps_abs: float = 0.05
    n_levels_listed: int = 3  # levels 1, 1/2, 1/4 are listed; finer ones are generated on demand
    problem_seed: int = 0

    def value(self, x_unit, hbar: float):  # pragma: no cover - abstract
        raise NotImplementedError

    def truth(self, x_unit) -> np.ndarray:
        """The converged value f0(x) at unit points x_unit (n, d)."""
        x = np.atleast_2d(np.asarray(x_unit, dtype=float))
        return np.array([self.value(xi[None, :], 0.0) for xi in x])

    def problem(self) -> Problem:
        names = tuple(f"x{i + 1}" for i in range(self.d))
        inputs = InputSpace(names, (0.0,) * self.d, (1.0,) * self.d, self.d)
        values = tuple(0.5**i for i in range(self.n_levels_listed))
        res = ResolutionSpec((ResolutionComponent("h", values, 2.0),))
        return Problem(
            inputs=inputs,
            resolution=res,
            tolerance=Tolerance(abs=self.eps_abs),
            budget=self.budget,
            transforms=("identity",),
            seed=self.problem_seed,
        )

    def oracle(self, seed: int = 0, cost_sigma: float | None = None) -> SyntheticOracle:
        return SyntheticOracle(self, seed, cost_sigma)


class SyntheticOracle:
    """Oracle (quote / submit / poll) of a SyntheticTruth. Results are immediate: submit computes them."""

    def __init__(self, truth: SyntheticTruth, seed: int = 0, cost_sigma: float | None = None):
        self.truth_obj = truth
        self.seed = int(seed)
        self.cost_sigma = truth.cost_sigma if cost_sigma is None else float(cost_sigma)
        self._done: list[RunResult] = []
        self.n_submitted = 0

    def quote(self, probes: Sequence[Probe]) -> list[float | None]:
        return [None] * len(probes)

    def _normals(self, probe: Probe) -> np.ndarray:
        key = zlib.crc32(probe.probe_id.encode())
        return np.random.default_rng(np.random.SeedSequence([self.seed, key])).standard_normal(2)

    def submit(self, probes: Sequence[Probe], caps: Sequence[float]) -> None:
        t = self.truth_obj
        inputs = t.problem().inputs
        for probe, cap in zip(probes, caps, strict=True):
            self.n_submitted += 1
            z_noise, z_cost = self._normals(probe)
            lev = _level_of(probe.h[0])
            cost = float(
                t.cost_unit * t.cost_base ** (t.cost_gamma * lev) * math.exp(self.cost_sigma * z_cost)
            )
            x = np.asarray(probe.u, dtype=float)[None, :]
            if cost > cap:
                d = x.shape[1]
                self._done.append(
                    RunResult(probe, np.zeros((0, d)), np.zeros(0), np.zeros(0, bool), float(cap), True)
                )
                continue
            y = t.value(inputs.to_unit(x), probe.h[0]) + t.noise_sd * z_noise
            self._done.append(RunResult(probe, x, np.atleast_1d(y), np.zeros(1, bool), cost, False))

    def poll(self) -> list[RunResult]:
        out, self._done = self._done, []
        return out


class A12Truth(SyntheticTruth):
    """f(x, hbar) = f0(x) + a(x) hbar^p with f0, a independent Matern 5/2 GP draws and p ~ U[0.7, 2.5]."""

    def __init__(self, seed: int, d: int = 2, **kw):
        super().__init__(d=d, **kw)
        self.seed = int(seed)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 12]))
        self.f0 = MaternDraw(rng, d, 1.0, 0.3)
        self.a = MaternDraw(rng, d, 0.5, 0.3)
        self.p = float(rng.uniform(0.7, 2.5))

    def value(self, x_unit, hbar: float):
        x = np.atleast_2d(np.asarray(x_unit, dtype=float))
        out = self.f0(x) + self.a(x) * hbar**self.p
        return float(out[0]) if out.size == 1 else out

    def truth(self, x_unit) -> np.ndarray:
        return self.f0(np.atleast_2d(np.asarray(x_unit, dtype=float)))


class RichardsonToy(SyntheticTruth):
    """The guide's Fig. T2 sequence at one condition: f = 1.4755 + 0.7865 hbar^1.5 (a dummy input x)."""

    F0, A, P = 1.4755, 0.7865, 1.5

    def __init__(self, **kw):
        kw.setdefault("noise_sd", 0.0)
        super().__init__(d=1, **kw)

    def value(self, x_unit, hbar: float):
        return float(self.F0 + self.A * hbar**self.P)


class OscillatingTruth(SyntheticTruth):
    """f(hbar) = 1 + 0.4 hbar^1.5 cos(2 pi hbar) (spec Fig. E caption), one condition (a dummy input x)."""

    def __init__(self, **kw):
        kw.setdefault("noise_sd", 0.0)
        super().__init__(d=1, **kw)

    def value(self, x_unit, hbar: float):
        return float(1.0 + 0.4 * hbar**1.5 * math.cos(2.0 * math.pi * hbar))


# ----------------------------------------------------------------------------------------------
# The A12 oracle ranking
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleRank:
    """Value of one candidate by full MCMC refits: mean gain of H per unit expected cost, and its s.e."""

    index: int  # position in the ``candidates`` argument
    value: float
    gain: float
    se: float
    n: int = 0  # fantasies (full refits) behind the value
    rel_se: float = float("nan")  # s.e. of the mean gain / mean gain
    rhat_p0: float = float("nan")  # worst rhat of log p0 over the refits
    raw_gains: tuple = ()  # H_ref - H_after of every refit, divergent ones included
    raw_rhats: tuple = ()  # rhat of log p0 of every refit
    n_discarded: int = 0  # divergent refits left out of the mean (see campaign.oracle_values)


def a12_oracle_ranking(
    truth: SyntheticTruth,
    campaign_state,
    candidates,
    n_refit_draws: int,
    key=None,
    source: str = "model",
    **kw,
) -> list[OracleRank]:
    """Rank ``candidates`` by the value of a probe with every posterior refitted by MCMC on its fantasy.

    For each candidate and each of n_refit_draws fantasies: draw an outcome, add it to the data of
    ``campaign_state`` (a gcbml.campaign.Campaign after its initial design), run the full MCMC fit of every
    structure again (the settings of the campaign), keep the stacking weights fixed (spec Step 5), and
    compute H_n of the refitted pool. The value of a candidate is (H_now - mean fantasy H_after) /
    cost_mean: the quantity the acquisition approximates by conditioning and reweighting.
    ``source`` = "model": the fantasy outcome comes from the posterior predictive of a random posterior draw
    (what Step 5 does); "truth": from ``truth`` itself (its f(x, h) plus noise), which asks what the real
    probe would do in this world. Returns OracleRank sorted by value, best first.
    """
    from gcbml.campaign import oracle_values  # local import: campaign does not import synthetic

    return oracle_values(truth, campaign_state, candidates, n_refit_draws, key, source, **kw)


# ---------------------------------------------------------------------------------------------------------
# Pre-asymptotic truths (benchmarks/preasymptotic): coarse levels are under-resolved, so flat in hbar.
# ---------------------------------------------------------------------------------------------------------


def preasymptotic_b(hbar, p: float, h_s: float, m: float):
    """b(hbar) = hbar^p / (1 + (hbar/h_s)^m)^(p/m).

    b(0) = 0. For hbar << h_s, b ~ hbar^p (asymptotic power law of order p).
    For hbar >> h_s, b -> h_s^p (flat).
    h_s = inf gives the pure power law hbar^p.
    """
    h = np.asarray(hbar, dtype=float)
    if np.isinf(h_s):
        return h**p
    return h**p / (1.0 + (h / h_s) ** m) ** (p / m)


@dataclass(frozen=True)
class PreasymptoticTruth:
    """z(x, hbar) = mu(x) - a(x) b(hbar), x in [0, 1]. z is the log of the quantity of interest.

    mu(x) = mu0 + mu1 x^2, a(x) = a0 (1 + a1 x^2). The truth at hbar = 0 is z0(x) = mu(x).
    """

    mu0: float
    mu1: float
    a0: float
    a1: float
    p: float
    h_s: float  # inf: no saturation (control)
    m: float

    def z0(self, x):
        x = np.asarray(x, dtype=float)
        return self.mu0 + self.mu1 * x**2

    def a(self, x):
        x = np.asarray(x, dtype=float)
        return self.a0 * (1.0 + self.a1 * x**2)

    def z(self, x, hbar):
        return self.z0(x) - self.a(x) * preasymptotic_b(hbar, self.p, self.h_s, self.m)

    def observe(self, x, hbar, noise_sd: float, seed: int = 0) -> np.ndarray:
        """One noisy run per (level, x): array (len(hbar), len(x)) of z + N(0, noise_sd^2)."""
        x = np.asarray(x, dtype=float)
        hbar = np.atleast_1d(np.asarray(hbar, dtype=float))
        clean = np.array([self.z(x, h) for h in hbar])
        return clean + noise_sd * np.random.default_rng(seed).standard_normal(clean.shape)


def preasymptotic_truth(
    seed: int,
    *,
    mu0: float = 0.0,
    mu1_range: tuple[float, float] = (0.5, 1.5),
    a0_range: tuple[float, float] = (0.2, 0.6),
    a1_range: tuple[float, float] = (0.0, 0.5),
    p_range: tuple[float, float] = (1.0, 2.0),
    h_s_range: tuple[float, float] | None = (0.2, 0.6),
    m_range: tuple[float, float] = (2.0, 8.0),
) -> PreasymptoticTruth:
    """Draw one truth. mu1, a0, a1, p, m are uniform on their ranges; h_s is log-uniform.

    h_s_range=None is the control family: h_s = inf, b = hbar^p.
    """
    rng = np.random.default_rng(seed)
    mu1 = rng.uniform(*mu1_range)
    a0 = rng.uniform(*a0_range)
    a1 = rng.uniform(*a1_range)
    p = rng.uniform(*p_range)
    m = rng.uniform(*m_range)
    h_s = math.inf
    if h_s_range is not None:
        h_s = math.exp(rng.uniform(math.log(h_s_range[0]), math.log(h_s_range[1])))
    return PreasymptoticTruth(mu0=mu0, mu1=mu1, a0=a0, a1=a1, p=p, h_s=h_s, m=m)
