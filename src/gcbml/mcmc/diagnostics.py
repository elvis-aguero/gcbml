"""Convergence diagnostics (Vehtari, Gelman, Simpson, Carpenter & Buerkner, arXiv 1903.08008).

NumPy only: diagnostics run outside jit. Input draws: array (n_chains, n_draws). Equation and
section numbers refer to arXiv 1903.08008v5.

split_rhat(x)        classic split-R-hat (Section 3.1, eqs 1-4): each chain is cut in two halves
                     (the middle draw is dropped when n_draws is odd), then B, W and R-hat as in eq 4.
rank_normalize(x)    normal scores z = Phi^{-1}((r - 3/8) / (S + 1/4)) of the average ranks r over all
                     S draws (Section 4.1, eq 14).
rhat(x)              max(rank-normalised split-R-hat, rank-normalised folded split-R-hat); the folded
                     draws are |x - median(x)| (Section 4.2, eq 15; the recommendation to report the
                     maximum is at the end of Section 4.2; threshold 1.01 in Section 2).
ess(x)               ESS of the split chains (Section 3.2): multi-chain autocorrelation of eq 10 (FFT
                     autocovariances, divisor N), Geyer's initial positive and monotone sequences
                     (eqs 11-13), the average of the sums ending at the odd and at the next even lag,
                     and the cap tau >= 1/log10(S), i.e. ESS <= S log10(S), that the paper states for
                     its software.
bulk_ess(x)          ess of the split, rank-normalised draws (Section 4.1).
tail_ess(x)          min of the ESS of the indicators I(x <= q05) and I(x <= q95) of the split
                     chains (Section 4.3, eq 16 and the definition of tail-ESS).
summary(samples)     dict name -> (rhat, bulk_ess, tail_ess) for a dict of (n_chains, n_draws) arrays.
converged(summary, rhat_max=1.01, ess_min=400) -> bool   (spec 2.5: R-hat < 1.01, bulk- and tail-ESS > 400).

Tests compare against ArviZ (arviz_stats), which implements the same paper.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def _split(x: np.ndarray) -> np.ndarray:
    """Split every chain in two halves (Section 3.1); a middle draw is dropped if n_draws is odd."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    half = x.shape[1] // 2
    return np.vstack((x[:, :half], x[:, x.shape[1] - half :]))


def _rhat_of_split_chains(x: np.ndarray) -> float:
    """Eqs 1-4 applied to the rows of x, which are already the (split) chains."""
    n = x.shape[1]
    b = n * np.var(x.mean(axis=1), ddof=1)  # eq 1
    w = np.mean(np.var(x, axis=1, ddof=1))  # eq 2
    var_plus = (n - 1) / n * w + b / n  # eq 3
    return float(np.sqrt(var_plus / w))  # eq 4


def split_rhat(x) -> float:
    return _rhat_of_split_chains(_split(x))


def rank_normalize(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    r = rankdata(x, method="average").reshape(x.shape)
    return ndtri((r - 3.0 / 8.0) / (x.size + 1.0 / 4.0))  # eq 14


def rhat(x) -> float:
    xs = _split(x)
    bulk = _rhat_of_split_chains(rank_normalize(xs))
    folded = _rhat_of_split_chains(rank_normalize(np.abs(xs - np.median(xs))))  # eq 15
    return max(bulk, folded)


def _autocov(x: np.ndarray) -> np.ndarray:
    """Biased (divisor N) autocovariance at all lags, per row, by FFT."""
    n = x.shape[1]
    xc = x - x.mean(axis=1, keepdims=True)
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, n=m, axis=1)
    return np.fft.irfft(f * np.conj(f), n=m, axis=1)[:, :n] / n


def _ess_split_chains(x: np.ndarray) -> float:
    """Eqs 10-13 on rows that are already the (split) chains."""
    m, n = x.shape
    if np.ptp(x) < np.finfo(float).resolution:
        return float(x.size)
    acov = _autocov(x)
    s2 = acov[:, 0] * n / (n - 1.0)  # s_m^2
    w = s2.mean()
    var_plus = w * (n - 1.0) / n
    if m > 1:
        var_plus += np.var(x.mean(axis=1), ddof=1)  # B / N
    # eq 10: combined autocorrelation at every lag
    rho = 1.0 - (w - acov.mean(axis=0)) / var_plus
    # Geyer's initial positive sequence on the pairs P_t = rho_{2t} + rho_{2t+1} (eq 13)
    pairs = rho[: (n // 2) * 2].reshape(-1, 2).sum(axis=1)
    pairs[0] = 1.0 + rho[1]  # rho_0 = 1 by definition
    k = 0
    while k + 1 < len(pairs) - 1 and pairs[k + 1] > 0.0:
        k += 1
    p = pairs[: k + 1].copy()
    for i in range(1, len(p)):  # initial monotone sequence: running minimum
        p[i] = min(p[i], p[i - 1])
    tau = -1.0 + 2.0 * p.sum()
    # "average of the truncated sum ending at the odd lag and the one ending at the next even lag"
    if k + 1 < len(pairs) and rho[2 * (k + 1)] > 0.0:
        tau += rho[2 * (k + 1)]
    s = m * n
    tau = max(tau, 1.0 / np.log10(s))
    return float(s / tau)


def ess(x) -> float:
    return _ess_split_chains(_split(x))


def bulk_ess(x) -> float:
    return _ess_split_chains(rank_normalize(_split(x)))


def tail_ess(x) -> float:
    x = np.asarray(x, dtype=float)
    out = []
    for q in (0.05, 0.95):
        ind = (x <= np.quantile(x, q)).astype(float)
        out.append(_ess_split_chains(_split(ind)))
    return min(out)


def summary(samples: dict) -> dict:
    return {k: (rhat(v), bulk_ess(v), tail_ess(v)) for k, v in samples.items()}


def converged(summ: dict, rhat_max: float = 1.01, ess_min: float = 400.0) -> bool:
    return all(r < rhat_max and b > ess_min and t > ess_min for r, b, t in summ.values())
