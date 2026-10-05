"""Per-run metrics (PROTOCOL Section 3), computed the same way for every method.

Inputs are the method's estimate of the converged value on Sigma_N: ``m_y`` and the epistemic sd
``sigma_epi``, and the truth on the same points (all in the units of the oracle).

  claimed      P1 met by the method's own numbers: max_x sigma_epi / eps_hat <= 1, eps_hat = rel * |m_y|
               (what a user can compute; the truth is not used).
  inside2      fraction of Sigma_N with |m_y - truth| <= 2 sigma_epi.
  success      claimed AND inside2 >= 0.95.
  false_claim  claimed AND NOT success.
  coverage95   fraction of Sigma_N with |m_y - truth| <= 1.96 sigma_epi.
  err_over_eps max_x |m_y - truth| / (rel * |truth|) (the real tolerance, from the truth).
  z            (m_y - truth) / sigma_epi at every point of Sigma_N.
A method that returns no estimate (nothing affordable) scores claimed = success = False.
"""

from __future__ import annotations

import numpy as np

COVER_FRACTION = 0.95


def score(m_y, sigma_epi, truth, rel_tol: float) -> dict:
    truth = np.asarray(truth, float)
    if m_y is None or sigma_epi is None or not np.all(np.isfinite(m_y)) or not np.all(np.isfinite(sigma_epi)):
        return {
            "has_estimate": False,
            "claimed": False,
            "success": False,
            "false_claim": False,
            "inside2": None,
            "coverage95": None,
            "err_over_eps": None,
            "max_sigma_over_eps": None,
            "z": None,
        }
    m_y, s = np.asarray(m_y, float), np.maximum(np.asarray(sigma_epi, float), 1e-300)
    eps_hat = np.maximum(rel_tol * np.abs(m_y), 1e-12)
    ratio = float(np.max(s / eps_hat))
    err = np.abs(m_y - truth)
    claimed = ratio <= 1.0
    inside2 = float(np.mean(err <= 2.0 * s))
    success = bool(claimed and inside2 >= COVER_FRACTION)
    return {
        "has_estimate": True,
        "claimed": bool(claimed),
        "success": success,
        "false_claim": bool(claimed and not success),
        "inside2": inside2,
        "coverage95": float(np.mean(err <= 1.96 * s)),
        "err_over_eps": float(np.max(err / (rel_tol * np.abs(truth)))),
        "max_sigma_over_eps": ratio,
        "z": [float(t) for t in (m_y - truth) / s],
    }
