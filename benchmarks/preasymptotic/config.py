"""Application-mimicking setting of the pre-asymptotic benchmark.

Every number here is a stated ASSUMPTION that mimics a two-phase CFD tau study. None is in src/gcbml.
"""

import math

import numpy as np

LEVELS = (6, 7, 8, 9, 10)  # assumption: grid levels L6..L10
HBAR = tuple(2.0 ** -(L - LEVELS[0]) for L in LEVELS)  # 1, 1/2, 1/4, 1/8, 1/16
X_GRID = np.linspace(0.0, 1.0, 9)  # assumption: 9 equispaced conditions
NOISE_SD = 0.015  # assumption: run noise in log units

# Random truth family (assumptions). a0 and a1 ranges are tuned in calibrate.py to hit TARGETS.
FAMILY = dict(
    mu0=math.log(0.002),
    mu1_range=(0.5, 1.5),
    a0_range=(1.4, 2.8),
    a1_range=(0.0, 1.0),
    p_range=(1.0, 2.0),
    h_s_range=(0.25, 0.6),  # tuned from the nominal (0.2, 0.6) to hit the L6->L7 band
    m_range=(2.0, 8.0),
)
# Control family: no saturation. a0/a1 ranges tuned so the L8->L9 change is 18%.
CONTROL = dict(FAMILY, h_s_range=None, a0_range=(1.1, 3.4), a1_range=(0.0, 1.0))

# Calibration targets: median over x of |f_L - f_{L-1}| / f_L, median over truths, with its band (lo, hi).
TARGETS = {
    "L6->L7": (0.08, (0.04, 0.12)),
    "L7->L8": (0.17, (0.12, 0.22)),
    "L8->L9": (0.18, (0.13, 0.23)),
    "L9->L10": (0.06, (0.02, 0.10)),
}
N_CALIB = 200

N_TRUTHS = 20
SEEDS_PRE = tuple(range(0, 20))
SEEDS_CONTROL = tuple(range(100, 120))
