"""Shared helpers for the MCMC tests: Monte Carlo standard errors by batch means."""

import numpy as np


def batch_se(series, n_batches=25):
    """Monte Carlo standard error of the mean of ``series`` (axis 0 is time) by non-overlapping batch means."""
    series = np.asarray(series, dtype=float)
    n = (series.shape[0] // n_batches) * n_batches
    b = series[:n].reshape(n_batches, -1, *series.shape[1:]).mean(axis=1)
    return b.std(axis=0, ddof=1) / np.sqrt(n_batches)


def assert_mean_close(series, truth, n_se=4.0, extra_var=0.0, floor=0.0):
    """Assert that the mean of ``series`` equals ``truth`` within ``n_se`` Monte Carlo standard errors."""
    series = np.asarray(series, dtype=float)
    se = np.sqrt(batch_se(series) ** 2 + extra_var)
    err = np.abs(series.mean(axis=0) - truth)
    assert np.all(err <= n_se * se + floor), (series.mean(axis=0), truth, se)
