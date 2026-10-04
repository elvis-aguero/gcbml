"""Heteroscedastic noise model (spec 2.4, 2.5): sites, variance map, latent covariance, prior."""

import jax.numpy as jnp
import numpy as np
from scipy import stats

import gcbml  # noqa: F401  (float64)
from gcbml import noise
from gcbml._config import bucket
from gcbml.data import PaddedData
from gcbml.priors import PriorScales, pc_matern_logpdf


def make_data(U, V, Hh, n_pad):
    """Rows: control u (1 column), output coordinate v (1 column), resolution h (1 column)."""
    n = len(U)
    X = np.column_stack([U, V])
    H = np.asarray(Hh, float)[:, None]
    pad = n_pad - n
    # padded rows hold garbage that would make new sites if they were not masked
    Xp = np.vstack([X, np.full((pad, 2), 0.777)])
    Hp = np.vstack([H, np.full((pad, 1), 0.123)])
    return PaddedData(
        X=Xp,
        H=Hp,
        y=np.zeros(n_pad),
        censored=np.zeros(n_pad, bool),
        run=np.concatenate([np.arange(n), np.full(pad, -1)]),
        mask=np.concatenate([np.ones(n, bool), np.zeros(pad, bool)]),
    )


def test_replicates_share_a_site_and_padded_rows_are_masked():
    # rows 0 and 1 are replicates (same u, same h); row 2 differs in h only; row 3 differs in u only;
    # rows 4, 5 share a site and differ in v (the output coordinate is not part of the site).
    U = [0.1, 0.1, 0.1, 0.4, 0.9, 0.9]
    V = [0.0, 0.0, 0.0, 0.0, 0.2, 0.7]
    Hh = [1.0, 1.0, 0.5, 1.0, 0.25, 0.25]
    data = make_data(U, V, Hh, n_pad=16)
    sites, row_site, site_mask = noise.build_sites(data, n_controls=1)
    sites, row_site, site_mask = np.asarray(sites), np.asarray(row_site), np.asarray(site_mask)
    assert site_mask.sum() == 4
    assert sites.shape == (bucket(4), 2)  # n_controls + k columns, padded to a bucket
    assert row_site[0] == row_site[1]
    assert len({row_site[0], row_site[2], row_site[3], row_site[4]}) == 4
    assert row_site[4] == row_site[5]
    assert np.all(row_site[6:] == 0)  # padded rows map to 0
    assert np.all(site_mask[:4]) and not np.any(site_mask[4:])  # real sites first
    for i in range(6):  # every real row finds its own (u, hbar) in its site
        np.testing.assert_array_equal(sites[row_site[i]], [U[i], Hh[i]])
    assert np.all(np.isfinite(sites))  # padded sites are finite


def test_noise_var_equals_hand_computation():
    U = [0.1, 0.1, 0.4, 0.9]
    Hh = [1.0, 1.0, 0.5, 0.25]
    data = make_data(U, [0.0] * 4, Hh, n_pad=8)
    sites, row_site, site_mask = noise.build_sites(data, n_controls=1)
    zeta = jnp.arange(sites.shape[0], dtype=float) * 0.1
    m_s, b_s = -2.0, jnp.array([0.7])
    s2 = np.asarray(noise.noise_var(m_s, b_s, zeta, sites, row_site, data.mask))
    rs = np.asarray(row_site)
    for i in range(4):
        assert np.isclose(s2[i], np.exp(-2.0 + 0.7 * Hh[i] + 0.1 * rs[i]), rtol=1e-13)
    np.testing.assert_array_equal(s2[4:], 1.0)  # padded rows get 1
    assert np.isclose(s2[0], s2[1])  # replicates: same variance


def np_matern52(r):
    return (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)


def test_zeta_cov_equals_ard_matern_with_padded_block_decoupled():
    U = [0.1, 0.4, 0.9, 0.9]
    Hh = [1.0, 0.5, 0.25, 1.0]
    data = make_data(U, [0.0] * 4, Hh, n_pad=8)
    sites, _, site_mask = noise.build_sites(data, n_controls=1)
    ls, le = np.log(0.8), np.log([0.3, 0.6])
    K = np.asarray(noise.zeta_cov(ls, le, sites, site_mask))
    S, m = np.asarray(sites), np.asarray(site_mask)
    ns = S.shape[0]
    ref = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            if m[i] and m[j]:
                r = np.sqrt(np.sum(((S[i] - S[j]) / np.exp(le)) ** 2))
                ref[i, j] = 0.8**2 * np_matern52(r)
            elif i == j:
                ref[i, j] = 1.0  # padded block: identity, as in linalg
    np.testing.assert_allclose(K, ref, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(K, K.T)
    assert np.all(np.linalg.eigvalsh(K) > 0)


def test_log_prior_equals_sum_of_documented_densities():
    sc = PriorScales(S_mu=1.0, S_c=0.5, S_delta=0.4, S_noise=0.2, ell0=0.15, b_s_sd=1.1)
    m_s, b_s = -2.5, jnp.array([0.3, -0.8])
    ls, le = -0.4, jnp.array([-1.0, -0.5, -2.0])
    ref = (
        stats.norm.logpdf(m_s, 2 * np.log(0.2), np.log(10.0))
        + stats.norm.logpdf(np.asarray(b_s), 0.0, 1.1).sum()
        + float(pc_matern_logpdf(ls, le, 1.0, 0.15))
    )
    assert np.isclose(float(noise.log_prior(m_s, b_s, ls, le, sc)), ref, rtol=1e-12)
