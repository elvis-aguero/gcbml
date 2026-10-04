"""model.joint_new: joint predictive of the m outputs of one new run, against a brute-force NumPy joint."""

import jax.numpy as jnp
import numpy as np
import pytest
from test_model import (
    BETA_C,
    make_problem,
    np_ard,
    np_basis,
    np_delta,
    np_noise,
    np_rho,
)

import gcbml  # noqa: F401
from gcbml import linalg, model
from gcbml.data import PaddedData

M_NEW = 3


def new_rows(seed=11, m=M_NEW, d=2, k=2):
    r = np.random.default_rng(seed)
    Xn = r.uniform(size=(m, d))
    Xn[:, 1] = np.sort(Xn[:, 1])  # output coordinate v of one run
    Hn = np.tile(r.choice([0.0, 0.25, 0.5], size=(1, k)), (m, 1))
    Pn = np.tile(r.uniform(1.0, 2.5, size=(1, k)), (m, 1))
    nvn = r.uniform(0.01, 0.05, size=m)
    return Xn, Hn, Pn, nvn


def extend(params, data, z, Xn, Hn, Pn, nvn, ya, run_id):
    """Enlarged (params, data, z): the new rows appended after the n_pad rows, in their own run."""
    m = len(Xn)
    data2 = PaddedData(
        X=np.vstack([data.X, Xn]),
        H=np.vstack([data.H, Hn]),
        y=np.concatenate([data.y, ya]),
        censored=np.concatenate([data.censored, np.zeros(m, bool)]),
        run=np.concatenate([data.run, np.full(m, run_id)]),
        mask=np.concatenate([data.mask, np.ones(m, bool)]),
    )
    params2 = params._replace(
        P=jnp.concatenate([params.P, jnp.asarray(Pn)]),
        noise_var=jnp.concatenate([params.noise_var, jnp.asarray(nvn)]),
    )
    return params2, data2, jnp.concatenate([z, jnp.asarray(ya)])


def brute_force_joint(ref, Xn, Hn, Pn, nvn, beta):
    """Mean and covariance of the new outputs given the data, from the full (data + new) covariance in NumPy.

    beta: (b0, bsd) Gaussian, or None for flat (generalised least squares, written out directly).
    """
    X, H, P, pr, dp, cfg, n = ref["X"], ref["H"], ref["P"], ref["pr"], ref["dp"], ref["cfg"], len(ref["X"])
    Xa, Ha, Pa = np.vstack([X, Xn]), np.vstack([H, Hn]), np.vstack([P, Pn])
    nva = np.concatenate([ref["nv"], nvn])
    runa = np.concatenate([ref["run"], np.full(len(Xn), 99)])
    rho0, rho1 = np_rho(pr["c0"], pr["c1"], Ha, Pa)
    Ma = np_basis(Xa, cfg.mean_basis)
    A = rho1[:, None] * Ma
    Kg = pr["sigma_mu"] ** 2 * np_ard(Xa, Xa, pr["ell_mu"], cfg.nu_x)
    Kd = np_delta(Xa, Ha, Xa, Ha, Pa, Pa, dp, cfg.h_kernel, cfg.nu_x, cfg.nu_h)
    va = Xa[:, cfg.v_index[0]] if cfg.within_run else np.zeros(len(Xa))
    K = rho1[:, None] * Kg * rho1[None, :] + Kd + np_noise(nva, runa, va, pr["ell_v"], cfg.within_run)
    z = ref["z"]
    if beta is not None:
        b0, bsd = np.array(beta[0]), np.array(beta[1])
        Kt = K + (A * bsd**2) @ A.T
        mean = rho0 + A @ b0
        Cdd, Cnd, Cnn = Kt[:n, :n], Kt[n:, :n], Kt[n:, n:]
        w = np.linalg.solve(Cdd, z - mean[:n])
        return mean[n:] + Cnd @ w, Cnn - Cnd @ np.linalg.solve(Cdd, Cnd.T)
    Kdd, Knd, Knn = K[:n, :n], K[n:, :n], K[n:, n:]
    Ad, An = A[:n], A[n:]
    Ki = lambda b: np.linalg.solve(Kdd, b)  # noqa: E731  (reference side only)
    G = Ad.T @ Ki(Ad)
    bhat = np.linalg.solve(G, Ad.T @ Ki(z - rho0[:n]))
    mean = rho0[n:] + An @ bhat + Knd @ Ki(z - rho0[:n] - Ad @ bhat)
    Rm = An.T - Ad.T @ Ki(Knd.T)
    cov = Knn - Knd @ Ki(Knd.T) + Rm.T @ np.linalg.solve(G, Rm)
    return mean, cov


@pytest.mark.parametrize("beta", [BETA_C, None])
@pytest.mark.parametrize("basis", ["constant", "linear"])
def test_joint_new_equals_brute_force_joint_within_run(basis, beta):
    pad = 6 if basis == "linear" else 0  # padding with garbage rows must not matter
    q = 1 if basis == "constant" else 3
    if beta is not None and q == 3:
        beta = ((0.4,) * 3, (2.0,) * 3)
    ref, data, params, z, cfg = make_problem(seed=21, basis=basis, beta=beta, pad=pad, within_run=True)
    assert cfg.within_run and M_NEW == 3
    Xn, Hn, Pn, nvn = new_rows()
    jn = model.joint_new(params, data, z, cfg, jnp.asarray(Xn), jnp.asarray(Hn), jnp.asarray(Pn), True, nvn)
    mean_ref, cov_ref = brute_force_joint(ref, Xn, Hn, Pn, nvn, beta)
    np.testing.assert_allclose(jn.mean, mean_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(jn.cov, cov_ref, rtol=1e-8, atol=1e-10)
    # the within-run correlation is really in the covariance (off-diagonal noise is not zero)
    off = ~np.eye(M_NEW, dtype=bool)
    assert np.all(np.abs(np.asarray(jn.cov)[off]) > 0)


def test_joint_new_flat_is_limit_of_gaussian_beta():
    ref, data, params, z, cfg = make_problem(seed=22, within_run=True)
    Xn, Hn, Pn, nvn = new_rows(3)
    args = (jnp.asarray(Xn), jnp.asarray(Hn), jnp.asarray(Pn), True, nvn)
    flat = model.joint_new(params, data, z, cfg, *args)
    cg = type(cfg)(**{**cfg.__dict__, "beta_prior": ((0.0,), (1e3,))})
    gau = model.joint_new(params, data, z, cg, *args)
    np.testing.assert_allclose(gau.mean, flat.mean, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(gau.cov, flat.cov, rtol=1e-4, atol=1e-5)


def test_joint_new_independent_noise_when_not_same_run_and_default_noise_free():
    ref, data, params, z, cfg = make_problem(seed=23, within_run=True, beta=BETA_C)
    Xn, Hn, Pn, nvn = new_rows(4)
    a = (jnp.asarray(Xn), jnp.asarray(Hn), jnp.asarray(Pn))
    same = model.joint_new(params, data, z, cfg, *a, True, nvn)
    diff = model.joint_new(params, data, z, cfg, *a, False, nvn)
    nofree = model.joint_new(params, data, z, cfg, *a)
    # only the noise block differs, and only off the diagonal when same_run=False drops the correlation
    d = np.asarray(same.cov) - np.asarray(diff.cov)
    assert np.allclose(np.diag(d), 0.0, atol=1e-12) and np.abs(d).max() > 1e-4
    np.testing.assert_allclose(
        np.asarray(same.cov) - np.asarray(nofree.cov),
        np.asarray(same.k_new) - np.asarray(nofree.k_new),
        atol=1e-10,
    )
    np.testing.assert_allclose(np.diag(np.asarray(diff.cov) - np.asarray(nofree.cov)), nvn, atol=1e-10)


@pytest.mark.parametrize("beta", [BETA_C, None])
def test_append_rows_equals_refit_factor_and_conditioning(beta):
    """linalg.append with joint_new's blocks gives the factor of the enlarged data, and the enlarged
    predict_mu equals the Gaussian conditioning of the old posterior on the new values."""
    ref, data, params, z, cfg = make_problem(seed=24, within_run=True, beta=beta, pad=5)
    Xn, Hn, Pn, nvn = new_rows(5)
    Xj, Hj, Pj = jnp.asarray(Xn), jnp.asarray(Hn), jnp.asarray(Pn)
    jn = model.joint_new(params, data, z, cfg, Xj, Hj, Pj, True, nvn)
    S = model._setup(params, data, cfg)
    K = S.K if beta is None else S.K + (S.A * jnp.asarray(beta[1]) ** 2) @ S.A.T
    F2 = linalg.append(linalg.factor(K, S.mask), jn.cross, jn.k_new, jnp.ones(M_NEW, bool))
    ya = np.array([0.3, -0.2, 0.5])
    params2, data2, z2 = extend(params, data, z, Xn, Hn, Pn, nvn, ya, run_id=77)
    S2 = model._setup(params2, data2, cfg)
    K2 = S2.K if beta is None else S2.K + (S2.A * jnp.asarray(beta[1]) ** 2) @ S2.A.T
    F2ref = linalg.factor(K2, S2.mask)
    np.testing.assert_allclose(F2.L, F2ref.L, rtol=1e-8, atol=1e-10)
    # chain rule: mu | data, ya  ==  (mu | data) conditioned on ya through the joint of (mu*, z_new)
    Xs = jnp.asarray(np.random.default_rng(0).uniform(size=(4, 2)))
    mean2, var2 = model.predict_mu(params2, data2, z2, cfg, Xs)
    # brute-force the posterior cross covariance of mu* with the new outputs (Gaussian beta: exact)
    if beta is not None:
        mean1, var1 = model.predict_mu(params, data, z, cfg, Xs)
        pr, bsd2 = ref["pr"], np.array(beta[1]) ** 2
        Xp, Hp, Pp = np.vstack([data.X, Xn]), np.vstack([data.H, Hn]), np.asarray(params2.P)
        real = np.flatnonzero(np.asarray(data2.mask))
        old, new = real[:-M_NEW], real[-M_NEW:]
        rho0, rho1 = np_rho(pr["c0"], pr["c1"], Hp, Pp)
        A = rho1[:, None] * np_basis(Xp, cfg.mean_basis)
        Ms = np_basis(np.asarray(Xs), cfg.mean_basis)
        Cs = pr["sigma_mu"] ** 2 * np_ard(np.asarray(Xs), Xp, pr["ell_mu"], cfg.nu_x) * rho1[None, :]
        Cs = Cs + (Ms * bsd2) @ A.T  # cov(mu*, z) over the padded layout
        Kt = np.asarray(K2)[np.ix_(real, real)]
        mean_p = rho0 + A @ np.array(beta[0])
        no, Kold = len(old), Kt[: len(old), : len(old)]
        Kno = Kt[no:, :no]
        Gcross = Cs[:, new] - Cs[:, old] @ np.linalg.solve(Kold, Kno.T)  # posterior cov(mu*, z_new)
        mean_new = mean_p[new] + Kno @ np.linalg.solve(Kold, np.asarray(z)[old] - mean_p[old])
        Snn = Kt[no:, no:] - Kno @ np.linalg.solve(Kold, Kno.T)
        np.testing.assert_allclose(jn.mean, mean_new, rtol=1e-8, atol=1e-10)
        mean_chain = np.asarray(mean1) + Gcross @ np.linalg.solve(Snn, ya - mean_new)
        var_chain = np.asarray(var1) - np.einsum("ij,ji->i", Gcross, np.linalg.solve(Snn, Gcross.T))
        np.testing.assert_allclose(mean2, mean_chain, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(var2, var_chain, rtol=1e-7, atol=1e-9)
