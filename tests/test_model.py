import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import gcbml  # noqa: F401
from gcbml import model
from gcbml.data import PaddedData
from gcbml.kernels import DeltaParams
from gcbml.model import ModelConfig, ModelParams

# ----------------------------------------------------------------------------------------------
# Independent NumPy reference, written from the formulas in the model.py docstring (spec 2.2-2.4)
# ----------------------------------------------------------------------------------------------


def np_matern(r, nu):
    if nu == 0.5:
        return np.exp(-r)
    if nu == 1.5:
        return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    if nu == 2.5:
        return (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)
    raise ValueError(nu)


def np_ard(X1, X2, ell, nu):
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = np_matern(np.sqrt(np.sum(((X1[i] - X2[j]) / ell) ** 2)), nu)
    return K


def np_hp(h, p):
    return h**p if h > 0 else 0.0


def np_kh(h1, h2, p1, p2, ell_h, gamma, kind, nu_h):
    K = np.zeros((len(h1), len(h2)))
    for i in range(len(h1)):
        for j in range(len(h2)):
            if kind == "twy2":
                K[i, j] = (
                    np_hp(h1[i], p1[i]) * np_hp(h2[j], p2[j]) * np_matern(abs(h1[i] - h2[j]) / ell_h, nu_h)
                )
            else:  # lifted Brownian, one shared order p1[0]
                p = p1[0]
                if h1[i] <= 0 or h2[j] <= 0:
                    K[i, j] = 0.0
                else:
                    K[i, j] = 0.5 * (
                        h1[i] ** (2 * p)
                        + h2[j] ** (2 * p)
                        - abs(h1[i] ** (p / gamma) - h2[j] ** (p / gamma)) ** (2 * gamma)
                    )
    return K


def np_delta(X1, H1, X2, H2, P1, P2, dp, kind, nu_x, nu_h):
    K = 0.0
    for j in range(len(dp["sigma"])):
        K = K + dp["sigma"][j] ** 2 * np_ard(X1, X2, dp["ell_x"][j], nu_x) * np_kh(
            H1[:, j], H2[:, j], P1[:, j], P2[:, j], dp["ell_h"][j], dp["gamma"][j], kind, nu_h
        )
    return K


def np_basis(X, kind):
    return np.ones((len(X), 1)) if kind == "constant" else np.hstack([np.ones((len(X), 1)), X])


def np_rho(c0, c1, H, P):
    rho0 = np.array([sum(c0[j] * np_hp(H[i, j], P[i, j]) for j in range(H.shape[1])) for i in range(len(H))])
    rho1 = np.array(
        [1 + sum(c1[j] * np_hp(H[i, j], P[i, j]) for j in range(H.shape[1])) for i in range(len(H))]
    )
    return rho0, rho1


def np_noise(nv, run, v, ell_v, within_run):
    n = len(nv)
    S = np.diag(nv).astype(float)
    if within_run:
        for i in range(n):
            for ll in range(n):
                if i != ll and run[i] == run[ll]:
                    S[i, ll] = np.sqrt(nv[i] * nv[ll]) * np_matern(abs(v[i] - v[ll]) / ell_v, 1.5)
    return S


def inv_via_solve(M):
    """Precision matrix of the reference (test side only; the package never forms an inverse)."""
    return np.linalg.solve(M, np.eye(len(M)))


def block_diag(*Ms):
    n = sum(M.shape[0] for M in Ms)
    out = np.zeros((n, n))
    o = 0
    for M in Ms:
        out[o : o + len(M), o : o + len(M)] = M
        o += len(M)
    return out


# ----------------------------------------------------------------------------------------------
# Problem builder (real rows, optional padding with garbage)
# ----------------------------------------------------------------------------------------------


def make_problem(seed=0, n=9, d=2, k=2, h_kernel="twy2", within_run=True, basis="constant", pad=0, beta=None):
    r = np.random.default_rng(seed)
    X = r.uniform(size=(n, d))
    H = r.choice([0.0, 0.125, 0.25, 0.5, 1.0], size=(n, k))
    H[0] = 0.0  # a row at the converged limit
    H[1, 0], H[1, 1] = 0.5, 0.0  # one component at its limit while the other is not
    run = np.repeat(np.arange(n // 3), 3)
    if h_kernel == "twy2":
        P = r.uniform(1.0, 2.5, size=(n, k))  # varies per row
    else:
        P = np.tile(r.uniform(1.0, 2.0, size=(1, k)), (n, 1))  # shared order
    nv = r.uniform(0.01, 0.05, size=n)
    q = 1 if basis == "constant" else d + 1
    dp = dict(
        sigma=np.array([0.5, 0.3][:k]),
        ell_x=r.uniform(0.4, 1.2, size=(k, d)),
        ell_h=np.array([0.8, 1.5][:k]),
        gamma=np.array([0.3, 0.7][:k]),
    )
    pr = dict(
        sigma_mu=0.8,
        ell_mu=r.uniform(0.4, 1.0, size=d),
        c0=np.array([0.3, -0.2][:k]),
        c1=np.array([0.2, 0.1][:k]),
        ell_v=0.5,
    )
    z = r.normal(size=n)
    v_index = (d - 1,)
    cfg = ModelConfig(
        h_kernel=h_kernel,
        mean_basis=basis,
        within_run=within_run,
        v_index=v_index if within_run else (),
        beta_prior=beta,
    )
    ref = dict(X=X, H=H, P=P, nv=nv, run=run, z=z, dp=dp, pr=pr, q=q, cfg=cfg)
    # padded jax inputs
    N = n + pad
    gar = np.random.default_rng(seed + 100)

    def padded(a, kind="float"):
        if pad == 0:
            return a
        g = 50 * gar.normal(size=(pad,) + a.shape[1:])
        return np.concatenate([a, g if kind == "float" else np.abs(g)])

    mask = np.concatenate([np.ones(n, bool), np.zeros(pad, bool)])
    data = PaddedData(
        X=padded(X),
        H=padded(H, "abs"),
        y=np.concatenate([z, np.full(pad, np.nan)]),
        censored=np.zeros(N, bool),
        run=np.concatenate([run, np.full(pad, -1)]),
        mask=mask,
    )
    params = ModelParams(
        sigma_mu=jnp.asarray(pr["sigma_mu"]),
        ell_mu=jnp.asarray(pr["ell_mu"]),
        c0=jnp.asarray(pr["c0"]),
        c1=jnp.asarray(pr["c1"]),
        P=jnp.asarray(padded(P, "abs")),
        delta=DeltaParams(*(jnp.asarray(dp[key]) for key in ("sigma", "ell_x", "ell_h", "gamma"))),
        noise_var=jnp.asarray(padded(nv, "abs")),
        ell_v=jnp.asarray(pr["ell_v"]),
    )
    zp = np.concatenate([z, np.full(pad, np.nan)])
    return ref, data, params, jnp.asarray(zp), cfg


def ref_joint(ref, Xs, beta_prior):
    """Full joint Gaussian of (z, mu*) from the latent components (beta, g_d, g_s, delta_d, e)."""
    X, H, P, cfg, pr, dp = ref["X"], ref["H"], ref["P"], ref["cfg"], ref["pr"], ref["dp"]
    n, s, q = len(X), len(Xs), ref["q"]
    b0, bsd = (np.array(beta_prior[0]), np.array(beta_prior[1]))
    M, Ms = np_basis(X, cfg.mean_basis), np_basis(Xs, cfg.mean_basis)
    rho0, rho1 = np_rho(pr["c0"], pr["c1"], H, P)
    Xall = np.vstack([X, Xs])
    Kg_all = pr["sigma_mu"] ** 2 * np_ard(Xall, Xall, pr["ell_mu"], cfg.nu_x)
    Kd = np_delta(X, H, X, H, P, P, dp, cfg.h_kernel, cfg.nu_x, cfg.nu_h)
    v = X[:, cfg.v_index[0]] if cfg.within_run else np.zeros(n)
    S = np_noise(ref["nv"], ref["run"], v, pr["ell_v"], cfg.within_run)
    # latent vector u = [beta (q); g at data and test points (n+s); delta_d (n); e (n)]
    Sigma_u = block_diag(np.diag(bsd**2), Kg_all, Kd, S)
    T = np.zeros((n + s, q + n + s + n + n))
    T[:n, :q] = rho1[:, None] * M  # A beta
    T[:n, q : q + n] = np.diag(rho1)  # rho1 g(x_i)
    T[:n, q + n + s : q + n + s + n] = np.eye(n)  # delta
    T[:n, q + n + s + n :] = np.eye(n)  # noise
    T[n:, :q] = Ms
    T[n:, q + n :] = 0.0
    T[n:, q + n : q + n + s] = np.eye(s)
    mean = np.concatenate([rho0 + (rho1[:, None] * M) @ b0, Ms @ b0])
    return mean, T @ Sigma_u @ T.T


def cond_ref(ref, Xs, beta_prior):
    n = len(ref["X"])
    mean, C = ref_joint(ref, Xs, beta_prior)
    Czz, Csz, Css = C[:n, :n], C[n:, :n], C[n:, n:]
    w = np.linalg.solve(Czz, ref["z"] - mean[:n])
    mu_s = mean[n:] + Csz @ w
    var_s = np.diag(Css) - np.einsum("ij,ji->i", Csz, np.linalg.solve(Czz, Csz.T))
    return mu_s, var_s, mean[:n], Czz


# ----------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------

BETA_C = ((0.4,), (2.0,))


@pytest.mark.parametrize("h_kernel", ["twy2", "lb"])
@pytest.mark.parametrize("basis", ["constant", "linear"])
def test_gaussian_beta_matches_brute_force_joint(h_kernel, basis):
    q = 1 if basis == "constant" else 3
    beta = ((0.4,) * q, (2.0,) * q) if q > 1 else BETA_C
    ref, data, params, z, cfg = make_problem(seed=1, h_kernel=h_kernel, basis=basis, beta=beta)
    Xs = np.random.default_rng(5).uniform(size=(4, 2))
    mu_s, var_s, m_z, Czz = cond_ref(ref, Xs, beta)
    mean, var = model.predict_mu(params, data, z, cfg, jnp.asarray(Xs))
    np.testing.assert_allclose(mean, mu_s, rtol=1e-8)
    np.testing.assert_allclose(var, var_s, rtol=1e-8)
    lm = float(model.log_marginal(params, data, z, cfg))
    refl = stats.multivariate_normal(m_z, Czz, allow_singular=True).logpdf(ref["z"])
    assert abs(lm - refl) < 1e-8 * abs(refl)


def test_gaussian_beta_brute_force_without_within_run():
    beta = BETA_C
    ref, data, params, z, cfg = make_problem(seed=2, within_run=False, beta=beta)
    mean, var = model.predict_mu(params, data, z, cfg, jnp.asarray(ref["X"][:3] * 0.9))
    mu_s, var_s, m_z, Czz = cond_ref(ref, ref["X"][:3] * 0.9, beta)
    np.testing.assert_allclose(mean, mu_s, rtol=1e-8)
    np.testing.assert_allclose(var, var_s, rtol=1e-8)
    lm = float(model.log_marginal(params, data, z, cfg))
    assert abs(lm - stats.multivariate_normal(m_z, Czz).logpdf(ref["z"])) < 1e-8 * abs(lm)


def test_covariance_matches_reference_blocks():
    ref, data, params, z, cfg = make_problem(seed=3, beta=BETA_C)
    K = np.asarray(model.covariance(params, data, cfg))
    _, C = ref_joint(ref, ref["X"][:1], (np.zeros(1), 1e-9 * np.ones(1)))  # beta term ~ 0
    np.testing.assert_allclose(K, C[:9, :9], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("basis", ["constant", "linear"])
def test_flat_beta_is_limit_of_gaussian_and_constant_offset_is_known(basis):
    q = 1 if basis == "constant" else 3
    ref, data, params, z, cfg = make_problem(seed=4, basis=basis)
    Xs = np.random.default_rng(6).uniform(size=(3, 2))
    mf, vf = model.predict_mu(params, data, z, cfg, jnp.asarray(Xs))
    lf = float(model.log_marginal(params, data, z, cfg))
    gaps = []
    for bsd in [1.0, 3.0, 10.0, 30.0, 1e3]:
        beta = ((0.0,) * q, (bsd,) * q)
        cg = ModelConfig(**{**cfg.__dict__, "beta_prior": beta})
        mg, vg = model.predict_mu(params, data, z, cg, jnp.asarray(Xs))
        lg = float(model.log_marginal(params, data, z, cg))
        # flat = gaussian + q log(bsd) + q/2 log(2 pi) in the limit
        gaps.append(abs(lf - (lg + q * np.log(bsd) + 0.5 * q * np.log(2 * np.pi))))
        last = (mg, vg)
    # the gap shrinks like 1/bsd^2 (factor ~9 per step of 3), until Gaussian round-off (cond ~ bsd^2) at 1e3
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3]
    assert gaps[1] / gaps[2] > 5 and gaps[2] / gaps[3] > 5
    assert gaps[4] < 1e-4
    np.testing.assert_allclose(last[0], mf, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(last[1], vf, rtol=1e-5)


def test_flat_log_marginal_matches_restricted_likelihood_brute_force():
    ref, data, params, z, cfg = make_problem(seed=7)
    n, q = 9, 1
    _, C = ref_joint(ref, ref["X"][:1], (np.zeros(1), 1e-9 * np.ones(1)))
    K = C[:n, :n]
    rho0, rho1 = np_rho(ref["pr"]["c0"], ref["pr"]["c1"], ref["H"], ref["P"])
    A = rho1[:, None] * np_basis(ref["X"], "constant")
    r = ref["z"] - rho0
    KiA, Kir = np.linalg.solve(K, A), np.linalg.solve(K, r)
    G = A.T @ KiA
    quad = r @ Kir - (A.T @ Kir) @ np.linalg.solve(G, A.T @ Kir)
    expect = (
        -0.5 * quad
        - 0.5 * np.linalg.slogdet(K)[1]
        - 0.5 * np.linalg.slogdet(G)[1]
        - 0.5 * (n - q) * np.log(2 * np.pi)
    )
    got = float(model.log_marginal(params, data, z, cfg))
    assert abs(got - expect) < 1e-8 * abs(expect)


@pytest.mark.parametrize("beta", [None, BETA_C])
def test_padding_with_garbage_equals_unpadded(beta):
    ref, d0, p0, z0, cfg = make_problem(seed=8, beta=beta)
    _, d1, p1, z1, _ = make_problem(seed=8, beta=beta, pad=7)
    Xs = jnp.asarray(np.random.default_rng(1).uniform(size=(3, 2)))
    assert (
        abs(float(model.log_marginal(p0, d0, z0, cfg)) - float(model.log_marginal(p1, d1, z1, cfg))) < 1e-10
    )
    for a, b in zip(
        model.predict_mu(p0, d0, z0, cfg, Xs), model.predict_mu(p1, d1, z1, cfg, Xs), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=1e-10)
    Hs, Ps = jnp.asarray([[0.5, 0.25], [0.0, 1.0], [1.0, 1.0]]), jnp.full((3, 2), 1.7)
    for a, b in zip(
        model.predict_level(p0, d0, z0, cfg, Xs, Hs, Ps),
        model.predict_level(p1, d1, z1, cfg, Xs, Hs, Ps),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-10)


def test_predict_level_matches_brute_force_and_equals_mu_at_h0():
    beta = BETA_C
    ref, data, params, z, cfg = make_problem(seed=9, beta=beta)
    Xs = np.random.default_rng(2).uniform(size=(3, 2))
    Hs = np.array([[0.5, 0.25], [1.0, 0.0], [0.125, 1.0]])
    Ps = np.random.default_rng(3).uniform(1.0, 2.0, size=(3, 2))
    nvs = np.array([0.0, 0.1, 0.2])
    mean, var = model.predict_level(
        params, data, z, cfg, jnp.asarray(Xs), jnp.asarray(Hs), jnp.asarray(Ps), jnp.asarray(nvs)
    )
    # brute force: f* = rho0* + rho1* mu* + delta*, joint with z over the same latent components
    n, s, q = 9, 3, 1
    X, H, P, pr, dp = ref["X"], ref["H"], ref["P"], ref["pr"], ref["dp"]
    b0, bsd = np.array(beta[0]), np.array(beta[1])
    rho0, rho1 = np_rho(pr["c0"], pr["c1"], H, P)
    rho0s, rho1s = np_rho(pr["c0"], pr["c1"], Hs, Ps)
    Xa, Ha, Pa = np.vstack([X, Xs]), np.vstack([H, Hs]), np.vstack([P, Ps])
    Kg = pr["sigma_mu"] ** 2 * np_ard(Xa, Xa, pr["ell_mu"], 2.5)
    Kd = np_delta(Xa, Ha, Xa, Ha, Pa, Pa, dp, "twy2", 2.5, 1.5)  # delta at data and test points jointly
    S = np_noise(ref["nv"], ref["run"], X[:, 1], pr["ell_v"], True)
    M, Ms = np_basis(X, "constant"), np_basis(Xs, "constant")
    Su = block_diag(np.diag(bsd**2), Kg, Kd, S)
    T = np.zeros((n + s, q + (n + s) + (n + s) + n))
    T[:n, :q] = rho1[:, None] * M
    T[:n, q : q + n] = np.diag(rho1)
    T[:n, q + n + s : q + n + s + n] = np.eye(n)
    T[:n, q + 2 * (n + s) :] = np.eye(n)
    T[n:, :q] = rho1s[:, None] * Ms
    T[n:, q + n : q + n + s] = np.diag(rho1s)
    T[n:, q + (n + s) + n : q + 2 * (n + s)] = np.eye(s)
    m = np.concatenate([rho0 + rho1 * (M @ b0), rho0s + rho1s * (Ms @ b0)])
    C = T @ Su @ T.T
    w = np.linalg.solve(C[:n, :n], ref["z"] - m[:n])
    em = m[n:] + C[n:, :n] @ w
    ev = np.diag(C[n:, n:]) - np.einsum("ij,ji->i", C[n:, :n], np.linalg.solve(C[:n, :n], C[:n, n:])) + nvs
    np.testing.assert_allclose(mean, em, rtol=1e-8)
    np.testing.assert_allclose(var, ev, rtol=1e-8)
    # at hbar = 0 the level is the converged value
    m0, v0 = model.predict_level(params, data, z, cfg, jnp.asarray(Xs), jnp.zeros((3, 2)), jnp.asarray(Ps))
    mm, vm = model.predict_mu(params, data, z, cfg, jnp.asarray(Xs))
    np.testing.assert_allclose(m0, mm, rtol=1e-9)
    np.testing.assert_allclose(v0, vm, rtol=1e-9)


def test_predict_level_flat_beta_is_limit_of_gaussian():
    ref, data, params, z, cfg = make_problem(seed=10)
    Xs = jnp.asarray(np.random.default_rng(4).uniform(size=(3, 2)))
    Hs, Ps = jnp.asarray([[0.5, 0.25], [1.0, 0.0], [0.125, 1.0]]), jnp.full((3, 2), 1.5)
    mf, vf = model.predict_level(params, data, z, cfg, Xs, Hs, Ps)
    cg = ModelConfig(**{**cfg.__dict__, "beta_prior": ((0.0,), (1e4,))})
    mg, vg = model.predict_level(params, data, z, cg, Xs, Hs, Ps)
    np.testing.assert_allclose(mg, mf, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(vg, vf, rtol=1e-5)


def test_richardson_toy_recovers_f0():
    # guide Fig. T2: f(h) = f0 + a h^1.5 at hbar = 1, 1/2, 1/4, 1/8
    f0, a = 1.4755, 0.7865
    hb = np.array([1.0, 0.5, 0.25, 0.125])
    y = f0 + a * hb**1.5
    np.testing.assert_allclose(
        np.round(y, 4), [2.2620, 1.7536, 1.5738, 1.5103]
    )  # the guide's 4-decimal values
    data = PaddedData(
        X=np.full((4, 1), 0.5),
        H=hb[:, None],
        y=y,
        censored=np.zeros(4, bool),
        run=np.arange(4),
        mask=np.ones(4, bool),
    )
    params = ModelParams(
        sigma_mu=jnp.asarray(1.0),
        ell_mu=jnp.asarray([0.5]),
        c0=jnp.zeros(1),
        c1=jnp.zeros(1),
        P=jnp.full((4, 1), 1.5),
        delta=DeltaParams(jnp.asarray([1.0]), jnp.asarray([[0.5]]), jnp.asarray([50.0]), jnp.asarray([0.5])),
        noise_var=jnp.full(4, 1e-12),
        ell_v=jnp.asarray(1.0),
    )
    cfg = ModelConfig(h_kernel="twy2", nu_h=1.5, mean_basis="constant", beta_prior=None)
    mean, var = model.predict_mu(params, data, jnp.asarray(y), cfg, jnp.asarray([[0.5]]))
    sd = float(jnp.sqrt(var[0]))
    assert abs(float(mean[0]) - f0) < 1e-3
    assert abs(float(mean[0]) - f0) < 2 * sd


def test_rho_zero_at_hbar_zero_and_hand_values():
    params = ModelParams(
        sigma_mu=jnp.asarray(1.0),
        ell_mu=jnp.ones(1),
        c0=jnp.asarray([0.3, -0.2]),
        c1=jnp.asarray([0.5, 0.1]),
        P=jnp.zeros((3, 2)),
        delta=DeltaParams(jnp.ones(2), jnp.ones((2, 1)), jnp.ones(2), jnp.ones(2)),
        noise_var=jnp.ones(3),
        ell_v=jnp.asarray(1.0),
    )
    H = jnp.asarray([[0.0, 0.0], [0.5, 0.25], [1.0, 0.0]])
    P = jnp.asarray([[2.0, 1.0], [2.0, 1.5], [3.0, 1.0]])
    rho0, rho1 = model.rho(params, H, P)
    assert float(rho0[0]) == 0.0 and float(rho1[0]) == 1.0
    e0 = 0.3 * 0.5**2 - 0.2 * 0.25**1.5
    e1 = 1 + 0.5 * 0.5**2 + 0.1 * 0.25**1.5
    np.testing.assert_allclose(rho0, [0.0, e0, 0.3], rtol=1e-14)
    np.testing.assert_allclose(rho1, [1.0, e1, 1.5], rtol=1e-14)
    g = jax.grad(lambda h: model.rho(params, h, P)[0].sum())(H)  # finite gradient at hbar = 0
    assert np.all(np.isfinite(g))


@pytest.mark.parametrize("beta", [None, BETA_C])
def test_log_marginal_is_minus_inf_not_nan_for_indefinite_covariance(beta):
    ref, data, params, z, cfg = make_problem(seed=11, beta=beta)
    assert np.isfinite(float(model.log_marginal(params, data, z, cfg)))
    bad = params._replace(noise_var=params.noise_var.at[2].set(-5.0))
    lm = float(model.log_marginal(bad, data, z, cfg))
    assert lm == -np.inf
    bad2 = params._replace(noise_var=params.noise_var.at[2].set(-5.0))
    cfg2 = ModelConfig(**{**cfg.__dict__, "within_run": False, "v_index": ()})
    assert float(model.log_marginal(bad2, data, z, cfg2)) == -np.inf


def test_jit_and_vmap_over_params_batch():
    ref, data, params, z, cfg = make_problem(seed=12, beta=None, pad=3)
    Xs = jnp.asarray(np.random.default_rng(0).uniform(size=(2, 2)))
    scales = jnp.asarray([0.8, 1.3])
    batch = jax.tree_util.tree_map(lambda a: jnp.stack([a * s for s in scales]), params)
    for c in (cfg, ModelConfig(**{**cfg.__dict__, "beta_prior": BETA_C})):
        lm = jax.jit(jax.vmap(lambda p, c=c: model.log_marginal(p, data, z, c)))(batch)
        mu = jax.jit(jax.vmap(lambda p, c=c: model.predict_mu(p, data, z, c, Xs)))(batch)
        for i, s in enumerate(scales):
            p_i = jax.tree_util.tree_map(lambda a, s=s: a * s, params)
            np.testing.assert_allclose(lm[i], model.log_marginal(p_i, data, z, c), rtol=1e-10)
            for a, b in zip(mu, model.predict_mu(p_i, data, z, c, Xs), strict=True):
                np.testing.assert_allclose(a[i], b, rtol=1e-9)
        assert np.all(np.isfinite(lm))


# ---------------------------- censored Gibbs sweep ----------------------------------------------


def _truncated_setup(seed, cens, increasing, beta, n_ref=400_000):
    ref, data, params, z, cfg = make_problem(seed=seed, beta=beta, pad=3)
    cfg = ModelConfig(**{**cfg.__dict__, "increasing": increasing})
    n = 9
    if beta is None:
        # flat: Q = K^-1 - K^-1 A G^-1 A^T K^-1, from the reference covariance
        _, C = ref_joint(ref, ref["X"][:1], (np.zeros(1), 1e-9 * np.ones(1)))
        K = C[:n, :n]
        rho0, rho1 = np_rho(ref["pr"]["c0"], ref["pr"]["c1"], ref["H"], ref["P"])
        A = rho1[:, None] * np_basis(ref["X"], "constant")
        KiA = np.linalg.solve(K, A)
        Q = inv_via_solve(K) - KiA @ np.linalg.solve(A.T @ KiA, KiA.T)  # test-side reference only
        r = ref["z"] - rho0
    else:
        mean, C = ref_joint(ref, ref["X"][:1], beta)
        mean, Czz = mean[:n], C[:n, :n]
        Q = inv_via_solve(Czz)
        r = ref["z"] - mean
    c = np.array(cens)
    # conditional of z_c given z_o from the precision matrix: N(z_c - Qcc^-1 (Q r)_c, Qcc^-1)
    Qcc = Q[np.ix_(c, c)]
    cov = inv_via_solve(Qcc)
    mu_c = ref["z"][c] - cov @ (Q @ r)[c]
    sgn = 1.0 if increasing else -1.0
    bounds = mu_c + sgn * 0.0  # truncate at the conditional mean of each coordinate (acceptance ~ 1/2^m)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mu_c, cov, size=n_ref)
    ok = np.all(sgn * (draws - bounds) >= 0, axis=1)
    refdraws = draws[ok]
    b_full = np.zeros(12)
    b_full[c] = bounds
    zc = np.asarray(z).copy()
    zc[c] = bounds + sgn * 0.1  # feasible start
    cens_mask = np.zeros(12, bool)
    cens_mask[c] = True
    data = PaddedData(**{**data.__dict__, "censored": cens_mask})
    return data, params, jnp.asarray(zc), cfg, jnp.asarray(b_full), c, refdraws


def _run_chains(data, params, z, cfg, bounds, c, n_chains, n_sweeps, seed=0):
    def chain(key):
        def body(zz, k):
            return model.censored_sweep(k, params, data, zz, cfg, bounds), None

        zz, _ = jax.lax.scan(body, z, jax.random.split(key, n_sweeps))
        return zz[jnp.asarray(c)]

    return np.asarray(jax.jit(jax.vmap(chain))(jax.random.split(jax.random.key(seed), n_chains)))


def _assert_matches(sample, refdraws):
    for j in range(sample.shape[1]):
        assert stats.ks_2samp(sample[:, j], refdraws[:, j]).pvalue > 0.01
        se = np.sqrt(refdraws[:, j].var() * (1 / len(sample) + 1 / len(refdraws)))
        assert abs(sample[:, j].mean() - refdraws[:, j].mean()) < 4 * se


@pytest.mark.parametrize(("increasing", "beta"), [(True, BETA_C), (False, None)])
def test_censored_sweep_one_row_matches_truncated_conditional(increasing, beta):
    data, params, z, cfg, bounds, c, refdraws = _truncated_setup(21, [3], increasing, beta)
    sample = _run_chains(data, params, z, cfg, bounds, c, n_chains=2000, n_sweeps=1)
    sgn = 1.0 if increasing else -1.0
    assert np.all(sgn * (sample[:, 0] - float(bounds[3])) >= 0)
    _assert_matches(sample, refdraws)


@pytest.mark.parametrize("increasing", [True, False])
def test_censored_sweep_two_rows_matches_truncated_conditional(increasing):
    data, params, z, cfg, bounds, c, refdraws = _truncated_setup(22, [2, 5], increasing, BETA_C)
    sample = _run_chains(data, params, z, cfg, bounds, c, n_chains=3000, n_sweeps=12)
    _assert_matches(sample, refdraws)


def test_censored_sweep_leaves_uncensored_and_padded_rows_alone():
    data, params, z, cfg, bounds, c, _ = _truncated_setup(23, [2, 5], True, BETA_C, n_ref=10)
    zn = model.censored_sweep(jax.random.key(0), params, data, z, cfg, bounds)
    keep = np.setdiff1d(np.arange(12), c)
    np.testing.assert_array_equal(np.asarray(zn)[keep], np.asarray(z)[keep])
    assert np.all(np.asarray(zn)[c] >= np.asarray(bounds)[c])


@pytest.mark.slow
@pytest.mark.parametrize("increasing", [True, False])
@pytest.mark.parametrize("beta", [BETA_C, None])
def test_censored_sweep_long_chains_two_rows(increasing, beta):
    data, params, z, cfg, bounds, c, refdraws = _truncated_setup(
        24, [1, 4], increasing, beta, n_ref=2_000_000
    )
    sample = _run_chains(data, params, z, cfg, bounds, c, n_chains=20_000, n_sweeps=30, seed=3)
    _assert_matches(sample, refdraws)


@pytest.mark.slow
def test_censored_sweep_deep_tail_is_finite_and_truncated():
    # bound 6 above the conditional mean (many conditional sds): the tail sampler must stay stable
    data, params, z, cfg, bounds, c, _ = _truncated_setup(25, [3], True, BETA_C, n_ref=10)
    zc = np.asarray(z).copy()
    b = np.asarray(bounds).copy()
    b[3] += 6.0
    zc[3] = b[3] + 0.1
    sample = _run_chains(data, params, jnp.asarray(zc), cfg, jnp.asarray(b), c, n_chains=2000, n_sweeps=1)
    assert np.all(np.isfinite(sample)) and np.all(sample[:, 0] >= b[3])
