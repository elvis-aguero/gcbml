import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

import gcbml  # noqa: F401
from gcbml import transforms

NAMES = ["identity", "log", "reciprocal"]


@pytest.mark.parametrize("name", NAMES)
def test_round_trip_forward_inverse(name):
    T = transforms.get(name)
    y = jnp.array([0.1, 0.5, 1.0, 2.3, 17.0])
    np.testing.assert_allclose(T.inverse(T.forward(y)), y, rtol=1e-13)
    z = T.forward(y)
    np.testing.assert_allclose(T.forward(T.inverse(z)), z, rtol=1e-13)


@pytest.mark.parametrize("name", NAMES)
def test_log_abs_jac_matches_finite_difference(name):
    T = transforms.get(name)
    y = np.array([0.2, 0.9, 1.7, 5.0])
    h = 1e-6 * y
    fd = (np.asarray(T.forward(y + h)) - np.asarray(T.forward(y - h))) / (2 * h)
    np.testing.assert_allclose(np.asarray(T.log_abs_jac(y)), np.log(np.abs(fd)), atol=1e-8)


@pytest.mark.parametrize("name", NAMES)
def test_log_abs_jac_matches_autodiff(name):
    T = transforms.get(name)
    y = jnp.array([0.2, 0.9, 1.7])
    g = jax.vmap(jax.grad(lambda t: T.forward(t)))(y)
    np.testing.assert_allclose(T.log_abs_jac(y), jnp.log(jnp.abs(g)), atol=1e-12)


@pytest.mark.parametrize(
    ("name", "mu", "sd"), [("log", 0.3, 0.7), ("reciprocal", 5.0, 0.5), ("identity", 1.0, 2.0)]
)
def test_change_of_variables_density_integrates_to_one(name, mu, sd):
    T = transforms.get(name)

    def dens(y):
        z = float(T.forward(jnp.asarray(y)))
        return stats.norm.pdf(z, mu, sd) * float(jnp.exp(T.log_abs_jac(jnp.asarray(y))))

    lo = -np.inf if name == "identity" else 0.0
    # split the range so quad sees the bulk of the mass
    pts = [lo, 0.0 if name == "identity" else 1.0, np.inf]
    total = sum(integrate.quad(dens, a, b, limit=200)[0] for a, b in zip(pts[:-1], pts[1:], strict=True))
    assert abs(total - 1.0) < 1e-8


def test_get_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        transforms.get("bad")


@pytest.mark.parametrize("name", ["log", "reciprocal"])
def test_domain_rejects_nonpositive(name):
    T = transforms.get(name)
    ok = np.asarray(T.domain_ok(jnp.array([-1.0, 0.0, 1e-300, 2.0])))
    assert ok.tolist() == [False, False, True, True]


def test_identity_domain_accepts_all_finite():
    T = transforms.get("identity")
    ok = np.asarray(T.domain_ok(jnp.array([-5.0, 0.0, 3.0])))
    assert ok.all()
    np.testing.assert_allclose(T.log_abs_jac(jnp.array([1.0, -2.0])), 0.0)


def test_transforms_jit_and_vmap():
    T = transforms.get("log")
    y = jnp.linspace(0.5, 3.0, 6)
    np.testing.assert_allclose(jax.jit(T.forward)(y), np.log(np.asarray(y)), rtol=1e-14)
    np.testing.assert_allclose(jax.vmap(T.log_abs_jac)(y), -np.log(np.asarray(y)), rtol=1e-14)
