"""Components of vintch_model.subunit: nonlinearity, convolution, pooling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vintch_model.subunit.model import Convolution, Pooling, tents

# Tents with bounds (-3, 3) and 4 nodes: nodes at -3, -1, 1, 3.
BOUNDS = jnp.array([-3.0, 3.0])
COEF = jnp.array([-1.0, 0.0, 2.0, 5.0])


@pytest.fixture
def x64():
    """Run a test in float64, restoring the global setting afterwards."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("jit", [False, True])
def test_tents_interpolate_between_nodes(x64, jit):
    nonlinearity = tents(4)
    call = jax.jit(nonlinearity) if jit else nonlinearity
    drive = jnp.array([-3.0, -2.0, 0.0, 2.5])
    np.testing.assert_allclose(call(COEF, drive, BOUNDS), [-1.0, -0.5, 1.0, 4.25], atol=1e-12)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_tents_clamp_holds_end_coefficients(x64, dtype):
    nonlinearity = jax.jit(tents(4, "clamp"))
    drive = jnp.array([-10.0, -3.0, 3.0, 10.0], dtype=dtype)
    out = nonlinearity(COEF.astype(dtype), drive, BOUNDS.astype(dtype))
    np.testing.assert_allclose(out, [-1.0, -1.0, 5.0, 5.0], rtol=1e-6)


@pytest.mark.xfail(reason="nemos basis.evaluate does not apply fill_value outside the bounds yet")
def test_tents_zero_outside(x64):
    out = jax.jit(tents(4, "zero"))(COEF, jnp.array([-10.0, 10.0]), BOUNDS)
    np.testing.assert_allclose(out, [0.0, 0.0])


def test_tents_tree_matches_per_channel(x64):
    nonlinearity = tents(4)
    coef = {"excitatory": COEF, "suppressive": -COEF}
    bounds = {"excitatory": BOUNDS, "suppressive": jnp.array([-1.0, 2.0])}
    drive = {"excitatory": jnp.linspace(-4, 4, 7), "suppressive": jnp.linspace(-2, 3, 5)}
    out = nonlinearity(coef, drive, bounds)
    for name in coef:
        np.testing.assert_allclose(out[name], nonlinearity(coef[name], drive[name], bounds[name]))


def test_convolution_equals_patch_sum(x64):
    rng = np.random.default_rng(0)
    stimulus = rng.standard_normal((3, 8, 16, 16))
    kernel = rng.standard_normal((8, 8, 8))
    drive = np.asarray(Convolution()(jnp.asarray(kernel), jnp.asarray(stimulus)))
    assert drive.shape == (3, 9, 9)
    expected = np.array([[[np.sum(kernel * stimulus[t, :, i:i + 8, j:j + 8])
                           for j in range(9)] for i in range(9)] for t in range(3)])
    np.testing.assert_allclose(drive, expected, atol=1e-10)


def test_convolution_tree_structure(x64):
    rng = np.random.default_rng(1)
    stimulus = jnp.asarray(rng.standard_normal((2, 8, 16, 16)))
    kernel = {"excitatory": jnp.asarray(rng.standard_normal((8, 8, 8))),
              "suppressive": jnp.asarray(rng.standard_normal((8, 4, 4)))}
    drive = Convolution()(kernel, stimulus)
    assert jax.tree.structure(drive) == jax.tree.structure(kernel)
    assert drive["excitatory"].shape == (2, 9, 9)
    assert drive["suppressive"].shape == (2, 13, 13)


def test_pooling_equals_weighted_sum(x64):
    rng = np.random.default_rng(2)
    weights = {"excitatory": rng.standard_normal((9, 9)), "suppressive": rng.standard_normal((9, 9))}
    responses = {"excitatory": rng.standard_normal((5, 9, 9)), "suppressive": rng.standard_normal((5, 9, 9))}
    pooled = Pooling()(jax.tree.map(jnp.asarray, weights), jax.tree.map(jnp.asarray, responses))
    for name in weights:
        expected = np.sum(responses[name] * weights[name], axis=(1, 2))
        np.testing.assert_allclose(pooled[name], expected, atol=1e-10)
