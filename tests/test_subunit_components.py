"""Components of vintch_model.subunit (nonlinearity, convolution, pooling) and the
model that composes them."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vintch_model.subunit import Convolution, Pooling, SubunitModel, SubunitParams, tents

# Tents with bounds (-3, 3) and 4 nodes: nodes at -3, -1, 1, 3.
BOUNDS = jnp.array([-3.0, 3.0])
COEF = jnp.array([-1.0, 0.0, 2.0, 5.0])


@pytest.mark.parametrize("jit", [False, True])
def test_tents_interpolate_between_nodes(x64, jit):
    nonlinearity = tents(4)
    call = jax.jit(nonlinearity) if jit else nonlinearity
    pre_activations = jnp.array([-3.0, -2.0, 0.0, 2.5])
    np.testing.assert_allclose(call(COEF, pre_activations, BOUNDS), [-1.0, -0.5, 1.0, 4.25], atol=1e-12)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_tents_clamp_holds_end_coefficients(x64, dtype):
    nonlinearity = jax.jit(tents(4, "clamp"))
    pre_activations = jnp.array([-10.0, -3.0, 3.0, 10.0], dtype=dtype)
    out = nonlinearity(COEF.astype(dtype), pre_activations, BOUNDS.astype(dtype))
    np.testing.assert_allclose(out, [-1.0, -1.0, 5.0, 5.0], rtol=1e-6)


@pytest.mark.xfail(reason="nemos basis.evaluate does not apply fill_value outside the bounds yet")
def test_tents_zero_outside(x64):
    out = jax.jit(tents(4, "zero"))(COEF, jnp.array([-10.0, 10.0]), BOUNDS)
    np.testing.assert_allclose(out, [0.0, 0.0])


def test_tents_tree_matches_per_channel(x64):
    nonlinearity = tents(4)
    coef = {"excitatory": COEF, "suppressive": -COEF}
    bounds = {"excitatory": BOUNDS, "suppressive": jnp.array([-1.0, 2.0])}
    pre_activations = {"excitatory": jnp.linspace(-4, 4, 7), "suppressive": jnp.linspace(-2, 3, 5)}
    out = nonlinearity(coef, pre_activations, bounds)
    for name in coef:
        np.testing.assert_allclose(out[name], nonlinearity(coef[name], pre_activations[name], bounds[name]))


def test_convolution_equals_patch_sum(x64):
    rng = np.random.default_rng(0)
    stimulus = rng.standard_normal((3, 8, 16, 16))
    kernel = rng.standard_normal((8, 8, 8))
    pre_activations = np.asarray(Convolution()(jnp.asarray(kernel), jnp.asarray(stimulus)))
    assert pre_activations.shape == (3, 9, 9)
    expected = np.array([[[np.sum(kernel * stimulus[t, :, i:i + 8, j:j + 8])
                           for j in range(9)] for i in range(9)] for t in range(3)])
    np.testing.assert_allclose(pre_activations, expected, atol=1e-10)


def test_convolution_tree_structure(x64):
    rng = np.random.default_rng(1)
    stimulus = jnp.asarray(rng.standard_normal((2, 8, 16, 16)))
    kernel = {"excitatory": jnp.asarray(rng.standard_normal((8, 8, 8))),
              "suppressive": jnp.asarray(rng.standard_normal((8, 4, 4)))}
    pre_activations = Convolution()(kernel, stimulus)
    assert jax.tree.structure(pre_activations) == jax.tree.structure(kernel)
    assert pre_activations["excitatory"].shape == (2, 9, 9)
    assert pre_activations["suppressive"].shape == (2, 13, 13)


def test_pooling_equals_weighted_sum(x64):
    rng = np.random.default_rng(2)
    weights = {"excitatory": rng.standard_normal((9, 9)), "suppressive": rng.standard_normal((9, 9))}
    activations = {"excitatory": rng.standard_normal((5, 9, 9)), "suppressive": rng.standard_normal((5, 9, 9))}
    pooled = Pooling()(jax.tree.map(jnp.asarray, weights), jax.tree.map(jnp.asarray, activations))
    for name in weights:
        expected = np.sum(activations[name] * weights[name], axis=(1, 2))
        np.testing.assert_allclose(pooled[name], expected, atol=1e-10)


@pytest.mark.parametrize("jit", [False, True])
def test_model_equals_composed_stages(x64, random_params, jit):
    """offset + sum over channels of pool(clamped tents(patch sums)), computed independently."""
    rng = np.random.default_rng(3)
    names = ("excitatory", "suppressive")
    params = random_params(rng, names)
    stimulus = rng.standard_normal((4, 8, 16, 16))
    bounds = {n: jnp.array([-20.0, 20.0]) for n in names}          # pre-activations fall inside and outside
    model = SubunitModel(Convolution(), tents(16), Pooling())
    call = eqx.filter_jit(model) if jit else model
    out = call(params, jnp.asarray(stimulus), bounds)

    expected = np.full(4, 0.7)
    nodes = np.linspace(-20.0, 20.0, 16)
    for n in names:
        kernel, coef, weights = (np.asarray(params.kernel[n]), np.asarray(params.coef[n]),
                                 np.asarray(params.pooling_weights[n]))
        pre_activations = np.array([[[np.sum(kernel * stimulus[t, :, i:i + 8, j:j + 8])
                            for j in range(9)] for i in range(9)] for t in range(4)])
        activation = np.interp(pre_activations, nodes, coef)                   # holds the end values = clamp
        expected += np.sum(activation * weights, axis=(1, 2))
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_regularizable_subtrees_are_pooling_and_coef(random_params):
    params = random_params(np.random.default_rng(4), ("excitatory",))
    selected = [where(params) for where in SubunitParams.regularizable_subtrees()]
    assert selected[0] is params.pooling_weights
    assert selected[1] is params.coef
