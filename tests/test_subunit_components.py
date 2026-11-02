"""Components of vintch_model.subunit (nonlinearity, convolution, pooling) and the
model that composes them."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vintch_model.subunit import (
    Convolution,
    Pooling,
    ReduceAndOffset,
    SubunitModel,
    SubunitParams,
    tents,
)

# Tents with bounds (-3, 3) and 4 nodes: nodes at -3, -1, 1, 3.
BOUNDS = jnp.array([-3.0, 3.0])
COEF = jnp.array([-1.0, 0.0, 2.0, 5.0])


def tol(dtype):
    """Absolute tolerance for ``dtype``: square root of the machine epsilon."""
    return jnp.sqrt(jnp.finfo(dtype).eps)


@pytest.fixture
def x64():
    """Run a test in float64, restoring the global setting afterwards."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(params=[jnp.float32, jnp.float64], ids=["float32", "float64"])
def dtype(request):
    """Run a test in float32 and in float64, restoring the global setting afterwards."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == jnp.float64)
    yield request.param
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("jit", [False, True])
def test_tents_interpolate_between_nodes(dtype, jit):
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


def test_tents_tree_matches_per_channel(dtype):
    nonlinearity = tents(4)
    coef = {"excitatory": COEF, "suppressive": -COEF}
    bounds = {"excitatory": BOUNDS, "suppressive": jnp.array([-1.0, 2.0])}
    pre_activations = {"excitatory": jnp.linspace(-4, 4, 7), "suppressive": jnp.linspace(-2, 3, 5)}
    out = nonlinearity(coef, pre_activations, bounds)
    for name in coef:
        np.testing.assert_allclose(out[name], nonlinearity(coef[name], pre_activations[name], bounds[name]))


def test_convolution_equals_patch_sum(dtype):
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    stimulus = jax.random.normal(keys[0], shape=(3, 8, 16, 16))
    kernel = jax.random.normal(keys[1], shape=(8, 8, 8))
    pre_activations = Convolution()(kernel, stimulus)
    assert pre_activations.shape == (3, 9, 9)
    expected = jnp.array([[[jnp.sum(kernel * stimulus[t, :, i:i + 8, j:j + 8])
                           for j in range(9)] for i in range(9)] for t in range(3)])
    np.testing.assert_allclose(pre_activations, expected, atol=tol(dtype))


def test_convolution_tree_structure(x64):
    keys = jax.random.split(jax.random.PRNGKey(10), 3)
    stimulus = jax.random.normal(keys[0], shape=(2, 8, 16, 16))
    kernel = {"excitatory": jax.random.normal(keys[1], shape=(8, 8, 8)),
              "suppressive": jax.random.normal(keys[2], shape=(8, 4, 4))}
    pre_activations = Convolution()(kernel, stimulus)
    assert jax.tree.structure(pre_activations) == jax.tree.structure(kernel)
    assert pre_activations["excitatory"].shape == (2, 9, 9)
    assert pre_activations["suppressive"].shape == (2, 13, 13)


def test_pooling_equals_weighted_sum(dtype):
    keys = jax.random.split(jax.random.PRNGKey(10), 4)
    weights = {"excitatory": jax.random.normal(keys[0], shape=(9, 9)), "suppressive": jax.random.normal(keys[1], shape=(9, 9))}
    activations = {"excitatory": jax.random.normal(keys[2], shape=(5, 9, 9)), "suppressive": jax.random.normal(keys[3], shape=(5, 9, 9))}
    pooled = Pooling()(weights, activations)
    for name in weights:
        expected = jnp.sum(activations[name] * weights[name], axis=(1, 2))
        np.testing.assert_allclose(pooled[name], expected, atol=tol(dtype))


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("as_list", [False, True])
def test_reduce_and_offset_sums_channels_plus_offset(dtype, jit, as_list):
    keys = jax.random.split(jax.random.PRNGKey(10), 2)
    pooled = {"excitatory": jax.random.normal(keys[0], shape=(5,)), "suppressive": jax.random.normal(keys[1], shape=(5,))}
    offset = jnp.asarray(0.7)
    expected = 0.7 + pooled["excitatory"] + pooled["suppressive"]
    if as_list:
        pooled = list(pooled.values())
    reduce = jax.jit(ReduceAndOffset()) if jit else ReduceAndOffset()
    np.testing.assert_allclose(reduce(offset, pooled), expected, atol=tol(dtype))


@pytest.fixture
def random_params(request):
    """Random SubunitParams; request.param = dict(seed, names, n_basis=16)."""
    seed, names, n_basis = request.param["seed"], request.param["names"], request.param.get("n_basis", 16)
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    return SubunitParams(
        kernel={n: jax.random.normal(jax.random.fold_in(keys[0], i), shape=(8, 8, 8)) for i, n in enumerate(names)},
        coef={n: jax.random.normal(jax.random.fold_in(keys[1], i), shape=(n_basis,)) for i, n in enumerate(names)},
        pooling_weights={n: jax.random.normal(jax.random.fold_in(keys[2], i), shape=(9, 9))
                         for i, n in enumerate(names)},
        offset=jnp.asarray(0.7),
    )


@pytest.mark.parametrize("random_params", [dict(seed=3, names=("excitatory", "suppressive"))], indirect=True)
@pytest.mark.parametrize("jit", [False, True])
def test_model_equals_composed_stages(dtype, random_params, jit):
    """offset + sum over channels of pool(clamped tents(patch sums)), computed independently."""
    names = ("excitatory", "suppressive")
    params = random_params
    stimulus = jax.random.normal(jax.random.PRNGKey(3), shape=(4, 8, 16, 16))
    bounds = {n: jnp.array([-20.0, 20.0]) for n in names}          # pre-activations fall inside and outside
    model = SubunitModel(Convolution(), tents(16), Pooling(), ReduceAndOffset())
    call = jax.jit(model) if jit else model
    out = call(params, stimulus, bounds)
    stimulus = np.asarray(stimulus)

    expected = np.full(4, 0.7)
    nodes = np.linspace(-20.0, 20.0, 16)
    for n in names:
        kernel, coef, weights = (np.asarray(params.kernel[n]), np.asarray(params.coef[n]),
                                 np.asarray(params.pooling_weights[n]))
        pre_activations = np.array([[[np.sum(kernel * stimulus[t, :, i:i + 8, j:j + 8])
                            for j in range(9)] for i in range(9)] for t in range(4)])
        activation = np.interp(pre_activations, nodes, coef)                   # holds the end values = clamp
        expected += np.sum(activation * weights, axis=(1, 2))
    np.testing.assert_allclose(out, expected, atol=tol(dtype))


@pytest.mark.parametrize("random_params", [dict(seed=4, names=("excitatory",))], indirect=True)
def test_regularizable_subtrees_are_pooling_and_coef(random_params):
    params = random_params
    selected = [where(params) for where in SubunitParams.regularizable_subtrees()]
    assert selected[0] is params.pooling_weights
    assert selected[1] is params.coef


def no_jit(f):
    return f


@pytest.mark.parametrize("wrap", [jax.jit, no_jit], ids=["jit", "no_jit"])
def test_end_to_end_stages_match_model(dtype, wrap):
    """Convolution -> tents -> pooling -> reduce, run stage by stage, equals the jitted model."""
    n_samples, n_lags, n_pix, k_size, n_basis = 20, 3, 10, 4, 6
    n_pos = n_pix - k_size + 1
    names = ("excitatory", "suppressive")
    keys = jax.random.split(jax.random.PRNGKey(121), 5)
    stimulus = jax.random.normal(keys[0], shape=(n_samples, n_lags, n_pix, n_pix))
    kernels = {n: jax.random.normal(jax.random.fold_in(keys[1], i), shape=(n_lags, k_size, k_size))
               for i, n in enumerate(names)}
    tent_coef = {n: jax.random.normal(jax.random.fold_in(keys[2], i), shape=(n_basis,)) for i, n in enumerate(names)}
    pooling_weights = {n: jax.random.normal(jax.random.fold_in(keys[3], i), shape=(n_pos, n_pos))
                       for i, n in enumerate(names)}
    offset = jax.random.normal(keys[4])
    tent_bounds = {"excitatory": jnp.array([-2.0, 2.0]), "suppressive": jnp.array([-3.0, 1.0])}

    conv = Convolution(padding="VALID")
    tent_nl = tents(n_basis, fill_strategy="clamp")
    pooling = Pooling()
    reduce = ReduceAndOffset()

    out_conv = wrap(conv)(kernels, stimulus)
    activations = wrap(tent_nl)(tent_coef, out_conv, bounds=tent_bounds)
    pooled = wrap(pooling)(pooling_weights, activations)
    result = wrap(reduce)(offset, pooled)
    assert result.shape == (n_samples,)

    params = SubunitParams(kernel=kernels, coef=tent_coef, pooling_weights=pooling_weights, offset=offset)
    model = SubunitModel(convolution=conv, nonlinearity=tent_nl, pooling=pooling, reduce=reduce)
    result_model = wrap(model)(params, stimulus, bounds=tent_bounds)
    np.testing.assert_allclose(result_model, result)
