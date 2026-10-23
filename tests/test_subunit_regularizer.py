"""Regularizers of vintch_model.subunit."""

import equinox as eqx
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest
from nemos.regularizer import Ridge

from vintch_model.subunit import Convolution, Loss, NoNormalization, Pooling, SubunitModel, tents
from vintch_model.subunit.regularizer import DifferencePenalty


def explicit_difference_penalty(x, alpha, ratio, order):
    x = np.asarray(x)
    D = np.diff(np.eye(x.size), n=order, axis=0)
    return 0.5 * alpha * ((1 - ratio) * np.sum((D @ x) ** 2) + ratio * np.sum(x ** 2))


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("ratio", [0.0, 0.1])
def test_difference_penalty_matches_explicit_matrix(x64, order, ratio):
    rng = np.random.default_rng(6)
    params = {"excitatory": jnp.asarray(rng.standard_normal(16)),
              "suppressive": jnp.asarray(rng.standard_normal(16))}
    penalty = DifferencePenalty(order).penalty_fn(params, (0.3, ratio))(params)
    expected = sum(explicit_difference_penalty(params[n], 0.3, ratio, order) for n in params)
    np.testing.assert_allclose(penalty, expected, rtol=1e-12)


def test_difference_penalty_scalar_strength_means_no_ridge(x64):
    x = jnp.asarray(np.random.default_rng(7).standard_normal(16))
    penalty = DifferencePenalty(2)
    np.testing.assert_allclose(penalty.penalty_fn(x, 0.3)(x), penalty.penalty_fn(x, (0.3, 0.0))(x))


@pytest.mark.parametrize("order", [1, 2])
def test_difference_penalty_null_space(x64, order):
    """Without the ridge term, polynomials of degree below the order are not penalized;
    with it, they are."""
    x = jnp.ones(16) if order == 1 else 2.0 * jnp.arange(16.0) - 3.0
    penalty = DifferencePenalty(order)
    np.testing.assert_allclose(penalty.penalty_fn(x, (1.0, 0.0))(x), 0.0, atol=1e-10)
    np.testing.assert_allclose(penalty.penalty_fn(x, (1.0, 0.1))(x), 0.05 * np.sum(np.asarray(x) ** 2))


@pytest.mark.parametrize(
    "strength, error",
    [(jnp.ones(16), ValueError), ((1.0, 1.0), ValueError), ((1.0, 0.1, 3.0), TypeError)],
)
def test_difference_penalty_rejects_bad_strength(x64, strength, error):
    x = jnp.ones(16)
    with pytest.raises(error):
        DifferencePenalty(2).penalty_fn(x, strength)


def test_difference_penalty_on_tent_block(x64, random_params):
    """In the tent block pooling_weights are fixed (None), so only coef is penalized."""
    params = random_params(np.random.default_rng(8), ("excitatory", "suppressive"))
    active = eqx.tree_at(lambda p: (p.kernel, p.pooling_weights), params,
                         ({n: None for n in params.kernel},) * 2, is_leaf=lambda x: x is None)
    penalty = DifferencePenalty(2).penalty_fn(active, (0.5, 0.1))(active)
    expected = sum(explicit_difference_penalty(c, 0.5, 0.1, 2) for c in params.coef.values())
    np.testing.assert_allclose(penalty, expected, rtol=1e-12)


def test_penalized_loss_is_loss_plus_penalty(x64, random_params):
    rng = np.random.default_rng(9)
    names = ("excitatory", "suppressive")
    params = random_params(rng, names)
    stimulus = jnp.asarray(rng.standard_normal((6, 8, 16, 16)))
    y = jnp.asarray(rng.standard_normal(6))
    bounds = {n: jnp.array([-20.0, 20.0]) for n in names}
    loss = Loss(SubunitModel(NoNormalization(), Convolution(), tents(16), Pooling()),
                nmo.observation_models.GaussianObservations())
    regularizer, strength = Ridge(), 0.2

    penalized = eqx.filter_jit(regularizer.penalized_loss(loss, params, strength))
    expected = loss(params, stimulus, y, bounds) + regularizer.penalty_fn(params, strength)(params)
    np.testing.assert_allclose(penalized(params, stimulus, y, bounds), expected, rtol=1e-12)
