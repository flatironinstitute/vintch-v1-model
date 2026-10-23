"""Regularizers of vintch_model.subunit."""

import equinox as eqx
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest
from nemos.regularizer import Ridge

from vintch_model.subunit import Convolution, NoNormalization, Loss, Pooling, SubunitModel, tents
from vintch_model.subunit.regularizer import CompositePenalty, DifferencePenalty


def explicit_difference_penalty(x, strength, order):
    D = np.diff(np.eye(x.size), n=order, axis=0)
    return 0.5 * strength * np.sum((D @ np.asarray(x)) ** 2)


@pytest.mark.parametrize("order", [1, 2])
def test_difference_penalty_matches_explicit_matrix(x64, order):
    rng = np.random.default_rng(6)
    params = {"excitatory": jnp.asarray(rng.standard_normal(16)),
              "suppressive": jnp.asarray(rng.standard_normal(16))}
    penalty = DifferencePenalty(order).penalty_fn(params, 0.3)(params)
    expected = sum(explicit_difference_penalty(params[n], 0.3, order) for n in params)
    np.testing.assert_allclose(penalty, expected, rtol=1e-12)


@pytest.mark.parametrize("order", [1, 2])
def test_difference_penalty_null_space(x64, order):
    """Polynomials of degree below the order are not penalized."""
    k = jnp.arange(16.0)
    x = jnp.ones(16) if order == 1 else 2.0 * k - 3.0
    np.testing.assert_allclose(DifferencePenalty(order).penalty_fn(x, 1.0)(x), 0.0, atol=1e-10)


def test_difference_penalty_rejects_array_strength(x64):
    x = jnp.ones(16)
    with pytest.raises(ValueError, match="scalar strengths"):
        DifferencePenalty(2).penalty_fn(x, jnp.ones(16))


def test_composite_routes_each_regularizer(x64, random_params):
    """Ridge on pooling_weights, differences on coef, nothing on kernel or offset."""
    params = random_params(np.random.default_rng(7), ("excitatory", "suppressive"))
    composite = CompositePenalty((Ridge(), DifferencePenalty(2)))
    penalty = composite.penalty_fn(params, (0.2, 0.5))(params)

    expected = sum(0.5 * 0.2 * np.sum(np.asarray(w) ** 2) for w in params.pooling_weights.values())
    expected += sum(explicit_difference_penalty(c, 0.5, 2) for c in params.coef.values())
    np.testing.assert_allclose(penalty, expected, rtol=1e-12)


def test_composite_rejects_wrong_number_of_strengths(x64, random_params):
    params = random_params(np.random.default_rng(8), ("excitatory",))
    with pytest.raises(TypeError, match="tuple of 2 strengths"):
        CompositePenalty((Ridge(), DifferencePenalty(2))).penalty_fn(params, 0.2)


def test_penalized_loss_is_loss_plus_penalty(x64, random_params):
    rng = np.random.default_rng(9)
    names = ("excitatory", "suppressive")
    params = random_params(rng, names)
    stimulus = jnp.asarray(rng.standard_normal((6, 8, 16, 16)))
    y = jnp.asarray(rng.standard_normal(6))
    bounds = {n: jnp.array([-20.0, 20.0]) for n in names}
    loss = Loss(SubunitModel(NoNormalization(), Convolution(), tents(16), Pooling()),
                nmo.observation_models.GaussianObservations())
    composite = CompositePenalty((Ridge(), DifferencePenalty(2)))
    strength = (0.2, 0.5)

    penalized = eqx.filter_jit(composite.penalized_loss(loss, params, strength))
    expected = loss(params, stimulus, y, bounds) + composite.penalty_fn(params, strength)(params)
    np.testing.assert_allclose(penalized(params, stimulus, y, bounds), expected, rtol=1e-12)
