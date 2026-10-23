"""Losses of vintch_model.subunit."""

import equinox as eqx
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest

from vintch_model.subunit import (
    Convolution,
    Loss,
    LossFromActivations,
    LossFromPreActivations,
    NoNormalization,
    Pooling,
    SubunitModel,
    tents,
)


@pytest.mark.parametrize("jit", [False, True])
def test_gaussian_loss_is_mean_squared_error(x64, random_params, jit):
    rng = np.random.default_rng(5)
    names = ("excitatory", "suppressive")
    params = random_params(rng, names)
    stimulus = jnp.asarray(rng.standard_normal((6, 8, 16, 16)))
    y = jnp.asarray(rng.standard_normal(6))
    bounds = {n: jnp.array([-20.0, 20.0]) for n in names}
    model = SubunitModel(NoNormalization(), Convolution(), tents(16), Pooling())
    loss = Loss(model, nmo.observation_models.GaussianObservations())

    call = eqx.filter_jit(loss) if jit else loss
    expected = np.mean((np.asarray(y) - np.asarray(model(params, stimulus, bounds))) ** 2)
    np.testing.assert_allclose(call(params, stimulus, y, bounds), expected, rtol=1e-12)


def test_losses_from_stages_match_full_loss(x64, random_params):
    """Entering the model at a later stage gives the same loss as from the stimulus."""
    rng = np.random.default_rng(14)
    names = ("excitatory", "suppressive")
    params = random_params(rng, names)
    stimulus = jnp.asarray(rng.standard_normal((6, 8, 16, 16)))
    y = jnp.asarray(rng.standard_normal(6))
    bounds = {n: jnp.array([-20.0, 20.0]) for n in names}
    model = SubunitModel(NoNormalization(), Convolution(), tents(16), Pooling())
    observations = nmo.observation_models.GaussianObservations()

    pre_activations = model.pre_activations(params, stimulus)
    activations = model.activations(params, pre_activations, bounds)
    full = Loss(model, observations)(params, stimulus, y, bounds)
    np.testing.assert_allclose(
        LossFromPreActivations(model, observations)(params, pre_activations, y, bounds), full, rtol=1e-12
    )
    np.testing.assert_allclose(
        LossFromActivations(model, observations)(params, activations, y), full, rtol=1e-12
    )
