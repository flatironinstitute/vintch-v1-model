"""Shared fixtures for the vintch_model.subunit tests."""

import jax
import jax.numpy as jnp
import pytest

from vintch_model.subunit import SubunitParams


@pytest.fixture
def x64():
    """Run a test in float64, restoring the global setting afterwards."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def random_params():
    """Factory for SubunitParams with 8x8x8 kernels, 9x9 pooling maps and offset 0.7."""

    def make(rng, names, n_basis=16):
        return SubunitParams(
            kernel={n: jnp.asarray(rng.standard_normal((8, 8, 8))) for n in names},
            coef={n: jnp.asarray(rng.standard_normal(n_basis)) for n in names},
            pooling_weights={n: jnp.asarray(rng.standard_normal((9, 9))) for n in names},
            offset=jnp.asarray(0.7),
        )

    return make
