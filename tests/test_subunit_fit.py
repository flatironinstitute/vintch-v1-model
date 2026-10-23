"""Fitting procedures of vintch_model.subunit."""

import equinox as eqx
import jax
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest
from nemos.regularizer import Ridge

from vintch_model.subunit import (
    Convolution,
    LossFromActivations,
    NoNormalization,
    Pooling,
    SubunitModel,
    SubunitParams,
    tents,
)
from vintch_model.subunit.fit import fix_params_from, replica_quantile_bounds, solve
from vintch_model.subunit.regularizer import CompositePenalty, DifferencePenalty

NAMES = ("excitatory", "suppressive")


def test_replica_quantile_bounds(x64):
    rng = np.random.default_rng(11)
    pre_activations = {"excitatory": rng.standard_normal((50, 9, 9)),
                       "suppressive": 3.0 * rng.standard_normal((50, 9, 9))}
    bounds = replica_quantile_bounds({n: jnp.asarray(z) for n, z in pre_activations.items()})
    for n, z in pre_activations.items():
        expected = np.quantile(z, [1 / 32, 31 / 32], method="nearest")
        np.testing.assert_allclose(bounds[n], expected, rtol=0)


def test_composite_penalty_on_active_tree(x64, random_params):
    """With coef and kernel held fixed, only the ridge term on pooling_weights remains."""
    params = random_params(np.random.default_rng(12), NAMES)
    active = eqx.tree_at(lambda p: (p.kernel, p.coef), params, ({n: None for n in NAMES},) * 2,
                         is_leaf=lambda x: x is None)
    penalty = CompositePenalty((Ridge(), DifferencePenalty(2))).penalty_fn(active, (0.2, 0.5))(active)
    expected = sum(0.5 * 0.2 * np.sum(np.asarray(w) ** 2) for w in params.pooling_weights.values())
    np.testing.assert_allclose(penalty, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "solver_name, solver_kwargs, rtol",
    [("Newton", {}, 1e-6), ("LBFGS", {"tol": 1e-10, "maxiter": 5000}, 1e-4)],
)
def test_pooling_block_matches_closed_form(x64, random_params, solver_name, solver_kwargs, rtol):
    """Pooling weights and offset solved with kernel and coef fixed = penalized least squares."""
    rng = np.random.default_rng(13)
    params = random_params(rng, NAMES)
    model = SubunitModel(NoNormalization(), Convolution(), tents(16), Pooling())
    loss = LossFromActivations(model, nmo.observation_models.GaussianObservations())
    stimulus = jnp.asarray(rng.standard_normal((300, 8, 16, 16)))
    y = jnp.asarray(rng.standard_normal(300))
    bounds = {n: jnp.array([-20.0, 20.0]) for n in NAMES}
    lam = 0.3

    # kernel and coef are fixed, so the activations are computed once
    activations = model.activations(params, model.pre_activations(params, stimulus), bounds)
    learned = SubunitParams(kernel=False, coef=False, pooling_weights=True, offset=True)
    fix_params = fix_params_from(params, learned)
    solved = solve(loss, params, fix_params, CompositePenalty((Ridge(), DifferencePenalty(2))),
                   (lam, 0.5), activations, y, solver_name=solver_name, **solver_kwargs)

    # closed form: minimize mean((y - b - A w)^2) + 0.5 * lam * ||w||^2, b unpenalized
    A = np.concatenate([np.asarray(activations[n]).reshape(300, -1) for n in NAMES], axis=1)
    X = np.column_stack([np.ones(300), A])
    P = np.diag([0.0] + [1.0] * A.shape[1])
    beta = np.linalg.solve(2 * X.T @ X / 300 + lam * P, 2 * X.T @ np.asarray(y) / 300)

    w = np.concatenate([np.asarray(solved.pooling_weights[n]).ravel() for n in NAMES])
    np.testing.assert_allclose(solved.offset, beta[0], rtol=rtol, atol=1e-8)
    np.testing.assert_allclose(w, beta[1:], rtol=rtol, atol=rtol * np.max(np.abs(beta[1:])))
    for n in NAMES:
        np.testing.assert_array_equal(solved.kernel[n], params.kernel[n])
        np.testing.assert_array_equal(solved.coef[n], params.coef[n])
