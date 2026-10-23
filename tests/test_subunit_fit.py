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
    LossFromPreActivations,
    NoNormalization,
    Pooling,
    SubunitModel,
    SubunitParams,
    tents,
)
from vintch_model.subunit.fit import fix_params_from, replica_quantile_bounds, solve
from vintch_model.subunit.regularizer import DifferencePenalty

NAMES = ("excitatory", "suppressive")


def test_replica_quantile_bounds(x64):
    rng = np.random.default_rng(11)
    pre_activations = {"excitatory": rng.standard_normal((50, 9, 9)),
                       "suppressive": 3.0 * rng.standard_normal((50, 9, 9))}
    bounds = replica_quantile_bounds({n: jnp.asarray(z) for n, z in pre_activations.items()})
    for n, z in pre_activations.items():
        expected = np.quantile(z, [1 / 32, 31 / 32], method="nearest")
        np.testing.assert_allclose(bounds[n], expected, rtol=0)


def test_ridge_on_pooling_block(x64, random_params):
    """With coef and kernel held fixed (None), Ridge penalizes only pooling_weights."""
    params = random_params(np.random.default_rng(12), NAMES)
    active = eqx.tree_at(lambda p: (p.kernel, p.coef), params, ({n: None for n in NAMES},) * 2,
                         is_leaf=lambda x: x is None)
    penalty = Ridge().penalty_fn(active, 0.2)(active)
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
    solved = solve(loss, params, fix_params, Ridge(), lam, activations, y,
                   solver_name=solver_name, **solver_kwargs)

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


def solve_tent_block(random_params, lam, ratio):
    """Solve coef and offset with kernel and pooling fixed; return what the checks need.

    The block is linear: prediction = X @ beta, with beta = (offset, coef per channel)
    and X = [1, F], F the tent features pooled with the fixed pooling weights.
    """
    rng = np.random.default_rng(15)
    params = random_params(rng, NAMES)
    model = SubunitModel(NoNormalization(), Convolution(), tents(16), Pooling())
    loss = LossFromPreActivations(model, nmo.observation_models.GaussianObservations())
    stimulus = jnp.asarray(rng.standard_normal((300, 8, 16, 16)))
    y = np.asarray(rng.standard_normal(300))

    # the kernel is fixed, so the pre-activations and the bounds are computed once
    pre_activations = model.pre_activations(params, stimulus)
    bounds = replica_quantile_bounds(pre_activations)
    learned = SubunitParams(kernel=False, coef=True, pooling_weights=False, offset=True)
    solved = solve(loss, params, fix_params_from(params, learned), DifferencePenalty(2),
                   (lam, ratio), pre_activations, jnp.asarray(y), bounds)

    features = model.nonlinearity.features(pre_activations, bounds)
    F = np.concatenate([np.einsum("tijl,ij->tl", np.asarray(features[n]),
                                  np.asarray(params.pooling_weights[n])) for n in NAMES], axis=1)
    X = np.column_stack([np.ones(300), F])
    D = np.diff(np.eye(16), n=2, axis=0)
    P = np.zeros((33, 33))
    P[1:17, 1:17] = P[17:, 17:] = (1 - ratio) * D.T @ D + ratio * np.eye(16)
    normal_matrix = 2 * X.T @ X / 300 + lam * P
    normal_rhs = 2 * X.T @ y / 300

    def objective(beta):
        return np.mean((y - X @ beta) ** 2) + 0.5 * lam * beta @ P @ beta

    def as_beta(p):
        return np.concatenate([[float(p.offset)]] + [np.asarray(p.coef[n]) for n in NAMES])

    prediction = np.asarray(model.output(solved, model.activations(solved, pre_activations, bounds)))
    return params, solved, X, normal_matrix, normal_rhs, objective, as_beta, prediction


def test_tent_block(x64, random_params):
    """coef and offset solved with kernel and pooling fixed, no ridge term.

    With clamped tents the basis sums to 1, so adding a constant to a channel's coef and
    subtracting it (times the channel's total pooling weight) from the offset changes
    neither the predictions nor the second-difference penalty: the minimizers form a
    line. The predictions and the objective value are unique, and Newton should not move
    along that line.
    """
    params, solved, X, A, b, objective, as_beta, prediction = solve_tent_block(
        random_params, lam=0.5, ratio=0.0)
    beta_direct = np.linalg.pinv(A) @ b
    beta_solved, beta_start = as_beta(solved), as_beta(params)

    np.testing.assert_allclose(prediction, X @ beta_direct, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(objective(beta_solved), objective(beta_direct), rtol=1e-10)

    # null directions: +1 on one channel's coef, -(its total pooling weight) on the offset
    null = np.zeros((33, 2))
    for c, n in enumerate(NAMES):
        null[0, c] = -np.sum(np.asarray(params.pooling_weights[n]))
        null[1 + 16 * c:17 + 16 * c, c] = 1.0
    np.testing.assert_allclose(X @ null, 0.0, atol=1e-8)
    step = beta_solved - beta_start
    along_null = null @ np.linalg.lstsq(null, step, rcond=None)[0]
    assert np.linalg.norm(along_null) <= 1e-6 * np.linalg.norm(step)

    for n in NAMES:
        np.testing.assert_array_equal(solved.kernel[n], params.kernel[n])
        np.testing.assert_array_equal(solved.pooling_weights[n], params.pooling_weights[n])


def test_tent_block_with_ridge(x64, random_params):
    """With a ridge ratio r > 0 the minimizer is unique: Newton gives the parameters
    of the direct solve, not only its predictions."""
    params, solved, X, A, b, objective, as_beta, prediction = solve_tent_block(
        random_params, lam=0.5, ratio=1e-3)
    beta_direct = np.linalg.solve(A, b)
    np.testing.assert_allclose(as_beta(solved), beta_direct, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(prediction, X @ beta_direct, rtol=1e-6, atol=1e-8)
