"""Fitting procedures of vintch_model.subunit."""

import equinox as eqx
import jax
import jax.numpy as jnp
import nemos as nmo
import numpy as np
import pytest
from nemos.regularizer import Ridge, UnRegularized

from vintch_model.subunit import (
    Convolution,
    L2Normalization,
    Loss,
    LossFromActivations,
    LossFromPreActivations,
    NoNormalization,
    Pooling,
    SubunitModel,
    SubunitParams,
    tents,
)
from vintch_model.subunit.fit import block_solver, replica_fit, replica_quantile_bounds, run_block
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
    solver = block_solver(loss, learned, Ridge(), lam, params,
                          solver_name=solver_name, **solver_kwargs)
    solved = run_block(solver, learned, params, activations, y)

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
    solver = block_solver(loss, learned, DifferencePenalty(2), (lam, ratio), params)
    solved = run_block(solver, learned, params, pre_activations, jnp.asarray(y), bounds)

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


def solve_kernel_block(random_params, normalization, start_scale=1.0):
    """Recover the kernels of a known model from noise-free data, other parameters at truth.

    Returns the true and solved parameters, and the loss at both.
    """
    rng = np.random.default_rng(16)
    # small problem: 2 lags of 4x4 kernels on 8x8 frames, 5x5 positions, 64 unknowns
    params_true = random_params(rng, NAMES, kernel_shape=(2, 4, 4), pool_shape=(5, 5))
    model = SubunitModel(normalization, Convolution(), tents(16), Pooling())
    loss = Loss(model, nmo.observation_models.GaussianObservations())
    stimulus = jnp.asarray(rng.standard_normal((400, 2, 8, 8)))
    bounds = replica_quantile_bounds(model.pre_activations(params_true, stimulus))
    y = model(params_true, stimulus, bounds)

    # start: each kernel scaled and perturbed by 10% of its own size
    start_kernel = {n: start_scale * k + 0.1 * jnp.std(k) * jnp.asarray(rng.standard_normal(k.shape))
                    for n, k in params_true.kernel.items()}
    start = eqx.tree_at(lambda p: p.kernel, params_true, start_kernel)
    learned = SubunitParams(kernel=True, coef=False, pooling_weights=False, offset=False)
    solver = block_solver(loss, learned, UnRegularized(), None, start,
                          solver_name="LBFGS", tol=1e-10, maxiter=5000)
    solved = run_block(solver, learned, start, stimulus, y, bounds)
    return params_true, solved, loss(solved, stimulus, y, bounds), float(jnp.var(y))


def test_kernel_block_recovers_kernel(x64, random_params):
    params_true, solved, final_loss, y_var = solve_kernel_block(random_params, NoNormalization())
    for n in NAMES:
        k_true, k_hat = np.asarray(params_true.kernel[n]), np.asarray(solved.kernel[n])
        assert np.linalg.norm(k_hat - k_true) <= 1e-6 * np.linalg.norm(k_true)
        np.testing.assert_array_equal(solved.coef[n], params_true.coef[n])
        np.testing.assert_array_equal(solved.pooling_weights[n], params_true.pooling_weights[n])
    assert final_loss <= 1e-12 * y_var


def test_kernel_block_recovers_normalized_kernel(x64, random_params):
    """With L2Normalization the raw kernel's scale is not identified: the start is three
    times the true kernel, and only the normalized kernel is compared."""
    params_true, solved, final_loss, y_var = solve_kernel_block(
        random_params, L2Normalization(), start_scale=3.0)
    normalize = L2Normalization()
    k_true, k_hat = normalize(params_true.kernel), normalize(solved.kernel)
    for n in NAMES:
        assert np.linalg.norm(np.asarray(k_hat[n] - k_true[n])) <= 1e-6
    assert final_loss <= 1e-12 * y_var


def test_replica_fit_lowers_the_loss(x64, random_params):
    """From perturbed kernels on noise-free data, the penalized loss decreases at every
    iteration and ends at least 1000 times below its first value."""
    rng = np.random.default_rng(17)
    params_true = random_params(rng, NAMES, kernel_shape=(2, 4, 4), pool_shape=(5, 5))
    model = SubunitModel(NoNormalization(), Convolution(), tents(16), Pooling())
    stimulus = jnp.asarray(rng.standard_normal((400, 2, 8, 8)))
    bounds = replica_quantile_bounds(model.pre_activations(params_true, stimulus))
    y = model(params_true, stimulus, bounds)
    start_kernel = {n: k + 0.1 * jnp.std(k) * jnp.asarray(rng.standard_normal(k.shape))
                    for n, k in params_true.kernel.items()}
    start = eqx.tree_at(lambda p: p.kernel, params_true, start_kernel)

    _, _, losses = replica_fit(model, nmo.observation_models.GaussianObservations(), start,
                               bounds, stimulus, y, 1e-6, (1e-6, 1e-3), n_inner=2, max_iter=30)
    losses = np.asarray(losses)
    losses = losses[~np.isnan(losses)]
    assert np.all(np.diff(losses) <= 1e-6 * losses[:-1])
    assert losses[-1] <= 1e-3 * losses[0]
