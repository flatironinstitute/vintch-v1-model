"""Fitting procedures for the subunit model.

Functions prefixed ``replica_`` belong to the original fit of Vintch et al. (2015)
and reproduce its choices; the others are the general fit.
"""

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from nemos.observation_models import Observations
from nemos.regularizer import Regularizer, Ridge, UnRegularized
from nemos.solvers import get_solver

from .loss import Loss, LossFromActivations, LossFromPreActivations
from .model import SubunitModel
from .params import SubunitParams
from .regularizer import DifferencePenalty

POOLING_BLOCK = SubunitParams(kernel=False, coef=False, pooling_weights=True, offset=True)
TENT_BLOCK = SubunitParams(kernel=False, coef=True, pooling_weights=False, offset=True)
KERNEL_BLOCK = SubunitParams(kernel=True, coef=False, pooling_weights=False, offset=False)


def block_solver(
    loss: Callable,
    learned: SubunitParams,
    regularizer: Regularizer,
    strength: Any,
    init_params: SubunitParams,
    solver_name: str | None = None,
    **solver_kwargs,
):
    """NeMoS solver for the learned parameters of a block, built once and reused.

    The fixed parameters are not captured: they are the first argument after the
    learned ones, so the same solver (and its compiled code) serves every call of
    ``run_block``.

    Parameters
    ----------
    loss :
        Unregularized loss, called as ``loss(params, *args)``.
    learned :
        Prefix tree of booleans marking what is learned, e.g.
        ``SubunitParams(kernel=False, coef=False, pooling_weights=True, offset=True)``.
    regularizer, strength :
        Penalty on the learned parameters, as for any NeMoS solver.
    init_params :
        Parameters with the structure of the ones to be solved.
    solver_name :
        A NeMoS solver name (e.g. ``"Newton"``, ``"LBFGS"``). Defaults to the
        regularizer's default solver.
    **solver_kwargs :
        Passed to the solver.
    """
    active, _ = eqx.partition(init_params, learned)

    def active_loss(p, frozen, *args):
        return loss(eqx.combine(p, frozen), *args)

    if solver_name is None:
        solver_name = regularizer.default_solver
    solver_cls = get_solver(solver_name).implementation
    return solver_cls(
        active_loss, regularizer, strength, has_aux=False, init_params=active, **solver_kwargs
    )


def run_block(solver, learned: SubunitParams, params: SubunitParams, *args) -> SubunitParams:
    """Solve the learned parameters with ``solver``, holding the others at their values in ``params``."""
    active, frozen = eqx.partition(params, learned)
    active, _, _ = solver.run(active, frozen, *args)
    return eqx.combine(active, frozen)


def replica_quantile_bounds(
    pre_activations: PyTree[Float[Array, "time p_rows p_cols"]],
) -> PyTree[Float[Array, "2"]]:
    """Tent bounds of the original fit (Vintch et al., 2015).

    Each channel's bounds are the 1/32 and 31/32 quantiles of its pre-activations, over
    all frames and positions. The original recomputes them before every tent solve.
    """
    q = jnp.array([1 / 32, 31 / 32])
    # Close to, but not exactly, the original rule: MATLAB takes the sorted value at
    # 0-based index round(q * n) - 1, "nearest" takes index round(q * (n - 1)). The two
    # can pick neighboring order statistics; the difference in the bounds is negligible.
    return jax.tree.map(lambda z: jnp.quantile(z, q, method="nearest"), pre_activations)


def replica_fit(
    model: SubunitModel,
    observation_model: Observations,
    params: SubunitParams,
    bounds: PyTree[Float[Array, "2"]],
    stimulus: Float[Array, "time lags rows cols"],
    y: Float[Array, " time"],
    pooling_strength: float,
    coef_strength: tuple[float, float],
    n_inner: int = 10,
    max_iter: int = 100,
    tol: float = 1e-8,
    kernel_solver_name: str = "LBFGS",
    kernel_solver_kwargs: dict | None = None,
    verbose: bool = False,
) -> tuple[SubunitParams, PyTree[Float[Array, "2"]], Float[Array, " max_iter"]]:
    """Block coordinate descent of the original fit (Vintch et al., 2015).

    Each iteration runs ``n_inner`` rounds of (pooling block, bounds re-placed at the
    1/32 and 31/32 quantiles, tent block), then one kernel block, as the original does.
    The model and objective are the original's; the block solves use NeMoS solvers
    (Newton for pooling and tents, L-BFGS for the kernel) instead of its heuristics.

    Parameters
    ----------
    model, observation_model :
        The subunit model and the NeMoS observation model of the loss.
    params, bounds :
        Starting parameters and tent bounds (initialization is a separate step).
    stimulus, y :
        Training data.
    pooling_strength :
        ``Ridge`` strength on ``pooling_weights``.
    coef_strength :
        ``DifferencePenalty`` strength ``(alpha, r)`` on ``coef``.
    n_inner :
        Rounds of pooling and tent blocks per iteration (10 in the original).
    max_iter :
        Maximum number of iterations.
    tol :
        Stop when the relative change of the penalized loss over an iteration,
        ``|previous - loss| / |previous|``, is below ``tol``; an increase larger than that
        does not stop the fit. The re-placed bounds change the objective, so the change is
        measured across whole iterations only.
    kernel_solver_name, kernel_solver_kwargs :
        Solver of the kernel block (any registered NeMoS name) and its options, e.g.
        ``{"maxiter": 50}``. The kernel need not be solved to full precision every
        iteration: the next iteration starts from it again.
    verbose :
        Print the penalized loss after each iteration (from inside the compiled loop).

    Returns
    -------
    params, bounds :
        Fitted parameters and the bounds they were fitted with.
    losses :
        Penalized loss after each iteration; ``nan`` for iterations not run.
    """
    # solvers and penalties are built once, outside the compiled loop
    pooling_solver = block_solver(LossFromActivations(model, observation_model), POOLING_BLOCK,
                                  Ridge(), pooling_strength, params, solver_name="Newton")
    tent_solver = block_solver(LossFromPreActivations(model, observation_model), TENT_BLOCK,
                               DifferencePenalty(2), coef_strength, params, solver_name="Newton")
    kernel_loss = Loss(model, observation_model)
    kernel_solver = block_solver(kernel_loss, KERNEL_BLOCK, UnRegularized(), None, params,
                                 solver_name=kernel_solver_name, **(kernel_solver_kwargs or {}))
    pooling_penalty = Ridge().penalty_fn(params, pooling_strength)
    tent_penalty = DifferencePenalty(2).penalty_fn(params, coef_strength)

    def penalized_loss(params, bounds):
        return kernel_loss(params, stimulus, y, bounds) + pooling_penalty(params) + tent_penalty(params)

    def inner_round(_, carry):
        params, bounds, pre_activations = carry
        activations = model.activations(params, pre_activations, bounds)
        params = run_block(pooling_solver, POOLING_BLOCK, params, activations, y)
        bounds = replica_quantile_bounds(pre_activations)
        params = run_block(tent_solver, TENT_BLOCK, params, pre_activations, y, bounds)
        return params, bounds, pre_activations

    def iteration(carry):
        i, params, bounds, losses, _ = carry
        pre_activations = model.pre_activations(params, stimulus)
        params, bounds, _ = jax.lax.fori_loop(
            0, n_inner, inner_round, (params, bounds, pre_activations)
        )
        params = run_block(kernel_solver, KERNEL_BLOCK, params, stimulus, y, bounds)
        loss = penalized_loss(params, bounds)
        previous = losses[i - 1]
        converged = (i > 0) & (jnp.abs(previous - loss) <= tol * jnp.abs(previous))
        if verbose:
            jax.debug.print("iteration {i}: penalized loss {loss}", i=i + 1, loss=loss)
        return i + 1, params, bounds, losses.at[i].set(loss), converged

    def keep_going(carry):
        i, _, _, _, converged = carry
        return (i < max_iter) & ~converged

    losses = jnp.full(max_iter, jnp.nan)
    _, params, bounds, losses, _ = jax.lax.while_loop(
        keep_going, iteration, (0, params, bounds, losses, False)
    )
    return params, bounds, losses
