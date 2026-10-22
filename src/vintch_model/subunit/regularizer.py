"""Regularizers for the subunit model, built on NeMoS ``Regularizer``."""

from typing import Any, Sequence

import jax
import jax.numpy as jnp
from nemos import tree_utils
from nemos._hess import LeafClaim, MatrixProperty, MatrixStructure
from nemos.regularizer import Regularizer


class DifferencePenalty(Regularizer):
    """Smoothness penalty ``0.5 * strength * sum(diff(x, n=order) ** 2)`` for each leaf.

    Penalizes the ``order``-th differences of each parameter leaf, so differences never
    cross from one leaf (one channel) to the next. Polynomials of degree below ``order``
    are not penalized.

    Parameters
    ----------
    order :
        Order of the differences: 1 penalizes slopes, 2 penalizes curvature.

    Notes
    -----
    Only scalar strengths are accepted: a strength per coefficient does not fit the
    differences, which have ``order`` fewer entries than the leaf.
    """

    _allowed_solvers = ("GradientDescent", "BFGS", "LBFGS", "NonlinearCG", "Newton")
    _default_solver = "Newton"
    # D'D is positive semidefinite and never definite: low-degree polynomials are its null space.
    _hess_property = MatrixProperty.POSITIVE_SEMI_DEFINITE
    _hess_structure = MatrixStructure.FULL
    _hess_leaf_kind = LeafClaim.UNCLAIMED

    def __init__(self, order: int = 2):
        self.order = order

    @staticmethod
    def _check_scalar(strength: Any):
        for leaf in jax.tree.leaves(strength):
            if jnp.ndim(leaf) != 0:
                raise ValueError(
                    f"DifferencePenalty takes scalar strengths, got shape {jnp.shape(leaf)}."
                )

    def _validate_strength(self, strength: Any):
        # Called by BaseRegressor's strength setter.
        strength = super()._validate_strength(strength)
        self._check_scalar(strength)
        return strength

    def _validate_strength_structure(self, params: Any, strength: Any):
        # Called by penalty_fn / penalized_loss, also without a BaseRegressor.
        structured = super()._validate_strength_structure(params, strength)
        self._check_scalar(structured)
        return structured

    def _penalty_on_subtree(self, subtree, strength: Any, **kwargs) -> jnp.ndarray:
        def penalty(coef, leaf_strength):
            return 0.5 * leaf_strength * jnp.sum(jnp.square(jnp.diff(coef, n=self.order)))

        return tree_utils.pytree_map_and_reduce(penalty, sum, subtree, strength)


class CompositePenalty(Regularizer):
    """One regularizer per regularizable subtree, in the order of ``regularizable_subtrees()``.

    For ``SubunitParams``, ``CompositePenalty((Ridge(), DifferencePenalty(2)))`` applies
    a ridge penalty to ``pooling_weights`` and a second-difference penalty to ``coef``.
    The strength is a tuple with one entry per regularizer.

    Parameters
    ----------
    regularizers :
        The regularizers, one per regularizable subtree.

    Notes
    -----
    The Hessian tag is conservative: positive semidefinite, with no leaf claimed
    definite. Combining the regularizers' own tags would be sharper.
    """

    _hess_property = MatrixProperty.POSITIVE_SEMI_DEFINITE
    _hess_structure = MatrixStructure.FULL
    _hess_leaf_kind = LeafClaim.UNCLAIMED

    def __init__(self, regularizers: Sequence[Regularizer]):
        self.regularizers = tuple(regularizers)
        # intersection of solvers
        self._allowed_solvers = tuple(
            solver
            for solver in self.regularizers[0].allowed_solvers
            if all(solver in r.allowed_solvers for r in self.regularizers)
        )
        self._default_solver = (
            "Newton" if "Newton" in self._allowed_solvers else self.regularizers[0].default_solver
        )

    def _validate_strength(self, strength: Any):
        # Called by BaseRegressor's strength setter, and below for penalty_fn / penalized_loss.
        if not isinstance(strength, tuple) or len(strength) != len(self.regularizers):
            raise TypeError(
                f"CompositePenalty takes a tuple of {len(self.regularizers)} strengths, "
                f"one per regularizer, got {strength!r}."
            )
        return tuple(r._validate_strength(s) for r, s in zip(self.regularizers, strength))

    def _validate_strength_structure(self, params: Any, strength: Any):
        strength = self._validate_strength(strength)
        return super()._validate_strength_structure(params, list(strength))

    def _penalization(self, params: Any, filter_kwargs: dict) -> jnp.ndarray:
        strength = filter_kwargs["strength"]
        return sum(
            r._penalty_on_subtree(where(params), strength=where(strength))
            for r, where in zip(self.regularizers, params.regularizable_subtrees())
        )

    def _penalty_on_subtree(self, subtree, **kwargs) -> jnp.ndarray:
        raise NotImplementedError("CompositePenalty routes each subtree in _penalization.")
