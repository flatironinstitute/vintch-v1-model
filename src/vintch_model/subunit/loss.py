"""Losses of the subunit model fits.

``Loss`` takes the stimulus; ``LossFromPreActivations`` and ``LossFromActivations``
enter the model at a later stage, for fits that hold the earlier stages fixed.
"""

import equinox as eqx
from jaxtyping import Array, Float, PyTree
from nemos.observation_models import Observations

from .model import SubunitModel
from .params import SubunitParams


class Loss(eqx.Module):
    """Negative log-likelihood of the observations under the model's prediction.

    The call signature ``(params, *args)`` is the one NeMoS solvers expect for the
    unregularized loss.

    Parameters
    ----------
    model :
        The subunit model.
    observation_model :
        A NeMoS observation model, e.g. ``GaussianObservations()``. With the Gaussian
        model the loss is the mean squared error.
    """

    model: SubunitModel
    observation_model: Observations = eqx.field(static=True)

    def __call__(
        self,
        params: SubunitParams,
        stimulus: Float[Array, "time lags rows cols"],
        y: Float[Array, " time"],
        bounds: PyTree[Float[Array, "2"]],
    ) -> Float[Array, ""]:
        prediction = self.model(params, stimulus, bounds)
        return self.observation_model._negative_log_likelihood(y, prediction)


class LossFromPreActivations(eqx.Module):
    """``Loss`` entered at the pre-activations, for fits that hold the kernel fixed.

    The pre-activations are computed once (``model.pre_activations``) and passed in, so
    the convolution is not repeated at every evaluation.

    Parameters
    ----------
    model :
        The subunit model.
    observation_model :
        A NeMoS observation model, e.g. ``GaussianObservations()``.
    """

    model: SubunitModel
    observation_model: Observations = eqx.field(static=True)

    def __call__(
        self,
        params: SubunitParams,
        pre_activations: PyTree[Float[Array, "time p_rows p_cols"]],
        y: Float[Array, " time"],
        bounds: PyTree[Float[Array, "2"]],
    ) -> Float[Array, ""]:
        activations = self.model.activations(params, pre_activations, bounds)
        prediction = self.model.output(params, activations)
        return self.observation_model._negative_log_likelihood(y, prediction)


class LossFromActivations(eqx.Module):
    """``Loss`` entered at the activations, for fits that hold the kernel and ``coef`` fixed.

    The activations are computed once (``model.activations``) and passed in, so the
    convolution and the nonlinearity are not repeated at every evaluation.

    Parameters
    ----------
    model :
        The subunit model.
    observation_model :
        A NeMoS observation model, e.g. ``GaussianObservations()``.
    """

    model: SubunitModel
    observation_model: Observations = eqx.field(static=True)

    def __call__(
        self,
        params: SubunitParams,
        activations: PyTree[Float[Array, "time p_rows p_cols"]],
        y: Float[Array, " time"],
    ) -> Float[Array, ""]:
        prediction = self.model.output(params, activations)
        return self.observation_model._negative_log_likelihood(y, prediction)
