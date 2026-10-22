"""Objective of the subunit model fits."""

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
