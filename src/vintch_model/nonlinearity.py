from typing import TypeVar, Generic, Literal
import einops
from .backend_config import BackendBase
import numpy as np

Tensor = TypeVar("Tensor")


class TentNonlinearity(Generic[Tensor]):
    """
    Tent-based nonlinearity.

    Parameters
    ----------
    n_basis_funcs :
        Number of basis functions to use. Must be a positive integer.
    backend_instance :
        Backend to use. Must be an instance of BackendBase.
    nonlinearity_mode :
        Type of nonlinearity to initialize weights for. Weights will be fitted to data.
    """

    def __init__(
        self,
        backend_instance: BackendBase,
        n_basis_funcs: int = 15,
        nonlinearity_mode: Literal["relu", "quadratic", "linear"] = "relu",
    ):
        if n_basis_funcs < 1:
            raise ValueError("basis functions must be a positive integer.")

        if nonlinearity_mode not in ["relu", "quadratic", "linear"]:
            raise ValueError(
                f"nonlinearity_mode must be 'relu', 'quadratic', or 'linear', got '{nonlinearity_mode}'"
            )

        self._n_basis_funcs = n_basis_funcs
        self._backend = backend_instance

        if nonlinearity_mode == "relu":
            self._initialize_weights = self._initialize_relu_weights
        elif nonlinearity_mode == "quadratic":
            self._initialize_weights = self._initialize_quadratic_weights
        elif nonlinearity_mode == "linear":
            self._initialize_weights = self._initialize_linear_weights

        self._initialize_basis_functions()
        self._initialize_weights()

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    @property
    def weights(self):
        """
        Get the weights of the nonlinearity.
        """
        return self._weights

    @weights.setter
    def weights(self, value):
        """
        Set the weights of the nonlinearity.
        """
        self._weights = value

    def _initialize_basis_functions(self):
        """
        Initialize the basis functions for the tent nonlinearity.
        """
        self._tent_centers = self._backend.lib.linspace(-1, 1, self._n_basis_funcs)

    def _initialize_relu_weights(self):
        """Initialize weights for relu mode."""
        zero_array = self._backend.convert_array(
            np.array([0.0]), dtype=self._tent_centers.dtype
        )
        self._weights = self._backend.lib.maximum(zero_array, self._tent_centers)

    def _initialize_quadratic_weights(self):
        """Initialize weights for quadratic mode."""
        self._weights = self._tent_centers**2

    def _initialize_linear_weights(self):
        """Initialize weights for linear mode."""
        self._weights = self._tent_centers

    def _tents_transform(self, x: Tensor) -> Tensor:
        """Compute tent basis features for input x."""
        flat_x = x.flatten()
        tent_width = self._tent_centers[1] - self._tent_centers[0]

        # Calculate the absolute difference and normalize to the tent width
        diff = self._backend.lib.abs(flat_x[:, None] - self._tent_centers[None, :])
        diff = diff / tent_width

        # Apply the tent function
        values = 1 - diff

        # Ensure non-negative values
        zero_array = self._backend.convert_array(np.array([0.0]), dtype=x.dtype)
        features = self._backend.lib.maximum(values, zero_array)
        return features

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the nonlinearity and weights to the input.

        Parameters
        ----------
        x :
            Input data.

        Returns
        -------
        Output after applying the nonlinearity and weights, reshaped to match input shape.
        """
        self._backend.check_input_type(x)
        self._weights = self._backend.set_dtype(self._weights, x.dtype)

        original_shape = x.shape

        features = self._tents_transform(x)

        output_flat = einops.einsum(
            features, self._weights, "extended_dim n_basis, n_basis -> extended_dim"
        )

        return output_flat.reshape(original_shape)
