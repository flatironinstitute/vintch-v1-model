from typing import Any, Literal, TypeVar, Generic, Optional
from .backend_config import get_backend
from .nonlinearity import TentNonlinearity
import einops

Tensor = TypeVar("Tensor")


class SubunitModel(Generic[Tensor]):
    """
    Implementation of the Vintch model.

    Parameters
    ----------
    subunit_kernel_shape :
        Tuple specifying the size of the subunit kernel filters.
    pooling_shape :
        Tuple specifying the shape of the pooling weights.
    n_basis_funcs :
        Number of basis functions for the nonlinearity.
    backend :
        Computational backend to use, either 'jax', 'torch' or 'numpy'.
    n_channels :
        Number of channels in the model.
    is_channel_excitatory :
        List indicating whether each channel is excitatory. Must match the number of channels.
    """

    def __init__(
        self,
        subunit_kernel_shape: tuple = (8, 8, 8),
        pooling_shape: tuple = (16, 16, 16),
        n_basis_funcs: int = 25,
        backend: Literal["jax", "torch", "numpy"] = "torch",
        n_channels: int = 2,
        is_channel_excitatory: Optional[list[bool]] = None,
    ):
        if is_channel_excitatory is None:
            is_channel_excitatory = [True] * n_channels
        assert (
            len(is_channel_excitatory) == n_channels
        ), "Length of is_channel_excitatory must match n_channels"

        self._backend = get_backend(backend)
        self._nonlinearities_mode = [
            "relu" if is_channel_excitatory[c] else "quadratic"
            for c in range(n_channels)
        ]
        self._nonlinearities_chan = [
            TentNonlinearity(
                backend_instance=self._backend,
                n_basis_funcs=n_basis_funcs,
                nonlinearity_mode=self._nonlinearities_mode[c],
            )
            for c in range(n_channels)
        ]
        self._nonlinearity_out = TentNonlinearity(
            backend_instance=self._backend,
            n_basis_funcs=n_basis_funcs,
            nonlinearity_mode="relu",
        )

        random_kernels = self._backend.randn((n_channels, 1, *subunit_kernel_shape))
        self._kernels = random_kernels / self._backend.l1_norm(random_kernels)

        random_pooling_weights = self._backend.randn((n_channels, *pooling_shape))
        self._pooling_weights = random_pooling_weights / self._backend.l1_norm(
            random_pooling_weights
        )
        if len(pooling_shape) == len(subunit_kernel_shape):
            self._pooling_dims = "n_channels time height width"
        elif len(pooling_shape) == len(subunit_kernel_shape) - 1:
            self._pooling_dims = "n_channels height width"
        else:
            raise ValueError(
                "Invalid pooling_shape length. Must match subunit_kernel length or be one less."
            )
        self._pooling_bias = self._backend.randn((1,))

        self.n_channels = n_channels
        self.n_basis_funcs = n_basis_funcs

    def __call__(self, *args, **kwds):
        return self._predict(*args, **kwds)

    @property
    def kernels(self):
        return self._kernels

    @kernels.setter
    def kernels(self, new_kernels: Any):
        """Setter for kernel filters."""
        assert (
            new_kernels.shape == self._kernels.shape
        ), f"Expected kernel shape to be {self._kernels.shape}, got {new_kernels.shape} instead."
        self._backend.check_input_type(new_kernels)
        self._kernels = new_kernels / self._backend.l1_norm(new_kernels)

    @property
    def pooling_weights(self):
        return self._pooling_weights

    @pooling_weights.setter
    def pooling_weights(self, weights: Any):
        """Setter for pooling weights."""
        assert (
            weights.shape == self._pooling_weights.shape
        ), f"Expected pooling weights shape to be {self._pooling_weights.shape}, got {weights.shape} instead."
        self._backend.check_input_type(weights)
        self._pooling_weights = weights / self._backend.l1_norm(weights)

    @property
    def pooling_bias(self):
        return self._pooling_bias

    @pooling_bias.setter
    def pooling_bias(self, bias: Any):
        """Setter for pooling bias."""
        assert (
            bias.shape == self._pooling_bias.shape
        ), f"Expected pooling bias shape to be {self._pooling_bias.shape}, got {bias.shape} instead."
        self._backend.check_input_type(bias)
        self._pooling_bias = bias

    def _predict(self, x: Tensor) -> Tensor:
        """
        Forward pass through the whole model.

        Parameters
        ----------
        x :
            Input tensor with shape [batch_size, 1, time, height, width].

        Returns
        -------
        out :
            Output tensor with shape [batch_size, time].
        """
        self._backend.check_input_type(x)
        assert (
            x.shape[1] == 1
        ), f"Dimension 1 of input must be 1 (grayscale), got {x.shape[1]} instead."

        x = self._backend.set_dtype(x, self._kernels.dtype)

        # check if the input range is between 0 and 1
        x_min, x_max = x.min(), x.max()
        assert (0 <= x_min <= 1) and (
            0 <= x_max <= 1
        ), f"Input values must be in the range [0, 1], got min: {x_min}, max: {x_max} instead."

        # [batch_size, n_channels, time, height, width]
        sub_convolved = self._convolve(x, self._kernels)
        # [batch_size, n_channels, time, height, width]
        sub_activated = self._apply_nonlinearities(
            sub_convolved, self._nonlinearities_chan
        )
        # [batch_size, time]
        weighted_pooled = self._weighted_pooling(sub_activated, self.pooling_weights)
        # [batch_size, time]
        generator_signal = self._apply_bias(weighted_pooled, self.pooling_bias)
        # [batch_size, time]
        out = self._nonlinearity_out.forward(generator_signal)
        return out

    def _convolve(self, x: Tensor, kernel: Optional[Tensor] = None) -> Tensor:
        """
        Perform convolution operation.

        Parameters
        ----------
        x :
            Input tensor with shape [batch_size, 1, time, height, width].
        kernel :
            Kernel tensor with shape matching the subunit kernel. [n_kernels, 1, kT, kH, kW].

        Returns
        -------
        convolved :
            Convolved tensor with shape [batch_size, n_channels, time, height, width].
        """

        if kernel is None:
            kernel = self._kernels
        return self._backend.convolve(x, kernel)

    def _apply_nonlinearities(
        self, x: Tensor, nonlinearities: Optional[list] = None
    ) -> Tensor:
        """
        Apply channel-specific nonlinearities.

        Parameters
        ----------
        x :
            Input tensor with shape [batch_size, n_channels, time, height, width].
        nonlinearities :
            List of nonlinearity objects, one for each channel.

        Returns
        -------
        transformed :
            Transformed tensor with shape [batch_size, n_channels, time, height, width].
        """
        if nonlinearities is None:
            nonlinearities = self._nonlinearities_chan
        transformed = [
            nonlinearity(x[:, c]) for c, nonlinearity in enumerate(nonlinearities)
        ]
        stacked_transformed, _ = einops.pack(
            transformed, pattern="batch_size * time height width"
        )
        return stacked_transformed

    def _weighted_pooling(
        self, x: Tensor, pooling_weights: Optional[Tensor] = None
    ) -> Tensor:
        """
        Perform weighted pooling operation.

        This operation computes a weighted sum of the input tensor using the
        provided pooling weights and sum over the spatial dimensions.

        Parameters
        ----------
        x :
            Input tensor with shape [batch_size, n_channels, time, height, width].
        pooling_weights :
            Pooling weights tensor with shape matching the pooling dimensions.

        Returns
        -------
        pooled :
            Pooled tensor with shape [batch_size, time].
        """
        if pooling_weights is None:
            pooling_weights = self._pooling_weights
        pooled = einops.einsum(
            x,
            pooling_weights,
            f"batch_size n_channels time height width, {self._pooling_dims} -> batch_size time",
        )
        return pooled

    def _apply_bias(self, x: Tensor, pooling_bias: Optional[Tensor] = None) -> Tensor:
        """
        Apply bias to the input tensor.

        Parameters
        ----------
        x :
            Input tensor with shape [batch_size, time].
        pooling_bias :
            Bias tensor with shape [1,].
        """
        if pooling_bias is None:
            pooling_bias = self._pooling_bias
        return x + pooling_bias
