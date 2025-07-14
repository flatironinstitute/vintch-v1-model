from typing import Any, Literal
from .backend_config import get_backend
from .nonlinearity import TentNonlinearity
import einops


class SubunitModel:
    """
    Implementation of the Vintch model.

    Parameters
    ----------
    subunit_kernel :
        Tuple specifying the size of the subunit kernel filters.
    n_dim :
        Number of dimension of the input. This is also the dimension
        of the pooling weights.
    n_basis_funcs :
        Number of basis functions for the nonlinearity.
    backend :
        Computational backend to use, either 'jax', 'torch' or 'numpy'.
    n_channels :
        Number of channels in the model.
    is_channel_excitatory :
        List indicating whether each channel is excitatory.
    pooling_in_time :
        Whether to perform weight pooling also in time.
    """

    def __init__(
        self,
        subunit_kernel: tuple = (8, 8, 8),
        pooling_shape: tuple = (16, 16, 16),
        n_basis_funcs: int = 25,
        backend: Literal["jax", "torch", "numpy"] = "torch",
        n_channels: int = 2,
        is_channel_excitatory: list = [True, False],
    ):
        assert (
            len(is_channel_excitatory) == n_channels
        ), "Length of is_channel_excitatory must match n_channels"
        self.subunit_kernel = subunit_kernel
        self.pooling_shape = pooling_shape
        self.n_channels = n_channels
        self._backend = get_backend(backend)
        self._kernels = self._backend.randn((n_channels, 1, *subunit_kernel))

        if len(pooling_shape) == len(subunit_kernel):
            self._pooling_weights = self._backend.randn((n_channels, *pooling_shape))
            self._pooling_dims = "out_channels time height width"
        elif len(pooling_shape) == len(subunit_kernel) - 1:
            self._pooling_weights = self._backend.randn((n_channels, *pooling_shape))
            self._pooling_dims = "out_channels height width"
        else:
            raise ValueError(
                "Invalid pooling_size length. Must match subunit_kernel length or be one less."
            )

        self._pooling_biases = self._backend.randn((n_channels, 1))
        self.nonlinearities_chan = [
            TentNonlinearity(backend_class=self._backend, n_basis_funcs=n_basis_funcs)
            for _ in range(n_channels)
        ]
        self.nonlinearity_out = TentNonlinearity(
            backend_class=self._backend, n_basis_funcs=n_basis_funcs
        )  # should check the number of basis functions

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

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
        self._kernels = new_kernels

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
        self._pooling_weights = weights

    @property
    def pooling_biases(self):
        return self._pooling_biases

    @pooling_biases.setter
    def pooling_biases(self, biases: Any):
        """Setter for pooling biases."""
        assert (
            biases.shape == self._pooling_biases.shape
        ), f"Expected pooling biases shape to be {self._pooling_biases.shape}, got {biases.shape} instead."
        self._backend.check_input_type(biases)
        self._pooling_biases = biases

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend_name: Literal["jax", "torch", "numpy"]):
        """Setter for backend class."""
        self._backend = get_backend(backend_name)

    def forward(self, x):
        """
        Forward pass through the whole model.
        """
        self._backend.check_input_type(x)
        # ensure the input is 5D and the second dimension is 1 (grayscale)
        if x.shape[1] != 1:
            raise ValueError(
                f"Dimension 1 of input must be 1 (grayscale), got {x.shape[1]} instead."
            )
        generator_signal = self.generator_signal_forward(x)  # [batch, time]
        out = self.nonlinearity_out.forward(generator_signal)  # [batch, time]
        return out

    def generator_signal_forward(self, x):
        """
        Compute the generator signal.

        The generator signal is sent through the final nonlinearity to produce
        the cell's firing rate.

        Parameters
        ----------
        x :
            Input tensor with shape [batch, channels, time, height, width].

        Returns
        -------
        output :
            Generator signal with shape [batch, time].
        """
        # [batch, channels, time, height, width]
        convolved = self.convolve(x, self.kernels)
        # [batch, channels, time, height, width]
        activated = self.nonlinearity_pass(convolved, self.nonlinearities_chan)
        # [batch, channels, time]
        pooled = self.weighted_pooling(
            activated, self.pooling_weights, biases=self.pooling_biases
        )
        # [batch, time]
        generator_signal = pooled.sum(axis=1)
        return generator_signal

    def convolve(self, x, kernel):
        return self._backend.convolve(x, kernel)

    def nonlinearity_pass(self, x, nonlinearities):
        transformed = [
            nonlinearity(x[:, c]) for c, nonlinearity in enumerate(nonlinearities)
        ]

        stacked_transformed, _ = einops.pack(
            transformed, pattern="batch * time height width"
        )
        return stacked_transformed

    def weighted_pooling(self, x, pooling_weights, biases):
        pooled = einops.einsum(
            x,
            pooling_weights,
            f"batch_size in_channels time height width, {self._pooling_dims} -> batch_size out_channels time",
        )

        pooled = pooled + biases
        return pooled
