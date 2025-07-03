from typing import Any
from .backend_config import get_backend
from .nonlinearity import TentNonlinearity
from einops import einsum

# type_ArrayLike = Union[np.ndarray, "jax.numpy.ndarray", "torch.Tensor"]


class SubunitModel:
    """
    Implementation of the Vintch model.

    Parameters
    ----------
    kernel_size :
        Size of the kernel filters. Defaults to 8.
    kernel_time_depth :
        Time depth of the kernel filters. Defaults to 8.
    n_dim :
        Number of dimension of the input, defaults to 16. This is also the dimension
        of the pooling weights.
    n_basis_funcs :
        Number of basis functions for the nonlinearity. Defaults to 25.
    backend :
        Computational backend to use, either 'jax' or 'torch'. Defaults to 'torch'.
    n_channels :
        Number of channels in the model. Defaults to 2.
    is_channel_excitatory :
        List indicating whether each channel is excitatory. Defaults to [True, False].
    pooling_in_time :
        Whether to perform weight pooling also in time. Defaults to True.
    """

    def __init__(
        self,
        kernel_size: int = 8,
        kernel_time_depth: int = 8,
        n_dim: int = 16,
        n_basis_funcs: int = 25,
        backend: str = "torch",
        n_channels: int = 2,
        is_channel_excitatory: list = [True, False],
        pooling_in_time: bool = True,
        kernels: Any = None,
        pooling_weights: Any = None,
        pooling_biases: Any = None,
    ):
        assert (
            len(is_channel_excitatory) == n_channels
        ), "Length of is_channel_excitatory must match n_channels"
        self.kernel_size = kernel_size
        self.kernel_time_depth = kernel_time_depth
        self.n_dim = n_dim
        self.n_channels = n_channels
        self.backend_class = get_backend(backend)
        self._kernels = self.backend_class.randn(
            (n_channels, kernel_time_depth, kernel_size, kernel_size)
        )
        if pooling_in_time:
            self._pooling_weights = self.backend_class.randn(
                (n_channels, n_dim, n_dim, n_dim)
            )  # which time to take
        else:
            self._pooling_weights = self.backend_class.randn((n_channels, n_dim, n_dim))
        self.pooling_in_time = pooling_in_time
        self._pooling_biases = self.backend_class.randn((n_channels,))
        self.nonlinearities_chn = [
            TentNonlinearity(n_basis_funcs, soft=True, backend=self.backend_class.name)
            for _ in range(n_channels)
        ]
        self.nonlinearity_out = TentNonlinearity(
            n_basis_funcs, soft=True, backend=self.backend_class.name
        )

        if kernels is not None:
            self.kernels = kernels
        if pooling_weights is not None:
            self.pooling_weights = pooling_weights
        if pooling_biases is not None:
            self.pooling_biases = pooling_biases

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    @property
    def kernels(self):
        return self._kernels

    @kernels.setter
    def kernels(self, new_kernels: Any):
        """Setter for kernel filters."""
        assert new_kernels.shape == self._kernels.shape, (
            f"Expected kernel shape to be {self._kernels.shape}, got {new_kernels.shape} instead."
        )
        self.backend_class.check_input_type(new_kernels)
        self._kernels = new_kernels

    @property
    def pooling_weights(self):
        return self._pooling_weights

    @pooling_weights.setter
    def pooling_weights(self, weights: Any):
        """Setter for pooling weights."""
        assert weights.shape == self._pooling_weights.shape, (
            f"Expected pooling weights shape to be {self._pooling_weights.shape}, got {weights.shape} instead."
        )
        self.backend_class.check_input_type(weights)
        self._pooling_weights = weights

    @property
    def pooling_biases(self):
        return self._pooling_biases

    @pooling_biases.setter
    def pooling_biases(self, biases: Any):
        """Setter for pooling biases."""
        assert biases.shape == self._pooling_biases.shape, (
            f"Expected pooling biases shape to be {self._pooling_biases.shape}, got {biases.shape} instead."
        )
        self.backend_class.check_input_type(biases)
        self._pooling_biases = biases

    def forward(self, x):
        """
        Forward pass through the whole model.
        """
        self.backend_class.check_input_type(x)
        generator_signal = self.generator_forward(x)  # [T]
        out = self.nonlinearity_pass(generator_signal, self.nonlinearity_out)  # [T]
        return out

    def generator_forward(self, x):
        outputs = []
        for c in range(self.n_channels):
            convolved = self.convolve(x, self.kernels[c])  # [T, H, W]
            activated = self.nonlinearity_pass(
                convolved, self.nonlinearities_chn[c]
            )  # [T, H, W]
            pooled = self.weighted_pooling(
                activated, self.pooling_weights[c], biases=self.pooling_biases[c]
            )  # [T]
            outputs.append(pooled)

        output = self.backend_class.lib.stack(outputs).sum(axis=0)  # [T]
        return output

    def convolve(self, x, kernel):
        return self.backend_class.convolve(x, kernel)

    def nonlinearity_pass(self, x, nonlinearity):
        return nonlinearity.transform(x)

    def weighted_pooling(self, x, pooling_weights, biases=None):
        if self.pooling_in_time:
            pooled = einsum(
                x, pooling_weights, "time height width, time height width -> time"
            )
        else:
            pooled = einsum(
                x, pooling_weights, "time height width, height width -> time"
            )
        if biases is not None:
            pooled = pooled + biases
        return pooled
