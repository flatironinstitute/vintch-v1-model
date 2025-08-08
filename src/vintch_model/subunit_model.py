import einops

try:
    import optax
    import jax
    import jax.numpy as jnp
except ImportError:
    optax = None
    jax = None
    jnp = None

from typing import Any, Literal, TypeVar, Generic, Optional
from .backend_config import get_backend
from .nonlinearity import TentNonlinearity
from .utils import train_validation_split

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
        List indicating whether each channel is excitatory. Must match the number of
        channels. None is the equivalent of setting True for all channels.
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

    def _predict(self, x: Tensor, **kwargs) -> Tensor:
        """
        Forward pass through the whole model.

        Parameters
        ----------
        x :
            Input tensor with shape [batch_size, 1, time, height, width].
        **kwargs :
            Additional keyword arguments.

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

        show_warning = True
        if jax is not None and hasattr(jax.core, "trace_state_clean"):
            show_warning = jax.core.trace_state_clean()

        if show_warning:
            x_min, x_max = x.min(), x.max()
            if not (0 <= x_min <= 1 and 0 <= x_max <= 1):
                print(
                    f"Input values are not in the range [0, 1], got min: {x_min}, max: {x_max} instead."
                )

        kernels = kwargs.get("kernels", self._kernels)
        pooling_weights = kwargs.get("pooling_weights", self._pooling_weights)

        # [batch_size, n_channels, time, height, width]
        sub_convolved = self._convolve(x, kernels)
        # [batch_size, n_channels, time, height, width]
        non_lin_response = self._apply_nonlinearities(
            sub_convolved, self._nonlinearities_chan
        )
        # [batch_size, time]
        weighted_pooled = self._weighted_pooling(non_lin_response, pooling_weights)
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

    def alternating_fit(
        self,
        input_data: Tensor,
        observed_rate: Tensor,
        kernel_lr: float = 0.01,
        pooling_lr: float = 0.01,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        iter: int = 50,
        verbose: bool = False,
        update_kernels: bool = True,
        update_pooling: bool = True,
        inner_iter: int = 1,
    ):
        """
        Fit the model using alternating optimization for kernels and pooling weights.

        Parameters
        ----------
        input_data :
            Input data.
        observed_rate :
            Observed firing rate.
        kernel_lr :
            Learning rate for the kernel optimizer.
        pooling_lr :
            Learning rate for the pooling weights optimizer.
        rtol :
            Relative tolerance for convergence.
        atol :
            Absolute tolerance for convergence.
        iter :
            Maximum number of outer iterations.
        verbose :
            Whether to print progress information during fitting.
        update_kernels :
            Whether to update kernels during optimization.
        update_pooling :
            Whether to update pooling weights during optimization.
        inner_iter :
            Number of inner iterations for each parameter update.

        Returns
        -------
        loss_history :
            Array of validation loss values at each iteration.
        """

        kernels = self._kernels
        pooling_weights = self._pooling_weights

        train_image, val_image, train_rate, val_rate = train_validation_split(
            input_data, observed_rate
        )

        def kernel_loss_fn(kernels, args):
            image, observed_rate, pooling_weights = args
            pred_rate = self._predict(
                image, kernels=kernels, pooling_weights=pooling_weights
            )
            mse = jnp.mean((pred_rate - observed_rate) ** 2)
            l1_norm = jnp.sum(self._backend.l1_norm(kernels))
            alpha = 1e-5
            penalty = alpha * jnp.abs(l1_norm - 1)
            return mse + penalty

        def pooling_loss_fn(pooling_weights, args):
            nl_response, observed_rate = args
            pooled = self._weighted_pooling(nl_response, pooling_weights)
            generator_signal = self._apply_bias(pooled, self.pooling_bias)
            pred_rate = self._nonlinearity_out(generator_signal)
            mse = jnp.mean((pred_rate - observed_rate) ** 2)
            l1_norm = jnp.sum(self._backend.l1_norm(pooling_weights))
            alpha = 1e-5
            penalty = alpha * jnp.abs(l1_norm - 1)
            return mse + penalty

        def full_loss(input, output, kernels, pooling_weights):
            pred_rate = self._predict(
                input, kernels=kernels, pooling_weights=pooling_weights
            )
            return jnp.mean((pred_rate - output) ** 2)

        def cond_fun(state):
            i, _, _, _, _, prev_val_loss, current_val_loss, _ = state
            loss_diff = jnp.abs(current_val_loss - prev_val_loss)
            not_converged = loss_diff > (atol + rtol * prev_val_loss)
            return jnp.logical_and(i < iter, jnp.logical_or(i < 2, not_converged))

        def body_fun(state):
            (
                i,
                kernels,
                pooling_weights,
                kernel_opt_state,
                pool_opt_state,
                prev_val_loss,
                learning_rate,
                loss_history,
            ) = state

            def kernel_update_body(_, carry):
                kernels, kernel_opt_state = carry
                kernel_loss, grad = jax.value_and_grad(kernel_loss_fn)(
                    kernels, (train_image, train_rate, pooling_weights)
                )
                grad = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grad)
                updates, kernel_opt_state = kernel_opt.update(
                    grad,
                    kernel_opt_state,
                    kernels,
                    value=kernel_loss,
                    grad=grad,
                    value_fn=kernel_loss_fn,
                    args=(train_image, train_rate, pooling_weights),
                )
                kernels = optax.apply_updates(kernels, updates)
                return kernels, kernel_opt_state

            def pool_update_body(j, carry):
                pooling_weights, pool_opt_state = carry
                subunit_response = self._convolve(train_image, kernels)
                nl_response = self._apply_nonlinearities(
                    subunit_response, self._nonlinearities_chan
                )
                loss_value, grad = jax.value_and_grad(pooling_loss_fn)(
                    pooling_weights, (nl_response, train_rate)
                )
                grad = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grad)
                updates, pool_opt_state = pool_opt.update(
                    grad,
                    pool_opt_state,
                    pooling_weights,
                    value=loss_value,
                    grad=grad,
                    value_fn=pooling_loss_fn,
                    args=(nl_response, train_rate),
                )
                pooling_weights = optax.apply_updates(pooling_weights, updates)
                return pooling_weights, pool_opt_state

            if update_kernels:
                kernels, kernel_opt_state = jax.lax.fori_loop(
                    0, inner_iter, kernel_update_body, (kernels, kernel_opt_state)
                )
            if update_pooling:
                pooling_weights, pool_opt_state = jax.lax.fori_loop(
                    0, inner_iter, pool_update_body, (pooling_weights, pool_opt_state)
                )

            train_loss = full_loss(train_image, train_rate, kernels, pooling_weights)
            val_loss = full_loss(val_image, val_rate, kernels, pooling_weights)

            loss_history = loss_history.at[i].set(val_loss)

            if verbose:
                jax.lax.cond(
                    (i % 1) == 0,
                    lambda _: jax.debug.print(
                        "step {i}, val_loss = {loss}, train_loss = {train_loss}, Δ = {delta}, lr = {lr}",
                        i=i,
                        loss=val_loss,
                        train_loss=train_loss,
                        delta=jnp.abs(val_loss - prev_val_loss),
                        lr=learning_rate,
                    ),
                    lambda _: None,
                    (i, val_loss, train_loss, prev_val_loss, learning_rate),
                )
            return (
                i + 1,
                kernels,
                pooling_weights,
                kernel_opt_state,
                pool_opt_state,
                val_loss,
                learning_rate,
                loss_history,
            )

        # Initialize optimization states
        kernel_opt = optax.lbfgs(kernel_lr)
        kernel_opt_state = kernel_opt.init(kernels)

        pool_opt = optax.lbfgs(pooling_lr)
        pool_opt_state = pool_opt.init(pooling_weights)

        initial_val_loss = full_loss(val_image, val_rate, kernels, pooling_weights)
        loss_history = jnp.zeros(iter)
        state = (
            0,
            kernels,
            pooling_weights,
            kernel_opt_state,
            pool_opt_state,
            initial_val_loss * 10,
            kernel_lr,
            loss_history,
        )

        # Run the JAX while loop
        final_state = jax.lax.while_loop(cond_fun, body_fun, state)

        (
            final_iter,
            final_kernels,
            final_pooling_weights,
            _,
            _,
            final_val_loss,
            final_learning_rate,
            final_loss_history,
        ) = final_state

        # Compute final train_loss and val_loss for return
        final_train_loss = full_loss(
            train_image, train_rate, final_kernels, final_pooling_weights
        )
        final_val_loss = full_loss(
            val_image, val_rate, final_kernels, final_pooling_weights
        )

        if final_iter < iter:
            print(
                f" Optimization converged in {final_iter} iterations. Final val loss: {final_val_loss:.6f}, Final train loss: {final_train_loss:.6f}, Final learning rate: {final_learning_rate:.2e}"
            )
        else:
            print(
                f" Reached max iterations ({iter}). Final val loss: {final_val_loss:.6f}, Final train loss: {final_train_loss:.6f}, Final learning rate: {final_learning_rate:.2e}"
            )

        result = {
            "kernels": final_kernels,
            "pooling_weights": final_pooling_weights,
            "final_val_loss": final_val_loss,
            "final_train_loss": final_train_loss,
            "final_learning_rate": final_learning_rate,
            "loss_history": final_loss_history[:final_iter],
        }

        self.kernels = result["kernels"]
        self.pooling_weights = result["pooling_weights"]

        return result["loss_history"]

    def fit_together(
        self,
        input_data: Tensor,
        observed_rate: Tensor,
        lr: float = 0.01,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        iter: int = 50,
        verbose: bool = False,
    ):
        """
        Fit the model using joint optimization for kernels and pooling weights.

        Parameters
        ----------
        input_data :
            Input data tensor for training and validation.
        observed_rate :
            Observed firing rates corresponding to the input data.
        lr :
            Learning rate for the optimizer.
        rtol :
            Relative tolerance for convergence.
        atol :
            Absolute tolerance for convergence.
        iter :
            Maximum number of iterations.
        verbose :
            Whether to print progress information during fitting.

        Returns
        -------
        loss_history :
            Array of validation loss values at each iteration.
        """
        kernels = self._kernels
        pooling_weights = self._pooling_weights

        train_image, val_image, train_rate, val_rate = train_validation_split(
            input_data, observed_rate
        )

        def loss_fn(parameters, args):
            image, target_rate = args
            kernels = parameters["kernels"]
            pooling_weights = parameters["pooling_weights"]
            pred_rate = self._predict(
                image, kernels=kernels, pooling_weights=pooling_weights
            )
            return jnp.mean((pred_rate - target_rate) ** 2)

        loss_history = jnp.zeros(iter)

        def cond_fun(state):
            i, _, _, prev_val_loss, current_val_loss, _ = state
            loss_diff = jnp.abs(current_val_loss - prev_val_loss)
            not_converged = loss_diff > atol + rtol * jnp.abs(current_val_loss)
            jax.debug.print(
                "Iteration {i}: Val Loss = {val_loss}, Δ = {loss_diff}",
                i=i,
                val_loss=current_val_loss,
                loss_diff=loss_diff,
            )
            return jnp.logical_and(i < iter, jnp.logical_or(i < 10, not_converged))

        def body_fun(state):
            i, parameters, optimizer_state, _, current_val_loss, loss_history = state
            loss, grad = jax.value_and_grad(loss_fn)(
                parameters, (train_image, train_rate)
            )
            grad = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grad)
            updates, optimizer_state = optimizer.update(
                grad,
                optimizer_state,
                parameters,
                value=loss,
                grad=grad,
                value_fn=loss_fn,
                args=(train_image, train_rate),
            )
            parameters = optax.apply_updates(parameters, updates)
            new_val_loss = loss_fn(parameters, (val_image, val_rate))
            loss_history = loss_history.at[i].set(new_val_loss)
            if verbose:
                jax.debug.print(
                    "Iteration {i}: Train Loss = {train_loss}, Val Loss = {val_loss}, Δ = {delta}",
                    i=i,
                    train_loss=loss,
                    val_loss=new_val_loss,
                    delta=jnp.abs(new_val_loss - current_val_loss),
                )
            return (
                i + 1,
                parameters,
                optimizer_state,
                current_val_loss,
                new_val_loss,
                loss_history,
            )

        optimizer = optax.lbfgs(lr)
        parameters = {"kernels": kernels, "pooling_weights": pooling_weights}
        optimizer_state = optimizer.init(parameters)
        initial_validation_loss = loss_fn(parameters, (val_image, val_rate))
        loss_history = jnp.zeros(iter)
        state = (
            0,
            parameters,
            optimizer_state,
            initial_validation_loss * 10,
            initial_validation_loss,
            loss_history,
        )

        final_state = jax.lax.while_loop(cond_fun, body_fun, state)
        (
            final_iteration,
            final_parameters,
            _,
            _,
            final_validation_loss,
            final_loss_history,
        ) = final_state

        final_training_loss = loss_fn(final_parameters, (train_image, train_rate))
        final_validation_loss = loss_fn(final_parameters, (val_image, val_rate))

        if verbose or final_iteration < iter:
            print(
                f"Optimization stopped at iteration {final_iteration}. Final val loss: {float(final_validation_loss):.6f}, Final train loss: {float(final_training_loss):.6f}"
            )

        self.kernels = final_parameters["kernels"]
        self.pooling_weights = final_parameters["pooling_weights"]

        return final_loss_history[:final_iteration]
