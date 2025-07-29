import einops

try:
    import optax
    import jax
    import jax.numpy as jnp
except ImportError:
    optax = None
    jax = None
    jnp = None

from typing import Any, Literal, TypeVar, Generic
from .backend_config import get_backend
from .nonlinearity import TentNonlinearity
from .utils import train_validation_split

Tensor = TypeVar("Tensor")


class SubunitModel(Generic[Tensor]):
    """
    Implementation of the Vintch model.

    Parameters
    ----------
    subunit_kernel :
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
        self.n_channels = n_channels
        self._backend = get_backend(backend)

        random_kernels = self._backend.randn((n_channels, 1, *subunit_kernel))
        self._kernels = random_kernels / self._backend.l1_norm(random_kernels)
        self.n_basis_funcs = n_basis_funcs

        random_pooling_weights = self._backend.randn((n_channels, *pooling_shape))
        self._pooling_weights = random_pooling_weights / self._backend.l1_norm(
            random_pooling_weights
        )
        if len(pooling_shape) == len(subunit_kernel):
            self._pooling_dims = "n_channels time height width"
        elif len(pooling_shape) == len(subunit_kernel) - 1:
            self._pooling_dims = "n_channels height width"
        else:
            raise ValueError(
                "Invalid pooling_shape length. Must match subunit_kernel length or be one less."
            )

        self._pooling_biases = self._backend.randn((1,))
        self._nonlinearities_chan = [
            TentNonlinearity(
                backend_instance=self._backend,
                n_basis_funcs=n_basis_funcs,
                nonlinearity_mode="relu" if is_channel_excitatory[c] else "quadratic",
            )
            for c in range(n_channels)
        ]
        self._nonlinearity_out = TentNonlinearity(
            backend_instance=self._backend,
            n_basis_funcs=n_basis_funcs,
            nonlinearity_mode="relu",
        )

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

        # Check if the input range is between 0 and 1
        # x_min, x_max = self._backend.lib.min(x), self._backend.lib.max(x)
        # margin = 0.1
        # if not (-margin <= x_min <= 1 + margin) or not (-margin <= x_max <= 1 + margin):
        #     raise ValueError(
        #         f"Input values must be in the range [0, 1] (with margin {margin}), got min: {x_min}, max: {x_max} instead."
        #     )
        kernels = kwargs.get("kernels", self._kernels)
        pooling_weights = kwargs.get("pooling_weights", self._pooling_weights)

        # [batch_size, n_channels, time, height, width]
        sub_convolved = self._convolve(x, kernels)
        # [batch_size, n_channels, time, height, width]
        non_lin_response = self._apply_nonlinearities(
            sub_convolved, self._nonlinearities_chan
        )
        # [batch_size, time]
        generator_signal = self._weighted_pooling(non_lin_response, pooling_weights)
        # [batch_size, time]
        out = self._nonlinearity_out.forward(generator_signal)
        return out

    def _convolve(self, x: Tensor, kernel: Tensor) -> Tensor:
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
        return self._backend.convolve(x, kernel)

    def _apply_nonlinearities(self, x: Tensor, nonlinearities: list) -> Tensor:
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
        transformed = [
            nonlinearity(x[:, c]) for c, nonlinearity in enumerate(nonlinearities)
        ]

        stacked_transformed, _ = einops.pack(
            transformed, pattern="batch_size * time height width"
        )
        return stacked_transformed

    def _weighted_pooling(self, x: Tensor, pooling_weights: Tensor) -> Tensor:
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
        pooled = einops.einsum(
            x,
            pooling_weights,
            f"batch_size n_channels time height width, {self._pooling_dims} -> batch_size time",
        )

        pooled = pooled + self._pooling_biases
        return pooled

    def alternative_fit(
        self,
        image: Tensor,
        observed_spikes: Tensor,
        kernel_lr=0.01,
        pooling_lr=0.01,
        rtol=1e-5,
        atol=1e-8,
        iter=50,
        verbose=False,
        update_kernels=True,
        update_pooling=True,
        inner_iter=1,
    ):
        kernels = self._kernels
        pooling_weights = self._pooling_weights

        # Split data
        train_image, val_image, train_spikes, val_spikes = train_validation_split(
            image, observed_spikes
        )

        def kernel_loss(kernels, args):
            image, observed_spikes, pooling_weights = args
            pred_rate = self._predict(
                image, kernels=kernels, pooling_weights=pooling_weights
            )
            return jnp.mean((pred_rate - observed_spikes) ** 2)

        def pooling_loss(pooling_weights, args):
            nl_response, observed_spikes = args
            pooled = self._weighted_pooling(nl_response, pooling_weights)
            pred_rate = self._nonlinearity_out(pooled)
            return jnp.mean((pred_rate - observed_spikes) ** 2)

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
                _, grad = jax.value_and_grad(kernel_loss)(
                    kernels, (train_image, train_spikes, pooling_weights)
                )
                grad = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grad)
                updates, kernel_opt_state = kernel_opt.update(
                    grad, kernel_opt_state, kernels
                )
                kernels = optax.apply_updates(kernels, updates)
                return kernels, kernel_opt_state

            def pool_update_body(j, carry):
                pooling_weights, pool_opt_state = carry
                subunit_response = self._convolve(train_image, kernels)
                nl_response = self._apply_nonlinearities(
                    subunit_response, self._nonlinearities_chan
                )
                loss_value, grad = jax.value_and_grad(pooling_loss)(
                    pooling_weights, (nl_response, train_spikes)
                )
                grad = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grad)
                updates, pool_opt_state = pool_opt.update(
                    grad,
                    pool_opt_state,
                    pooling_weights,
                    value=loss_value,
                    grad=grad,
                    value_fn=pooling_loss,
                    args=(nl_response, train_spikes),
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

            train_loss = full_loss(train_image, train_spikes, kernels, pooling_weights)
            val_loss = full_loss(val_image, val_spikes, kernels, pooling_weights)

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
        kernel_opt = optax.adam(kernel_lr)
        kernel_opt_state = kernel_opt.init(kernels)

        pool_opt = optax.lbfgs(pooling_lr)
        pool_opt_state = pool_opt.init(pooling_weights)

        initial_val_loss = full_loss(val_image, val_spikes, kernels, pooling_weights)
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
            train_image, train_spikes, final_kernels, final_pooling_weights
        )
        final_val_loss = full_loss(
            val_image, val_spikes, final_kernels, final_pooling_weights
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
        image: Tensor,
        observed_spikes: Tensor,
        lr=0.01,
        rtol=1e-5,
        atol=1e-8,
        iter=50,
        verbose=False,
    ):
        """
        Fit the model using joint optimization for kernels and pooling weights.
        Returns: final_params, loss_history
        """
        kernels = self._kernels
        pooling_weights = self._pooling_weights

        train_image, val_image, train_spikes, val_spikes = train_validation_split(
            image, observed_spikes
        )

        def loss_fn(params, args):
            train_image, train_spikes = args
            kernels = params["kernels"]
            pooling_weights = params["pooling_weights"]
            pred_rate = self._predict(
                train_image, kernels=kernels, pooling_weights=pooling_weights
            )
            return jnp.mean((pred_rate - train_spikes) ** 2)

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
            i, params, opt_state, _, current_val_loss, loss_history = state
            loss, grad = jax.value_and_grad(loss_fn)(
                params, (train_image, train_spikes)
            )
            grad = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grad)
            updates, opt_state = opt.update(
                grad,
                opt_state,
                params,
                value=loss,
                grad=grad,
                value_fn=loss_fn,
                args=(train_image, train_spikes),
            )
            params = optax.apply_updates(params, updates)
            new_val_loss = loss_fn(params, (val_image, val_spikes))
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
                params,
                opt_state,
                current_val_loss,
                new_val_loss,
                loss_history,
            )

        opt = optax.lbfgs(lr)
        params = {"kernels": kernels, "pooling_weights": pooling_weights}
        opt_state = opt.init(params)
        initial_val_loss = loss_fn(params, (val_image, val_spikes))
        loss_history = jnp.zeros(iter)
        state = (
            0,
            params,
            opt_state,
            initial_val_loss * 10,
            initial_val_loss,
            loss_history,
        )

        final_state = jax.lax.while_loop(cond_fun, body_fun, state)
        final_iter, final_params, _, _, final_val_loss, final_loss_history = final_state

        final_train_loss = loss_fn(final_params, (train_image, train_spikes))
        final_val_loss = loss_fn(final_params, (val_image, val_spikes))

        if verbose or final_iter < iter:
            print(
                f"Optimization stopped at iteration {final_iter}. Final val loss: {float(final_val_loss):.6f}, Final train loss: {float(final_train_loss):.6f}"
            )

        self.kernels = final_params["kernels"]
        self.pooling_weights = final_params["pooling_weights"]

        return final_loss_history[:final_iter]
