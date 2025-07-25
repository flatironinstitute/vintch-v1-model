import numpy as np
from matplotlib import pyplot as plt

try:
    import jax
    import jax.numpy as jnp
    import optax
except ImportError:
    jax = None
    jnp = None
    optax = None


def create_gabor_filter(
    frequency=0.25,
    theta=np.pi / 4,
    sigma_x=3.0,
    sigma_y=3.0,
    n_stds=3,
    offset=0,
    size=None,
):
    """
    Generate a Gabor filter.

    Returns:
        gabor :
            The generated complex Gabor filter.
    """

    ct = np.cos(theta)
    st = np.sin(theta)

    if size is None:
        x0 = np.round(max(abs(n_stds * sigma_x * ct), abs(n_stds * sigma_y * st), 1))
        y0 = np.round(max(abs(n_stds * sigma_y * ct), abs(n_stds * sigma_x * st), 1))
    else:
        y0 = (size[0] - 1) // 2
        x0 = (size[1] - 1) // 2

    y, x = np.meshgrid(
        np.arange(-y0, y0 + 1), np.arange(-x0, x0 + 1), indexing="ij", sparse=True
    )
    rotx = x * ct + y * st
    roty = -x * st + y * ct

    g = np.empty(roty.shape, dtype=np.complex128)
    np.exp(
        -0.5 * (rotx**2 / sigma_x**2 + roty**2 / sigma_y**2)
        + 1j * (2 * np.pi * frequency * rotx + offset),
        out=g,
    )
    g *= 1 / (2 * np.pi * sigma_x * sigma_y)

    g = (g - np.min(g)) / (np.max(g) - np.min(g))
    return g


def generate_grating(size=10, spatial_freq=10, orientation=0, phase=0):
    """
    Generate an oriented sinusoidal grating or a stack of gratings if orientation or phase is an array.

    Parameters:
        size (int): Width and height of the square image (in pixels)
        spatial_freq (float): Spatial frequency (cycles per image)
        orientation (float or array-like): Orientation of grating (in degrees, counter-clockwise)
        phase (float or array-like): Phase shift (in degrees)

    Returns:
        2D NumPy array or 3D NumPy array: Single grating or stack of gratings
    """
    # Ensure orientation and phase are arrays for consistent processing
    orientation = np.atleast_1d(orientation)
    phase = np.atleast_1d(phase)

    # Create a coordinate grid centered at (0,0)
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    # Generate gratings for all combinations of orientation and phase
    gratings = []
    for ori in orientation:
        for ph in phase:
            gradient = np.sin(ori * np.pi / 180) * x - np.cos(ori * np.pi / 180) * y
            grating = np.sin((2 * np.pi * gradient) / spatial_freq + (ph * np.pi) / 180)
            grating_norm = (grating - np.min(grating)) / (
                np.max(grating) - np.min(grating)
            )
            gratings.append(grating_norm)

    # Stack gratings along a new axis if multiple gratings are generated
    return np.stack(gratings, axis=0) if len(gratings) > 1 else gratings[0]


def alternating_fit_jax(
    kernels,
    pooling_weights,
    image,
    observed_spikes,
    predict_fn,
    weighted_pooling_fn,
    nonlinearity_out_fn,
    generate_subunit_convolutions,
    apply_nonlinearities,
    learning_rate=0.01,
    rtol=1e-5,
    atol=1e-8,
    max_iter=50,
    verbose=False,
    lr_decay_factor=0.5,
    lr_decay_threshold=1e-7,
):
    """
    Alternating optimization for kernels and pooling weights using JAX while loop.
    Includes learning rate decay if converging.
    """

    def kernel_loss(kernels, args):
        image, observed_spikes, pooling_weights = args
        pred_rate = predict_fn(image, kernels=kernels, pooling_weights=pooling_weights)
        return jnp.mean((pred_rate - observed_spikes) ** 2)

    def pooling_loss(pooling_weights, args):
        nl_response, observed_spikes = args
        pooled = weighted_pooling_fn(nl_response, pooling_weights)
        pred_rate = nonlinearity_out_fn(pooled)
        return jnp.mean((pred_rate - observed_spikes) ** 2)

    def cond_fun(state):
        i, _, _, _, _, prev_loss, current_loss, _ = state
        loss_diff = jnp.abs(current_loss - prev_loss)
        not_converged = loss_diff > (atol + rtol * prev_loss)
        return jnp.logical_and(i < max_iter, jnp.logical_or(i < 2, not_converged))

    def body_fun(state):
        (
            i,
            kernels,
            pooling_weights,
            kernel_opt_state,
            pool_opt_state,
            prev_loss,
            current_loss,
            learning_rate,
        ) = state

        # Use JAX-compatible modulus operation
        is_kernel_step = jnp.equal(jnp.mod(i, 2), 0)

        def kernel_step():
            loss_value, grad = jax.value_and_grad(kernel_loss)(
                kernels, (image, observed_spikes, pooling_weights)
            )
            grad = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grad)
            updates, new_kernel_opt_state = kernel_opt.update(
                grad, kernel_opt_state, kernels
            )
            new_kernels = optax.apply_updates(kernels, updates)
            return (
                new_kernels,
                pooling_weights,
                new_kernel_opt_state,
                pool_opt_state,
                loss_value,
            )

        def pooling_step():
            subunit_response = generate_subunit_convolutions(image, kernels)
            nl_response = apply_nonlinearities(subunit_response)
            loss_value, grad = jax.value_and_grad(pooling_loss)(
                pooling_weights, (nl_response, observed_spikes)
            )
            grad = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grad)
            updates, new_pool_opt_state = pool_opt.update(
                grad, pool_opt_state, pooling_weights
            )
            new_pooling_weights = optax.apply_updates(pooling_weights, updates)
            return (
                kernels,
                new_pooling_weights,
                kernel_opt_state,
                new_pool_opt_state,
                loss_value,
            )

        kernels, pooling_weights, kernel_opt_state, pool_opt_state, loss_value = (
            jax.lax.cond(
                is_kernel_step,
                lambda _: kernel_step(),
                lambda _: pooling_step(),
                operand=None,
            )
        )

        if verbose:
            jax.lax.cond(
                (i % 50) == 0,
                lambda args: jax.debug.print(
                    "Kernel step {i}, loss = {loss:.6f}, Δ = {delta:.2e}",
                    i=i,
                    loss=loss_value,
                    delta=jnp.abs(loss_value - prev_loss),
                ),
                lambda args: None,
                (i, loss_value, prev_loss),
            )
        return (
            i + 1,
            kernels,
            pooling_weights,
            kernel_opt_state,
            pool_opt_state,
            current_loss,
            loss_value,
            learning_rate,
        )

    # Initialize optimization states
    kernel_opt = optax.adam(learning_rate)
    kernel_opt_state = kernel_opt.init(kernels)

    pool_opt = optax.adam(learning_rate)
    pool_opt_state = pool_opt.init(pooling_weights)

    initial_loss = jnp.mean(
        (
            predict_fn(image, kernels=kernels, pooling_weights=pooling_weights)
            - observed_spikes
        )
        ** 2
    )
    state = (
        0,
        kernels,
        pooling_weights,
        kernel_opt_state,
        pool_opt_state,
        initial_loss * 10,
        initial_loss,
        learning_rate,
    )

    # Run the JAX while loop
    final_state = jax.lax.while_loop(cond_fun, body_fun, state)

    (
        final_iter,
        final_kernels,
        final_pooling_weights,
        _,
        _,
        _,
        final_loss,
        final_learning_rate,
    ) = final_state

    if final_iter < max_iter:
        print(
            f" Optimization converged in {final_iter} iterations. Final loss: {final_loss:.6f}, Final learning rate: {final_learning_rate:.2e}"
        )
    else:
        print(
            f" Reached max iterations ({max_iter}). Final loss: {final_loss:.6f}, Final learning rate: {final_learning_rate:.2e}"
        )

    return {
        "kernels": final_kernels,
        "pooling_weights": final_pooling_weights,
        "final_loss": final_loss,
        "final_learning_rate": final_learning_rate,
    }


def optimize_kernels(
    kernels,
    image,
    observed_spikes,
    pooling_weights,
    learning_rate,
    predict_fn,
    iter=1000,
    atol=1e-6,
    rtol=1e-6,
    verbose=False,
):

    jax.debug.print("Starting kernel optimization...")
    kernel_opt = optax.adam(learning_rate)
    kernel_opt_state = kernel_opt.init(kernels)

    def kernel_loss(kernels, args):
        image, observed_counts, pooling_weights = args
        pred_rate = predict_fn(image, kernels=kernels, pooling_weights=pooling_weights)
        return jnp.mean((pred_rate - observed_counts) ** 2)

    initial_loss, _ = jax.value_and_grad(kernel_loss)(
        kernels, (image, observed_spikes, pooling_weights)
    )
    init_state = (0, kernels, kernel_opt_state, initial_loss * 10, initial_loss)

    def cond_fun(state):
        i, _, _, prev_loss, current_loss = state
        loss_diff = jnp.abs(current_loss - prev_loss)

        not_converged = loss_diff > atol + rtol * prev_loss
        return jnp.logical_and(i < iter, jnp.logical_or(i < 2, not_converged))

    def body_fun(state):
        i, current_kernels, opt_state, prev_loss, current_loss = state

        loss_value, grad = jax.value_and_grad(kernel_loss)(
            current_kernels, (image, observed_spikes, pooling_weights)
        )

        grad = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grad)
        updates, new_opt_state = kernel_opt.update(grad, opt_state, current_kernels)
        new_kernels = optax.apply_updates(current_kernels, updates)

        if verbose:
            jax.lax.cond(
                (i % 50) == 0,
                lambda args: jax.debug.print(
                    "Kernel step {i}, loss = {loss:.6f}, Δ = {delta:.2e}",
                    i=i,
                    loss=loss_value,
                    delta=jnp.abs(loss_value - prev_loss),
                ),
                lambda args: None,
                (i, loss_value, prev_loss),
            )
        return (i + 1, new_kernels, new_opt_state, current_loss, loss_value)

    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

    final_iter, final_kernels, _, _, final_loss = final_state

    if final_iter < iter:
        print(
            f" Optimization converged in {final_iter} iterations. Final loss: {final_loss:.6f}"
        )
    else:
        print(f" Reached max iterations ({iter}). Final loss: {final_loss:.6f}")

    return {
        "kernels": final_kernels,
        "final_loss": final_loss,
        "initial_loss": initial_loss,
        "iterations": final_iter,
    }


def optimize_pooling_weights(
    nonlinear_response,
    observed_spikes,
    pooling_weights,
    learning_rate,
    weighted_pooling_fn,
    nonlinearity_out_fn,
    pool_iter=1000,
    atol=1e-6,
    rtol=1e-6,
    verbose=False,
):
    print_every = pool_iter // 10
    jax.debug.print("Starting pooling weights optimization...")

    pool_opt = optax.adam(learning_rate)
    pool_opt_state = pool_opt.init(pooling_weights)

    def loss_fn(pooling_weights):
        signal = weighted_pooling_fn(nonlinear_response, pooling_weights)
        out = nonlinearity_out_fn(signal)
        return jnp.mean((observed_spikes - out) ** 2)

    initial_loss, _ = jax.value_and_grad(loss_fn)(pooling_weights)
    state = (0, pooling_weights, pool_opt_state, initial_loss * 10, initial_loss)

    print(f"Initial loss: {initial_loss:.6f}")

    def cond(s):
        i, _, _, prev_loss, current_loss = s
        loss_diff = jnp.abs(current_loss - prev_loss)

        not_converged = loss_diff > atol + rtol * prev_loss
        return jnp.logical_and(i < pool_iter, jnp.logical_or(i < 2, not_converged))

    def body(s):
        i, w, opt_state, prev_loss, current_loss = s
        loss, grad = jax.value_and_grad(loss_fn)(w)
        grad = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grad)
        updates, opt_state = pool_opt.update(grad, opt_state, w)
        w = optax.apply_updates(w, updates)

        if verbose:
            jax.lax.cond(
                (i % print_every) == 0,
                lambda args: jax.debug.print(
                    "Pooling step {i}, loss = {loss:.6f}, Δ = {delta:.2e}",
                    i=i,
                    loss=loss,
                    delta=jnp.abs(loss - prev_loss),
                ),
                lambda args: None,
                (i, _, loss, prev_loss),
            )

        return (i + 1, w, opt_state, current_loss, loss)

    final_state = jax.lax.while_loop(cond, body, state)
    final_iter, final_weights, _, _, final_loss = final_state

    if final_iter < pool_iter:
        print(
            f"Pooling optimization converged in {final_iter} iterations. Final loss: {final_loss:.6f}"
        )
    else:
        print(
            f"Reached max pooling iterations ({pool_iter}). Final loss: {final_loss:.6f}"
        )

    return {
        "pooling_weights": final_weights,
        "final_loss": final_loss,
        "initial_loss": initial_loss,
        "iterations": final_iter,
    }
