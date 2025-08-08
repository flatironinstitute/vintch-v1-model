import jax
import jax.numpy as jnp
import numpy as np
from vintch_model.subunit_model import SubunitModel
from vintch_model.backend_config import JaxBackend
from vintch_model.utils import create_gabor_filter, create_gaussian_map
from vintch_model.nonlinearity import TentNonlinearity
from vintch_model.plotting import plot_pooling_weights, plot_kernels
from datetime import datetime

jax.config.update("jax_enable_x64", True)

input_size = 15
kernel_size = 7
batch_size = 1000

jax_backend = JaxBackend()

# CREATE INPUT
inputs = jax.random.normal(
    jax.random.PRNGKey(0), (batch_size, 1, 1, input_size, input_size)
)
inputs = (inputs - jnp.min(inputs)) / (jnp.max(inputs) - jnp.min(inputs))
inputs = inputs.astype(jnp.float64)

# INITIALIZE THE MODEL
subunit_model = SubunitModel(
    n_basis_funcs=15,
    backend="jax",
    pooling_shape=(1, input_size, input_size),
    n_channels=1,
    is_channel_excitatory=[False],
    subunit_kernel_shape=(1, kernel_size, kernel_size),
)

subunit_model._nonlinearity_out = TentNonlinearity(
    backend_instance=jax_backend, nonlinearity_mode="linear"
)

# SET BIAS TO 0
subunit_model.pooling_bias = jnp.array((0,))

# GENERATE TARGET KERNELS AND POOLING WEIGHTS
initial_kernels = subunit_model._kernels
target_kernels = (
    create_gabor_filter(
        frequency=0.25,
        theta=-np.pi / 4,
        sigma_x=2.0,
        sigma_y=2.0,
        offset=0 * np.pi,
        size=(kernel_size, kernel_size),
    )
    .reshape((1, 1, 1, kernel_size, kernel_size))
    .real
)
target_kernels = jnp.array(target_kernels, dtype=jnp.float64)
subunit_model.kernels = jax_backend.convert_array(target_kernels)
target_kernels = subunit_model._kernels

initial_pooling_weights = subunit_model._pooling_weights

target_pooling_weights = jax_backend.convert_array(
    create_gaussian_map(shape=(1, input_size, input_size), sigma=1.5)
)

subunit_model.pooling_weights = target_pooling_weights.reshape(
    (1, 1, input_size, input_size)
)
target_pooling_weights = subunit_model._pooling_weights

observed_rates = subunit_model(inputs)

# INITIAL PARAMS
# initialize as ones in jax
subunit_model.pooling_weights = jnp.ones_like(subunit_model._pooling_weights)
subunit_model.kernels = jnp.ones_like(subunit_model._kernels)

initial_rates = subunit_model(inputs)

start_time = datetime.now()

loss_history = subunit_model.alternating_fit(
    input_data=inputs,
    observed_rate=observed_rates,
    rtol=1e-10,
    atol=1e-10,
    kernel_lr=1e-4,
    pooling_lr=1e-4,
    verbose=True,
    update_kernels=True,
    update_pooling=True,
    iter=int(400),
    inner_iter=5,
)

end_time = datetime.now()
print(f"Fitting took {end_time - start_time} seconds.")

# loss_history = subunit_model.fit_together(
#      input_data=inputs,
#      observed_rate=observed_rates,
#      rtol=1e-15,
#      atol=1e-15,
#      lr=1e-4,
#      verbose=True,
#      iter=100,
#      )

fit_rates = subunit_model(inputs)

# correlation between fit and target rate
correlation_fit = jnp.corrcoef(fit_rates.flatten(), observed_rates.flatten())[0, 1]
print(f"Correlation between fit rate and target output: {correlation_fit:.4f}")

# mean squared error between fit and target rate
mse_fit = jnp.mean((fit_rates - observed_rates) ** 2)
print(f"Mean Squared Error between fit rate and target output: {mse_fit}")

plot_kernels(
    subunit_model._kernels,
    filename="fitting_kernels.svg",
)

plot_pooling_weights(
    subunit_model._pooling_weights,
    filename="fitting_pooling_weights.svg",
)
