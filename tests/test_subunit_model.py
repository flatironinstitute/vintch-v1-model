import pytest
import numpy as np
from vintch_model.subunit_model import SubunitModel
from vintch_model.backend_config import get_backend


@pytest.mark.parametrize("x_shape", [(1, 1, 15, 15, 15), (5, 1, 30, 30, 30)])
@pytest.mark.parametrize("subunit_kernel", [(3, 3, 3), (5, 5, 5), (8, 8, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_channels", [2])
@pytest.mark.parametrize("n_basis_funcs", [25])
def test_subunit_model_forward_pass(
    x_shape, subunit_kernel, dtype, n_channels, n_basis_funcs
):
    backends_names = ["torch", "jax"]
    outputs = []
    pooling_shape = x_shape[2:]

    x = np.random.randn(*x_shape).astype(dtype)
    kernels = np.random.randn(n_channels, 1, *subunit_kernel).astype(dtype)
    pooling_weights = np.random.randn(n_channels, *pooling_shape).astype(dtype)
    pooling_biases = np.random.randn(n_channels, 1).astype(dtype)
    nonlinearities_weights = np.random.randn(n_channels + 1, n_basis_funcs).astype(
        dtype
    )

    for backend_name in backends_names:
        backend = get_backend(backend_name)
        x_backend = backend.get_array(arr=x)
        kernels_backend = backend.get_array(arr=kernels)
        pooling_weights_backend = backend.get_array(arr=pooling_weights)
        pooling_biases_backend = backend.get_array(arr=pooling_biases)
        model = SubunitModel(
            backend=backend_name,
            subunit_kernel=subunit_kernel,
            pooling_shape=pooling_shape,
            n_channels=n_channels,
            n_basis_funcs=n_basis_funcs,
        )
        model.kernels = kernels_backend
        model.pooling_weights = pooling_weights_backend
        model.pooling_biases = pooling_biases_backend

        for i, nonlinearity in enumerate(model.nonlinearities_chan):
            nonlinearity.weights = backend.get_array(arr=nonlinearities_weights[i])
        model.nonlinearity_out.weights = backend.get_array(
            arr=nonlinearities_weights[-1]
        )

        output = model(x_backend)
        outputs.append(output)

    output_shapes = [o.shape for o in outputs]
    assert all(
        shape == output_shapes[0] for shape in output_shapes
    ), "Output shapes do not match across backends."
    assert np.allclose(
        outputs[0], outputs[1], atol=1e-3
    ), f"Outputs from {backends_names[0]} and {backends_names[1]} do not match."
