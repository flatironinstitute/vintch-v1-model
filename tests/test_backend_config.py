import pytest
import numpy as np
from vintch_model.backend_config import (
    TorchBackend,
    JaxBackend,
    NumpyBackend,
)


@pytest.mark.parametrize(
    "backend_cls",
    [
        TorchBackend,
        JaxBackend,
        NumpyBackend,
    ],
)
class TestBackend:
    @pytest.mark.parametrize(
        "input_shape",
        [
            (2, 3),
            (1, 1, 10, 10, 10),
        ],
    )
    def test_check_input_type(self, backend_cls, input_shape):
        backend = backend_cls()

        x = backend.randn(input_shape)
        backend.check_input_type(x)

        # Manual check to ensure the type is as expected
        if backend_cls == NumpyBackend:
            assert isinstance(
                x, np.ndarray
            ), f"{backend_cls.__name__}: Expected np.ndarray, but got {type(x)}"
        elif backend_cls == TorchBackend:
            assert isinstance(
                x, backend.lib.Tensor
            ), f"{backend_cls.__name__}: Expected backend.lib.Tensor, but got {type(x)}"
        elif backend_cls == JaxBackend:
            assert isinstance(
                x, backend.lib.ndarray
            ), f"{backend_cls.__name__}: Expected backend.lib.ndarray, but got {type(x)}"

    def test_setitem(self, backend_cls):
        backend = backend_cls()

        arr = backend.lib.zeros(3)
        arr2 = backend.setitem(arr, 1, 5.0)
        assert arr2[1] == 5.0

    def test_to_numpy(self, backend_cls):
        backend = backend_cls()

        arr = backend.lib.zeros(3)
        np_arr = backend.to_numpy(arr)
        assert isinstance(np_arr, np.ndarray)

    @pytest.mark.parametrize(
        "shape",
        [(2, 2)],
    )
    def test_randn(self, backend_cls, shape):
        backend = backend_cls()
        rand = backend.randn(shape, key=123)
        assert rand.shape == shape, "Output shape mismatch for randn"

    def test_convolve(self, backend_cls):
        backend = backend_cls()

        x = backend.lib.ones((1, 1, 5, 5, 5))
        kernel = backend.lib.ones((1, 1, 3, 3, 3))
        out = backend.convolve(x, kernel, padding="same")
        assert out.shape == x.shape, "Output shape mismatch for convolve"

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 3),
            (1, 1, 10, 10, 10),
        ],
    )
    def test_reshape(self, backend_cls, shape):
        backend = backend_cls()
        array = backend.randn(shape)
        reshaped = array.flatten().reshape(array.shape)
        arrays_equal = np.array_equal(
            backend.to_numpy(array), backend.to_numpy(reshaped)
        )

        assert (
            arrays_equal
        ), f"{backend_cls.__name__}: Reshape failed or values changed!"

    @pytest.mark.parametrize(
        "input_shape",
        [
            (3, 3, 3),
            (1, 1, 5, 5),
            (1, 10, 3),
        ],
    )
    def test_l1_norm_multiple_shapes(self, backend_cls, input_shape):
        backend = backend_cls()
        array = backend.randn(input_shape)

        l1_norm_backend = backend.l1_norm(array)

        l1_norm_numpy = np.sum(np.abs(backend.to_numpy(array)))

        assert np.isclose(
            l1_norm_backend, l1_norm_numpy
        ), f"{backend_cls.__name__}: L1 norm mismatch for shape {input_shape}. Backend: {l1_norm_backend}, NumPy: {l1_norm_numpy}"


def test_convolve_consistency_across_backends():
    backends = [TorchBackend(), JaxBackend(), NumpyBackend()]
    x_shapes = (1, 1, 25, 25, 25)
    kernel_shapes = (2, 1, 3, 3, 3)

    kernel = np.random.rand(*kernel_shapes)
    x = np.random.rand(*x_shapes)

    outputs = []
    for backend in backends:
        x = backend.convert_array(x)
        kernel = backend.convert_array(kernel)
        out = backend.convolve(x, kernel, padding="same")
        outputs.append(backend.to_numpy(out))

    assert np.allclose(
        outputs[0], outputs[1]
    ), f"Convolution outputs differ across backends. First output: {outputs[0][0][0]}, Second output: {outputs[1][0][0]}, Third output: {outputs[2][0][0]}"
