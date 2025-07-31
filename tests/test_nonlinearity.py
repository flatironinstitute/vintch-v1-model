import pytest
from vintch_model.backend_config import (
    TorchBackend,
    JaxBackend,
    NumpyBackend,
)
from vintch_model.nonlinearity import TentNonlinearity


class TestNonlinearity:
    @pytest.mark.parametrize(
        "backend_cls",
        [
            TorchBackend,
            JaxBackend,
            NumpyBackend,
        ],
    )
    @pytest.mark.parametrize(
        "nonlinearity_mode",
        ["relu", "quadratic", "linear"],
    )
    @pytest.mark.parametrize(
        "input_shape",
        [
            (5,),
            (2, 3),
            (1, 1, 10, 10),
            (1, 1, 10, 10, 10),
        ],
    )
    def test_nonlinearity(self, backend_cls, nonlinearity_mode, input_shape):
        backend = backend_cls()
        non_linearity = TentNonlinearity(
            backend_instance=backend,
            nonlinearity_mode=nonlinearity_mode,
            n_basis_funcs=10000,
        )
        x = backend.randn(input_shape)
        x = (x - x.min()) / (x.max() - x.min())
        non_lin_output = non_linearity(x)
        x_np = backend.to_numpy(x)

        if nonlinearity_mode == "relu":
            expected_output = x_np.clip(min=0)
        elif nonlinearity_mode == "quadratic":
            expected_output = x_np**2
        elif nonlinearity_mode == "linear":
            expected_output = x_np

        assert (
            non_lin_output.shape == x.shape
        ), f"Output shape {non_lin_output.shape} does not match input shape {x.shape}"

        assert backend.to_numpy(non_lin_output) == pytest.approx(
            expected_output, abs=1e-3
        ), f"{expected_output} {non_lin_output}"
