"""
Tests for DeepExtractorModel and the top-level reconstruct/extract API.

No pretrained weights are downloaded — load_torch_model is patched to return
a zero-output mock model so only the pre/post-processing logic is exercised.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

import deepextractor
from deepextractor.model import DeepExtractorModel

LENGTH = 8192  # 2 s at 4096 Hz


def _make_mock_model(output_shape):
    """Return a torch.nn.Module mock that outputs zeros of ``output_shape``."""
    mock = MagicMock(spec=torch.nn.Module)
    mock.return_value = torch.zeros(*output_shape)
    mock.eval.return_value = mock
    mock.to.return_value = mock
    return mock


def _build_model(dummy_scaler_path, batch_size=1):
    mock_nn = _make_mock_model((batch_size, 2, 257, 257))
    with patch("deepextractor.model.load_torch_model", return_value=mock_nn):
        return DeepExtractorModel(device="cpu", scaler_path=dummy_scaler_path)


# ---------------------------------------------------------------------------
# DeepExtractorModel
# ---------------------------------------------------------------------------

class TestDeepExtractorModel:

    def test_reconstruct_1d_returns_same_shape(self, dummy_scaler_path):
        model = _build_model(dummy_scaler_path)
        x = np.random.randn(LENGTH)
        assert model.reconstruct(x).shape == (LENGTH,)

    def test_background_1d_returns_same_shape(self, dummy_scaler_path):
        model = _build_model(dummy_scaler_path)
        x = np.random.randn(LENGTH)
        assert model.background(x).shape == (LENGTH,)

    def test_reconstruct_equals_input_minus_background(self, dummy_scaler_path):
        model = _build_model(dummy_scaler_path)
        x = np.random.randn(LENGTH)
        bg = model.background(x)
        reconstructed = model.reconstruct(x)
        np.testing.assert_allclose(reconstructed, x - bg, rtol=1e-5)

    def test_batch_reconstruct_returns_batch_shape(self, dummy_scaler_path):
        mock_nn = _make_mock_model((4, 2, 257, 257))
        with patch("deepextractor.model.load_torch_model", return_value=mock_nn):
            model = DeepExtractorModel(device="cpu", scaler_path=dummy_scaler_path)
        batch = np.random.randn(4, LENGTH)
        assert model.reconstruct(batch).shape == (4, LENGTH)

    def test_batch_background_returns_batch_shape(self, dummy_scaler_path):
        mock_nn = _make_mock_model((4, 2, 257, 257))
        with patch("deepextractor.model.load_torch_model", return_value=mock_nn):
            model = DeepExtractorModel(device="cpu", scaler_path=dummy_scaler_path)
        batch = np.random.randn(4, LENGTH)
        assert model.background(batch).shape == (4, LENGTH)

    def test_failed_checkpoint_raises_runtime_error(self, dummy_scaler_path):
        with patch("deepextractor.model.load_torch_model", return_value=None):
            with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
                DeepExtractorModel(device="cpu", scaler_path=dummy_scaler_path)

    def test_missing_scaler_raises_file_not_found(self):
        with pytest.raises((FileNotFoundError, OSError)):
            DeepExtractorModel(scaler_path="/nonexistent/path/scaler.pkl")

    def test_reconstruct_output_is_float_array(self, dummy_scaler_path):
        model = _build_model(dummy_scaler_path)
        x = np.random.randn(LENGTH).astype(np.float32)
        result = model.reconstruct(x)
        assert result.dtype in (np.float32, np.float64)
        assert isinstance(result, np.ndarray)

    def test_device_auto_detection(self, dummy_scaler_path):
        with patch("deepextractor.model.load_torch_model", return_value=_make_mock_model((1, 2, 257, 257))):
            model = DeepExtractorModel(device=None, scaler_path=dummy_scaler_path)
        expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert model.device == expected


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

class TestTopLevelAPI:

    def test_reconstruct_is_exported(self):
        assert callable(deepextractor.reconstruct)

    def test_extract_is_exported(self):
        assert callable(deepextractor.extract)

    def test_extract_is_alias_for_reconstruct(self):
        assert deepextractor.extract is deepextractor.reconstruct

    def test_deepextractormodel_is_exported(self):
        assert deepextractor.DeepExtractorModel is DeepExtractorModel

    def test_reconstruct_delegates_to_model(self, dummy_scaler_path):
        fake_result = np.zeros(LENGTH)
        with patch("deepextractor.api.DeepExtractorModel") as MockClass:
            instance = MockClass.return_value
            instance.reconstruct.return_value = fake_result
            result = deepextractor.reconstruct(
                np.random.randn(LENGTH),
                device="cpu",
                scaler_path=dummy_scaler_path,
            )
        instance.reconstruct.assert_called_once()
        np.testing.assert_array_equal(result, fake_result)
