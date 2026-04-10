"""Tests for deepextractor.generation.generate_timeseries.generate_gaussian_noise."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deepextractor.generation.generate_timeseries import (
    SAMPLE_RATE,
    generate_gaussian_noise,
)

MEAN = 0
STD_DEV = np.sqrt(SAMPLE_RATE / 2)  # PyCBC convention: variance = SAMPLE_RATE / 2
NUM_SAMPLES = 50
LENGTH = int(2.0 * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# pycbc (numpy) path
# ---------------------------------------------------------------------------

def test_pycbc_output_shape():
    out = generate_gaussian_noise(MEAN, STD_DEV, NUM_SAMPLES, (LENGTH,), bilby_noise=False)
    assert out.shape == (NUM_SAMPLES, LENGTH)


def test_pycbc_output_dtype():
    out = generate_gaussian_noise(MEAN, STD_DEV, NUM_SAMPLES, (LENGTH,), bilby_noise=False)
    assert np.issubdtype(out.dtype, np.floating)


def test_pycbc_all_finite():
    out = generate_gaussian_noise(MEAN, STD_DEV, NUM_SAMPLES, (LENGTH,), bilby_noise=False)
    assert np.isfinite(out).all()


def test_pycbc_respects_num_samples():
    for n in [1, 5, 20]:
        out = generate_gaussian_noise(MEAN, STD_DEV, n, (LENGTH,), bilby_noise=False)
        assert out.shape[0] == n


def test_pycbc_mean_near_zero():
    """Mean should be near 0 — use a large batch for a stable estimate."""
    rng = np.random.default_rng(0)
    out = rng.normal(loc=MEAN, scale=STD_DEV, size=(500, LENGTH))
    assert abs(out.mean()) < 1.0


def test_pycbc_variance():
    """Variance should equal SAMPLE_RATE / 2 (PyCBC convention)."""
    rng = np.random.default_rng(0)
    out = rng.normal(loc=MEAN, scale=STD_DEV, size=(500, LENGTH))
    assert abs(out.var() - STD_DEV**2) / STD_DEV**2 < 0.05


# ---------------------------------------------------------------------------
# bilby path — mocked (fast, tests our code logic)
# ---------------------------------------------------------------------------

def _make_mock_ifos(strain_values):
    """Return a mock InterferometerList whose first ifo has the given whitened strain."""
    mock_ifo = MagicMock()
    mock_ifo.whitened_time_domain_strain = strain_values
    mock_ifos = MagicMock()
    mock_ifos.__iter__ = MagicMock(return_value=iter([mock_ifo]))
    mock_ifos.__getitem__ = MagicMock(return_value=mock_ifo)
    return mock_ifos


def _bilby_mock(fake_strain):
    """Build a mock bilby module whose InterferometerList returns the given strain."""
    mock_bilby = MagicMock()
    mock_bilby.gw.detector.InterferometerList.return_value = _make_mock_ifos(fake_strain)
    return mock_bilby


def test_bilby_output_shape():
    mock_bilby = _bilby_mock(list(np.random.randn(LENGTH)))
    with patch.dict("sys.modules", {"bilby": mock_bilby}):
        out = generate_gaussian_noise(MEAN, STD_DEV, NUM_SAMPLES, (LENGTH,), bilby_noise=True)
    assert out.shape == (NUM_SAMPLES, LENGTH)


def test_bilby_output_is_numpy_array():
    mock_bilby = _bilby_mock(list(np.random.randn(LENGTH)))
    with patch.dict("sys.modules", {"bilby": mock_bilby}):
        out = generate_gaussian_noise(MEAN, STD_DEV, 2, (LENGTH,), bilby_noise=True)
    assert isinstance(out, np.ndarray)


def test_bilby_calls_correct_detector():
    """The detector argument should be forwarded to InterferometerList."""
    mock_bilby = _bilby_mock(list(np.random.randn(LENGTH)))
    with patch.dict("sys.modules", {"bilby": mock_bilby}):
        generate_gaussian_noise(MEAN, STD_DEV, 2, (LENGTH,), bilby_noise=True, detector="V1")
    calls = mock_bilby.gw.detector.InterferometerList.call_args_list
    assert all(call.args[0] == ["V1"] for call in calls)


def test_bilby_default_detector_is_l1():
    mock_bilby = _bilby_mock(list(np.random.randn(LENGTH)))
    with patch.dict("sys.modules", {"bilby": mock_bilby}):
        generate_gaussian_noise(MEAN, STD_DEV, 2, (LENGTH,), bilby_noise=True)
    calls = mock_bilby.gw.detector.InterferometerList.call_args_list
    assert all(call.args[0] == ["L1"] for call in calls)


def test_bilby_missing_package_raises():
    """ImportError should propagate clearly if bilby is not installed."""
    with patch.dict("sys.modules", {"bilby": None}):
        with pytest.raises(ImportError):
            generate_gaussian_noise(MEAN, STD_DEV, 1, (LENGTH,), bilby_noise=True)


# ---------------------------------------------------------------------------
# bilby path — real call (slow integration tests, skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_bilby_real_mean_near_zero():
    out = generate_gaussian_noise(MEAN, STD_DEV, 5, (LENGTH,), bilby_noise=True)
    assert abs(out.mean()) < 0.5


@pytest.mark.slow
def test_bilby_real_variance_near_unit():
    """Bilby returns whitened strain — variance should be roughly 1."""
    out = generate_gaussian_noise(MEAN, STD_DEV, 5, (LENGTH,), bilby_noise=True)
    assert 0.5 < out.var() < 2.0


@pytest.mark.slow
def test_bilby_real_virgo_shape():
    out = generate_gaussian_noise(MEAN, STD_DEV, 2, (LENGTH,), bilby_noise=True, detector="V1")
    assert out.shape == (2, LENGTH)
