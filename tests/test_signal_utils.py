import numpy as np
import pytest

from deepextractor.utils.signal import (
    butter_filter,
    generate_gaussian_noise,
    quality_factor_conversion,
    rescale,
    whitened_snr_scaling,
)


def test_whitened_snr_scaling_preserves_shape():
    glitch = np.random.randn(8192)
    scaled = whitened_snr_scaling(glitch, snr=10.0)
    assert scaled.shape == glitch.shape


def test_whitened_snr_scaling_none_snr():
    glitch = np.random.randn(100)
    result = whitened_snr_scaling(glitch, snr=None)
    np.testing.assert_array_equal(result, glitch)


def test_whitened_snr_scaling_dtype():
    glitch = np.random.randn(1024)
    scaled = whitened_snr_scaling(glitch, snr=5.0)
    assert isinstance(scaled, np.ndarray)


def test_rescale_range():
    x = np.random.randn(3, 100)
    out = rescale(x)
    assert out.min() >= -1.0 - 1e-9
    assert out.max() <= 1.0 + 1e-9


def test_rescale_shape_preserved():
    x = np.random.randn(5, 200)
    out = rescale(x)
    assert out.shape == x.shape


def test_quality_factor_conversion():
    tau = quality_factor_conversion(Q=10.0, f_0=100.0)
    assert tau > 0


def test_butter_filter_shape():
    data = np.random.randn(8192)
    filtered = butter_filter(data, fs=4096)
    assert filtered.shape == data.shape


def test_generate_gaussian_noise_shape():
    noise = generate_gaussian_noise(0, 1, 5, (8192,))
    assert noise.shape == (5, 8192)
