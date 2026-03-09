import numpy as np
import pytest

from deepextractor.utils.metrics import (
    calculate_mae,
    calculate_mape,
    calculate_mse,
    calculate_psnr,
    calculate_r2,
    calculate_rmse,
    calculate_snr,
)


def test_mse_identical():
    x = np.random.randn(100)
    assert calculate_mse(x, x) == pytest.approx(0.0)


def test_mse_shape_mismatch():
    with pytest.raises(ValueError):
        calculate_mse(np.ones(10), np.ones(20))


def test_rmse_identical():
    x = np.random.randn(100)
    assert calculate_rmse(x, x) == pytest.approx(0.0)


def test_mae_identical():
    x = np.random.randn(100)
    assert calculate_mae(x, x) == pytest.approx(0.0)


def test_r2_perfect():
    x = np.random.randn(100)
    assert calculate_r2(x, x) == pytest.approx(1.0)


def test_r2_zero_model():
    x = np.random.randn(100)
    # Predicting the mean gives R² ≈ 0
    y_pred = np.full_like(x, np.mean(x))
    assert calculate_r2(x, y_pred) == pytest.approx(0.0, abs=1e-10)


def test_psnr_identical():
    x = np.random.randn(100)
    # MSE = 0 → PSNR = inf, so just test it runs without error for non-identical inputs
    x2 = x + 0.001
    psnr = calculate_psnr(x, x2)
    assert np.isfinite(psnr)


def test_snr_shape_preserved():
    x = np.random.randn(100)
    y = x + 0.1 * np.random.randn(100)
    snr = calculate_snr(x, y)
    assert np.isfinite(snr)


def test_mape_shape_mismatch():
    with pytest.raises(ValueError):
        calculate_mape(np.ones(5), np.ones(10))


def test_mape_identical():
    x = np.ones(50)
    assert calculate_mape(x, x) == pytest.approx(0.0)
