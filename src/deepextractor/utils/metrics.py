import numpy as np


def calculate_mse(target, output):
    """Mean Squared Error between target and output arrays."""
    if target.shape != output.shape:
        raise ValueError("Target and output must have the same shape.")
    return np.mean((target.flatten() - output.flatten()) ** 2)


def calculate_rmse(target, output):
    """Root Mean Squared Error between target and output arrays."""
    return np.sqrt(calculate_mse(target, output))


def calculate_mae(target, output):
    """Mean Absolute Error between target and output arrays."""
    return np.mean(np.abs(target.flatten() - output.flatten()))


def calculate_snr(target, output):
    """Signal-to-Noise Ratio (dB) of the reconstruction."""
    signal_power = np.mean(target.flatten() ** 2)
    noise_power = np.mean((target.flatten() - output.flatten()) ** 2)
    return 10 * np.log10(signal_power / noise_power)


def calculate_psnr(target, output, max_value=1.0):
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = calculate_mse(target, output)
    return 10 * np.log10(max_value**2 / mse)


def calculate_r2(target, output):
    """Coefficient of determination (R²)."""
    target_flat = target.flatten()
    output_flat = output.flatten()
    ss_res = np.sum((target_flat - output_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    return 1 - (ss_res / ss_tot)


def calculate_mape(target, output):
    """
    Mean Absolute Percentage Error (MAPE), expressed as a percentage.

    Parameters
    ----------
    target : numpy.ndarray
        The target values.
    output : numpy.ndarray
        The predicted values.

    Returns
    -------
    float
        MAPE value as a percentage.
    """
    if target.shape != output.shape:
        raise ValueError("Target and output values must have the same shape.")
    epsilon = 1e-10
    target = np.where(target == 0, epsilon, target)
    return np.mean(np.abs((target - output) / target)) * 100
