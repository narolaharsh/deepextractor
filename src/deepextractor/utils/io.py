import os
import logging

import numpy as np
import torch
import torch.nn as nn
from gwpy.timeseries import TimeSeries
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


def get_loaders(
    train_dir,
    train_target_dir,
    val_dir,
    val_target_dir,
    batch_size,
    train_transform=False,
    val_transform=False,
    num_workers=4,
    pin_memory=True,
    time_domain=True,
):
    """Return train and validation DataLoaders."""
    from deepextractor.data.datasets import SpectrogramDataset, TimeSeriesDataset
    if time_domain:
        train_ds = TimeSeriesDataset(
            input_npy=train_dir,
            target_npy=train_target_dir,
            transform=train_transform,
        )
        val_ds = TimeSeriesDataset(
            input_npy=val_dir,
            target_npy=val_target_dir,
            transform=val_transform,
        )
    else:
        print("Got spec datasets")
        train_ds = SpectrogramDataset(
            input_npy=train_dir,
            target_npy=train_target_dir,
            transform=train_transform,
        )
        val_ds = SpectrogramDataset(
            input_npy=val_dir,
            target_npy=val_target_dir,
            transform=val_transform,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def check_accuracy(loader, model, model_name, device="cuda"):
    """Compute MSE loss on the validation set and return average losses."""
    mse_loss_fn = nn.MSELoss()
    total_mse_loss = 0
    total_noise_loss = 0
    total_constraint_loss = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            predictions = model(x)

            if model_name == "UNET1D_diff":
                noise_pred = predictions[:, 0:1, :]
                residual_pred = predictions[:, 1:2, :]
                reconstructed = noise_pred + residual_pred
                noise_loss = mse_loss_fn(noise_pred, y)
                constraint_loss = mse_loss_fn(reconstructed, x)
                total_noise_loss += noise_loss.item() * x.size(0)
                total_constraint_loss += constraint_loss.item() * x.size(0)
                total_loss = noise_loss + constraint_loss
            else:
                total_loss = mse_loss_fn(predictions, y)

            total_mse_loss += total_loss.item() * x.size(0)
            num_samples += x.size(0)

    model.train()

    avg_mse_loss = total_mse_loss / num_samples
    avg_noise_loss = total_noise_loss / num_samples if model_name == "UNET1D_diff" else None
    avg_constraint_loss = (
        total_constraint_loss / num_samples if model_name == "UNET1D_diff" else None
    )

    if model_name == "UNET1D_diff":
        print(
            f"Validation Losses - Total: {avg_mse_loss:.6f}, "
            f"Noise: {avg_noise_loss:.6f}, Constraint: {avg_constraint_loss:.6f}"
        )
    else:
        print(f"Validation Loss: {avg_mse_loss:.6f}")

    return avg_mse_loss, avg_noise_loss, avg_constraint_loss


def numpy_to_gwf(strain, sample_times, channel, output_filename):
    """
    Write a strain time series to a GWF (frame) file.

    Parameters
    ----------
    strain : array-like
        The time-domain strain data.
    sample_times : array-like
        The corresponding time array.
    channel : str
        Channel name, e.g. ``'L1:STRAIN'``.
    output_filename : str
        Path for the output GWF file.
    """
    frame_file = TimeSeries(strain, times=sample_times, channel=channel)
    frame_file.write(output_filename)
    return None


def gwf_to_lcf(start_time, duration, channel_name, gwf_file_location):
    """Write a minimal LCF (frame cache) file alongside the GWF file."""
    output_string = (
        f"{channel_name[0]} {channel_name} {int(start_time)} {int(duration - 2)} "
        f"file://localhost{gwf_file_location}"
    )
    os.system(
        "echo %s > %s"
        % (output_string, f"{gwf_file_location.replace('gwf', 'lcf')}")
    )
    return None


def load_tf_model(path, model_name):
    """
    Load a TensorFlow/Keras SavedModel.

    Requires the ``[generative]`` optional dependencies:
    ``pip install deepextractor[generative]``.
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required for this function. "
            "Install it with: pip install deepextractor[generative]"
        ) from e
    return tf.keras.models.load_model(os.path.join(path, model_name))
