import torch
import torch.nn as nn
from tqdm import tqdm

from deepextractor.utils.checkpoints import load_checkpoint, load_optimizer, save_checkpoint
from deepextractor.utils.io import check_accuracy, get_loaders
from deepextractor.utils.visualization import save_predictions_as_plots


def train_fn(loader, model, model_name, optimizer, loss_fn, scaler, device):
    """Train the model for one epoch and return average losses."""
    loop = tqdm(loader, desc="Training on batch")

    epoch_loss = 0
    epoch_noise_loss = 0
    epoch_constraint_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().to(device=device)

        autocast_device = "cuda" if device.startswith("cuda") else "cpu"
        with torch.amp.autocast(autocast_device):
            predictions = model(data)

            if model_name == "UNET1D_diff":
                noise_pred = predictions[:, 0:1, :]
                residual_pred = predictions[:, 1:2, :]
                reconstructed = noise_pred + residual_pred
                constraint_loss = loss_fn(reconstructed, data)
                noise_loss = loss_fn(noise_pred, targets)
                loss = constraint_loss + noise_loss
                epoch_noise_loss += noise_loss.item()
                epoch_constraint_loss += constraint_loss.item()
            else:
                loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        if model_name == "UNET1D_diff":
            loop.set_postfix(
                total_loss=loss.item(),
                constraint_loss=constraint_loss.item(),
                noise_loss=noise_loss.item(),
            )
        else:
            loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(loader)
    avg_noise_loss = epoch_noise_loss / len(loader) if model_name == "UNET1D_diff" else 0
    avg_constraint_loss = (
        epoch_constraint_loss / len(loader) if model_name == "UNET1D_diff" else 0
    )

    return avg_loss, avg_noise_loss, avg_constraint_loss
