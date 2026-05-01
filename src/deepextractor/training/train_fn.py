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


def train_fn_td(
    loader,
    model,
    optimizer,
    loss_fn,
    scaler,
    device,
    *,
    residual_channels: bool = False,
    residual_weight: float = 1.0,
    residual_mode: str = "true",
    use_amp: bool = False,
):
    """Train the two-detector time-domain separation model for one epoch.

    Expects the DataLoader to yield:
        data    — (B, 2, T)  H1+L1 strain (standard-scaled inputs)
        targets — (B, 4, T)  [bg_H1, bg_L1, sig_H1, sig_L1] (whitened)

    Args:
        residual_channels: If True, model outputs 6 channels; the extra 2 are
            a residual term that enforces input reconstruction.
        residual_weight: Weight applied to the residual loss term.
        residual_mode: How the residual loss is computed —
            "true"       : residual target = data - (tgt_bg + tgt_sig)
            "sum"        : loss on (pred_bg + pred_sig + pred_res) vs data
            "sum_detach" : same but bg+sig gradients are detached
        use_amp: Enable mixed-precision autocast. Disabled by default —
            Snellius training found AMP unstable with whitened targets.

    Returns:
        (avg_total, avg_bg, avg_sig)           if residual_channels=False
        (avg_total, avg_bg, avg_sig, avg_res)  if residual_channels=True
    """
    loop = tqdm(loader, desc="Training on batch")
    tot = bg_acc = sig_acc = res_acc = 0.0

    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    for data, targets in loop:
        data = data.to(device)
        targets = targets.float().to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(autocast_device, enabled=use_amp):
            preds = model(data)

            if residual_channels:
                if preds.shape[1] != 6:
                    raise ValueError(f"residual_channels=True requires 6 output channels, got {preds.shape[1]}")
                pred_bg  = preds[:, 0:2]
                pred_sig = preds[:, 2:4]
                pred_res = preds[:, 4:6]
            else:
                if preds.shape[1] != 4:
                    raise ValueError(f"residual_channels=False requires 4 output channels, got {preds.shape[1]}")
                pred_bg  = preds[:, 0:2]
                pred_sig = preds[:, 2:4]
                pred_res = None

            tgt_bg  = targets[:, 0:2]
            tgt_sig = targets[:, 2:4]

            bg_loss  = loss_fn(pred_bg,  tgt_bg)
            sig_loss = loss_fn(pred_sig, tgt_sig)

            if residual_channels:
                if residual_mode == "true":
                    res_tgt  = data - (tgt_bg + tgt_sig)
                    res_loss = loss_fn(pred_res, res_tgt)
                elif residual_mode == "sum":
                    res_loss = loss_fn(pred_bg + pred_sig + pred_res, data)
                elif residual_mode == "sum_detach":
                    res_loss = loss_fn((pred_bg + pred_sig).detach() + pred_res, data)
                else:
                    raise ValueError(f"Unknown residual_mode '{residual_mode}'. Choose 'true', 'sum', or 'sum_detach'.")
                loss = 0.5 * (bg_loss + sig_loss) + residual_weight * res_loss
            else:
                res_loss = None
                loss = 0.5 * (bg_loss + sig_loss)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tot     += loss.item()
        bg_acc  += bg_loss.item()
        sig_acc += sig_loss.item()
        if res_loss is not None:
            res_acc += res_loss.item()

        postfix = {"total": loss.item(), "bg": bg_loss.item(), "sig": sig_loss.item()}
        if res_loss is not None:
            postfix["res"] = res_loss.item()
        loop.set_postfix(**postfix)

    n = max(1, len(loader))
    if residual_channels:
        return tot / n, bg_acc / n, sig_acc / n, res_acc / n
    return tot / n, bg_acc / n, sig_acc / n
