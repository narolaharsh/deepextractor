import os

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import torch
from gwpy.timeseries import TimeSeries


def save_predictions_as_plots(loader, model, folder="saved_predictions/", device="cuda"):
    """Save model prediction vs target plots for each sample in the loader."""
    model.eval()
    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x).cpu().numpy()

        for i, pred in enumerate(preds):
            plt.figure(figsize=(10, 4))
            plt.plot(pred.squeeze(), label="Prediction", color="b")
            plt.plot(y[i].cpu().numpy().squeeze(), label="Target", color="r", linestyle="--")
            plt.title(f"Time Series Prediction vs Target {idx}_{i}")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{folder}/plot_{idx}_{i}.png")
            plt.close()

    model.train()


def plot_examples(
    Difference_ts,
    clean_glitch_subtract,
    snrs,
    signal_type,
    PLOTS_PATH,
    indices_to_plot,
    noisy=False,
):
    """Plot up to 3 example time series and save to disk."""
    plt.figure(figsize=(18, 5))
    for i, idx in enumerate(indices_to_plot):
        plt.subplot(1, 3, i + 1)
        plt.plot(Difference_ts[idx], label="Difference_ts", color="red", alpha=0.7)
        plt.plot(
            clean_glitch_subtract[idx],
            label="Clean Glitch Subtract",
            color="blue",
            alpha=0.5,
        )
        plt.title(f"Example {i + 1} for {signal_type} with SNR={np.round(snrs[idx], 2)}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    suffix = "noisy_example" if noisy else "example"
    plt.savefig(os.path.join(PLOTS_PATH, f"{signal_type}_{suffix}"))
    plt.close()


def plot_q_transform(data, srate=4096.0, crop=None, whiten=True, ax=None, colourbar=True):
    """
    Plot the Q-transform of a time series using gwpy.

    Parameters
    ----------
    data : array-like
        Input time-domain data.
    srate : float
        Sample rate in Hz.
    crop : tuple or list, optional
        ``(center_time, duration)`` window in seconds for the Q-transform.
    whiten : bool
        If True, apply whitening before the Q-transform.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot. A new figure is created if not provided.
    colourbar : bool
        If True, add a colorbar to the plot.
    """
    data = TimeSeries(data, sample_rate=srate)

    q_scan = data.q_transform(
        qrange=[4, 64],
        frange=[1, 1290],
        tres=0.002,
        fres=0.5,
        whiten=whiten,
    )

    if isinstance(crop, (list, tuple)):
        t_center, dur = crop
        t_center = t_center + data.t0.value
        q_scan = q_scan.crop(t_center - dur / 2, t_center + dur / 2)
        xticklabels = np.linspace(0, 2, 5)

    if ax is None:
        fig, ax = plt.subplots(dpi=120)

    im = ax.imshow(q_scan, aspect="auto", extent=[0, 2, 10, 1290])
    ax.set_yscale("log", base=2)
    ax.set_xscale("linear")

    if isinstance(crop, (list, tuple)):
        ax.set_xticks(xticklabels)
        ax.set_xticklabels(xticklabels)

    ax.set_ylabel("Frequency (Hz)", fontsize=14)
    ax.set_xlabel("Time (s)", labelpad=0.1, fontsize=14)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="both", which="major", labelsize=14)

    im.set_clim(0, 25.5)
    if colourbar:
        cb = ax.figure.colorbar(im, ax=ax, label="Normalized energy", pad=0.01)
        cb.ax.tick_params(labelsize=18)
        cb.set_label("Normalized energy", fontsize=24)
