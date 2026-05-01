"""
Top-level convenience functions for DeepExtractor inference.

For one-shot use. For repeated inference on many signals, instantiate
:class:`DeepExtractorModel` or :class:`DeepExtractorSeparator` directly
to amortise the model load cost.
"""

from typing import Union
from pathlib import Path

import numpy as np

from deepextractor.model import DeepExtractorModel, DeepExtractorSeparator, SeparationResult
from deepextractor.utils.checkpoints import CHECKPOINT_BILBY


def reconstruct(
    noisy_input: np.ndarray,
    checkpoint: str = "DeepExtractor_257",
    checkpoint_filename: str = CHECKPOINT_BILBY,
    checkpoint_dir: str | None = None,
    device: str | None = None,
    scaler_path: str | None = None,
) -> np.ndarray:
    """
    Extract the transient signal from a noisy gravitational-wave strain.

    Loads a DeepExtractor model, runs inference, and returns the reconstructed
    signal. For repeated calls, prefer instantiating :class:`DeepExtractorModel`
    directly to avoid reloading weights on each call.

    Parameters
    ----------
    noisy_input : np.ndarray
        1-D array of shape ``(T,)`` or 2-D batch of shape ``(N, T)``.
    checkpoint : str
        Model name. Default ``"DeepExtractor_257"``.
    checkpoint_filename : str
        Checkpoint filename. Defaults to the bilby-noise checkpoint.
    checkpoint_dir : str | None
        Local checkpoint directory. Falls back to HuggingFace Hub if None.
    device : str | None
        Torch device string. Auto-detected if None.
    scaler_path : str | None
        Path to scaler .pkl. Uses bundled asset if None.

    Returns
    -------
    np.ndarray
        Reconstructed signal, same shape as ``noisy_input``.
    """
    model = DeepExtractorModel(
        checkpoint=checkpoint,
        checkpoint_filename=checkpoint_filename,
        checkpoint_dir=checkpoint_dir,
        device=device,
        scaler_path=scaler_path,
    )
    return model.reconstruct(noisy_input)


# `extract` and `reconstruct` are synonyms at the API level.
extract = reconstruct


def separate(
    h1: np.ndarray,
    l1: np.ndarray,
    checkpoint_path: Union[str, Path],
    scaler=None,
    device: str | None = None,
    model_kwargs: dict | None = None,
) -> SeparationResult:
    """Separate H1 and L1 strain into signal and background in the time domain.

    Loads a :class:`DeepExtractorSeparator`, runs inference, and returns the
    separated components. For repeated calls, instantiate
    :class:`DeepExtractorSeparator` directly to avoid reloading weights.

    Parameters
    ----------
    h1 : np.ndarray
        H1 strain. Shape ``(T,)`` or ``(N, T)``.
    l1 : np.ndarray
        L1 strain. Same shape as ``h1``.
    checkpoint_path : str | Path
        Path to the ``.pth.tar`` checkpoint saved during training.
    scaler : ChannelStandardScaler | str | Path | None
        Per-channel input scaler. Pass a fitted
        :class:`~deepextractor.data.ChannelStandardScaler` instance, a path to
        a pickled scaler, or ``None`` to skip scaling.
    device : str | None
        Torch device string. Auto-detected if None.
    model_kwargs : dict | None
        Override keyword arguments forwarded to :class:`UNET1D_LSTM_ATT`.

    Returns
    -------
    SeparationResult
        Named tuple with fields ``h1_signal``, ``l1_signal``,
        ``h1_background``, ``l1_background``.
    """
    separator = DeepExtractorSeparator(
        checkpoint_path=checkpoint_path,
        scaler=scaler,
        device=device,
        model_kwargs=model_kwargs,
    )
    return separator.separate(h1, l1)
