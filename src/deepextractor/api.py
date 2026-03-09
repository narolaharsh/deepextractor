"""
Top-level convenience functions for DeepExtractor inference.

For one-shot use. For repeated inference on many signals, instantiate
:class:`DeepExtractorModel` directly to amortise the model load cost.
"""

import numpy as np

from deepextractor.model import DeepExtractorModel
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
