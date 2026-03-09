"""High-level DeepExtractor model wrapper for inference."""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from deepextractor.models.architectures import UNET2D
from deepextractor.utils.checkpoints import CHECKPOINT_BILBY, load_torch_model
from deepextractor.utils.stft import apply_istft, apply_stft

logger = logging.getLogger(__name__)


def _default_scaler_path() -> str:
    """Resolve the bundled scaler path relative to the package source tree."""
    # src/deepextractor/model.py -> src/deepextractor/ -> src/ -> repo root
    candidate = Path(__file__).parents[2] / "assets" / "scaler_bilby.pkl"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        f"Could not find scaler_bilby.pkl at {candidate}. "
        "Pass scaler_path= explicitly or ensure the assets/ directory is present."
    )


class DeepExtractorModel:
    """
    High-level wrapper around a pretrained DeepExtractor UNET2D model.

    Bundles the PyTorch model, StandardScaler, and STFT parameters into a
    single object so callers don't need to manage them separately.

    Parameters
    ----------
    checkpoint : str
        Model name / checkpoint key. Defaults to ``"DeepExtractor_257"``.
    checkpoint_filename : str
        Checkpoint file name within the model subdirectory on HuggingFace Hub
        or local ``checkpoint_dir``. Defaults to ``CHECKPOINT_BILBY``.
    checkpoint_dir : str | None
        Local directory to search for checkpoint files before falling back to
        HuggingFace Hub. Pass ``None`` to always use the Hub.
    device : str | torch.device | None
        Compute device. Auto-detects CUDA if available when ``None``.
    scaler_path : str | None
        Path to the scaler ``.pkl`` file. Defaults to the bundled
        ``assets/scaler_bilby.pkl``.
    n_fft : int
        STFT FFT size. Default 512.
    win_length : int
        STFT window length. Default 64.
    hop_length : int
        STFT hop length. Default 32.
    """

    def __init__(
        self,
        checkpoint: str = "DeepExtractor_257",
        checkpoint_filename: str = CHECKPOINT_BILBY,
        checkpoint_dir: str | None = None,
        device: str | torch.device | None = None,
        scaler_path: str | None = None,
        n_fft: int = 512,
        win_length: int = 64,
        hop_length: int = 32,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self._window = torch.hann_window(win_length)

        scaler_path = scaler_path or _default_scaler_path()
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        model_dict = {checkpoint: UNET2D(in_channels=2, out_channels=2)}
        self._model = load_torch_model(
            checkpoint,
            model_dict,
            checkpoint_dir=checkpoint_dir,
            device=self.device,
            checkpoint_filename=checkpoint_filename,
        )
        if self._model is None:
            raise RuntimeError(
                f"Failed to load checkpoint '{checkpoint}'. "
                "Check that the checkpoint name is correct and the weights are accessible."
            )

    def _scale(self, x: np.ndarray) -> np.ndarray:
        original_shape = x.shape
        return self._scaler.transform(x.reshape(-1, 1)).reshape(original_shape)

    def _unscale(self, x: np.ndarray) -> np.ndarray:
        original_shape = x.shape
        return self._scaler.inverse_transform(x.reshape(-1, 1)).reshape(original_shape)

    def background(self, noisy_input: np.ndarray) -> np.ndarray:
        """
        Estimate the background (noise-only) component.

        Parameters
        ----------
        noisy_input : np.ndarray
            1-D array of shape ``(T,)`` or 2-D batch of shape ``(N, T)``.

        Returns
        -------
        np.ndarray
            Background estimate, same shape as ``noisy_input``.
        """
        noisy_input = np.asarray(noisy_input, dtype=np.float64)
        single = noisy_input.ndim == 1
        if single:
            noisy_input = noisy_input[np.newaxis, :]  # (1, T)

        scaled = np.stack([self._scale(row) for row in noisy_input])  # (N, T)
        stft_tensor = apply_stft(
            scaled, self.n_fft, self.hop_length, self.win_length,
            self._window,
        ).to(self.device)  # (N, 2, F, frames)

        with torch.no_grad():
            output = self._model(stft_tensor)  # (N, 2, F, frames)

        time_domain = apply_istft(
            output, self.n_fft, self.hop_length, self.win_length, self._window,
        ).cpu().numpy()  # (N, T)

        bg = np.stack([self._unscale(row) for row in time_domain])  # (N, T)
        return bg[0] if single else bg

    def reconstruct(self, noisy_input: np.ndarray) -> np.ndarray:
        """
        Extract the transient signal by subtracting the predicted background.

        Parameters
        ----------
        noisy_input : np.ndarray
            1-D array of shape ``(T,)`` or 2-D batch of shape ``(N, T)``.

        Returns
        -------
        np.ndarray
            Reconstructed signal, same shape as ``noisy_input``.
        """
        noisy_input = np.asarray(noisy_input, dtype=np.float64)
        return noisy_input - self.background(noisy_input)
