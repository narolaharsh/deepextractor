"""High-level DeepExtractor model wrapper for inference."""

import logging
import pickle
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import torch

from deepextractor.models.architectures import UNET1D_LSTM_ATT, UNET2D
from deepextractor.utils.checkpoints import CHECKPOINT_BILBY, load_torch_model
from deepextractor.utils.stft import apply_istft, apply_stft

logger = logging.getLogger(__name__)


def _default_scaler_path() -> str:
    """Resolve the bundled scaler path using importlib.resources."""
    import importlib.resources as pkg_resources
    ref = pkg_resources.files("deepextractor") / "assets" / "scaler_bilby.pkl"
    path = Path(str(ref))
    if path.is_file():
        return str(path)
    raise FileNotFoundError(
        f"Could not find bundled scaler_bilby.pkl. "
        "Pass scaler_path= explicitly or ensure the package was installed correctly."
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


class SeparationResult(NamedTuple):
    """Outputs of :meth:`DeepExtractorSeparator.separate`.

    All arrays have shape ``(T,)`` for single inputs or ``(N, T)`` for batches.
    """

    h1_signal: np.ndarray
    l1_signal: np.ndarray
    h1_background: np.ndarray
    l1_background: np.ndarray


class DeepExtractorSeparator:
    """Two-detector time-domain signal/glitch separator.

    Wraps a pretrained :class:`~deepextractor.models.UNET1D_LSTM_ATT` model and
    a :class:`~deepextractor.data.ChannelStandardScaler` to expose a clean
    inference API for separating H1+L1 strain into signal and background
    components in the time domain.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the ``.pth.tar`` checkpoint saved during training.
    scaler : ChannelStandardScaler | str | Path | None
        Per-channel input scaler. Pass a fitted
        :class:`~deepextractor.data.ChannelStandardScaler` instance, a path to
        a pickled scaler, or ``None`` to skip scaling (not recommended —
        the model expects standard-scaled inputs).
    device : str | torch.device | None
        Compute device. Auto-detects CUDA if available when ``None``.
    model_kwargs : dict | None
        Override keyword arguments forwarded to :class:`UNET1D_LSTM_ATT`.
        By default uses ``in_channels=2, out_channels=4``.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        scaler=None,
        device: Union[str, torch.device, None] = None,
        model_kwargs: dict | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load scaler
        if scaler is None:
            self._scaler = None
            logger.warning(
                "No scaler provided — inputs will not be normalised. "
                "The model expects standard-scaled inputs."
            )
        elif isinstance(scaler, (str, Path)):
            with open(scaler, "rb") as f:
                self._scaler = pickle.load(f)
        else:
            self._scaler = scaler  # assume pre-fitted ChannelStandardScaler

        # Build model
        kwargs = {"in_channels": 2, "out_channels": 4}
        if model_kwargs:
            kwargs.update(model_kwargs)
        self._model = UNET1D_LSTM_ATT(**kwargs)
        self._model.to(self.device)

        # Load checkpoint weights
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        self._model.load_state_dict(state)
        self._model.eval()
        logger.info("Loaded separator checkpoint from %s", checkpoint_path)

    def separate(
        self, h1: np.ndarray, l1: np.ndarray
    ) -> SeparationResult:
        """Separate H1 and L1 strain into signal and background components.

        Parameters
        ----------
        h1 : np.ndarray
            H1 strain. Shape ``(T,)`` or ``(N, T)``.
        l1 : np.ndarray
            L1 strain. Same shape as ``h1``.

        Returns
        -------
        SeparationResult
            Named tuple with fields ``h1_signal``, ``l1_signal``,
            ``h1_background``, ``l1_background``, each of the same shape as
            the inputs.
        """
        h1 = np.asarray(h1, dtype=np.float32)
        l1 = np.asarray(l1, dtype=np.float32)

        single = h1.ndim == 1
        if single:
            h1 = h1[np.newaxis, :]
            l1 = l1[np.newaxis, :]

        # Stack to (N, 2, T)
        x = np.stack([h1, l1], axis=1)

        if self._scaler is not None:
            x = self._scaler.transform(x)

        tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            out = self._model(tensor)  # (N, 4, T)

        out_np = out.cpu().numpy()

        # Output channel layout: [h1_bg, l1_bg, h1_sig, l1_sig]
        h1_bg  = out_np[:, 0, :]
        l1_bg  = out_np[:, 1, :]
        h1_sig = out_np[:, 2, :]
        l1_sig = out_np[:, 3, :]

        if single:
            h1_bg  = h1_bg[0]
            l1_bg  = l1_bg[0]
            h1_sig = h1_sig[0]
            l1_sig = l1_sig[0]

        return SeparationResult(
            h1_signal=h1_sig,
            l1_signal=l1_sig,
            h1_background=h1_bg,
            l1_background=l1_bg,
        )
