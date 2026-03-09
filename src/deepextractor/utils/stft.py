"""STFT / iSTFT helpers for magnitude+phase representation."""

import torch


def apply_stft(array, n_fft, hop_length, win_length, window):
    """
    Compute STFT and return a magnitude+phase tensor.

    Parameters
    ----------
    array : array-like, shape (N, T) or (T,)
        Input time-series. Batched (N, T) is recommended; a 1D input will
        produce a (F, 2, T) tensor which is not compatible with UNET2D.
    n_fft, hop_length, win_length : int
        STFT parameters.
    window : torch.Tensor
        Analysis window (e.g. Hann), length ``win_length``.

    Returns
    -------
    torch.Tensor, shape (N, 2, F, frames)
        Channel 0 = magnitude, channel 1 = phase.
    """
    tensor = torch.tensor(array, dtype=torch.float32)
    stft_result = torch.stft(
        tensor, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True,
    )
    magnitude = torch.abs(stft_result)
    phase = torch.angle(stft_result)
    return torch.stack([magnitude, phase], dim=1)


def apply_istft(stft_array, n_fft, hop_length, win_length, window):
    """
    Reconstruct time-domain signal from a magnitude+phase tensor.

    Parameters
    ----------
    stft_array : torch.Tensor, shape (N, 2, F, frames)
        Channel 0 = magnitude, channel 1 = phase.
    n_fft, hop_length, win_length : int
        STFT parameters (must match the forward transform).
    window : torch.Tensor
        Synthesis window, moved to the correct device automatically.

    Returns
    -------
    torch.Tensor, shape (N, T)
    """
    magnitude = stft_array[:, 0, :, :]
    phase = stft_array[:, 1, :, :]
    real_part = magnitude * torch.cos(phase)
    imag_part = magnitude * torch.sin(phase)
    stft_complex = torch.complex(real_part, imag_part)
    window = window.to(stft_complex.device)
    return torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
