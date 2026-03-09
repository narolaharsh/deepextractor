"""
Convert time-domain .npy arrays to STFT spectrograms (magnitude + phase).

Also provides a utility to concatenate chunked spectrogram files.

Usage::

    deepextractor-specgen --input-dir data/pycbc_noise/time_domain/ --output-dir data/pycbc_noise/spectrogram_domain/

"""

import argparse
import os

import numpy as np
import torch


# Default STFT parameters (257x257 output shape)
DEFAULT_N_FFT = 256 * 2
DEFAULT_WIN_LENGTH = DEFAULT_N_FFT // 8
DEFAULT_HOP_LENGTH = DEFAULT_WIN_LENGTH // 2


def apply_stft_and_save(
    array_path, save_path, n_fft, hop_length, win_length, window, chunk_size=5000
):
    """Apply STFT to a .npy array in chunks and save the result."""
    array = np.load(array_path)
    print(f"Loaded {array_path}, shape: {array.shape}")

    total_chunks = array.shape[0] // chunk_size
    stft_list = []

    for i in range(0, array.shape[0], chunk_size):
        chunk = array[i : i + chunk_size]
        tensor = torch.tensor(chunk, dtype=torch.float32)
        stft_result = torch.stft(
            tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        magnitude = torch.abs(stft_result)
        phase = torch.angle(stft_result)
        stft_mag_phase = torch.stack([magnitude, phase], dim=1)
        stft_list.append(stft_mag_phase)

        del tensor, stft_result, magnitude, phase
        torch.cuda.empty_cache()

        print(f"Processed chunk {i // chunk_size + 1}/{max(total_chunks, 1)}")

    stft_final = torch.cat(stft_list, dim=0)
    stft_numpy = stft_final.cpu().numpy()
    np.save(save_path, stft_numpy)
    print(f"STFT saved to {save_path}.npy, final shape: {stft_numpy.shape}")

    del array, stft_list, stft_final, stft_numpy
    torch.cuda.empty_cache()


def load_and_concatenate_chunks(data_dir, base_filename, total_chunks):
    """Load and concatenate chunked numpy arrays saved as ``{base}_chunk_{i}.npy``."""
    stft_list = []
    for i in range(total_chunks):
        chunk_filename = f"{base_filename}_chunk_{i}.npy"
        chunk_path = os.path.join(data_dir, chunk_filename)
        if os.path.exists(chunk_path):
            print(f"Loading {chunk_filename}...")
            stft_list.append(np.load(chunk_path))
        else:
            print(f"Chunk {chunk_filename} not found. Skipping.")
    print("Concatenating chunks...")
    return np.concatenate(stft_list, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Convert time-domain .npy arrays to STFT spectrogram arrays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing the time-domain .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the spectrogram .npy files.",
    )
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--win-length", type=int, default=DEFAULT_WIN_LENGTH)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--combine-chunks",
        action="store_true",
        help="Combine pre-existing chunk files instead of generating new spectrograms.",
    )
    parser.add_argument(
        "--chunks-glitch-train", type=int, default=16,
        help="Number of chunks for glitch_train (used with --combine-chunks).",
    )
    parser.add_argument(
        "--chunks-background-train", type=int, default=16,
        help="Number of chunks for background_train (used with --combine-chunks).",
    )
    parser.add_argument(
        "--chunks-glitch-val", type=int, default=2,
        help="Number of chunks for glitch_val (used with --combine-chunks).",
    )
    parser.add_argument(
        "--chunks-background-val", type=int, default=2,
        help="Number of chunks for background_val (used with --combine-chunks).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    window = torch.hann_window(args.win_length)

    if args.combine_chunks:
        for base, n_chunks in [
            ("glitch_train_scaled_mag_phase", args.chunks_glitch_train),
            ("background_train_scaled_mag_phase", args.chunks_background_train),
            ("glitch_val_scaled_mag_phase", args.chunks_glitch_val),
            ("background_val_scaled_mag_phase", args.chunks_background_val),
        ]:
            combined = load_and_concatenate_chunks(args.output_dir, base, n_chunks)
            out_path = os.path.join(args.output_dir, f"{base}_combined.npy")
            np.save(out_path, combined)
            print(f"Saved combined {base} to {out_path}")
        print("All combined datasets saved.")
    else:
        datasets = [
            ("glitch_train_scaled.npy", "glitch_train_scaled_mag_phase"),
            ("background_train_scaled.npy", "background_train_scaled_mag_phase"),
            ("glitch_val_scaled.npy", "glitch_val_scaled_mag_phase"),
            ("background_val_scaled.npy", "background_val_scaled_mag_phase"),
        ]
        for in_name, out_name in datasets:
            in_path = os.path.join(args.input_dir, in_name)
            out_path = os.path.join(args.output_dir, out_name)
            apply_stft_and_save(
                in_path, out_path,
                args.n_fft, args.hop_length, args.win_length, window,
                args.chunk_size,
            )
        print("All STFT results saved.")


if __name__ == "__main__":
    main()
