"""
Investigate and plot whitened background samples.

Usage
-----
    python scripts/plot_backgrounds.py --pkl backgrounds_test.pkl --out plots/
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deepextractor.utils.visualization import plot_q_transform

SAMPLE_RATE     = 4096
SAMPLE_DURATION = 8
RUNS            = ['O3a', 'O3b', 'O4a', 'O4b']
IFOS            = ['H1', 'L1']
N_EXAMPLES      = 5   # time series examples to plot per IFO/run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pkl', type=Path, required=True, help='Path to backgrounds pickle')
    p.add_argument('--out', type=Path, default=Path('plots'), help='Output directory')
    return p.parse_args()


def get_samples(entry) -> np.ndarray:
    """Handle both old (array) and new (dict with gps_starts) pickle formats."""
    if isinstance(entry, dict):
        return entry['samples']
    return entry


def print_stats(backgrounds: dict) -> None:
    print(f"\n{'Run':<6} {'IFO':<4} {'N':>6}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    print('─' * 55)
    for run in RUNS:
        if run not in backgrounds:
            continue
        for ifo in IFOS:
            samples = get_samples(backgrounds[run][ifo])
            n = len(samples)
            if n == 0:
                print(f"{run:<6} {ifo:<4} {'0':>6}")
                continue
            flat = samples.ravel()
            print(f"{run:<6} {ifo:<4} {n:>6}  {flat.mean():>8.4f}  {flat.std():>8.4f}"
                  f"  {flat.min():>8.3f}  {flat.max():>8.3f}")


def plot_timeseries(backgrounds: dict, out: Path) -> None:
    t = np.linspace(0, SAMPLE_DURATION, SAMPLE_RATE * SAMPLE_DURATION, endpoint=False)

    for run in RUNS:
        if run not in backgrounds:
            continue
        fig, axes = plt.subplots(len(IFOS), N_EXAMPLES,
                                 figsize=(3 * N_EXAMPLES, 3 * len(IFOS)),
                                 sharey='row')
        fig.suptitle(f'Whitened background samples — {run}', fontsize=12)

        for row, ifo in enumerate(IFOS):
            samples = get_samples(backgrounds[run][ifo])
            n = len(samples)
            idxs = np.random.choice(n, size=min(N_EXAMPLES, n), replace=False)
            for col, idx in enumerate(idxs):
                ax = axes[row, col]
                ax.plot(t, samples[idx], lw=0.5, color='steelblue')
                ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
                ax.set_xlim(0, SAMPLE_DURATION)
                if col == 0:
                    ax.set_ylabel(ifo, fontsize=10)
                if row == len(IFOS) - 1:
                    ax.set_xlabel('Time (s)', fontsize=8)
                ax.tick_params(labelsize=7)

        fig.tight_layout()
        path = out / f'timeseries_{run}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Saved {path}')


def plot_amplitude_distributions(backgrounds: dict, out: Path) -> None:
    x = np.linspace(-6, 6, 300)
    gauss = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    fig, axes = plt.subplots(len(IFOS), len(RUNS),
                             figsize=(4 * len(RUNS), 3 * len(IFOS)),
                             sharey=True)
    fig.suptitle('Amplitude distributions (should be ~N(0,1))', fontsize=12)

    for row, ifo in enumerate(IFOS):
        for col, run in enumerate(RUNS):
            ax = axes[row, col]
            if run not in backgrounds:
                ax.set_visible(False)
                continue
            samples = get_samples(backgrounds[run][ifo])
            if len(samples) == 0:
                ax.set_visible(False)
                continue
            ax.hist(samples.ravel(), bins=100, density=True,
                    color='steelblue', alpha=0.7, label='data')
            ax.plot(x, gauss, 'r--', lw=1.5, label='N(0,1)')
            ax.set_xlim(-6, 6)
            ax.set_title(f'{ifo} {run}', fontsize=9)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel('Density', fontsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    fig.tight_layout()
    path = out / 'amplitude_distributions.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def plot_psds(backgrounds: dict, out: Path) -> None:
    freqs = np.fft.rfftfreq(SAMPLE_RATE * SAMPLE_DURATION, d=1.0 / SAMPLE_RATE)

    fig, axes = plt.subplots(len(IFOS), len(RUNS),
                             figsize=(4 * len(RUNS), 3 * len(IFOS)),
                             sharey=True)
    fig.suptitle('Mean PSD of whitened samples (should be ~flat at 1)', fontsize=12)

    for row, ifo in enumerate(IFOS):
        for col, run in enumerate(RUNS):
            ax = axes[row, col]
            if run not in backgrounds:
                ax.set_visible(False)
                continue
            samples = get_samples(backgrounds[run][ifo])
            if len(samples) == 0:
                ax.set_visible(False)
                continue
            psds = np.abs(np.fft.rfft(samples, axis=1))**2 / (SAMPLE_RATE * samples.shape[1])
            mean_psd = psds.mean(axis=0)
            ax.semilogy(freqs[1:], mean_psd[1:], lw=0.8, color='steelblue')
            ax.axhline(1.0, color='r', lw=1, ls='--', alpha=0.7)
            ax.set_xlim(10, SAMPLE_RATE // 2)
            ax.set_title(f'{ifo} {run}', fontsize=9)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel('PSD', fontsize=8)
            if row == len(IFOS) - 1:
                ax.set_xlabel('Frequency (Hz)', fontsize=8)

    fig.tight_layout()
    path = out / 'psds.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def plot_qscans(backgrounds: dict, out: Path,
                frange: list = [10, 1200], qrange: list = [4, 64]) -> None:
    for run in RUNS:
        if run not in backgrounds:
            continue
        fig, axes = plt.subplots(len(IFOS), N_EXAMPLES,
                                 figsize=(3.5 * N_EXAMPLES, 3.5 * len(IFOS)))
        fig.suptitle(f'Q-scans of whitened background samples — {run}', fontsize=12)

        for row, ifo in enumerate(IFOS):
            samples = get_samples(backgrounds[run][ifo])
            n = len(samples)
            idxs = np.random.choice(n, size=min(N_EXAMPLES, n), replace=False)
            for col, idx in enumerate(idxs):
                ax = axes[row, col]
                plot_q_transform(
                    samples[idx],
                    srate=SAMPLE_RATE,
                    whiten=False,   # already whitened
                    ax=ax,
                    colourbar=False,
                    qrange=qrange,
                    frange=frange,
                )
                ax.set_title(f'{ifo} sample {idx}', fontsize=8)
                if col != 0:
                    ax.set_ylabel('')
                if row != len(IFOS) - 1:
                    ax.set_xlabel('')

        fig.tight_layout()
        path = out / f'qscans_{run}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Saved {path}')


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.pkl, 'rb') as f:
        backgrounds = pickle.load(f)

    print_stats(backgrounds)
    print(f'\nSaving plots to {args.out}/')
    plot_timeseries(backgrounds, args.out)
    plot_amplitude_distributions(backgrounds, args.out)
    plot_psds(backgrounds, args.out)
    plot_qscans(backgrounds, args.out)


if __name__ == '__main__':
    main()
