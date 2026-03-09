"""
Synthetic glitch signal generators.

The CDVGAN and gengli generators require optional dependencies:
    pip install deepextractor[generative]
"""

import numpy as np
from scipy.signal import chirp, gausspulse

from deepextractor.utils.signal import quality_factor_conversion, rescale

SRATE = 4096
NYQUIST_FREQ = SRATE // 2


def generate_chirp(duration, sample_rate=4096, f0_min=1, f0_max=NYQUIST_FREQ,
                   f1_min=1, f1_max=NYQUIST_FREQ):
    f0 = np.random.uniform(f0_min, f0_max)
    f1 = np.random.uniform(f1_min, f1_max)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = chirp(t, f0=f0, f1=f1, t1=duration, method="linear")
    return t, signal


def generate_sine(duration, sample_rate=4096, freq_min=1, freq_max=NYQUIST_FREQ):
    frequency = np.random.uniform(freq_min, freq_max)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal


def generate_sine_gaussian(duration, sample_rate=4096, freq_min=1, freq_max=NYQUIST_FREQ):
    tau = np.random.uniform(duration / 200, duration / 4)
    frequency = np.random.uniform(freq_min, freq_max)
    t = np.linspace(-duration / 2, duration / 2, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t) * np.exp(-(t**2) / (2 * tau**2))
    return t, signal


def generate_gaussian_pulse(duration, sample_rate=4096, fc_min=1, fc_max=NYQUIST_FREQ,
                             bw_min=0.1, bw_max=1.0, bwr_min=-10, bwr_max=0,
                             tpr_min=0.5, tpr_max=2.0):
    """
    Generate a Gaussian pulse with random parameters.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : int
        Sampling rate in Hz.
    fc_min, fc_max : float
        Range for the center frequency (Hz).
    bw_min, bw_max : float
        Range for the fractional bandwidth.
    bwr_min, bwr_max : float
        Range for the bandwidth reference level (dB).
    tpr_min, tpr_max : float
        Range for the taper reference level (dB).
    """
    bw = np.random.uniform(bw_min, bw_max)
    fc = np.random.uniform(fc_min, fc_max)
    bwr = np.random.uniform(bwr_min, bwr_max)
    tpr = np.random.uniform(tpr_min, tpr_max)
    t = np.linspace(-duration / 2, duration / 2, int(sample_rate * duration), endpoint=False)
    signal = gausspulse(t, fc=fc, bw=bw, bwr=bwr, tpr=tpr)
    return t, signal


def ringdown(duration, sample_rate=4096, n_signals=1):
    t = np.linspace(0, duration, int(sample_rate * duration))
    phi = np.random.uniform(0, 2 * np.pi)
    A = 1.0
    f_0 = np.random.uniform(10, NYQUIST_FREQ, n_signals)
    t_0 = np.random.uniform(t[-1] / 4, 3 * t[-1] / 4, n_signals)
    Q = np.random.uniform(5, 150, n_signals)
    tau = np.maximum(quality_factor_conversion(Q, f_0), 0.01)
    f_0 = np.expand_dims(f_0, axis=1)
    t_0 = np.expand_dims(t_0, axis=1)
    tau = np.expand_dims(tau, axis=1)
    h_1 = A * np.exp(-1.0 * ((t - t_0) / (tau))) * np.sin(2 * np.pi * f_0 * (t - t_0) + phi)
    h_1 = ((t - t_0) > 0) * h_1
    h_1 = rescale(h_1)
    if np.random.rand() < 0.5:
        h_1 = np.flip(h_1)
    return t, h_1[0]


def generate_gengli_glitch(ifo):
    """
    Generate a glitch sample using the gengli library.

    Requires the ``[generative]`` optional dependencies:
    ``pip install deepextractor[generative]``
    """
    try:
        import gengli
    except ImportError as e:
        raise ImportError(
            "gengli is required for this function. "
            "Install it with: pip install deepextractor[generative]"
        ) from e
    g = gengli.glitch_generator(ifo)
    glitch = g.get_glitch(1, srate=4096, snr=10, alpha=0.2, fhigh=1024)
    return None, glitch


def generate_cdvgan_glitch(gtype, cdvgan_generator):
    """
    Generate a glitch sample using a pretrained CDVGAN TensorFlow model.

    Parameters
    ----------
    gtype : str
        Glitch type: one of ``'blip'``, ``'tomte'``, ``'bbh'``,
        ``'simplex'``, ``'uniform'``.
    cdvgan_generator : tf.keras.Model
        The loaded CDVGAN generator model.
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required for CDVGAN glitches. "
            "Install it with: pip install deepextractor[generative]"
        ) from e

    latent_dim = 100
    random_ints = np.random.randint(0, 100, size=(1, 3))
    simplex_classes = random_ints / np.sum(random_ints, axis=1).reshape(1, 1)
    uniform_classes = np.random.uniform(low=0.0, high=1.0, size=(1, 3))

    class_vector_map = {
        "blip": [[1, 0, 0]],
        "tomte": [[0, 1, 0]],
        "bbh": [[0, 0, 1]],
        "simplex": simplex_classes,
        "uniform": uniform_classes,
    }

    class_vector = np.array(class_vector_map[gtype])
    latent_vector = tf.random.normal(shape=(1, latent_dim))
    generated_glitch = cdvgan_generator([latent_vector, class_vector]).numpy()
    return None, generated_glitch
