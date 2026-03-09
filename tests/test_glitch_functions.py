import numpy as np
import pytest

from deepextractor.generation.glitch_functions import (
    generate_chirp,
    generate_gaussian_pulse,
    generate_sine,
    generate_sine_gaussian,
    ringdown,
)


DURATION = 1.0
SAMPLE_RATE = 4096
EXPECTED_LEN = int(DURATION * SAMPLE_RATE)


def test_chirp_shape():
    t, sig = generate_chirp(DURATION, sample_rate=SAMPLE_RATE)
    assert len(sig) == EXPECTED_LEN
    assert len(t) == EXPECTED_LEN


def test_sine_shape():
    t, sig = generate_sine(DURATION, sample_rate=SAMPLE_RATE)
    assert len(sig) == EXPECTED_LEN


def test_sine_gaussian_shape():
    t, sig = generate_sine_gaussian(DURATION, sample_rate=SAMPLE_RATE)
    assert len(sig) == EXPECTED_LEN


def test_gaussian_pulse_shape():
    t, sig = generate_gaussian_pulse(DURATION, sample_rate=SAMPLE_RATE)
    assert len(sig) == EXPECTED_LEN


def test_ringdown_shape():
    t, sig = ringdown(DURATION, sample_rate=SAMPLE_RATE)
    assert len(sig) == EXPECTED_LEN


def test_chirp_output_finite():
    _, sig = generate_chirp(DURATION)
    assert np.all(np.isfinite(sig))


def test_ringdown_output_range():
    _, sig = ringdown(DURATION)
    # rescale should place output in [-1, 1]
    assert sig.min() >= -1.0 - 1e-9
    assert sig.max() <= 1.0 + 1e-9


def test_gengli_requires_package():
    gengli = pytest.importorskip("gengli", reason="gengli not installed")
    from deepextractor.generation.glitch_functions import generate_gengli_glitch
    _, glitch = generate_gengli_glitch(ifo="H1")
    assert glitch is not None
