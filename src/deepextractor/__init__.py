"""
DeepExtractor: Deep learning for gravitational-wave glitch reconstruction.

Quick usage::

    import deepextractor
    reconstructed = deepextractor.reconstruct(noisy_strain)

    # Or with explicit model control:
    model = deepextractor.DeepExtractorModel(checkpoint="DeepExtractor_257")
    signal = model.reconstruct(noisy_strain)
    background = model.background(noisy_strain)

Paper: https://arxiv.org/abs/2501.18423
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deepextractor")
except PackageNotFoundError:
    # Running from source without pip install -e .
    __version__ = "0.0.0.dev"

from deepextractor.model import DeepExtractorModel
from deepextractor.api import extract, reconstruct

__all__ = ["__version__", "DeepExtractorModel", "reconstruct", "extract"]
