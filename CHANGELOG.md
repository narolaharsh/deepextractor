# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial pip-installable package structure
- `deepextractor.models` — U-Net 1D/2D, DnCNN 1D, Autoencoder 1D/2D architectures
- `deepextractor.data` — PyTorch Dataset classes for time-series and spectrogram data
- `deepextractor.training` — training loop and trainer CLI entry point
- `deepextractor.generation` — synthetic glitch signal generators and data generation scripts
- `deepextractor.evaluation` — simulated evaluation metrics
- `deepextractor.utils` — checkpoints, signal processing, metrics, I/O, visualization
- Sphinx documentation scaffold with ReadTheDocs configuration
- CLI entry points: `deepextractor-train`, `deepextractor-generate`, `deepextractor-specgen`, `deepextractor-evaluate`
- Pretrained checkpoint weights tracked via git-lfs
