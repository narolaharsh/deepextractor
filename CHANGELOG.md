# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-04-01

### Fixed
- Bundled assets (`scaler_bilby.pkl`, `scaler.pkl`, `data_o3a_sample.csv`) were missing from the pip wheel — moved into `src/deepextractor/assets/` and declared as package data
- `_default_scaler_path()` now uses `importlib.resources` instead of a relative path that broke after installation

## [0.1.0] - 2026-04-01

### Added
- Initial pip-installable package structure
- `deepextractor.models` — U-Net 1D/2D, DnCNN 1D, Autoencoder 1D/2D architectures
- `deepextractor.data` — PyTorch Dataset classes for time-series and spectrogram data
- `deepextractor.training` — training loop and trainer CLI entry point
- `deepextractor.generation` — synthetic glitch signal generators and data generation scripts
- `deepextractor.evaluation` — simulated evaluation metrics
- `deepextractor.utils` — checkpoints, signal processing, metrics, I/O, visualization
- Sphinx documentation with Furo dark theme
- CLI entry points: `deepextractor-train`, `deepextractor-generate`, `deepextractor-specgen`, `deepextractor-evaluate`
- Pretrained checkpoint weights hosted on Hugging Face Hub, downloaded automatically on first use
- Bundled GravitySpy O3a sample dataset (`assets/data_o3a_sample.csv`) — 170 high-confidence glitch examples across 17 classes
- Jupyter notebook tutorials: minimal example, glitch reconstruction, training from scratch
