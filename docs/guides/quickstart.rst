Quickstart
==========

Pretrained model variants
-------------------------

Two pretrained variants of ``DeepExtractor_257`` are available, each paired with a
specific scaler:

.. list-table::
   :header-rows: 1

   * - Variant
     - Checkpoint constant
     - Scaler file
     - Use case
   * - Bilby noise
     - ``CHECKPOINT_BILBY``
     - ``assets/scaler_bilby.pkl``
     - Simulated LIGO/Virgo noise
   * - Real O3 noise
     - ``CHECKPOINT_REAL``
     - ``assets/scaler.pkl``
     - Real LIGO O3 detector data

Load a pretrained model and reconstruct
----------------------------------------

Weights are downloaded automatically from Hugging Face Hub on first use and cached locally.

.. code-block:: python

   import numpy as np
   import deepextractor

   # Load model (bilby noise variant by default)
   model = deepextractor.DeepExtractorModel()

   # Extract the transient signal from a noisy strain array
   noisy_strain = np.random.randn(8192)              # replace with real data
   reconstructed = model.reconstruct(noisy_strain)   # extracted signal
   background    = model.background(noisy_strain)    # noise estimate

   # One-liner convenience function (reloads model on each call)
   reconstructed = deepextractor.reconstruct(noisy_strain)

To use the real O3 noise variant:

.. code-block:: python

   from deepextractor.utils.checkpoints import CHECKPOINT_REAL

   model = deepextractor.DeepExtractorModel(
       checkpoint_filename=CHECKPOINT_REAL,
       scaler_path="assets/scaler.pkl",
   )

Notebooks
---------

* :doc:`../../notebooks/deepextractor_minimal` — minimal end-to-end example: simulate noise,
  inject a sine-Gaussian wavelet, and reconstruct with ``DeepExtractorModel``

* :doc:`../../notebooks/glitch_reconstruction_tutorial` — reconstruct real LIGO O3a glitches
  using the bundled GravitySpy sample dataset

Bundled dataset
---------------

The package ships a sample of the `GravitySpy LIGO O3a high-confidence catalogue
<https://doi.org/10.5281/zenodo.1476551>`_ at ``assets/data_o3a_sample.csv``.
It contains **10 high-SNR H1 examples per glitch class** (19 classes, 190 rows total)
and is used by the glitch reconstruction tutorial notebook.

.. code-block:: python

   import pandas as pd
   import importlib.resources as resources

   with resources.path("deepextractor", "assets") as assets:
       df = pd.read_csv(assets / "data_o3a_sample.csv")

   print(df["label"].value_counts())

CLI tools
---------

After installation, the following CLI entry points are available:

.. code-block:: bash

   # Train a model
   deepextractor-train --model DeepExtractor_257 --data-dir data/pycbc_noise/spectrogram_domain/

   # Generate training data
   deepextractor-generate --output-dir data/ --num-train 250000

   # Convert time-domain data to spectrograms
   deepextractor-specgen --input-dir data/pycbc_noise/time_domain/ --output-dir data/pycbc_noise/spectrogram_domain/

   # Evaluate a trained model
   deepextractor-evaluate --model DeepExtractor_257 --checkpoint-dir checkpoints/ --data-dir data/
