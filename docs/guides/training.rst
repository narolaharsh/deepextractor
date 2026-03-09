Training
========

Data preparation
----------------

1. Generate time-domain data:

.. code-block:: bash

   deepextractor-generate --output-dir data/ --num-train 250000 --num-val 25000

   # Or with bilby noise:
   deepextractor-generate --output-dir data/ --num-train 250000 --bilby-noise

2. Convert to spectrograms (for 2D models):

.. code-block:: bash

   deepextractor-specgen \
       --input-dir data/pycbc_noise/time_domain/ \
       --output-dir data/pycbc_noise/spectrogram_domain/

Expected directory layout after data generation::

   data/
   └── pycbc_noise/
       ├── time_domain/
       │   ├── glitch_train_scaled_pycbc.npy
       │   ├── background_train_scaled_pycbc.npy
       │   ├── glitch_val_scaled_pycbc.npy
       │   └── background_val_scaled_pycbc.npy
       └── spectrogram_domain/
           ├── glitch_train_scaled_mag_phase.npy
           ├── background_train_scaled_mag_phase.npy
           ├── glitch_val_scaled_mag_phase.npy
           └── background_val_scaled_mag_phase.npy

Training a model
----------------

.. code-block:: bash

   deepextractor-train \
       --model DeepExtractor_257 \
       --data-dir data/pycbc_noise/spectrogram_domain/ \
       --checkpoint-dir checkpoints/ \
       --batch-size 32 \
       --epochs 150

Available models
----------------

.. list-table::
   :header-rows: 1

   * - Model name
     - Architecture
     - Domain
   * - ``DeepExtractor_257``
     - UNET2D (257×257 spectrograms)
     - Spectrogram
   * - ``DeepExtractor_129``
     - UNET2D (129×129 spectrograms)
     - Spectrogram
   * - ``UNET1D``
     - 1D U-Net
     - Time-domain
   * - ``DnCNN1D``
     - 1D DnCNN
     - Time-domain
   * - ``Autoencoder1D``
     - 1D Autoencoder
     - Time-domain

Hyperparameter options
----------------------

Run ``deepextractor-train --help`` for the full list of arguments.
