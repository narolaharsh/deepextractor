Overview
========

DeepExtractor is a deep learning framework for reconstructing transient noise artefacts
(*glitches*) and gravitational-wave signals in LIGO detector strain data.

The key idea
------------

LIGO strain data contains both astrophysical signals and instrumental glitches — short-duration
noise transients that can mimic or obscure real gravitational-wave events. DeepExtractor addresses
this by framing glitch reconstruction as a **supervised denoising problem**:

Given a 2-second stretch of whitened strain containing a glitch, the model learns to output
a clean background estimate. Subtracting that background from the input recovers the glitch
(or signal) waveform.

.. code-block:: text

    Input:  h(t)  = background noise  +  glitch/signal
    Output: n̂(t)  = predicted background
    Result: ĝ(t)  = h(t) − n̂(t)   ← reconstructed glitch or signal

Architecture
------------

The model is a **U-Net** operating on STFT spectrograms (magnitude + phase). The input strain
is transformed into a 2-channel time-frequency representation, processed by the U-Net encoder-
decoder, then converted back to the time domain via iSTFT. The default model,
``DeepExtractor_257``, uses a 4-level U-Net with feature maps ``[64, 128, 256, 512]`` and
produces 257×257 spectrograms.

Pretrained models
-----------------

Two pretrained variants of ``DeepExtractor_257`` are provided, both fine-tuned on LIGO O3 data:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Variant
     - Trained on
     - Best for
   * - ``bilby_noise``
     - Simulated LIGO/Virgo noise (bilby)
     - Simulated data, injection studies
   * - ``real_noise``
     - Real LIGO O3 strain
     - Real LIGO O3 detector data

Weights are downloaded automatically from Hugging Face Hub on first use.

Citation
--------

If you use DeepExtractor in your work, please cite:

.. code-block:: bibtex

   @article{dooney2025deepextractor,
     title   = {DeepExtractor: ...},
     author  = {Dooney, Tom and others},
     journal = {arXiv preprint arXiv:2501.18423},
     year    = {2025},
     url     = {https://arxiv.org/abs/2501.18423}
   }
