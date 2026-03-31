Installation
============

Requirements
------------

- Python ≥ 3.10
- PyTorch ≥ 2.1

Basic install
-------------

.. code-block:: bash

   pip install deepextractor

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/tomdooney/deepextractor.git
   cd deepextractor
   pip install -e ".[dev]"

Optional dependencies
---------------------

Generative glitch models (TensorFlow, gengli, bilby):

.. code-block:: bash

   pip install deepextractor[generative]

Documentation dependencies:

.. code-block:: bash

   pip install deepextractor[docs]

Pretrained weights
------------------

Pretrained model weights are hosted on `Hugging Face Hub <https://huggingface.co/tomdooney/deepextractor>`_
and are downloaded automatically on first use — no manual step required.
