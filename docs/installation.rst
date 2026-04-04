Installation
============

Requirements
------------

cellPMVI requires Python 3.10 or later and depends on:

- `scvi-tools <https://scvi-tools.org>`_ >= 1.1
- `anndata <https://anndata.readthedocs.io>`_ >= 0.10
- `scanpy <https://scanpy.readthedocs.io>`_ >= 1.9
- `PyTorch <https://pytorch.org>`_ >= 2.0

Install from source
-------------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/mdmanurung/cellPMVI.git
   cd cellPMVI
   pip install -e .

This installs the package in development mode, so changes to the source code
are immediately reflected.

Install with development dependencies
--------------------------------------

To install with testing and linting tools:

.. code-block:: bash

   pip install -e ".[dev]"

Verify installation
-------------------

.. code-block:: python

   import cellpmvi
   print(cellpmvi.__version__)
   # 0.1.0

   from cellpmvi import CellPMVI
   print("cellPMVI imported successfully")

GPU support
-----------

cellPMVI uses PyTorch for computation. To use GPU acceleration, install
PyTorch with CUDA support following the
`PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

The model will automatically use GPU if available. You can control device
placement through scvi-tools' training arguments:

.. code-block:: python

   model.train(max_epochs=200, accelerator="gpu", devices=1)
