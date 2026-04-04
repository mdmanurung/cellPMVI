Module Classes
==============

Neural network modules implementing the VAE architectures. These follow
the scvi-tools ``BaseModuleClass`` interface with ``inference``,
``generative``, and ``loss`` methods.

.. module:: cellpmvi.module

VAE Modules
-----------

CellPMVIVAE
~~~~~~~~~~~~

.. autoclass:: cellpmvi.module.CellPMVIVAE
   :members:
   :show-inheritance:

CellPMVAE
~~~~~~~~~~

.. autoclass:: cellpmvi.module.CellPMVAE
   :members:
   :show-inheritance:

CellPMVAECiteseq
~~~~~~~~~~~~~~~~~

.. autoclass:: cellpmvi.module.CellPMVAECiteseq
   :members:
   :show-inheritance:

ProteinVAE
~~~~~~~~~~

.. autoclass:: cellpmvi.module.ProteinVAE
   :members:
   :show-inheritance:

Encoders
--------

CellPMVIEncoder
~~~~~~~~~~~~~~~~

.. autoclass:: cellpmvi.module.CellPMVIEncoder
   :members:
   :show-inheritance:

MultiModalEncoder
~~~~~~~~~~~~~~~~~

.. autoclass:: cellpmvi.module.MultiModalEncoder
   :members:
   :show-inheritance:

Fusion
------

ProductOfExperts
~~~~~~~~~~~~~~~~~

.. autoclass:: cellpmvi.module.ProductOfExperts
   :members:
   :show-inheritance:

MixtureOfExperts
~~~~~~~~~~~~~~~~~

.. autoclass:: cellpmvi.module.MixtureOfExperts
   :members:
   :show-inheritance:
