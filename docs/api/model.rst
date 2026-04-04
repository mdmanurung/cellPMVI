Model Classes
=============

High-level model classes that provide the user-facing API for training,
inference, and sampling. These follow the scvi-tools
``BaseModelClass`` pattern.

.. module:: cellpmvi.model

CellPMVI
--------

.. autoclass:: cellpmvi.model.CellPMVI
   :members: setup_anndata, get_latent_representation, posterior_predictive_sample, prior_predictive_sample, transfer_predictive_sample
   :show-inheritance:

CellPMVICiteseq
----------------

.. autoclass:: cellpmvi.model.CellPMVICiteseq
   :members: setup_anndata, posterior_predictive_sample, prior_predictive_sample, transfer_predictive_sample
   :show-inheritance:
