cellPMVI Documentation
======================

**cellPMVI** is a multi-modal variational autoencoder for joint analysis of
single-cell RNA and surface protein (CITE-seq) data. Built on
`scvi-tools <https://scvi-tools.org>`_, it provides:

- **Separate encoders/decoders** per modality with a shared latent space
- **Flexible posterior fusion** via Product-of-Experts (PoE) or Mixture-of-Experts (MoE)
- **Multiple latent distributions** including Normal and Laplace priors
- **Negative binomial likelihoods** for count data (NB, ZINB, Poisson)
- **Batch correction** through conditional generation
- **Full scvi-tools compatibility** — standard ``setup_anndata`` / ``train`` / ``save`` / ``load`` workflow

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage
   fusion_strategies

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
