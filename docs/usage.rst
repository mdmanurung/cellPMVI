Usage Guide
===========

This guide covers the main features and usage patterns of cellPMVI in detail.

Data requirements
-----------------

cellPMVI requires an ``AnnData`` object with the following structure:

.. list-table::
   :header-rows: 1

   * - Field
     - Location
     - Description
   * - RNA counts
     - ``adata.X`` or ``adata.layers["counts"]``
     - Raw (unnormalized) count matrix
   * - Protein expression
     - ``adata.obsm["protein_expression"]``
     - Protein count matrix as a DataFrame
   * - Batch labels
     - ``adata.obs["batch"]``
     - (Optional) Categorical batch variable
   * - Categorical covariates
     - ``adata.obs[key]``
     - (Optional) Additional covariates

.. important::

   Both RNA and protein data should be **raw counts** (not normalized or
   log-transformed). cellPMVI models the data using negative binomial
   likelihoods internally.

Registering data
----------------

Before creating a model, register the relevant data fields:

.. code-block:: python

   from cellpmvi import CellPMVI

   CellPMVI.setup_anndata(
       adata,
       layer="counts",                              # RNA counts layer
       protein_expression_obsm_key="protein_expression",  # Protein key
       batch_key="batch",                            # Batch variable
       categorical_covariate_keys=["donor"],          # Extra covariates
   )

This follows the standard scvi-tools data registration pattern and validates
that all required fields are present.

Model architectures
-------------------

cellPMVI offers two model architectures:

**CellPMVI** (default)
   Separate encoder/decoder pairs for RNA and protein with a shared latent
   space. Supports PoE/MoE posterior fusion and cross-modal reconstruction.

   .. code-block:: python

      from cellpmvi import CellPMVI

      model = CellPMVI(
          adata,
          n_hidden=128,
          n_latent=20,
          n_layers=2,
          gene_likelihood="nb",
          fusion_method="poe",
      )

**CellPMVICiteseq**
   Uses a shared TotalVI-style decoder that jointly models gene expression
   and protein abundance. Suitable when you want tighter coupling between
   modalities.

   .. code-block:: python

      from cellpmvi import CellPMVICiteseq

      model = CellPMVICiteseq(
          adata,
          n_hidden=256,
          n_latent=20,
          n_layers_encoder=2,
          n_layers_decoder=1,
          gene_likelihood="nb",
      )

Training
--------

Training uses PyTorch Lightning under the hood:

.. code-block:: python

   model.train(
       max_epochs=200,
       batch_size=256,
       early_stopping=True,
       early_stopping_patience=15,
   )

   # Access training history
   model.history["train_loss"].plot()

Key training parameters:

- ``max_epochs``: Maximum training epochs (default varies)
- ``batch_size``: Minibatch size (default 128)
- ``early_stopping``: Enable early stopping (default False)
- ``train_size``: Fraction of data for training (default 0.9)

Latent representation
---------------------

Extract the learned latent representation:

.. code-block:: python

   # Posterior mean (deterministic)
   latent = model.get_latent_representation(give_mean=True)

   # Posterior sample (stochastic)
   latent_sample = model.get_latent_representation(give_mean=False)

When using PoE or MoE fusion, ``get_latent_representation`` returns the
**joint** posterior (fused across modalities). With ``fusion_method="none"``,
it returns the RNA encoder's posterior.

Sampling
--------

**Posterior predictive sampling** — reconstruct observed cells:

.. code-block:: python

   rna_recon, protein_recon = model.posterior_predictive_sample(n_samples=1)

**Prior predictive sampling** — generate new synthetic cells:

.. code-block:: python

   rna_new, protein_new = model.prior_predictive_sample(n_samples=100)

**Transfer predictive sampling** — counterfactual generation with different
covariates:

.. code-block:: python

   rna_transfer, protein_transfer = model.transfer_predictive_sample(
       adata=adata,
       cat_covs=[1],  # Target batch index
   )

Evaluation metrics
------------------

cellPMVI provides built-in evaluation utilities:

.. code-block:: python

   from cellpmvi.utils import (
       compute_reconstruction_metrics,
       compute_latent_metrics,
   )

   # Reconstruction quality
   rna_metrics = compute_reconstruction_metrics(
       observed=adata.layers["counts"],
       predicted=rna_recon,
   )
   # Returns: {"spearman_mean", "spearman_var", "rmse"}

   # Latent space quality
   latent_metrics = compute_latent_metrics(
       latent=latent,
       labels=adata.obs["cell_type"].values,
       batch=adata.obs["batch"].values,
   )
   # Returns: {"silhouette_labels", "silhouette_batch"}

Preprocessing
-------------

A convenience preprocessing function is available:

.. code-block:: python

   from cellpmvi.data import preprocess_anndata

   adata = preprocess_anndata(
       adata,
       n_top_genes=4000,
       min_counts=3,
       flavor="seurat_v3",
       counts_layer="counts",
   )

This applies normalization, log-transformation, gene filtering, and HVG
selection, then restores raw counts in ``.X`` for model input.

Saving and loading
------------------

Models can be saved and loaded following the scvi-tools pattern:

.. code-block:: python

   # Save model with data
   model.save("my_model/", save_anndata=True)

   # Load model
   loaded_model = CellPMVI.load("my_model/")

   # Load with new data (same features)
   loaded_model = CellPMVI.load("my_model/", adata=new_adata)

Latent distributions
--------------------

cellPMVI supports two latent prior distributions:

**Normal** (default)
   Standard Gaussian prior with KL divergence in closed form.

   .. code-block:: python

      model = CellPMVI(adata, latent_distribution="normal")

**Laplace**
   Heavier-tailed prior that can capture sparser latent representations.
   Uses softplus parameterization for the scale parameter.

   .. code-block:: python

      model = CellPMVI(adata, latent_distribution="lp")
