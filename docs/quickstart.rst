Quickstart
==========

This guide walks through a minimal cellPMVI workflow on CITE-seq data.

Loading data
------------

cellPMVI expects an ``AnnData`` object with:

- **RNA counts** in ``.X`` or a specified layer
- **Protein expression** in ``.obsm`` as a DataFrame

.. code-block:: python

   import scvi

   # Load example CITE-seq PBMC dataset
   adata = scvi.data.pbmcs_10x_cite_seq(save_path="data/")
   adata.obs_names_make_unique()

Preprocessing
-------------

Select highly variable genes while keeping raw counts for the model:

.. code-block:: python

   import scanpy as sc

   sc.pp.filter_genes(adata, min_counts=3)
   adata.layers["counts"] = adata.X.copy()
   sc.pp.normalize_total(adata)
   sc.pp.log1p(adata)
   sc.pp.highly_variable_genes(
       adata, n_top_genes=4000, flavor="seurat_v3", layer="counts"
   )
   adata = adata[:, adata.var.highly_variable].copy()
   adata.X = adata.layers["counts"].copy()

Setting up the model
--------------------

Register the data fields, then create and train the model:

.. code-block:: python

   from cellpmvi import CellPMVI

   CellPMVI.setup_anndata(
       adata,
       layer="counts",
       protein_expression_obsm_key="protein_expression",
       batch_key="batch",
   )

   model = CellPMVI(
       adata,
       n_hidden=128,
       n_latent=20,
       n_layers=2,
       gene_likelihood="nb",
       fusion_method="poe",
   )

Training
--------

.. code-block:: python

   model.train(max_epochs=200, batch_size=256)

Extracting the latent representation
-------------------------------------

.. code-block:: python

   import scanpy as sc

   latent = model.get_latent_representation()
   adata.obsm["X_cellpmvi"] = latent

   # Visualize with UMAP
   sc.pp.neighbors(adata, use_rep="X_cellpmvi")
   sc.tl.umap(adata)
   sc.pl.umap(adata, color=["batch", "cell_type"])

Posterior predictive sampling
-----------------------------

Generate reconstructed expression from the learned model:

.. code-block:: python

   rna_samples, protein_samples = model.posterior_predictive_sample(n_samples=1)

Saving and loading
------------------

.. code-block:: python

   model.save("my_model/", save_anndata=True)
   loaded_model = CellPMVI.load("my_model/")
