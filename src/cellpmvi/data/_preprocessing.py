"""Data preprocessing pipeline for cellPMVI."""

from __future__ import annotations

import logging

import numpy as np
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)


def preprocess_anndata(
    adata: AnnData,
    n_top_genes: int = 5000,
    min_counts: int = 3,
    flavor: str = "seurat_v3",
    counts_layer: str = "counts",
    subset_hvg: bool = True,
    copy: bool = True,
) -> AnnData:
    """Preprocess AnnData for cellPMVI.

    Applies standard single-cell preprocessing:
    1. Library-size normalization
    2. Log-transformation
    3. Gene filtering by minimum counts
    4. Highly-variable gene selection (on raw counts)

    Parameters
    ----------
    adata
        AnnData object with raw counts.
    n_top_genes
        Number of highly-variable genes to select.
    min_counts
        Minimum total counts for gene filtering.
    flavor
        HVG selection flavor (passed to ``scanpy.pp.highly_variable_genes``).
    counts_layer
        Layer containing raw counts. Used for HVG selection with ``seurat_v3``.
    subset_hvg
        Whether to subset to HVGs.
    copy
        Whether to return a copy.

    Returns
    -------
    AnnData
        Preprocessed AnnData with raw counts restored in ``.X``.
    """
    if copy:
        adata = adata.copy()

    # Store raw counts
    adata.raw = adata

    # Normalize and log-transform
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Filter low-count genes
    sc.pp.filter_genes(adata, min_counts=min_counts)

    # Select highly-variable genes
    hvg_kwargs = dict(
        n_top_genes=n_top_genes,
        subset=subset_hvg,
        flavor=flavor,
    )
    if flavor == "seurat_v3" and counts_layer in adata.layers:
        hvg_kwargs["layer"] = counts_layer
    sc.pp.highly_variable_genes(adata, **hvg_kwargs)

    logger.info(f"Selected {adata.n_vars} genes after HVG filtering.")

    # Restore raw counts for model input
    if adata.raw is not None:
        adata.X = adata.raw[:, adata.var_names].X

    return adata
