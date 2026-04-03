"""Shared test fixtures for cellPMVI."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


@pytest.fixture
def synthetic_adata():
    """Create synthetic AnnData with RNA + protein for testing."""
    np.random.seed(42)
    n_cells, n_genes, n_proteins = 200, 100, 20

    rna_counts = np.random.negative_binomial(
        5, 0.3, size=(n_cells, n_genes)
    ).astype(np.float32)
    protein_counts = np.random.negative_binomial(
        10, 0.5, size=(n_cells, n_proteins)
    ).astype(np.float32)

    adata = AnnData(X=rna_counts)
    adata.layers["counts"] = rna_counts.copy()
    adata.obsm["protein"] = pd.DataFrame(
        protein_counts,
        index=adata.obs_names,
        columns=[f"protein_{i}" for i in range(n_proteins)],
    )
    adata.obs["batch"] = np.random.choice(["batch_0", "batch_1"], n_cells)
    adata.obs["cell_type"] = np.random.choice(["A", "B", "C"], n_cells)

    return adata
