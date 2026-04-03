"""Tests for data preprocessing."""

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData

from cellpmvi.data import preprocess_anndata
from cellpmvi.utils import compute_reconstruction_metrics, compute_latent_metrics


class TestPreprocessing:
    """Tests for the preprocessing pipeline."""

    def test_basic_preprocessing(self):
        """Test basic preprocessing pipeline."""
        np.random.seed(0)
        adata = AnnData(
            X=np.random.negative_binomial(5, 0.3, size=(100, 500)).astype(np.float32)
        )
        adata.layers["counts"] = adata.X.copy()

        result = preprocess_anndata(adata, n_top_genes=50, counts_layer="counts")
        assert result.n_vars <= 50
        assert result is not adata  # copy=True by default

    def test_preprocessing_no_copy(self):
        """Test in-place preprocessing."""
        np.random.seed(0)
        adata = AnnData(
            X=np.random.negative_binomial(5, 0.3, size=(100, 500)).astype(np.float32)
        )
        adata.layers["counts"] = adata.X.copy()
        result = preprocess_anndata(adata, n_top_genes=50, copy=False, counts_layer="counts")
        assert result is adata


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_reconstruction_metrics(self):
        """Test reconstruction metrics computation."""
        np.random.seed(0)
        observed = np.random.randn(100, 50)
        # Predicted is observed + noise
        predicted = observed + np.random.randn(100, 50) * 0.1

        metrics = compute_reconstruction_metrics(observed, predicted)
        assert "spearman_mean" in metrics
        assert "spearman_var" in metrics
        assert "rmse" in metrics
        assert metrics["spearman_mean"] > 0.9  # Should be highly correlated
        assert metrics["rmse"] < 0.5

    def test_latent_metrics(self):
        """Test latent space metrics computation."""
        np.random.seed(0)
        # Create well-separated clusters
        latent = np.vstack([
            np.random.randn(50, 10) + 5,
            np.random.randn(50, 10) - 5,
        ])
        labels = np.array(["A"] * 50 + ["B"] * 50)

        metrics = compute_latent_metrics(latent, labels)
        assert "silhouette_labels" in metrics
        assert metrics["silhouette_labels"] > 0.5  # Well-separated clusters
