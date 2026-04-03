"""Tests for CellPMVI model class."""

import os
import tempfile

import numpy as np
import pytest

from cellpmvi import CellPMVI


class TestCellPMVI:
    """Tests for the high-level CellPMVI model."""

    def _setup_model(self, adata, **kwargs):
        """Helper to setup anndata and create model."""
        CellPMVI.setup_anndata(
            adata,
            layer="counts",
            protein_expression_obsm_key="protein",
            batch_key="batch",
        )
        defaults = dict(n_hidden=32, n_latent=10, n_layers=1)
        defaults.update(kwargs)
        return CellPMVI(adata, **defaults)

    def test_setup_anndata(self, synthetic_adata):
        """Test AnnData registration."""
        CellPMVI.setup_anndata(
            synthetic_adata,
            layer="counts",
            protein_expression_obsm_key="protein",
            batch_key="batch",
        )

    def test_model_init(self, synthetic_adata):
        """Test model initialization."""
        model = self._setup_model(synthetic_adata)
        assert model.module is not None
        assert model.n_latent == 10

    def test_train(self, synthetic_adata):
        """Test model training runs without error."""
        model = self._setup_model(synthetic_adata)
        model.train(max_epochs=2)

    def test_get_latent_representation(self, synthetic_adata):
        """Test latent representation extraction."""
        model = self._setup_model(synthetic_adata)
        model.train(max_epochs=1)
        latent = model.get_latent_representation()
        assert latent.shape == (200, 10)
        assert not np.isnan(latent).any()

    def test_posterior_predictive_sample(self, synthetic_adata):
        """Test posterior predictive sampling."""
        model = self._setup_model(synthetic_adata)
        model.train(max_epochs=1)
        samples = model.posterior_predictive_sample(n_samples=1)
        assert len(samples) == 2
        assert samples[0].shape[0] == 200  # RNA
        assert samples[1].shape[0] == 200  # Protein

    def test_prior_predictive_sample(self, synthetic_adata):
        """Test prior predictive sampling."""
        model = self._setup_model(synthetic_adata)
        model.train(max_epochs=1)
        samples = model.prior_predictive_sample(n_samples=5)
        assert len(samples) == 2
        assert samples[0].shape[0] == 5
        assert samples[1].shape[0] == 5

    def test_save_load(self, synthetic_adata):
        """Test model save and load."""
        model = self._setup_model(synthetic_adata)
        model.train(max_epochs=1)
        latent_before = model.get_latent_representation()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path, save_anndata=True)
            loaded = CellPMVI.load(save_path)
            latent_after = loaded.get_latent_representation()

        assert latent_before.shape == latent_after.shape
        np.testing.assert_allclose(latent_before, latent_after, atol=1e-5)

    def test_laplace_distribution(self, synthetic_adata):
        """Test training with Laplace latent distribution."""
        model = self._setup_model(synthetic_adata, latent_distribution="lp")
        model.train(max_epochs=1)
        latent = model.get_latent_representation()
        assert latent.shape == (200, 10)

    @pytest.mark.parametrize("fusion_method", ["poe", "moe", "none"])
    def test_fusion_methods(self, synthetic_adata, fusion_method):
        """Test all fusion methods train without error."""
        model = self._setup_model(synthetic_adata, fusion_method=fusion_method)
        model.train(max_epochs=1)
        latent = model.get_latent_representation()
        assert latent.shape == (200, 10)
