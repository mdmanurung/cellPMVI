"""Tests for cellPMVI neural network modules."""

import torch
import pytest

from cellpmvi.module._encoder import CellPMVIEncoder, MultiModalEncoder
from cellpmvi.module._fusion import ProductOfExperts, MixtureOfExperts
from cellpmvi.module._vae import CellPMVIVAE


class TestCellPMVIEncoder:
    """Tests for the custom encoder."""

    def test_normal_forward(self):
        """Test normal distribution encoder."""
        enc = CellPMVIEncoder(100, 10, distribution="normal", return_dist=False)
        x = torch.randn(32, 100)
        q_m, q_v, z = enc(x)
        assert q_m.shape == (32, 10)
        assert q_v.shape == (32, 10)
        assert z.shape == (32, 10)
        assert (q_v > 0).all()

    def test_laplace_forward(self):
        """Test Laplace distribution encoder."""
        enc = CellPMVIEncoder(100, 10, distribution="lp", return_dist=False)
        x = torch.randn(32, 100)
        q_m, q_v, z = enc(x)
        assert q_m.shape == (32, 10)
        assert q_v.shape == (32, 10)
        assert z.shape == (32, 10)
        # Laplace scale must be positive (softplus)
        assert (q_v > 0).all()

    def test_return_dist(self):
        """Test return_dist=True mode."""
        enc = CellPMVIEncoder(100, 10, distribution="normal", return_dist=True)
        x = torch.randn(32, 100)
        dist, z = enc(x)
        assert z.shape == (32, 10)
        assert hasattr(dist, "rsample")


class TestMultiModalEncoder:
    """Tests for the multi-modal encoder."""

    def test_forward(self):
        """Test multi-modal encoder forward pass."""
        enc = MultiModalEncoder(100, 20, 10, n_layers=1, n_hidden=32)
        genes = torch.randn(16, 100)
        proteins = torch.randn(16, 20)
        qz_m, qz_v, ql_m, ql_v, latent, untran = enc(genes, proteins)

        assert "gene" in qz_m and "protein" in qz_m
        assert qz_m["gene"].shape == (16, 10)
        assert qz_m["protein"].shape == (16, 10)
        assert ql_m.shape == (16, 1)
        assert latent["z_gene"].shape == (16, 10)
        assert latent["l"].shape == (16, 1)


class TestFusion:
    """Tests for fusion modules."""

    def test_poe(self):
        """Test Product-of-Experts."""
        poe = ProductOfExperts()
        means = [torch.randn(32, 10), torch.randn(32, 10)]
        variances = [torch.ones(32, 10), torch.ones(32, 10) * 2]
        joint_mean, joint_var = poe(means, variances)
        assert joint_mean.shape == (32, 10)
        assert joint_var.shape == (32, 10)
        # PoE should produce smaller variance than any input
        assert (joint_var < variances[0]).all()
        assert (joint_var < variances[1]).all()

    def test_moe(self):
        """Test Mixture-of-Experts."""
        moe = MixtureOfExperts()
        means = [torch.zeros(32, 10), torch.ones(32, 10) * 2]
        variances = [torch.ones(32, 10), torch.ones(32, 10)]
        joint_mean, joint_var = moe(means, variances)
        assert joint_mean.shape == (32, 10)
        assert joint_var.shape == (32, 10)
        # Mean should be between the two input means
        assert (joint_mean >= 0).all()
        assert (joint_mean <= 2).all()


class TestCellPMVIVAE:
    """Tests for the standalone VAE module."""

    def test_init(self):
        """Test VAE initialization."""
        vae = CellPMVIVAE(n_input=100, n_batch=2, n_latent=10)
        assert vae.n_latent == 10

    def test_inference_shape(self):
        """Test inference output shapes."""
        vae = CellPMVIVAE(n_input=100, n_batch=2, n_latent=10)
        x = torch.randn(16, 100).abs()
        batch = torch.zeros(16, 1, dtype=torch.long)
        out = vae.inference(x, batch)
        assert out["z"].shape == (16, 10)
        assert out["qz_m"].shape == (16, 10)
        assert out["library"].shape == (16, 1)

    def test_generative_shape(self):
        """Test generative output shapes."""
        vae = CellPMVIVAE(n_input=100, n_batch=2, n_latent=10)
        z = torch.randn(16, 10)
        lib = torch.ones(16, 1) * 7
        batch = torch.zeros(16, 1, dtype=torch.long)
        out = vae.generative(z, lib, batch)
        assert out["px_rate"].shape == (16, 100)
        assert out["px_r"].shape == (100,)
