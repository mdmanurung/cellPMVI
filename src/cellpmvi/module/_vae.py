"""Standalone VAE module for cellPMVI (not subclassing scvi.module.VAE).

This avoids fragile coupling to scvi-tools internal API changes while
following the same BaseModuleClass interface pattern.
"""

from __future__ import annotations

from typing import Callable, Iterable, Literal, Optional

import numpy as np
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, FCLayers

from cellpmvi.module._encoder import CellPMVIEncoder


class CellPMVIVAE(BaseModuleClass):
    """VAE with support for Normal, Logistic-Normal, and Laplace latent distributions.

    A standalone VAE implementation following scvi-tools' BaseModuleClass
    interface (inference/generative/loss pattern).

    Parameters
    ----------
    n_input
        Number of input features.
    n_batch
        Number of batches.
    n_hidden
        Hidden layer width.
    n_latent
        Latent space dimensionality.
    n_layers
        Number of hidden layers.
    n_continuous_cov
        Number of continuous covariates.
    n_cats_per_cov
        Number of categories per categorical covariate.
    dropout_rate
        Dropout rate.
    dispersion
        Gene dispersion mode.
    log_variational
        Whether to log-transform input for the encoder.
    gene_likelihood
        Reconstruction likelihood.
    latent_distribution
        Latent distribution type: ``"normal"``, ``"ln"``, or ``"lp"``.
    encode_covariates
        Whether to pass covariates to the encoder.
    deeply_inject_covariates
        Whether to inject covariates at every hidden layer.
    use_batch_norm
        Where to apply batch normalization.
    use_layer_norm
        Where to apply layer normalization.
    use_observed_lib_size
        Whether to use observed library size.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
        latent_distribution: str = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_observed_lib_size: bool = True,
    ):
        super().__init__()

        self.n_input = n_input
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size

        # Dispersion parameters
        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, 1))
        else:
            raise ValueError(f"Unknown dispersion: {dispersion}")

        use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        use_layer_norm_decoder = use_layer_norm in ("decoder", "both")

        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None

        self.z_encoder = CellPMVIEncoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=False,
        )
        self.l_encoder = CellPMVIEncoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=False,
        )

        n_input_decoder = n_latent + n_continuous_cov
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def _get_inference_input(self, tensors: dict) -> dict:
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None)
        return dict(x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs)

    def _get_generative_input(self, tensors: dict, inference_outputs: dict) -> dict:
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None)
        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)
        if size_factor is not None:
            size_factor = torch.log(size_factor)
        return dict(
            z=z, library=library, batch_index=batch_index, y=y,
            cont_covs=cont_covs, cat_covs=cat_covs, size_factor=size_factor,
        )

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict:
        """Run the encoder to get latent representation."""
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log1p(x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        qz_m, qz_v, z = self.z_encoder(encoder_input, batch_index, *categorical_input)

        if not self.use_observed_lib_size:
            ql_m, ql_v, library = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
        else:
            ql_m, ql_v = None, None

        if n_samples > 1:
            qz_m_exp = qz_m.unsqueeze(0).expand(n_samples, -1, -1)
            qz_v_exp = qz_v.unsqueeze(0).expand(n_samples, -1, -1)
            z = Normal(qz_m_exp, qz_v_exp.sqrt()).rsample()
            z = self.z_encoder.z_transformation(z)
            library = library.unsqueeze(0).expand(n_samples, -1, -1)

        return dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library)

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> dict:
        """Run the decoder to get reconstruction parameters."""
        if cont_covs is not None:
            decoder_input = torch.cat([z, cont_covs], dim=-1)
        else:
            decoder_input = z

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if size_factor is None:
            size_factor = library

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, decoder_input, size_factor, batch_index, *categorical_input, y
        )

        if self.dispersion == "gene":
            px_r = self.px_r
        elif self.dispersion == "gene-batch":
            px_r = torch.nn.functional.linear(
                torch.nn.functional.one_hot(
                    batch_index.squeeze(-1).long(), self.n_batch
                ).float(),
                self.px_r,
            )
        px_r = torch.exp(px_r)

        return dict(px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout)

    def loss(
        self,
        tensors: dict,
        inference_outputs: dict,
        generative_outputs: dict,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Compute the VAE ELBO loss."""
        x = tensors[REGISTRY_KEYS.X_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        # KL divergence
        kl_divergence_z = kl(
            Normal(qz_m, qz_v.sqrt()), Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        ).sum(dim=-1)

        kl_divergence_l = torch.tensor(0.0, device=x.device)
        if not self.use_observed_lib_size and inference_outputs["ql_m"] is not None:
            ql_m = inference_outputs["ql_m"]
            ql_v = inference_outputs["ql_v"]
            kl_divergence_l = kl(
                Normal(ql_m, ql_v.sqrt()), Normal(torch.zeros_like(ql_m), torch.ones_like(ql_v))
            ).sum(dim=1)

        # Reconstruction loss
        reconst_loss = self._reconstruction_loss(x, px_rate, px_r, px_dropout)

        weighted_kl = kl_weight * kl_divergence_z + kl_divergence_l
        loss = torch.mean(reconst_loss + weighted_kl)

        kl_local = dict(kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z)
        return LossOutput(loss, reconst_loss, kl_local, torch.tensor(0.0))

    def _reconstruction_loss(
        self,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood of observed data."""
        if self.gene_likelihood == "nb":
            reconst_loss = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(-1)
        elif self.gene_likelihood == "zinb":
            reconst_loss = -ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout
            ).log_prob(x).sum(-1)
        elif self.gene_likelihood == "poisson":
            reconst_loss = -torch.distributions.Poisson(px_rate).log_prob(x).sum(-1)
        else:
            raise ValueError(f"Unknown gene_likelihood: {self.gene_likelihood}")
        return reconst_loss
