"""Multi-modal VAE module integrating RNA and protein data."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch.distributions import Normal

from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data

from cellpmvi.module._fusion import MixtureOfExperts, ProductOfExperts
from cellpmvi.module._protein_vae import ProteinVAE
from cellpmvi.module._vae import CellPMVIVAE


class CellPMVAE(BaseModuleClass):
    """Multi-modal VAE integrating RNA and protein modalities.

    Uses separate encoder/decoder pairs for each modality with optional
    posterior fusion via Product-of-Experts or Mixture-of-Experts.

    Parameters
    ----------
    n_input_rna
        Number of RNA features.
    n_input_pro
        Number of protein features.
    n_batch
        Number of batches.
    n_hidden
        Hidden layer width.
    n_latent
        Latent space dimensionality.
    dropout_rate
        Dropout rate.
    gene_likelihood
        Reconstruction likelihood: ``"nb"``, ``"zinb"``, or ``"poisson"``.
    n_layers
        Number of hidden layers.
    latent_distribution
        Latent distribution: ``"normal"`` or ``"lp"`` (Laplace).
    fusion_method
        Multi-modal posterior fusion: ``"none"`` (cross-modal matrix),
        ``"poe"`` (product-of-experts), or ``"moe"`` (mixture-of-experts).
    """

    def __init__(
        self,
        n_input_rna: int,
        n_input_pro: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["nb", "zinb", "poisson"] = "nb",
        n_layers: int = 1,
        latent_distribution: str = "normal",
        fusion_method: Literal["none", "poe", "moe"] = "poe",
        **model_kwargs,
    ):
        super().__init__()

        self.gene_likelihood = gene_likelihood
        self.n_latent = n_latent
        self.latent_distribution = latent_distribution
        self.fusion_method = fusion_method

        self.rna_vae = CellPMVIVAE(
            n_input=n_input_rna,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            gene_likelihood=gene_likelihood,
            dropout_rate=dropout_rate,
            n_layers=n_layers,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )

        self.protein_vae = ProteinVAE(
            n_input=n_input_pro,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            gene_likelihood=gene_likelihood,
            dropout_rate=dropout_rate,
            n_layers=n_layers,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )

        self.vaes = [self.rna_vae, self.protein_vae]

        # Fusion module
        if fusion_method == "poe":
            self.fusion = ProductOfExperts()
        elif fusion_method == "moe":
            self.fusion = MixtureOfExperts()
        else:
            self.fusion = None

    def _get_inference_input(self, tensors: dict) -> dict:
        rna_input = self.rna_vae._get_inference_input(tensors)
        protein_input = self.protein_vae._get_inference_input(tensors)

        return dict(
            rna_x=rna_input["x"],
            pro_x=protein_input["x"],
            batch_index=rna_input["batch_index"],
            cont_covs=rna_input["cont_covs"],
            cat_covs=rna_input["cat_covs"],
        )

    def _get_generative_input(self, tensors: dict, inference_outputs) -> dict:
        rna_input = self.rna_vae._get_generative_input(tensors, inference_outputs[0])
        protein_input = self.protein_vae._get_generative_input(
            tensors, inference_outputs[1]
        )

        return dict(
            rna_z=rna_input["z"],
            rna_library=rna_input["library"],
            rna_y=rna_input["y"],
            rna_size_factor=rna_input["size_factor"],
            pro_z=protein_input["z"],
            pro_library=protein_input["library"],
            pro_y=protein_input["y"],
            pro_size_factor=protein_input["size_factor"],
            batch_index=rna_input["batch_index"],
            cont_covs=rna_input["cont_covs"],
            cat_covs=rna_input["cat_covs"],
        )

    @auto_move_data
    def inference(
        self,
        rna_x: torch.Tensor,
        pro_x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None,
        cat_covs: torch.Tensor | None,
        n_samples: int = 1,
    ) -> list[dict]:
        """Run inference (encoding) for both modalities.

        Parameters
        ----------
        rna_x
            RNA expression tensor.
        pro_x
            Protein expression tensor.
        batch_index
            Batch indices.
        cont_covs
            Continuous covariates.
        cat_covs
            Categorical covariates.
        n_samples
            Number of posterior samples.

        Returns
        -------
        list of dict
            ``[rna_outputs, protein_outputs]`` each containing
            ``z``, ``qz_m``, ``qz_v``, ``library``, etc.
        """
        rna_outputs = self.rna_vae.inference(
            rna_x, batch_index, cont_covs, cat_covs, n_samples=n_samples
        )
        protein_outputs = self.protein_vae.inference(
            pro_x, batch_index, cont_covs, cat_covs, n_samples=n_samples
        )

        # Apply posterior fusion if configured
        if self.fusion is not None:
            rna_mean = rna_outputs["qz_m"]
            rna_var = rna_outputs["qz_v"]
            pro_mean = protein_outputs["qz_m"]
            pro_var = protein_outputs["qz_v"]

            joint_mean, joint_var = self.fusion(
                means=[rna_mean, pro_mean],
                variances=[rna_var, pro_var],
            )

            # Sample from joint posterior
            joint_z = Normal(joint_mean, joint_var.sqrt()).rsample()

            # Override both modalities' z with joint z
            rna_outputs["z"] = joint_z
            protein_outputs["z"] = joint_z
            # Store joint posterior params
            rna_outputs["qz_m_joint"] = joint_mean
            rna_outputs["qz_v_joint"] = joint_var
            protein_outputs["qz_m_joint"] = joint_mean
            protein_outputs["qz_v_joint"] = joint_var

        return [rna_outputs, protein_outputs]

    @auto_move_data
    def generative(
        self,
        rna_z: torch.Tensor,
        pro_z: torch.Tensor,
        rna_library: torch.Tensor,
        pro_library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        rna_size_factor: torch.Tensor | None = None,
        pro_size_factor: torch.Tensor | None = None,
        rna_y: torch.Tensor | None = None,
        pro_y: torch.Tensor | None = None,
        transform_batch: int | None = None,
    ) -> list[list[dict]]:
        """Run the generative model (decoding) for all modality combinations.

        Returns a 2x2 cross-modal reconstruction matrix.

        Returns
        -------
        list of list of dict
            ``px_zs[encoder_modality][decoder_modality]`` containing
            reconstruction parameters.
        """
        # RNA with own z
        rna_out_own = self.rna_vae.generative(
            rna_z, rna_library, batch_index, cont_covs, cat_covs,
            rna_size_factor, rna_y,
        )
        # RNA with protein z
        rna_out_other = self.rna_vae.generative(
            pro_z, rna_library, batch_index, cont_covs, cat_covs,
            rna_size_factor, rna_y,
        )
        # Protein with own z
        protein_out_own = self.protein_vae.generative(
            pro_z, pro_library, batch_index, cont_covs, cat_covs,
            pro_size_factor, pro_y,
        )
        # Protein with RNA z
        protein_out_other = self.protein_vae.generative(
            rna_z, pro_library, batch_index, cont_covs, cat_covs,
            pro_size_factor, pro_y,
        )

        # Cross-modal matrix: [encoder_modality][decoder_modality]
        px_zs = [
            [rna_out_own, rna_out_other],
            [protein_out_other, protein_out_own],
        ]

        return px_zs

    def loss(
        self,
        tensors: dict,
        inference_outputs: list[dict],
        generative_outputs: list[list[dict]],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Compute the multi-modal ELBO loss.

        Uses the naive ELBO: average over all modality pairs of
        (reconstruction_loss + KL_divergence).

        Parameters
        ----------
        tensors
            Data tensors.
        inference_outputs
            Per-modality encoder outputs.
        generative_outputs
            Cross-modal reconstruction matrix.
        kl_weight
            KL divergence weight for annealing.

        Returns
        -------
        LossOutput
        """
        per_cell_losses = []
        for r, vae in enumerate(self.vaes):
            for d, gen_out in enumerate(generative_outputs[r]):
                # Compute per-cell loss directly
                if isinstance(vae, ProteinVAE):
                    x = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
                else:
                    x = tensors[REGISTRY_KEYS.X_KEY]

                inf_out = inference_outputs[r]
                qz_m = inf_out["qz_m"]
                qz_v = inf_out["qz_v"]

                px_rate = gen_out["px_rate"]
                px_r = gen_out["px_r"]
                px_dropout = gen_out["px_dropout"]

                # KL divergence (per cell)
                from torch.distributions import kl_divergence as kl_fn
                kl_z = kl_fn(
                    Normal(qz_m, qz_v.sqrt()),
                    Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)),
                ).sum(dim=-1)

                # Reconstruction loss (per cell)
                reconst = vae._reconstruction_loss(x, px_rate, px_r, px_dropout)

                per_cell_losses.append(reconst + kl_weight * kl_z)

        # Average across all modality combinations, then mean over cells
        stacked = torch.stack(per_cell_losses, dim=0)  # (n_combos, n_cells)
        mean_per_cell = stacked.mean(dim=0) / len(self.vaes)  # (n_cells,)
        loss = mean_per_cell.mean()

        return LossOutput(
            loss=loss,
            reconstruction_loss=mean_per_cell,
            kl_local=torch.tensor(0.0),
            kl_global=torch.tensor(0.0),
        )

    @torch.no_grad()
    def sample(
        self,
        tensors: dict,
        n_samples: int = 1,
        library_size: int = 1,
    ) -> list[torch.Tensor]:
        """Sample from the posterior predictive distribution.

        Parameters
        ----------
        tensors
            Data tensors.
        n_samples
            Number of samples per cell.
        library_size
            Not used (kept for API compatibility).

        Returns
        -------
        list of torch.Tensor
            ``[rna_samples, protein_samples]``.
        """
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, px_zs = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        exprs_list = []
        for e, row in enumerate(px_zs):
            for d, generative_outputs in enumerate(row):
                if e == d:
                    px_r = generative_outputs["px_r"]
                    px_rate = generative_outputs["px_rate"]
                    px_dropout = generative_outputs["px_dropout"]

                    if self.gene_likelihood == "poisson":
                        l_train = torch.clamp(px_rate, max=1e8)
                        dist = torch.distributions.Poisson(l_train)
                    elif self.gene_likelihood == "nb":
                        dist = NegativeBinomial(mu=px_rate, theta=px_r)
                    elif self.gene_likelihood == "zinb":
                        dist = ZeroInflatedNegativeBinomial(
                            mu=px_rate, theta=px_r, zi_logits=px_dropout
                        )
                    else:
                        raise ValueError(
                            f"Unknown gene_likelihood: {self.gene_likelihood}"
                        )

                    if n_samples > 1:
                        exprs = dist.sample().permute([1, 2, 0])
                    else:
                        exprs = dist.sample()
                    exprs_list.append(exprs.cpu())

        return exprs_list
