"""Protein-specific VAE module that reads from the protein expression key."""

from __future__ import annotations

import torch

from scvi import REGISTRY_KEYS

from cellpmvi.module._vae import CellPMVIVAE


class ProteinVAE(CellPMVIVAE):
    """VAE for protein data.

    Identical to :class:`CellPMVIVAE` but reads input from
    ``REGISTRY_KEYS.PROTEIN_EXP_KEY`` instead of ``REGISTRY_KEYS.X_KEY``.
    """

    def _get_inference_input(self, tensors: dict) -> dict:
        x = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None)
        return dict(x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs)

    def loss(
        self,
        tensors: dict,
        inference_outputs: dict,
        generative_outputs: dict,
        kl_weight: float = 1.0,
    ):
        """Compute loss using protein expression as ground truth."""
        # Override X_KEY with PROTEIN_EXP_KEY for reconstruction loss
        modified_tensors = dict(tensors)
        modified_tensors[REGISTRY_KEYS.X_KEY] = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        return super().loss(modified_tensors, inference_outputs, generative_outputs, kl_weight)
