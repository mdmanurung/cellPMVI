"""High-level CellPMVI CITE-seq model class."""

from __future__ import annotations

import logging
from typing import Literal, Optional, Sequence

import numpy as np
import torch
from anndata import AnnData
from torch.distributions import Normal

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ProteinObsmField,
)
from scvi.distributions import NegativeBinomial, NegativeBinomialMixture
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin
from scvi.utils._docstrings import setup_anndata_dsp

from cellpmvi.module._cellpmvae_citeseq import CellPMVAECiteseq

logger = logging.getLogger(__name__)


class CellPMVICiteseq(UnsupervisedTrainingMixin, BaseModelClass):
    """Multi-modal VAE for CITE-seq data using TotalVI-style decoder.

    Uses separate per-modality encoders with a shared TotalVI decoder
    that jointly models gene expression and protein abundance.

    Parameters
    ----------
    adata
        AnnData object registered via :meth:`setup_anndata`.
    n_hidden
        Hidden layer width.
    n_latent
        Latent space dimensionality.
    n_layers_encoder
        Number of encoder hidden layers.
    n_layers_decoder
        Number of decoder hidden layers.
    dropout_rate
        Dropout rate.
    gene_likelihood
        Gene reconstruction likelihood.
    latent_distribution
        Latent distribution type.
    encode_covariates
        Whether to condition encoder on covariates.
    **model_kwargs
        Additional kwargs for :class:`CellPMVAECiteseq`.
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 1,
        dropout_rate: float = 0.2,
        gene_likelihood: Literal["nb", "zinb"] = "nb",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        **model_kwargs,
    ):
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        self.module = CellPMVAECiteseq(
            n_input_genes=self.summary_stats["n_vars"],
            n_input_proteins=self.summary_stats["n_proteins"],
            n_batch=self.summary_stats["n_batch"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            gene_likelihood=gene_likelihood,
            dropout_rate_decoder=dropout_rate,
            dropout_rate_encoder=dropout_rate,
            encode_covariates=encode_covariates,
            n_cats_per_cov=n_cats_per_cov,
            **model_kwargs,
        )

        self.n_latent = n_latent
        self._model_summary_string = (
            f"CellPMVI-CITEseq Model: n_latent={n_latent}, n_hidden={n_hidden}, "
            f"n_layers_encoder={n_layers_encoder}, n_layers_decoder={n_layers_decoder}, "
            f"gene_likelihood={gene_likelihood}"
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        protein_expression_obsm_key: str,
        protein_names_uns_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[list[str]] = None,
        continuous_covariate_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        """Register AnnData fields for CellPMVI-CITEseq.

        Parameters
        ----------
        %(param_adata)s
        protein_expression_obsm_key
            Key in ``adata.obsm`` for protein expression data.
        protein_names_uns_key
            Key in ``adata.uns`` for protein names.
        %(param_batch_key)s
        %(param_layer)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        batch_field = CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            batch_field,
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
            ProteinObsmField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_expression_obsm_key,
                use_batch_mask=True,
                batch_field=batch_field,
                colnames_uns_key=protein_names_uns_key,
                is_count_data=True,
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Get latent representation for cells.

        Parameters
        ----------
        adata
            AnnData to use. Defaults to training data.
        indices
            Cell indices. Defaults to all cells.
        give_mean
            If ``True``, return posterior mean; otherwise sample.
        batch_size
            Minibatch size.

        Returns
        -------
        np.ndarray
            Latent representation of shape ``(n_cells, n_latent)``.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size or 128
        )

        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            if give_mean:
                # Average gene and protein latent means
                z = (outputs["qz_m"]["gene"] + outputs["qz_m"]["protein"]) / 2
            else:
                z = outputs["z"]["gene"]
            latent.append(z.cpu().numpy())

        return np.concatenate(latent, axis=0)

    @torch.no_grad()
    def posterior_predictive_sample(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        n_samples: int = 1,
    ) -> list[np.ndarray]:
        """Sample from the posterior predictive distribution.

        Returns
        -------
        list of np.ndarray
            ``[rna_samples, protein_samples]``.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=128)

        x_new, y_new = [], []
        for tensors in scdl:
            rna_sample, protein_sample = self.module.sample(
                tensors, n_samples=n_samples
            )
            x_new.append(rna_sample)
            y_new.append(protein_sample)

        x_new = torch.cat(x_new)
        y_new = torch.cat(y_new)

        return [x_new.numpy(), y_new.numpy()]

    @torch.no_grad()
    def prior_predictive_sample(
        self,
        n_samples: int = 1,
        cat_covs: list | None = None,
    ) -> list[np.ndarray]:
        """Sample from the prior predictive distribution."""
        z = Normal(
            torch.zeros(n_samples, self.n_latent),
            torch.ones(n_samples, self.n_latent),
        ).sample()
        z_dict = {"gene": z, "protein": z}

        library_gene = torch.ones(n_samples, 1) * 7
        batch_index = torch.zeros(n_samples, 1)
        y = torch.zeros(n_samples, 1)

        if cat_covs is not None:
            cat_covs = torch.tensor([[float(v)] * n_samples for v in cat_covs]).T

        generative_outputs = self.module.generative(
            z=z_dict,
            library_gene=library_gene,
            batch_index=batch_index,
            label=y,
            cat_covs=cat_covs,
        )

        px_ = generative_outputs["px_dict"]["gene"]
        py_ = generative_outputs["py_dict"]["protein"]

        rna_dist = NegativeBinomial(mu=px_["rate"], theta=px_["r"])
        protein_dist = NegativeBinomialMixture(
            mu1=py_["rate_back"],
            mu2=py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )

        return [
            rna_dist.sample().cpu().numpy(),
            protein_dist.sample().cpu().numpy(),
        ]

    @torch.no_grad()
    def transfer_predictive_sample(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        cat_covs: list | None = None,
        n_samples: int = 1,
    ) -> list[np.ndarray]:
        """Sample with transferred covariate conditions."""
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=128)

        if cat_covs is not None:
            cat_covs = torch.tensor([[float(v)] * n_samples for v in cat_covs]).T

        x_new, y_new = [], []
        for tensors in scdl:
            inference_kwargs = dict(n_samples=n_samples)
            inference_outputs, _ = self.module.forward(
                tensors, inference_kwargs=inference_kwargs, compute_loss=False,
            )

            dec_input = self.module._get_generative_input(tensors, inference_outputs)
            dec_input["cat_covs"] = cat_covs
            generative_outputs = self.module.generative(**dec_input)

            px_ = generative_outputs["px_dict"]["gene"]
            py_ = generative_outputs["py_dict"]["protein"]

            rna_dist = NegativeBinomial(mu=px_["rate"], theta=px_["r"])
            protein_dist = NegativeBinomialMixture(
                mu1=py_["rate_back"],
                mu2=py_["rate_fore"],
                theta1=py_["r"],
                mixture_logits=py_["mixing"],
            )

            rna_sample = rna_dist.sample().cpu().numpy()
            protein_sample = protein_dist.sample().cpu().numpy()

            x_new.append(rna_sample)
            y_new.append(protein_sample)

            if n_samples > 1:
                x_new[-1] = np.transpose(x_new[-1], (1, 2, 0))
                y_new[-1] = np.transpose(y_new[-1], (1, 2, 0))

        return [
            np.concatenate(x_new, axis=0),
            np.concatenate(y_new, axis=0),
        ]
