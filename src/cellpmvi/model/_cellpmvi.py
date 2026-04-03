"""High-level CellPMVI model class."""

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
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin
from scvi.utils._docstrings import setup_anndata_dsp

from cellpmvi.module._cellpmvae import CellPMVAE

logger = logging.getLogger(__name__)


class CellPMVI(UnsupervisedTrainingMixin, BaseModelClass):
    """Multi-modal VAE for joint RNA and protein analysis.

    Integrates RNA and protein expression data using separate
    encoder/decoder pairs with optional posterior fusion.

    Parameters
    ----------
    adata
        AnnData object registered via :meth:`setup_anndata`.
    n_hidden
        Number of hidden units per layer.
    n_latent
        Latent space dimensionality.
    n_layers
        Number of hidden layers.
    dropout_rate
        Dropout rate.
    gene_likelihood
        Reconstruction likelihood: ``"nb"``, ``"zinb"``, or ``"poisson"``.
    latent_distribution
        Latent distribution: ``"normal"`` or ``"lp"`` (Laplace).
    fusion_method
        Multi-modal posterior fusion strategy:
        ``"none"``, ``"poe"`` (product-of-experts), or ``"moe"`` (mixture-of-experts).
    encode_covariates
        Whether to condition the encoder on covariates.
    **model_kwargs
        Additional keyword arguments for :class:`CellPMVAE`.

    Examples
    --------
    >>> CellPMVI.setup_anndata(adata, layer="counts", protein_expression_obsm_key="protein")
    >>> model = CellPMVI(adata)
    >>> model.train(max_epochs=200)  # doctest: +SKIP
    >>> latent = model.get_latent_representation()  # doctest: +SKIP
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 20,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["nb", "zinb", "poisson"] = "nb",
        latent_distribution: str = "normal",
        fusion_method: Literal["none", "poe", "moe"] = "poe",
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

        self.module = CellPMVAE(
            n_input_rna=self.summary_stats["n_vars"],
            n_input_pro=self.summary_stats["n_proteins"],
            n_batch=self.summary_stats["n_batch"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            gene_likelihood=gene_likelihood,
            dropout_rate=dropout_rate,
            latent_distribution=latent_distribution,
            fusion_method=fusion_method,
            encode_covariates=encode_covariates,
            n_cats_per_cov=n_cats_per_cov,
            **model_kwargs,
        )

        self.n_latent = n_latent
        self._model_summary_string = (
            f"CellPMVI Model with params: "
            f"n_latent={n_latent}, n_hidden={n_hidden}, n_layers={n_layers}, "
            f"gene_likelihood={gene_likelihood}, dropout_rate={dropout_rate}, "
            f"latent_distribution={latent_distribution}, fusion_method={fusion_method}"
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
        """Register AnnData fields for CellPMVI.

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
            # Use RNA encoder's posterior (or joint if fusion is applied)
            rna_out = outputs[0]
            if give_mean:
                z = rna_out.get("qz_m_joint", rna_out["qz_m"])
            else:
                z = rna_out["z"]
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
            samples = self.module.sample(tensors, n_samples=n_samples)
            x_new.append(samples[0])
            y_new.append(samples[1])

        x_new = torch.cat(x_new)
        y_new = torch.cat(y_new)

        return [x_new.numpy(), y_new.numpy()]

    @torch.no_grad()
    def prior_predictive_sample(
        self,
        ls_constant_genes: float = 7.0,
        ls_constant_protein: float = 7.0,
        n_samples: int = 1,
        cat_covs: list | None = None,
    ) -> list[np.ndarray]:
        """Sample from the prior predictive distribution.

        Parameters
        ----------
        ls_constant_genes
            Log-library size for gene sampling.
        ls_constant_protein
            Log-library size for protein sampling.
        n_samples
            Number of samples.
        cat_covs
            Categorical covariate values for conditional generation.

        Returns
        -------
        list of np.ndarray
            ``[rna_samples, protein_samples]``.
        """
        z = Normal(
            torch.zeros(n_samples, self.n_latent),
            torch.ones(n_samples, self.n_latent),
        ).sample()

        library_genes = torch.ones(n_samples, 1) * ls_constant_genes
        library_proteins = torch.ones(n_samples, 1) * ls_constant_protein
        batch_index = torch.zeros(n_samples, 1, dtype=torch.long)
        y = torch.zeros(n_samples, 1, dtype=torch.long)

        if cat_covs is not None:
            cat_covs = torch.tensor([[float(v)] * n_samples for v in cat_covs]).T

        dec_input = dict(
            rna_z=z, pro_z=z,
            rna_library=library_genes, pro_library=library_proteins,
            batch_index=batch_index,
            rna_y=y, pro_y=y,
            cont_covs=None, cat_covs=cat_covs,
        )

        px_zs = self.module.generative(**dec_input)

        data = []
        for e, px_z in enumerate(px_zs):
            for i, generative_outputs in enumerate(px_z):
                if e != i:
                    continue

                px_r = generative_outputs["px_r"]
                px_rate = generative_outputs["px_rate"]
                px_dropout = generative_outputs["px_dropout"]

                if self.module.gene_likelihood == "poisson":
                    dist = torch.distributions.Poisson(
                        torch.clamp(px_rate, max=1e8)
                    )
                elif self.module.gene_likelihood == "nb":
                    dist = NegativeBinomial(mu=px_rate, theta=px_r)
                elif self.module.gene_likelihood == "zinb":
                    dist = ZeroInflatedNegativeBinomial(
                        mu=px_rate, theta=px_r, zi_logits=px_dropout
                    )
                else:
                    raise ValueError(
                        f"Unknown gene_likelihood: {self.module.gene_likelihood}"
                    )
                data.append(dist.sample().cpu().detach().numpy())

        return data

    @torch.no_grad()
    def transfer_predictive_sample(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        cat_covs: list | None = None,
        n_samples: int = 1,
    ) -> list[np.ndarray]:
        """Sample with transferred covariate conditions.

        Encodes cells from ``adata`` but decodes with different
        categorical covariates for counterfactual generation.

        Parameters
        ----------
        adata
            AnnData to encode.
        indices
            Cell indices.
        cat_covs
            Target categorical covariate values.
        n_samples
            Number of samples per cell.

        Returns
        -------
        list of np.ndarray
            ``[rna_samples, protein_samples]``.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=128)

        if cat_covs is not None:
            cat_covs = torch.tensor([[float(v)] * n_samples for v in cat_covs]).T

        data = [[], []]
        for tensors in scdl:
            inference_kwargs = dict(n_samples=n_samples)
            inference_outputs, _ = self.module.forward(
                tensors, inference_kwargs=inference_kwargs, compute_loss=False,
            )

            dec_input_dict = self.module._get_generative_input(
                tensors, inference_outputs
            )
            dec_input_dict["cat_covs"] = cat_covs

            px_zs = self.module.generative(**dec_input_dict)

            for e, px_z in enumerate(px_zs):
                for i, generative_outputs in enumerate(px_z):
                    if e != i:
                        continue

                    px_r = generative_outputs["px_r"]
                    px_rate = generative_outputs["px_rate"]
                    px_dropout = generative_outputs["px_dropout"]

                    if self.module.gene_likelihood == "nb":
                        dist = NegativeBinomial(mu=px_rate, theta=px_r)
                    elif self.module.gene_likelihood == "zinb":
                        dist = ZeroInflatedNegativeBinomial(
                            mu=px_rate, theta=px_r, zi_logits=px_dropout
                        )
                    elif self.module.gene_likelihood == "poisson":
                        dist = torch.distributions.Poisson(
                            torch.clamp(px_rate, max=1e8)
                        )
                    else:
                        raise ValueError(
                            f"Unknown gene_likelihood: {self.module.gene_likelihood}"
                        )
                    data[e].append(dist.sample().cpu().detach())

        x_new = torch.cat(data[0])
        y_new = torch.cat(data[1])

        return [x_new.numpy(), y_new.numpy()]
