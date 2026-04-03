"""Custom encoders for cellPMVI with support for Normal, Laplace, and Logistic-Normal."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Laplace, Normal

from scvi.nn import Encoder, FCLayers


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


class CellPMVIEncoder(Encoder):
    """Encoder supporting Normal, Laplace, and Logistic-Normal latent distributions.

    Extends scvi's Encoder with proper Laplace distribution parameterization
    using softplus for the scale parameter.

    Parameters
    ----------
    n_input
        Dimensionality of the input.
    n_output
        Dimensionality of the output (latent space).
    n_cat_list
        List of number of categories for each categorical covariate.
    n_layers
        Number of hidden layers.
    n_hidden
        Number of hidden units per layer.
    dropout_rate
        Dropout rate.
    distribution
        Distribution type: ``"normal"``, ``"ln"`` (logistic normal), or ``"lp"`` (Laplace).
    var_eps
        Minimum variance for numerical stability.
    var_activation
        Activation function for variance. Default is ``torch.exp``.
    return_dist
        If ``True``, return ``(distribution, sample)`` for compatibility with
        scvi-tools >= 1.1. If ``False``, return ``(mean, var, sample)``.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        return_dist: bool = True,
        **kwargs,
    ):
        super().__init__(n_input, n_output, return_dist=return_dist)

        self.distribution = distribution
        self.var_eps = var_eps
        self.return_dist = return_dist
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity

        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):
        """Forward pass: encode input to latent distribution parameters and sample.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch_size, n_input)``.
        cat_list
            Categorical covariate indices.

        Returns
        -------
        If ``return_dist=True``:
            ``(distribution, latent)`` for scvi-tools compatibility.
        If ``return_dist=False``:
            ``(q_m, q_v, latent)`` - mean, variance/scale, and sample.
        """
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)

        if self.distribution == "lp":
            q_v = F.softplus(self.var_encoder(q)) + self.var_eps
            dist = Laplace(q_m, q_v)
        else:
            q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
            dist = Normal(q_m, q_v.sqrt())

        latent = self.z_transformation(dist.rsample())

        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent


class MultiModalEncoder(nn.Module):
    """Multi-modal encoder for joint RNA and protein latent space.

    Encodes gene expression and protein abundance into separate latent
    representations, plus a library size estimate for genes. Based on
    the EncoderTOTALVI design.

    Parameters
    ----------
    n_input_gene
        Number of gene features.
    n_input_protein
        Number of protein features.
    n_output
        Latent space dimensionality.
    n_cat_list
        Number of categories per categorical covariate.
    n_layers
        Number of hidden layers.
    n_hidden
        Hidden layer width.
    dropout_rate
        Dropout rate.
    distribution
        Latent distribution: ``"normal"`` or ``"ln"`` (logistic normal).
    use_batch_norm
        Whether to use batch normalization.
    use_layer_norm
        Whether to use layer normalization.
    """

    def __init__(
        self,
        n_input_gene: int,
        n_input_protein: int,
        n_output: int,
        n_cat_list: Iterable[int] | None = None,
        n_layers: int = 2,
        n_hidden: int = 256,
        dropout_rate: float = 0.1,
        distribution: str = "ln",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.encoder_gene = FCLayers(
            n_in=n_input_gene,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean_encoder_gene = nn.Linear(n_hidden, n_output)
        self.z_var_encoder_gene = nn.Linear(n_hidden, n_output)

        self.encoder_protein = FCLayers(
            n_in=n_input_protein,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean_encoder_protein = nn.Linear(n_hidden, n_output)
        self.z_var_encoder_protein = nn.Linear(n_hidden, n_output)

        self.l_gene_encoder = FCLayers(
            n_in=n_input_gene,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.l_gene_mean_encoder = nn.Linear(n_hidden, 1)
        self.l_gene_var_encoder = nn.Linear(n_hidden, 1)

        self.distribution = distribution

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity

        self.l_transformation = torch.exp

    def _reparameterize(
        self, mu: torch.Tensor, var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = self.z_transformation(untran_z)
        return z, untran_z

    def forward(
        self, data_genes: torch.Tensor, data_proteins: torch.Tensor, *cat_list: int
    ) -> tuple[dict, dict, torch.Tensor, torch.Tensor, dict, dict]:
        """Encode gene and protein data into latent representations.

        Parameters
        ----------
        data_genes
            Gene expression tensor ``(batch_size, n_genes)``.
        data_proteins
            Protein expression tensor ``(batch_size, n_proteins)``.
        cat_list
            Categorical covariate indices.

        Returns
        -------
        tuple
            ``(qz_m, qz_v, ql_m, ql_v, latent, untran_latent)`` where
            ``qz_m``/``qz_v`` are dicts with "gene" and "protein" keys.
        """
        # Gene latent
        q_gene = self.encoder_gene(data_genes, *cat_list)
        qz_m_gene = self.z_mean_encoder_gene(q_gene)
        qz_v_gene = torch.exp(self.z_var_encoder_gene(q_gene)) + 1e-4
        z_gene, untran_z_gene = self._reparameterize(qz_m_gene, qz_v_gene)

        # Protein latent
        q_protein = self.encoder_protein(data_proteins, *cat_list)
        qz_m_protein = self.z_mean_encoder_protein(q_protein)
        qz_v_protein = torch.exp(self.z_var_encoder_protein(q_protein)) + 1e-4
        z_protein, untran_z_protein = self._reparameterize(qz_m_protein, qz_v_protein)

        # Library size
        ql_gene = self.l_gene_encoder(data_genes, *cat_list)
        ql_m = self.l_gene_mean_encoder(ql_gene)
        ql_v = torch.exp(self.l_gene_var_encoder(ql_gene)) + 1e-4
        log_library_gene = torch.clamp(Normal(ql_m, ql_v.sqrt()).rsample(), max=15)
        library_gene = self.l_transformation(log_library_gene)

        qz_m = {"gene": qz_m_gene, "protein": qz_m_protein}
        qz_v = {"gene": qz_v_gene, "protein": qz_v_protein}
        latent = {"z_gene": z_gene, "z_protein": z_protein, "l": library_gene}
        untran_latent = {
            "z_gene": untran_z_gene,
            "z_protein": untran_z_protein,
            "l": log_library_gene,
        }

        return qz_m, qz_v, ql_m, ql_v, latent, untran_latent
