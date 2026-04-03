"""Multi-modal posterior fusion strategies (PoE, MoE)."""

from __future__ import annotations

import torch
from torch import nn


class ProductOfExperts(nn.Module):
    """Product-of-Experts fusion for combining per-modality posteriors.

    Combines Gaussian posteriors from multiple modalities into a joint
    posterior using the product-of-experts formulation::

        1/sigma^2_joint = sum(1/sigma^2_i) + 1/sigma^2_prior
        mu_joint = sigma^2_joint * sum(mu_i / sigma^2_i)

    This produces a sharper joint posterior that leverages information
    from all modalities.
    """

    def forward(
        self,
        means: list[torch.Tensor],
        variances: list[torch.Tensor],
        prior_mean: torch.Tensor | None = None,
        prior_var: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse multiple Gaussian posteriors via Product-of-Experts.

        Parameters
        ----------
        means
            List of posterior means, each ``(batch_size, n_latent)``.
        variances
            List of posterior variances, each ``(batch_size, n_latent)``.
        prior_mean
            Prior mean. Defaults to zeros.
        prior_var
            Prior variance. Defaults to ones.

        Returns
        -------
        tuple of torch.Tensor
            Joint posterior ``(mean, variance)``.
        """
        # Start with prior precision
        if prior_mean is None:
            prior_mean = torch.zeros_like(means[0])
        if prior_var is None:
            prior_var = torch.ones_like(variances[0])

        # Accumulate precisions
        precision = 1.0 / prior_var
        weighted_mean = prior_mean / prior_var

        for mu, var in zip(means, variances):
            precision = precision + 1.0 / var
            weighted_mean = weighted_mean + mu / var

        joint_var = 1.0 / precision
        joint_mean = joint_var * weighted_mean

        return joint_mean, joint_var


class MixtureOfExperts(nn.Module):
    """Mixture-of-Experts fusion for combining per-modality posteriors.

    Averages posteriors from multiple modalities. More robust to missing
    modalities than PoE but produces a broader joint posterior.
    """

    def forward(
        self,
        means: list[torch.Tensor],
        variances: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse multiple Gaussian posteriors via Mixture-of-Experts.

        Parameters
        ----------
        means
            List of posterior means.
        variances
            List of posterior variances.

        Returns
        -------
        tuple of torch.Tensor
            Averaged posterior ``(mean, variance)``.
        """
        stacked_means = torch.stack(means)
        stacked_vars = torch.stack(variances)

        joint_mean = stacked_means.mean(dim=0)
        # Var of mixture = mean of vars + mean of squared means - square of mean of means
        joint_var = (stacked_vars + stacked_means.pow(2)).mean(dim=0) - joint_mean.pow(2)

        return joint_mean, joint_var
