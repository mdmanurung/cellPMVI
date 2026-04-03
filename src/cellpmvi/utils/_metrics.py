"""Evaluation metrics for cellPMVI models."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def compute_reconstruction_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Compute reconstruction quality metrics.

    Parameters
    ----------
    observed
        Observed expression matrix ``(n_cells, n_features)``.
    predicted
        Predicted expression matrix ``(n_cells, n_features)``.

    Returns
    -------
    dict
        Dictionary with ``"spearman_mean"``, ``"spearman_var"``,
        ``"rmse"`` metrics.
    """
    # Per-feature means
    obs_mean = observed.mean(axis=0)
    pred_mean = predicted.mean(axis=0)
    corr_mean, _ = spearmanr(obs_mean, pred_mean)

    # Per-feature variances
    obs_var = observed.var(axis=0)
    pred_var = predicted.var(axis=0)
    corr_var, _ = spearmanr(obs_var, pred_var)

    # RMSE
    rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))

    return {
        "spearman_mean": float(corr_mean),
        "spearman_var": float(corr_var),
        "rmse": rmse,
    }


def compute_latent_metrics(
    latent: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute latent space quality metrics.

    Parameters
    ----------
    latent
        Latent representation ``(n_cells, n_latent)``.
    labels
        Cell type labels for each cell.
    batch
        Batch labels for each cell (optional, for batch mixing metric).

    Returns
    -------
    dict
        Dictionary with ``"silhouette_labels"`` and optionally
        ``"silhouette_batch"`` scores.
    """
    from sklearn.metrics import silhouette_score

    results = {}

    if len(np.unique(labels)) > 1:
        results["silhouette_labels"] = float(
            silhouette_score(latent, labels, sample_size=min(5000, len(labels)))
        )

    if batch is not None and len(np.unique(batch)) > 1:
        results["silhouette_batch"] = float(
            silhouette_score(latent, batch, sample_size=min(5000, len(batch)))
        )

    return results
