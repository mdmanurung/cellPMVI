"""Custom training plan with cyclical KL annealing and proper logging."""

from __future__ import annotations

import math

from scvi.train import TrainingPlan


class CellPMVITrainingPlan(TrainingPlan):
    """Training plan with cyclical KL annealing support.

    Extends scvi-tools' TrainingPlan with:
    - Cyclical KL annealing (Fu et al., 2019)
    - Proper epoch-level metric logging for Lightning 2.x

    Parameters
    ----------
    module
        The VAE module to train.
    n_epochs_kl_warmup
        Number of epochs for KL warmup (linear ramp). If set, overrides
        ``n_steps_kl_warmup``. Default is 400.
    kl_annealing
        KL annealing strategy: ``"monotonic"`` or ``"cyclical"``.
    n_cycles
        Number of cycles for cyclical annealing.
    **kwargs
        Additional arguments passed to ``TrainingPlan``.
    """

    def __init__(
        self,
        module,
        n_epochs_kl_warmup: int = 400,
        kl_annealing: str = "monotonic",
        n_cycles: int = 4,
        **kwargs,
    ):
        super().__init__(module, n_epochs_kl_warmup=n_epochs_kl_warmup, **kwargs)
        self.kl_annealing = kl_annealing
        self.n_cycles = n_cycles

    @property
    def kl_weight(self) -> float:
        """Compute KL weight based on annealing schedule."""
        if self.kl_annealing == "cyclical" and self.n_epochs_kl_warmup is not None:
            epoch = self.current_epoch
            cycle_length = max(self.n_epochs_kl_warmup // self.n_cycles, 1)
            position_in_cycle = epoch % cycle_length
            proportion = position_in_cycle / cycle_length
            return min(1.0, proportion * 2)
        # Default monotonic from parent
        return super().kl_weight

    def training_step(self, batch, batch_idx):
        """Training step with KL weight update and metric logging."""
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        reconstruction_loss = scvi_loss.reconstruction_loss

        self.log(
            "train_loss",
            scvi_loss.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {
            "loss": scvi_loss.loss,
            "reconstruction_loss_sum": reconstruction_loss.sum()
            if torch.is_tensor(reconstruction_loss)
            else sum(v.sum() for v in reconstruction_loss.values()),
            "kl_local_sum": scvi_loss.kl_local.sum()
            if torch.is_tensor(scvi_loss.kl_local)
            else sum(v.sum() for v in scvi_loss.kl_local.values()),
            "kl_global": scvi_loss.kl_global,
            "n_obs": batch[list(batch.keys())[0]].shape[0]
            if isinstance(batch, dict)
            else reconstruction_loss.shape[0],
        }

    def on_train_epoch_end(self):
        """Log epoch-level metrics (Lightning 2.x compatible)."""
        self.log("kl_weight", self.kl_weight, prog_bar=True)


# Need this for the training_step
import torch  # noqa: E402
