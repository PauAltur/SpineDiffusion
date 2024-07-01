from typing import Optional

import diffusers
import pytorch_lightning as pl
import torch
from torchmetrics import MetricCollection
from tqdm import tqdm


class UnconditionalDiffusionModel(pl.LightningModule):
    """Pytorch lignting module that wraps a diffusion model composed of a
    noise prediction model and a noise scheduler coming from the diffusers
    library: https://huggingface.co/docs/diffusers/index."""

    def __init__(
        self,
        model,
        scheduler,
        loss,
        metrics,
        **kwargs,
    ):
        """Initializes the model.

        Args:
            model (diffusers.ModelMixin): Noise prediction model.
            Must be a subclass of torch.nn.Module.
            scheduler (diffusers.SchedulerMixin): Noise scheduler.
            loss (torch.nn.Module): Loss function.
            metrics (list): List of metrics to compute.
        """
        super().__init__(**kwargs)
        self.model = model
        self.scheduler = scheduler
        self.scheduler.device = self.device
        self.loss = loss
        self.metrics = MetricCollection(metrics)

    def forward(self, batch: dict) -> torch.Tensor:
        """Computes a training/validation/test step.

        Args:
            batch (dict): A batch of data.

        Returns:
            loss (torch.Tensor): The loss value.
        """
        x = batch[0]

        noise = torch.randn(x.shape, dtype=torch.float32, device=self.device)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (x.shape[0],),
            dtype=torch.int32,
            device=self.device,
        )

        noisy_x = self.scheduler.add_noise(x, noise, timesteps)
        noise_pred = self.model(
            noisy_x.type(torch.float32), timesteps, return_dict=False
        )[0]

        loss = self.loss(noise_pred, noise)
        metrics = self.metrics(noise_pred, noise)

        return loss, metrics

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Computes a training step.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The loss value.
        """
        loss, metrics = self.forward(batch)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Computes a validation step.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The loss value.
        """
        loss, metrics = self.forward(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(metrics)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Computes a test step.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The loss value.
        """
        loss, metrics = self.forward(batch)
        self.log("test_loss", loss)
        self.log_dict(metrics)
        return loss

    def predict_step(self, batch: dict) -> torch.Tensor:
        """Performs an unconditioned inference step.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The loss value.
        """
        x = batch.to(self.device)

        for t in tqdm(self.scheduler.timesteps):
            noisy_residual = self.model(x, t).sample
            previous_noisy_x = self.scheduler.step(noisy_residual, t, x).prev_sample
            x = previous_noisy_x

        return x
