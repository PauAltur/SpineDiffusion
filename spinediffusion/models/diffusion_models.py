import pytorch_lightning as pl
import torch


class DepthMapDiffusionModel(pl.LightningModule):
    """ """

    def __init__(self, model, scheduler, lr: float, loss, max_epochs: int, **kwargs):
        """

        Args:
            model (_type_): Noise prediction model. Must be a subclass of torch.nn.Module.
            scheduler (_type_): Noise scheduler.
            lr (_type_): Learning rate.
            loss (_type_): Loss function.
            max_epochs (_type_): Maximum number of epochs.
        """
        super().__init__(**kwargs)
        self.model = model
        self.scheduler = scheduler
        self.lr = lr
        self.loss = loss
        self.max_epochs = max_epochs

    def step(self, batch: dict) -> torch.Tensor:
        """Computes a training/validation/test step.

        Args:
            batch (dict): A batch of data.

        Returns:
            loss (torch.Tensor): The loss value.
        """
        x = batch["inputs"]

        noise = torch.randn(x.shape)
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (x.shape[0],), dtype=torch.int64
        )

        noisy_x = self.scheduler.add_noise(x, noise, timesteps)
        noise_pred = self.mode(noisy_x, timesteps, return_dict=False)[0]
        loss = self.loss(noise_pred, noise)

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Computes a training step.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The loss value.
        """
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Computes a validation step.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The loss value.
        """
        loss = self.step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Computes a test step.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The loss value.
        """
        loss = self.step(batch)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Predicts the output of the model.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The loss value.
        """
        x = batch["inputs"]

        for t in self.scheduler.timesteps:
            noisy_residual = self.model(x, t).sample
            previous_noisy_x = self.scheduler.step(noisy_residual, t, x)
            x = previous_noisy_x

        return x
