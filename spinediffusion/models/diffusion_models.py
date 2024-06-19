import pytorch_lightning as pl
import torch


class DepthMapDiffusionModel(pl.LightningModule):
    def __init__(self, model, scheduler, lr, loss, max_epochs, **kwargs):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.lr = lr
        self.loss = loss
        self.max_epochs = max_epochs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    # def configure_optimizers(self):
    #     self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
    #     self.lr_scheduler = self.lr_scheduler(
    #         optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps
    #     )
    #     return {
    #         "optimizer": self.optimizer,
    #         "lr_scheduler": {
    #             "scheduler": self.lr_scheduler,
    #             "interval": "step",
    #         },
    #     }
