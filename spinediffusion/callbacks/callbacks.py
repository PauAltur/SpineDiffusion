import torch
from pytorch_lightning.callbacks import Callback


class LogGeneratedImages(Callback):
    """Pytorch Lightning callback that generates images at the end of each training epoch."""

    def __init__(
        self,
        every_n_epochs: int = 10,
        num_images: int = 16,
        num_channels: int = 1,
        height: int = 128,
        width: int = 128,
    ):
        """Initializes the GenerateImages callback.

        Args:
            every_n_epochs (int, optional): The number of epochs between image generations.
            Defaults to 10.
            num_images (int, optional): The number of images to generate. Defaults to 16.
            num_channels (int, optional): The number of channels in the images. Defaults to 1.
            height (int, optional): The height of the images. Defaults to 128.
            width (int, optional): The width of the images. Defaults to 128.
        """
        self.every_n_epochs = every_n_epochs
        self.num_images = num_images
        self.num_channels = num_channels
        self.height = height
        self.width = width

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        noise = [
            torch.randn(self.num_images, self.num_channels, self.height, self.width).to(
                pl_module.device
            )
        ]
        generated = pl_module.predict_step(noise, 1)
        trainer.logger.experiment.add_image(
            "train/generated/",
            generated,
            global_step=trainer.global_step,
            dataformats="NCHW",
        )
