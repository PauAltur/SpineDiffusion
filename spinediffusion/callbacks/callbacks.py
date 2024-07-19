import glob
from typing import Optional

import pandas as pd
import torch
from pytorch_lightning.callbacks.callback import Callback
from tensorflow.python.summary.summary_iterator import summary_iterator
from torch.utils.data import TensorDataset

from spinediffusion.datamodule.sl_generator import SLGenerator


class LogGeneratedImages(Callback):
    """Pytorch Lightning callback that generates images at the end of each training epoch."""

    def __init__(
        self,
        every_n_epochs: int = 10,
        num_images: int = 16,
        num_channels: int = 1,
        height: int = 128,
        width: int = 128,
        sl_args: Optional[dict] = None,
    ):
        """Initializes the GenerateImages callback.

        Args:
            every_n_epochs (int, optional): The number of epochs between image generations.
                Defaults to 10.
            num_images (int, optional): The number of images to generate. Defaults to 16.
            num_channels (int, optional): The number of channels in the images. Defaults to 1.
            height (int, optional): The height of the images. Defaults to 128.
            width (int, optional): The width of the images. Defaults to 128.
            sl_args (Optional[dict], optional): The arguments for the SLGenerator.
                Defaults to None.
        """
        self.every_n_epochs = every_n_epochs
        self.num_images = num_images
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.sl_args = sl_args

    def on_train_epoch_end(self, trainer, pl_module):
        """Generates images at the end of each training epoch.

        Args:
            trainer (pytorch_lightning.trainer): The trainer object.
            pl_module (pytorch_lighnting.module): The module object.
        """
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        noise = [
            torch.randn(
                self.num_images,
                self.num_channels,
                self.height,
                self.width,
                device=pl_module.device,
            )
        ]

        if self.sl_args:
            sl_generator = SLGenerator(**self.sl_args)
            generated_sl = [
                torch.tensor(
                    next(sl_generator).copy(),
                    dtype=torch.float32,
                    device=pl_module.device,
                ).unsqueeze(0)
                for _ in range(self.num_images)
            ]
            predict_batch = [noise, None, generated_sl]
            predict_batch = TensorDataset(predict_batch)
        else:
            predict_batch = noise

        generated = pl_module.predict_step(predict_batch, 1)
        trainer.logger.experiment.add_image(
            "train/generated/",
            generated,
            global_step=trainer.global_step,
            dataformats="NCHW",
        )


class GenerateCSVLog(Callback):
    """Pytorch Lightning callback that generates CSV logs at the end of training."""

    def on_train_end(self, trainer, pl_module):
        """Transforms tf events files to csv files and saves
        them in the same directory as the tf events files.

        Args:
            trainer (pytorch_lightning.trainer): The trainer object.
            pl_module (pytorch_lighnting.module): The module object.
        """
        events_dir = trainer.logger.log_dir
        events_file = glob.glob(events_dir + "/events.out.tfevents.*")[0]

        df = pd.DataFrame(columns=["time", "tag", "value"])
        for event in summary_iterator(events_file):
            for value in event.summary.value:
                df.loc[len(df)] = [event.wall_time, value.tag, value.simple_value]

        df = df.sort_values(by=["tag", "time"])
        df["step"] = df.groupby("tag").cumcount()

        df.to_csv(f"{events_dir}/events.csv", index=False)
        print(f"Saved {events_dir}/events.csv")
