import glob
import os
from typing import Optional

import pandas as pd
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.callbacks.callback import Callback
from tensorflow.python.summary.summary_iterator import summary_iterator
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image

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
        batch_size: int = 10,
        save_dir: Optional[str] = None,
    ):
        """Initializes the LogGeneratedImages callback.

        Args:
            every_n_epochs (int, optional): The number of epochs between image generations.
                Defaults to 10.
            num_images (int, optional): The number of images to generate. Defaults to 16.
            num_channels (int, optional): The number of channels in the images. Defaults to 1.
            height (int, optional): The height of the images. Defaults to 128.
            width (int, optional): The width of the images. Defaults to 128.
            sl_args (Optional[dict], optional): The arguments for the SLGenerator.
                Defaults to None.
            batch_size (int, optional): The batch size for the generated images.
                Defaults to 10.
            save_dir (Optional[str], optional): The directory to save the images.
                Defaults to None.
        """
        self.every_n_epochs = every_n_epochs
        self.num_images = num_images
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.sl_args = sl_args
        self.batch_size = batch_size
        self.save_dir = save_dir

    def _unconditional_input_generation(self, device):
        """Generates unconditional input for the generator.

        Args:
            device (torch.device): The device to use for the input.

        Returns:
            torch.Tensor: The generated input.
        """
        predict_batch = torch.randn(
            self.num_images,
            self.num_channels,
            self.height,
            self.width,
            device=device,
        )
        return predict_batch

    def _conditional_input_generation(self, device):
        """Generates conditional input for the generator.

        Args:
            device (torch.device): The device to use for the input.

        Returns:
            torch.Tensor: The generated input.
        """
        noise = torch.randn(
            self.num_images,
            self.num_channels,
            self.height,
            self.width,
            device=device,
        )

        sl_generator = SLGenerator(**self.sl_args)
        generated_sl = torch.concatenate(
            [
                torch.tensor(
                    next(sl_generator).copy(),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                for _ in range(self.num_images)
            ]
        )

        predict_batch = TensorDataset(
            noise, torch.zeros(noise.size(), device=device), generated_sl
        )

        return predict_batch


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


class CustomImageWriter(BasePredictionWriter):
    """Pytorch Lightning callback that writes predictions to disk."""

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        torch.save(
            prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt")
        )

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))


class ComputeFID(Callback):
    """Pytorch Lightning callback that computes the FID score at the end of training."""

    def __init__(self):
        super().__init__()
        self.fid = FrechetInceptionDistance(normalize=True)
        self.num_images = 0

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        # stack the predictions along the second axis
        outputs = torch.cat((outputs, outputs, outputs), dim=1)
        self.fid.to(device=outputs.device)
        self.fid.update(outputs, real=False)
        self.num_images += outputs.shape[0]

    def on_predict_epoch_end(self, trainer, pl_module):

        import pdb

        pdb.set_trace()
        train_dataloader = trainer.train_dataloader

        num_real_images = 0
        while num_real_images < self.num_images:
            real_images = next(iter(train_dataloader))[0]
            real_images = torch.cat((real_images, real_images, real_images), dim=1)
            num_real_images += real_images.shape[0]
            self.fid.update(real_images, real=True)

        fid_score = self.fid.compute()
        print(f"FID score: {fid_score}")
        self.fid.reset()
