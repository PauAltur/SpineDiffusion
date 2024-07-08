import diffusers  # noqa: F401
import pytorch_lightning.callbacks  # noqa: F401
import pytorch_lightning.loggers  # noqa: F401
import torch.nn  # noqa: F401
import torchmetrics  # noqa: F401
import torchmetrics.image  # noqa: F401
from pytorch_lightning.cli import LightningCLI

import spinediffusion.data.datamodule  # noqa: F401
import spinediffusion.data.transforms  # noqa: F401
import spinediffusion.models.diffusion_models  # noqa: F401
import spinediffusion.utils.callbacks  # noqa: F401


def main():
    LightningCLI()


if __name__ == "__main__":
    main()
