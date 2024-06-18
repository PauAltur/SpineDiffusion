from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.demos.boring_classes import BoringDataModule, DemoModel

from spinediffusion.datamodule.datamodule import SpineDataModule


def main():
    LightningCLI()


if __name__ == "__main__":
    main()
