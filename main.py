from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.demos.boring_classes import BoringDataModule, DemoModel


def main():
    LightningCLI(DemoModel, BoringDataModule)


if __name__ == "__main__":
    main()
