from pytorch_lightning.cli import LightningCLI

import spinediffusion.data.transforms  # noqa: F401


def main():
    LightningCLI()


if __name__ == "__main__":
    main()
