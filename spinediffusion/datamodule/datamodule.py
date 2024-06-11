from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

TRANSFORMS = {
    "ToTensor": v2.ToTensor,
}


class SpineDataModule(pl.LightningDataModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        train_ratio: float | None,
        val_ratio: float | None,
        test_ratio: float | None,
        train_indices: list | None,
        val_indices: list | None,
        test_indices: list | None,
        transform_args: dict,
    ):
        """_summary_

        Args:
            data_dir (str): _description_
            batch_size (int): _description_
            train_ratio (float | None): _description_
            val_ratio (float | None): _description_
            test_ratio (float | None): _description_
            train_indices (list | None): _description_
            val_indices (list | None): _description_
            test_indices (list | None): _description_
            transform_args (dict): _description_
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.transform_args = transform_args

    def setup(self, stage: str | None):
        """_summary_

        Args:
            stage (str | None): _description_
        """
        self._load_data()
        self._preprocess_data()
        self._check_split_args(
            [self.train_ratio, self.val_ratio, self.test_ratio],
            [self.train_indices, self.val_indices, self.test_indices],
        )
        self._split_data()

    def _load_data(self):
        """_summary_"""
        assert Path(self.data_dir).exists(), "Data directory does not exist."
        self.data = np.load(self.data_dir)

    def _check_split_args(self, ratio_args, indices_args):
        """_summary_

        Args:
            ratio_args (_type_): _description_
            indices_args (_type_): _description_
        """
        ratio_bool = np.all([ratio is not None for ratio in ratio_args])
        indices_bool = np.all(
            [indices is not None for indices in indices_args]
        )  # noqa: E501

        msg = "Either ratios or indices should be provided, not both."
        assert np.xor(ratio_bool, indices_bool), msg

        if ratio_bool:
            msg = "Sum of ratios should be equal to 1."
            assert np.sum(ratio_args) == 1, msg
            self.train_indices = np.random.choice(
                np.arange(len(self.data)),
                size=int(self.train_ratio * len(self.data)),
                replace=False,
            )
            self.val_indices = np.random.choice(
                np.setdiff1d(np.arange(len(self.data)), self.train_indices),
                size=int(self.val_ratio * len(self.data)),
                replace=False,
            )
            self.test_indices = np.setdiff1d(
                np.arange(len(self.data)),
                np.concatenate([self.train_indices, self.val_indices]),
            )

    def _preprocess_data(self):
        """_summary_"""
        transforms = []
        for i in range(len(self.transform_args)):
            for key, value in self.transform_args.items():
                if value["transform_number"] == i:
                    transforms.append(TRANSFORMS[key](**value))
        transforms = v2.Compose(transforms)
        self.data = transforms(self.data)

    def _split_data(self):
        """_summary_"""
        self.train_data = Dataset(self.data[self.train_indices])
        self.val_data = Dataset(self.data[self.val_indices])
        self.test_data = Dataset(self.data[self.test_indices])

    def train_dataloader(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        train_dataloader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )
        return train_dataloader

    def val_dataloader(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        val_dataloader = DataLoader(
            self.val_data, batch_size=len(self.val_data), shuffle=False
        )
        return val_dataloader

    def test_dataloader(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        test_dataloader = DataLoader(
            self.test_data, batch_size=len(self.test_data), shuffle=False
        )
        return test_dataloader
