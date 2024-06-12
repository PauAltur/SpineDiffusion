import glob
import json
import open3d as o3d
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from typing import Optional
from .transforms.normalizing import Normalize
from .transforms.resampling import Resample3DCurve
from .transforms.cropping import Crop
from .transforms.centering import Center

TRANSFORMS = {
    "ToTensor": v2.ToTensor,
    "Normalize": Normalize,
    "Resample3DCurve": Resample3DCurve,
    "Crop": Crop,
    "Center": Center,
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
        transform_args: dict,
        train_fraction: Optional[float] = None,
        val_fraction: Optional[float] = None,
        test_fraction: Optional[float] = None,
        train_indices: Optional[list] = None,
        val_indices: Optional[list] = None,
        test_indices: Optional[list] = None,
        n_subjects: Optional[float] = None,
    ):
        """_summary_

        Args:
            data_dir (str): _description_
            batch_size (int): _description_
            transform_args (dict): _description_
            train_fraction (Optional[float], optional): _description_. Defaults to None.
            val_fraction (Optional[float], optional): _description_. Defaults to None.
            test_fraction (Optional[float], optional): _description_. Defaults to None.
            train_indices (Optional[list], optional): _description_. Defaults to None.
            val_indices (Optional[list], optional): _description_. Defaults to None.
            test_indices (Optional[list], optional): _description_. Defaults to None.
            n_subjects (Optional[float], optional): _description_. Defaults to None.
        """        
        
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.transform_args = transform_args
        self.data = {"backscans": {},"metadata" : {}}
        self.n_subjects = n_subjects

    def setup(self, stage: Optional[str]):
        """_summary_

        Args:
            stage (str | None): _description_
        """
        self._parse_datapaths()
        self._load_data("metadata", self.dirs_metadata)
        self._load_data("backscans", self.dirs_backscan)
        self._preprocess_data()
        self._split_data()

    def _parse_datapaths(self):
        """_summary_"""
        print(f"Parsing data paths from {self.data_dir}...")

        assert self.data_dir.exists(), "Data directory does not exist."

        backscan_dir = str(self.data_dir / "**" / f"*processed.ply")
        metadata_dir = str(self.data_dir / "**" / f"*processed.json")

        # making every path string a Path object is useful for cross-OS compatibility
        self.dirs_backscan = [
            Path(path) for path in natsorted(glob.glob(backscan_dir))
        ]
        self.dirs_metadata = [
            Path(path) for path in natsorted(glob.glob(metadata_dir))
        ]

    def _load_data(self, file_ext: str, data_dirs: list):
        """_summary_

        Args:
            file (str): _description_
            data_dirs (list): _description_
        """        
        print(f"Loading {file_ext}...")
        for path in tqdm(data_dirs[:self.n_subjects]):
            unique_id = path.parts[-2]
            if path.suffix == ".ply":
                self.data[file_ext][unique_id] = o3d.io.read_point_cloud(str(path))
            elif path.suffix == ".json":
                with open(path, "r") as f:
                    self.data[file_ext][unique_id] = json.load(f)

    def _check_split_args(self):
        """_summary_
        """
        fraction_args = [self.train_fraction, self.val_fraction, self.test_fraction]
        indices_args = [self.train_indices, self.val_indices, self.test_indices]

        fraction_bool = np.all([fraction is not None for fraction in fraction_args])
        indices_bool = np.all(
            [indices is not None for indices in indices_args]
        )

        msg = "Either fractions or indices should be provided, not both."
        assert np.xor(fraction_bool, indices_bool), msg

        if fraction_bool:
            msg = "Sum of fractions should be equal to 1."
            assert np.sum(fraction_args) == 1, msg
            self.train_indices = np.random.choice(
                np.arange(len(self.data)),
                size=int(self.train_fraction * len(self.data)),
                replace=False,
            )
            self.val_indices = np.random.choice(
                np.setdiff1d(np.arange(len(self.data)), self.train_indices),
                size=int(self.val_fraction * len(self.data)),
                replace=False,
            )
            self.test_indices = np.setdiff1d(
                np.arange(len(self.data)),
                np.concatenate([self.train_indices, self.val_indices]),
            )

    def _preprocess_data(self):
        """_summary_"""
        print("Preprocessing data...")
        transforms = []
        for i in range(len(self.transform_args)):
            for key, value in self.transform_args.items():
                if value["transform_number"] == i:
                    transforms.append(TRANSFORMS[key](**value))
        transforms = v2.Compose(transforms)
        self.data = transforms(self.data)

    def _split_data(self):
        """_summary_"""
        print("Splitting data...")
        self._check_split_args()
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
