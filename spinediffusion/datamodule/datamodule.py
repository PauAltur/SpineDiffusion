import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from natsort import natsorted
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
        surface_rotation: bool,
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
            surface_rotation (bool): _description_
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
        self.surface_rotation = surface_rotation

    def setup(self, stage: str | None):
        """_summary_

        Args:
            stage (str | None): _description_
        """
        self._parse_datapaths()
        self._load_metadata()
        self._load_scandata()
        self._preprocess_data()
        self._check_split_args()
        self._split_data()

    def _parse_datapaths(self):
        """_summary_"""
        assert Path(self.data_dir).exists(), "Data directory does not exist."
        data_dir = Path(self.data_dir)

        dirs_backscan = natsorted(
            glob.glob(str(data_dir / "**/*_backscan.ply"))
        )
        dirs_metadata = natsorted(
            glob.glob(str(data_dir / "**/*_metadata.json"))
        )

        self.metadata = pd.DataFrame(
            list(zip(dirs_backscan, dirs_metadata)),
            columns=['dirs_backscan', 'dirs_metadata']
        )

    def _load_metadata(self):
        """_summary_"""
        dirs_metadata = self.metadata['dirs_metadata'].tolist()

        subject_idx = np.empty(len(dirs_metadata), dtype=object)
        age_list = np.zeros(len(dirs_metadata), dtype=int)
        gender = np.empty((len(dirs_metadata)), dtype=str)
        rec_time = np.empty((len(dirs_metadata)), dtype=datetime)
        esl = np.zeros(len(dirs_metadata), dtype=object)
        isl = np.zeros(len(dirs_metadata), dtype=object)
        fix_points = np.zeros(len(dirs_metadata), dtype=dict)
        apex_points = np.zeros(len(dirs_metadata), dtype=object)
        if self.surface_rotation:
            surface_rotation = np.zeros(len(dirs_metadata), dtype=float)

        for idx, dir in enumerate(dirs_metadata):
            # Opening JSON file
            f = open(dir)
            data = json.load(f)

            # some age entries are wrong, they need to be corrected
            if isinstance(data['age'], str):
                converted_date = datetime.strptime(data['age'], "%d.%m.%Y").year
                year_diff = converted_date - 2000
                age_list[idx] = year_diff
            else:
                age_list[idx] = data['age']

            subject_idx[idx] = data['subject_index']
            gender[idx] = data['gender']
            rec_time[idx] = data['recording_datetime']
            esl[idx] = np.array(data['esl_formetric'])
            isl[idx] = np.array(data['isl_formetric'])
            fix_points[idx] = data['fix_pts']
            apex_points[idx] = data['apex_pts']
            if self.surface_rotation:
                surface_rotation[idx] = data['surface_rotation_rms_deg']

            # Closing file
            f.close()

        # convert "w" to 0 and "m" to 1 in gender
        gender_list = np.asarray([0 if x == "w" else 1 for x in gender])

        self.metadata.insert(0, 'subject_id', subject_idx)
        self.metadata = self.metadata.assign(gender=gender_list, age=age_list, rec_time=rec_time, esl=esl, isl=isl, fix_points=fix_points,
                                apex_points=apex_points)

        if self.surface_rotation:
            self.metadata['surface_rotation'] = surface_rotation

        self.metadata['rec_time'] = pd.to_datetime(self.metadata['rec_time'], format='%d-%b-%Y %H:%M:%S')
    
    def _load_scandata(self):
        """_summary_"""
        pass


    def _check_split_args(self):
        """_summary_
        """
        ratio_args = [self.train_ratio, self.val_ratio, self.test_ratio]
        indices_args = [self.train_indices, self.val_indices, self.test_indices]

        ratio_bool = np.all([ratio is not None for ratio in ratio_args])
        indices_bool = np.all(
            [indices is not None for indices in indices_args]
        )

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
