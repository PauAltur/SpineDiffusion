import glob
import json
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
from natsort import natsorted
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2
from tqdm import tqdm

from spinediffusion.utils.hashing import hash_dict
from spinediffusion.utils.misc import dumper

from .transforms.closing import Closing
from .transforms.normalizing import ConstantNormalization, SpineLengthNormalization
from .transforms.projecting import ProjectToPlane
from .transforms.resampling import Resample3DCurve, ResamplePointCloud
from .transforms.tensoring import Tensorize

TRANSFORMS = {
    "constant_normalize": ConstantNormalization,
    "spine_length_normalize": SpineLengthNormalization,
    "resample_3d_curve": Resample3DCurve,
    "project_to_plane": ProjectToPlane,
    "resample_point_cloud": ResamplePointCloud,
    "close_depthmap": Closing,
    "tensorize": Tensorize,
}

HASH_ARGS = {
    "data_dir",
    "transform_args",
    "num_subjects",
    "exclude_patients",
}


class SpineDataModule(pl.LightningDataModule):
    """Data module for the 3D backscan dataset of the Laboratory for
    Movement Biomechanics at D-HEST ETH Zurich. This dataset is composed
    of four different subdatasets coming from different medical centers
    that are exploring the use of 3D backscans as an alternative for the
    diagnosis and monitoring of scoliosis. To learn more about the dataset,
    please refer to the following sources:
        - Source 1
        - Source 2
        - Source 3
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        transform_args: dict,
        train_fraction: Optional[float] = None,
        val_fraction: Optional[float] = None,
        test_fraction: Optional[float] = None,
        train_keys: Optional[list] = None,
        val_keys: Optional[list] = None,
        test_keys: Optional[list] = None,
        num_subjects: Optional[int] = None,
        exclude_patients: Optional[dict] = None,
        use_cache: bool = True,
        cache_dir: str = "../../cache/",
        num_workers: int = 0,
    ):
        """Constructor for the SpineDataModule class.

        Args:
            data_dir (str): The root directory of the dataset.
            batch_size (int): The batch size for the training dataloader.
            transform_args (dict): A dictionary containing the arguments for the
                data preprocessing transforms. For more information on the
                available transforms, please refer to the README.md in the
                transforms subdirectory.
            train_fraction (Optional[float], optional): The fraction of the dataset
                to use as a training set. Defaults to None.
            val_fraction (Optional[float], optional):  The fraction of the dataset
                to use as a validation set. Defaults to None.
            test_fraction (Optional[float], optional):  The fraction of the dataset
                to use as a test set. Defaults to None.
            train_keys (Optional[list], optional): The exact keys of the samples to
                use as a training set. The keys have the format dataset_id. Defaults
                to None.
            val_keys (Optional[list], optional): The exact keys of the samples to use
                as a validation set. The keys have the format dataset_id. Defaults to
                None.
            test_keys (Optional[list], optional): The exact keys of the samples to use
                as a test set. The keys have the format dataset_id. Defaults to None.
            num_subjects (Optional[float], optional): The number of subjects to load from
                the dataset. Defaults to None, which loads all.
            exclude_patients (Optional[dict], optional): A dictionary containing the
                keys of the datasets and the ids of the patients to exclude from the
                dataset. Defaults to None.
            use_cache (bool, optional): Whether to use the cache if it exists. Defaults
                to True.
            cache_dir (str, optional): The directory to save the cache. Defaults to
            "../../cache/".
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.test_keys = test_keys
        self.transform_args = transform_args
        self.meta = {}
        self.backs = {}
        self.data = {}
        self.num_subjects = num_subjects
        self.exclude_patients = exclude_patients
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.num_workers = num_workers
        self.save_hyperparameters()

    def setup(self, stage: Optional[str]):
        """Setup method for the SpineDataModule class. This method is called
        by PyTorch Lightning to setup the data module before training.

        Args:
            stage (str | None): The stage of the training process. Can be "fit",
                "validate", "test", or None. Defaults to None.
        """
        self._check_cache()

        if self.cache_exists and self.use_cache:
            print("Cache found for this parameter combination.")
            self._load_cache()
        else:
            print("Cache not found or not used for this parameter combination.")
            self._parse_datapaths()
            self._exclude_patients()
            self._load_data()
            self._reformat_data()
            self._preprocess_data()
            self._save_cache()

        self._split_data()

    def _parse_datapaths(self):
        """Parses the data paths from the data directory.

        The data directory should contain two subdirectories: one for the backscans
        and one for the metadata. The backscans should be stored as .ply files and
        the metadata as .json files. The backscans and metadata files should have
        the same name, with the only difference being the file extension.
        """
        print(f"Parsing data paths from {self.data_dir}...")

        assert self.data_dir.exists(), "Data directory does not exist."

        back_dir = str(self.data_dir / "**" / "*processed.ply")
        meta_dir = str(self.data_dir / "**" / "*processed.json")

        # iglob is more memory-efficient than glob and allows to use tqdm
        back_iter = tqdm(glob.iglob(back_dir, recursive=True), desc="Parsing backscans")
        meta_iter = tqdm(glob.iglob(meta_dir, recursive=True), desc="Parsing metadata")

        # making every path string a Path object is useful for cross-OS compatibility
        self.dirs_back = [Path(path) for path in natsorted(back_iter)]
        self.dirs_meta = [Path(path) for path in natsorted(meta_iter)]

    def _exclude_patients(self):
        """Excludes patients from the dataset based on the provided dictionary."""
        if self.exclude_patients is None:
            return

        ex_patients_list = []
        for key in self.exclude_patients:
            for id in self.exclude_patients[key]:
                ex_patients_list.append(f"{key}_{id}")

        self.dirs_back = [
            path
            for path in self.dirs_back
            if not any(patient in str(path) for patient in ex_patients_list)
        ]
        self.dirs_meta = [
            path
            for path in self.dirs_meta
            if not any(patient in str(path) for patient in ex_patients_list)
        ]

    def _check_cache(self):
        """Checks if a cache exists for the current parameter combination."""
        print("Checking cache...")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dict = {key: self.__dict__[key] for key in HASH_ARGS}
        self.cache_file = self.cache_dir / f"{hash_dict(self.cache_dict)}.pt"
        self.cache_exists = (self.cache_file).exists()

    def _load_cache(self):
        """Loads the cache from the cache file."""
        print(f"Loading cache from {self.cache_file}...")
        self.data = torch.load(self.cache_file)

    def _load_data(self):
        """Loads both backscan point clouds and metadata from the data directories
        found in _parse_datapaths().
        """
        print("Loading data...")

        # auxiliary selection for quick testing and debugging
        self._select_n_subj_per_dataset()

        for back_path, meta_path in tqdm(
            zip(self.dirs_back, self.dirs_meta),  # noqa: B905
            total=len(self.dirs_back),
        ):
            msg = (
                f"Backscan and metadata files do not match: {back_path} and {meta_path}"
            )
            assert back_path.parent == meta_path.parent, msg

            with open(meta_path, "r") as f:
                meta_id = json.load(f)
                unique_id = f"{meta_id['dataset']}_{meta_id['id']}"
                self.meta[unique_id] = meta_id
            self.backs[unique_id] = o3d.io.read_point_cloud(str(back_path))

    def _select_n_subj_per_dataset(self):
        """Selects the first num_subjects from each dataset for quick testing and debugging."""
        DATASETS = ["balgrist", "croatian", "italian", "ukbb"]
        if self.num_subjects is not None:
            back_paths = []
            meta_paths = []
            for dataset in DATASETS:
                back_paths += [path for path in self.dirs_back if dataset in str(path)][
                    : self.num_subjects
                ]
                meta_paths += [path for path in self.dirs_meta if dataset in str(path)][
                    : self.num_subjects
                ]
            self.dirs_back = back_paths
            self.dirs_meta = meta_paths

    def _reformat_data(self):
        """Reformats the data into a more convenient structure for
        the rest of the pipeline.

        Each sample is stored as a dictionary with the following keys:
        - backscan: the backscan point cloud
        - dataset: the dataset the sample belongs to
        - id: the unique identifier of the sample
        - age: the age of the patient
        - gender: the gender of the patient
        - status: the status of the patient
        - pipeline_steps: the pipeline steps applied to the backscan
        - special_points: the special points of the backscan
        - esl: the external spinal line
        - isl: the internal spinal line

        All samples are stored in self.data as a dictionary with the
        unique_id as the key.
        """
        print("Reformatting data...")
        for unique_id in tqdm(self.meta.keys()):
            self.data[unique_id] = {}
            self.data[unique_id]["backscan"] = self.backs[unique_id]
            self.data[unique_id]["dataset"] = self.meta[unique_id]["dataset"]
            self.data[unique_id]["id"] = self.meta[unique_id]["id"]
            self.data[unique_id]["age"] = self.meta[unique_id]["age"]
            self.data[unique_id]["gender"] = self.meta[unique_id]["gender"]
            self.data[unique_id]["status"] = self.meta[unique_id]["status"]

            # change from camelCase to snake_case for python conventions
            self.data[unique_id]["pipeline_steps"] = self.meta[unique_id][
                "pipelineSteps"
            ].copy()
            self.data[unique_id]["special_points"] = self.meta[unique_id][
                "specialPts"
            ].copy()

            for point_id, point in self.data[unique_id]["special_points"].items():
                self.data[unique_id]["special_points"][point_id] = np.asarray(point)

            if self.data[unique_id]["dataset"] == "croatian":
                pc_type = "formetric"
            else:
                pc_type = "pcdicomapp"

            self.data[unique_id]["esl"] = np.asarray(
                self.meta[unique_id]["esl"][pc_type]
            )
            self.data[unique_id]["isl"] = np.asarray(
                self.meta[unique_id]["isl"][pc_type]
            )

    def _preprocess_data(self):
        """Preprocesses the data using the transforms provided
        in the transform_args dictionary.

        The transforms are applied order according to the transform_number
        key in the transform_args dictionary. The transforms are then composed
        into a single transform using torchvision.transforms.Compose.
        """
        print("Preprocessing data...")
        transforms = []
        for i in range(len(self.transform_args)):
            for key, value in self.transform_args.items():
                if value["transform_number"] == i:
                    transforms.append(TRANSFORMS[key](**value))
                    print(f"{key}")
                    print("---------------------------")
                    for k, v in value.items():
                        print(f"{k}: {v}")
                    print("\n")

        self.transforms = v2.Compose(transforms)
        for unique_id in tqdm(self.data.keys()):
            self.data[unique_id] = self.transforms(self.data[unique_id])

    def _save_cache(self):
        """Saves the data to a cache file for future use. The cache file
        is saved in the cache directory with the name being the hash of
        the cache_dict.

        The cache_dict is a dictionary containing the following keys:
        - data_dir: the root directory of the dataset
        - transform_args: the arguments for the data preprocessing transforms
        - num_subjects: the number of subjects to load from the dataset

        The cache file is saved using torch.save as a .pt file.
        """
        print(f"Saving cache to {self.cache_file}...")
        try:
            torch.save(self.data, self.cache_file)
            with open(self.cache_file.with_suffix(".json"), "w") as f:
                json.dump(self.cache_dict, f, default=dumper)
            print("Saved!")
        except TypeError:
            print("Error saving cache!")

    def _split_data(self):
        """Splits the data into training, validation, and test sets according
        to the provided fractions or keys. The data is split into three SpineDataset
        objects: train_data, val_data, and test_data.

        The train_data, val_data, and test_data objects are then used to create the
        training, validation, and test dataloaders, respectively.
        """
        print("Splitting data...")
        self._check_split_args()
        self.train_data = self._compose_dataset(self.train_keys)
        self.val_data = self._compose_dataset(self.val_keys)
        self.test_data = self._compose_dataset(self.test_keys)

    def _check_split_args(self):
        """Checks the arguments for splitting the data into training, validation,
        and test sets. In case fractions are provided, the data is split randomly.

        Raises:
            AssertionError: If both fractions and keys are provided, or if the
                sum of the fractions is not equal to 1.
        """
        fraction_args = [self.train_fraction, self.val_fraction, self.test_fraction]
        keys_args = [self.train_keys, self.val_keys, self.test_keys]

        fraction_bool = np.all([fraction is not None for fraction in fraction_args])
        keys_bool = np.all([keys is not None for keys in keys_args])

        msg = "Either fractions or keys should be provided, not both."
        assert np.logical_xor(fraction_bool, keys_bool), msg

        if fraction_bool:
            msg = "Sum of fractions should be equal to 1."
            assert np.sum(fraction_args) == 1, msg

            data_keys = list(self.data.keys())
            self.train_keys = list(
                np.random.choice(
                    data_keys,
                    size=int(self.train_fraction * len(data_keys)),
                    replace=False,
                )
            )
            self.val_keys = list(
                np.random.choice(
                    np.setdiff1d(data_keys, self.train_keys),
                    size=int(self.val_fraction * len(data_keys)),
                    replace=False,
                )
            )
            self.test_keys = list(
                np.setdiff1d(
                    data_keys,
                    np.concatenate([self.train_keys, self.val_keys]),
                )
            )

    def _compose_dataset(self, keys: list):
        """Composes a dataset from the provided keys.

        Args:
            keys (list): The keys of the samples to include in the dataset.

        Returns:
            dataset (SpineDataset): The composed dataset.
        """
        if "project_to_plane" in self.transform_args:
            input_key = "depth_map"
        else:
            input_key = "backscan"

        inputs, esls, isls = [], [], []

        for key in keys:
            inputs.append(self.data[key][input_key])
            esls.append(self.data[key]["esl"])
            isls.append(self.data[key]["isl"])

        return TensorDataset(
            torch.tensor(np.stack(inputs)),
            torch.tensor(np.stack(esls)),
            torch.tensor(np.stack(isls)),
        )

    def train_dataloader(self):
        """Creates the training dataloader.

        Returns:
            train_dataloader (torch.utils.data.DataLoader): The training dataloader.
        """
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Creates the validation dataloader.

        Returns:
            val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        """
        return DataLoader(
            self.val_data,
            batch_size=len(self.val_data),
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """Creates the test dataloader.

        Returns:
            test_data_loader (torch.utils.data.DataLoader): The test dataloader.
        """
        return DataLoader(
            self.test_data,
            batch_size=len(self.test_data),
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
