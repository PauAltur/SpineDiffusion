import glob
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import open3d as o3d
import pandas as pd
import pytorch_lightning as pl
import torch
from natsort import natsorted
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2
from tqdm import tqdm

from spinediffusion.data.dataclass import SpineSample
from spinediffusion.data.sl_generator import SLGenerator
from spinediffusion.data.transforms import (
    Closing,
    ConstantNormalization,
    ProjectToPlane,
    RandomRotationAugmentation,
    Resample3DCurve,
    ResamplePointCloud,
    SpineLengthNormalization,
    Tensorize,
)
from spinediffusion.utils.hashing import hash_dict, hash_dict_keys
from spinediffusion.utils.misc import dumper

TRANSFORMS = {
    "constant_normalize": ConstantNormalization,
    "spine_length_normalize": SpineLengthNormalization,
    "resample_3d_curve": Resample3DCurve,
    "project_to_plane": ProjectToPlane,
    "resample_point_cloud": ResamplePointCloud,
    "close_depthmap": Closing,
    "tensorize": Tensorize,
    "random_rotation": RandomRotationAugmentation,
}

HASH_ARGS = {
    "data_dir",
    "transform_args",
    "augment_args",
    "num_subjects",
    "exclude_patients",
    "sl_args",
}


class SpineDataModule(pl.LightningDataModule):
    """Data module for the 3D backscan dataset of the Laboratory for
    Movement Biomechanics at D-HEST ETH Zurich. This dataset is composed
    of four different subdatasets coming from different medical centers
    that are exploring the use of 3D backscans as an alternative for the
    diagnosis and monitoring of scoliosis. To learn more about the dataset,
    please refer to the following sources:
        - Implementation and Investigation of Machine Learning-based Spinal
        Curvature Estimation Models, Master Thesis, Sandhu, 2023.
        - Estimating the Spinal Curvature from Back Surface Scans using Deep
        Learning, Master Thesis, Ruegg, 2023.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        transform_args: dict,
        augment_args: Optional[dict] = None,
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
        cache_size: int = np.inf,
        num_workers: int = 0,
        predict_size: int = 1,
        sl_args: Optional[dict] = None,
    ):
        """Constructor for the SpineDataModule class.

        Args:
            data_dir (str): The root directory of the dataset.
            batch_size (int): The batch size for the training dataloader.
            transform_args (dict): A dictionary containing the arguments for the
                data preprocessing transforms. For more information on the
                available transforms, please refer to the README.md in the
                transforms subdirectory.
            augment_args (Optional[dict], optional): A dictionary containing the
                arguments for the data augmentation transforms. Defaults to None.
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
            cache_size (int, optional): The number of samples to save per cache file. This
                is useful for large datasets that do not fit in memory. Defaults to np.inf.
            num_workers (int, optional): The number of workers to use for the dataloaders.
                Defaults to 0.
            predict_size (int, optional): The number of samples to generate for the
                predict dataloader. Defaults to 1.
            sl_args (Optional[dict], optional): The arguments for the SLGenerator class. It also
            acts as the control variable on whether the dataset is conditional (i.e. inputs
            to the UNet model are composed of noise and a prior, which in our case is the ISL
            line, but it could also be a segmentation mask or any other informations).
            Defaults to None.
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
        self.augment_args = augment_args
        self.meta = {}
        self.backs = {}
        self.data = {}
        self.num_subjects = num_subjects
        self.exclude_patients = exclude_patients
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_size = cache_size
        self.num_workers = num_workers
        self.predict_size = predict_size
        self.sl_args = sl_args
        self.save_hyperparameters()

    def prepare_data(self):
        """Method for preparing the data. This method is called by PyTorch Lightning
        before the data module is set up. It is used to load the data augment it and
        preprocess it before saving it to disk (optionally in mini-batches).
        """
        cache_hash = hash_dict({key: self.__dict__[key] for key in HASH_ARGS})
        cache_folder = self.cache_dir / f"{cache_hash}"

        if cache_folder.exists() and self.use_cache:
            print("Using existing cache for this parameter combination.")
            pass
        else:
            if self.use_cache:
                print("Cache not found for this parameter combination.")
            elif not self.use_cache:
                print("Reprocessing data since use_cache attribute is set to False.")

            data = self._load_data()
            self._augment_data(data, cache_folder)
            self._preprocess_data(cache_folder)

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
            self._augment_data()
            self._preprocess_data()
            self._save_cache()

        self._split_data()

    def _load_data(self) -> Dict:
        """Loads both backscan point clouds and metadata from the data directories
        found in _parse_datapaths().

        Returns:
            data (dict): A dictionary containing the loaded data.
        """
        print(f"Parsing data paths from {self.data_dir}...")
        msg = f"Data directory {self.data_dir} does not exist or is inaccessible."
        assert self.data_dir.exists(), msg

        sample_dir = str(self.data_dir / "**")
        sample_iter = tqdm(glob.iglob(sample_dir), desc="Parsing sample directories")
        sample_dirs = [Path(path) for path in natsorted(sample_iter)]

        if self.num_subjects is not None:
            datasets = [path.name.split("_")[0] for path in sample_dirs]
            sample_dirs_df = pd.DataFrame({"path": sample_dirs, "dataset": datasets})
            sample_dirs = (
                sample_dirs_df.groupby("dataset")
                .head(self.num_subjects)["path"]
                .tolist()
            )

        print(f"Loading data from {self.data_dir}...")
        data = {}
        for sample in tqdm(sample_dirs, desc="Loading data"):

            back_path = sample / f"{sample.name}_processed.ply"
            meta_path = sample / f"{sample.name}_metadata_processed.json"

            with open(meta_path, "r") as f:
                meta_sample = json.load(f)
            back_sample = o3d.io.read_point_cloud(str(back_path))

            if "croatian" in sample.name:
                pc_type = "formetric"
            else:
                pc_type = "pcdicomapp"

            data[sample.name] = SpineSample(
                backscan=back_sample,
                special_points={
                    k: np.asarray(v) for k, v in meta_sample["specialPts"].items()
                },
                esl=np.asarray(meta_sample["esl"][pc_type]),
                isl=np.asarray(meta_sample["isl"][pc_type]),
            )

        return data

    def _augment_data(self, data, cache_folder):
        """Augments the data using the augment_args dictionary.

        Args:
            data (dict): The data to augment.
            cache_folder (Path): The folder to save the augmented data.
        """
        if self.augment_args is None:
            self._save_minibatch(data, cache_folder)
            return

        augmentations = []

        for i in range(len(self.augment_args)):
            for key, value in self.augment_args.items():
                if value["transform_number"] == i:
                    augmentations.append(TRANSFORMS[key](**value))
                    print(f"{key}")
                    print("---------------------------")
                    for k, v in value.items():
                        print(f"{k}: {v}")
                    print("\n")

        data_aug = {}
        aug_count_global = 0
        for unique_id, data_id in tqdm(data.items(), desc="Augmenting data"):
            aug_count_id = 0
            for augmentation in augmentations:
                for _ in range(augmentation.num_aug):
                    aug_id = f"{unique_id}_{aug_count_id}"
                    data_aug[aug_id] = augmentation(data_id)
                    # change to make data_aug serializable
                    data_aug[aug_id].backscan = np.asarray(
                        data_aug[aug_id].backscan.points
                    )
                    aug_count_id += 1
                    aug_count_global += 1

                    if aug_count_global >= self.cache_size:
                        self._save_minibatch(data_aug, cache_folder)
                        data_aug = {}
                        aug_count_global = 0

        if len(data_aug) > 0:
            self._save_minibatch(data_aug, cache_folder)

    def _save_minibatch(self, minibatch, cache_folder):
        """Saves a minibatch of data to a cache file. The cache file is saved
        in the cache directory with the name being the hash of the minibatch.

        Args:
            minibatch (dict): The minibatch of data to save.
        """
        cache_hash = hash_dict_keys(minibatch)
        cache_folder.mkdir(parents=True, exist_ok=True)
        cache_file = cache_folder / f"{cache_hash}.pt"
        print(f"Saving minibatch to {cache_file}...")
        torch.save(minibatch, cache_file)

    def _preprocess_data(self, cache_folder):
        """Preprocesses the data using the transforms provided
        in the transform_args dictionary.

        The transforms are applied order according to the transform_number
        key in the transform_args dictionary. The transforms are then composed
        into a single transform using torchvision.transforms.Compose.
        """
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

        minibatches = glob.glob(str(cache_folder / "*.pt"))

        for minibatch in tqdm(minibatches, desc="Preprocessing minibatches"):
            data = torch.load(minibatch)
            for unique_id, data_id in tqdm(
                data.items(), desc="Preprocessing samples in minibatch"
            ):
                points = o3d.utility.Vector3dVector(data_id.backscan)
                data_id.backscan = o3d.geometry.PointCloud(points=points)
                self.transforms(data_id)

            self._save_minibatch(data, cache_folder)

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
        self.predict_data = self._compose_predict_dataset()

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
            if self.sl_args is not None:
                isl_key = "isl_depth_map"
                esl_key = "esl_depth_map"
        else:
            input_key = "backscan"
            if self.sl_args is not None:
                isl_key = "isl"
                esl_key = "esl"

        inputs, esls, isls = [], [], []

        for key in keys:
            inputs.append(self.data[key][input_key])
            if self.sl_args is not None:
                esls.append(self.data[key][esl_key])
                isls.append(self.data[key][isl_key])

        if self.sl_args is not None:
            return TensorDataset(
                torch.tensor(np.stack(inputs)),
                torch.tensor(np.expand_dims(np.stack(esls), 1)),
                torch.tensor(np.expand_dims(np.stack(isls), 1)),
            )
        else:
            return TensorDataset(
                torch.tensor(np.stack(inputs)),
            )

    def _compose_predict_dataset(self):
        """Composes the predict dataset.

        Returns:
            predict_data (torch.utils.data.TensorDataset) : The predict
            dataset.
        """
        if self.sl_args is not None:
            noise = torch.randn(
                self.predict_size,
                *self.train_data[0][0].shape,
                dtype=torch.float32,
            )

            sl_generator = SLGenerator(**self.sl_args)
            generated_sl = torch.concatenate(
                [
                    torch.tensor(
                        next(sl_generator).copy(),
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    for _ in range(self.predict_size)
                ]
            )

            predict_dataset = TensorDataset(
                noise, torch.zeros(noise.size()), generated_sl
            )

        else:
            predict_dataset = TensorDataset(
                torch.randn(
                    (self.predict_size, 1, *self.train_data[0][0].shape[1:]),
                    dtype=torch.float32,
                )
            )

        return predict_dataset

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
            batch_size=self.batch_size,
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
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        """Creates the predict dataloader.

        Returns:
            predict_dataloader (torch.utils.data.DataLoader): The predict dataloader.
        """
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
