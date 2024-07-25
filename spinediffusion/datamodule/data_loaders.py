import glob
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader
from tqdm import tqdm

from spinediffusion.datamodule.dataclass import SpineSample


class SpineDataloader(DataLoader):
    """DataLoader for spine data. It assumes that the data is stored inside of a folder
    data_dir, which contains a number of subfolders, each named with a unique_id. Each
    subfolder should contain a processed point cloud file named unique_id_processed.ply
    and a metadata file named unique_id_metadata_processed.json."""

    def __init__(self, data_dir: str):
        """Initializes the SpineDataloader.

        Args:
            data_dir (str): The path to the directory containing the spine data.
        """
        super().__init__()
        self.data_dir = data_dir
        self.samples = []

    def _parse_data_paths(self):
        """Parses the data paths from the data directory."""
        msg = f"Data directory {self.data_dir} does not exist."
        assert Path(self.data_dir).is_dir(), msg

        self.data_paths = tqdm(
            [
                Path(path).parent
                for path in glob.iglob(self.data_dir + "/*_processed.ply")
            ],
            desc="Parsing data paths",
        )

    def _load_sample(self, data_path: Path) -> SpineSample:
        """Loads a single sample from a data path.

        Args:
            data_path (Path): The path to the data.

        Returns:
            (SpineSample): The loaded sample.
        """
        unique_id = data_path.stem

        back_path = data_path / f"{unique_id}_processed.ply"
        meta_path = data_path / f"{unique_id}_metadata_processed.json"

        backscan = o3d.io.read_point_cloud(str(back_path))

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        special_pts = metadata["specialPts"]
        for key, value in special_pts.items():
            special_pts[key] = np.asarray(value)

        sl_key = "pcdicomapp"
        if metadata["dataset"] == "croatian":
            sl_key = "formetric"

        isl = np.asarray(metadata[sl_key]["isl"])
        esl = np.asarray(metadata[sl_key]["esl"])

        return SpineSample(unique_id, backscan, special_pts, isl, esl)

    def _load_data(self):
        """Loads all of the data that conforms to the naming convention in the data directory."""
        for data_path in tqdm(self.data_paths, desc="Loading data"):
            self.samples.append(self._load_sample(data_path))

    def __len__(self):
        """Returns the number of samples in the dataloader.

        Returns:
            (int): The number of samples in the dataloader.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns the sample at the given index.

        Args:
            idx (int): The index of the sample to return.

        Returns:
            (SpineSample): The sample at the given index.
        """
        return self.samples[idx]
