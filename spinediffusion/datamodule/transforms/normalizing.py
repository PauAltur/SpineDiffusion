import numpy as np
import open3d as o3d
import torch.nn as nn
from scipy.spatial.distance import euclidean


class ConstantNormalization(nn.Module):
    """Normalizes all geometric data by a user-defined constant."""

    def __init__(self, norm: float, **kwargs):
        """Initializes the ConstantNormalization class.

        Args:
            norm (float): The constant by which to normalize the data.
        """
        super().__init__()
        self.norm = norm

    def forward(self, data) -> dict:
        """Normalizes the data by the user-defined constant.

        Args:
            data (dict): The data to normalize. Must have the keys
            'backscan', 'esl', 'isl', and 'fix_points' which store
            geometric data corresponding to point clouds (or lines).

        Returns:
            data (dict): The normalized data.
        """
        for unique_id, data_id in data.items():
            data_id["backscan"].points = o3d.utility.Vector3dVector(
                np.asarray(data_id["backscan"].points) / self.norm
            )
            data_id["esl"] = data_id["esl"] / self.norm
            data_id["isl"] = data_id["isl"] / self.norm
            for point_id, point in data_id["special_points"].items():
                data_id["special_points"][point_id] = point / self.norm

        return data


class SpineLengthNormalization(nn.Module):
    """Normalizes all geometric data per sample so that the spine length defined
    as the distance from C7 (7th cervical vertebrae) to DM (the dimple middle) is
    equal to some length (usually 1). DM is defined as the middle point between the
    two posterior superior iliac spines (PSIS) which are called DR and DL.
    """

    def __init__(self, norm_length: float, **kwargs):
        """Initializes the SpineLengthNormalization class."""
        super().__init__()
        self.norm_length = norm_length

    def forward(self, data) -> dict:
        for unique_id, data_id in data.items():
            # Compute DM
            C7 = data_id["special_points"]["C7"]
            DR = data_id["special_points"]["DR"]
            DL = data_id["special_points"]["DL"]

            DM = (DR + DL) / 2

            # Compute normalization constant
            spine_length = euclidean(C7, DM)
            norm = spine_length / self.norm_length

            # Normalize the data
            data_id["backscan"].points = o3d.utility.Vector3dVector(
                np.asarray(data_id["backscan"].points) / norm
            )
            data_id["esl"] = data_id["esl"] / norm
            data_id["isl"] = data_id["isl"] / norm
            for point_id, point in data_id["special_points"].items():
                data_id["special_points"][point_id] = point / norm

        return data
