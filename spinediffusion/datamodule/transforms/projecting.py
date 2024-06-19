from collections import defaultdict
from typing import Optional

import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch.nn as nn


class ProjectToPlane(nn.Module):
    """Project a point cloud to a plane and create a depth map."""

    def __init__(
        self,
        height: int,
        width: int,
        spine_factor: float,
        intensity: float,
        method: str,
        z_lims: Optional[list] = None,
        **kwargs,
    ):
        """Initialize the ProjectToPlane class.

        Args:
            height (int): The height of the depth map.
            width (int): The width of the depth map.
            intensity (float): The maximum intensity value to use for the depth map.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.spine_factor = spine_factor
        self.intensity = intensity
        self.method = method
        self.z_lims = z_lims

    def _scale_point_cloud(self, pc: npt.NDArray, special_points: dict) -> npt.NDArray:
        """Scale the point cloud to the desired dimensions.

        Args:
            pc (np.ndarray): The point cloud to scale.

        Returns:
            np.ndarray: The scaled point cloud.
        """
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

        C7 = special_points["C7"]
        DM = special_points["DR"] + special_points["DL"] / 2
        spine_length = np.linalg.norm(C7 - DM)

        factor = (self.spine_factor * self.height) / spine_length

        x = factor * x
        y = factor * y

        if self.z_lims:
            z_min, z_max = self.z_lims
            z = self.intensity * (z - z_min) / (z_max - z_min)
        else:
            z_min, z_max = np.min(z), np.max(z)
            z = self.intensity * (z - z_min) / (z_max - z_min)

        return np.stack((x, y, z), axis=1)

    def _compute_depth_map(self, pc: npt.NDArray) -> dict:
        """Compute the depth map from the point cloud by
        averaging the z values of the points that fall into
        the same pixel.

        Args:
            pc (np.ndarray): The point cloud to project.

        Returns:
            np.ndarray: The depth map.
        """
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

        h_axis = np.arange(-int(self.height / 2), int(self.height / 2))
        w_axis = np.arange(-int(self.width / 2), int(self.width / 2))

        # -1 to make the bins start from 0
        x_bins = np.digitize(x, w_axis) - 1
        y_bins = np.digitize(y, h_axis) - 1

        depth_map = np.zeros((self.height, self.width))

        # Calculate the mean where count is not zero
        assert self.method in [
            "mean",
            "median",
        ], "Method should be either 'mean' or 'median'"
        if self.method == "mean":
            hw_pixels_sum = np.zeros((self.height, self.width))
            hw_pixels_count = np.zeros((self.height, self.width))

            np.add.at(hw_pixels_sum, (y_bins, x_bins), z)
            np.add.at(hw_pixels_count, (y_bins, x_bins), 1)

            nonzero_mask = hw_pixels_count > 0
            depth_map[nonzero_mask] = (
                hw_pixels_sum[nonzero_mask] / hw_pixels_count[nonzero_mask]
            )

        elif self.method == "median":
            binned_data = np.vstack((y_bins, x_bins, z)).T
            sorted_data = binned_data[np.lexsort((x_bins, y_bins))]
            _, idx_start, counts = np.unique(
                sorted_data[:, :2], axis=0, return_index=True, return_counts=True
            )

            for start, count in zip(idx_start, counts):
                z_values = sorted_data[start : start + count, 2]
                x, y = sorted_data[start, :2].astype(int)
                depth_map[x, y] = np.median(z_values)

        return depth_map

    def forward(self, data_id: dict) -> dict:
        """Project the point cloud to a plane and create a depth map.

        Args:
            data_id (dict): A single sample of the data to project.

        Returns:
            data_id (dict): The projected data sample. The key 'depth_map' will
            store the depth map.
        """
        pc = np.asarray(data_id["backscan"].points)
        special_points = data_id["special_points"]
        pc = self._scale_point_cloud(pc, special_points)
        data_id["depth_map"] = np.flip(self._compute_depth_map(pc), axis=0)
        return data_id
