from collections import defaultdict

import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch.nn as nn


class ProjectToPlane(nn.Module):
    """Project a point cloud to a plane and create a depth map."""

    def __init__(self, height: int, width: int, intensity: float, **kwargs):
        """Initialize the ProjectToPlane class.

        Args:
            height (int): The height of the depth map.
            width (int): The width of the depth map.
            intensity (float): The maximum intensity value to use for the depth map.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.intensity = intensity

    def _scale_point_cloud(self, pc: npt.NDArray) -> npt.NDArray:
        """Scale the point cloud to the desired dimensions.

        Args:
            pc (np.ndarray): The point cloud to scale.

        Returns:
            np.ndarray: The scaled point cloud.
        """
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)

        x = (self.width - 1) * (x - x_min) / (x_max - x_min)
        y = (self.height - 1) * (y - y_min) / (y_max - y_min)
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

        h_axis = np.arange(self.height)
        w_axis = np.arange(self.width)

        # -1 to make the bins start from 0
        x_bins = np.digitize(x, w_axis) - 1
        y_bins = np.digitize(y, h_axis) - 1

        depth_map = np.zeros((self.height, self.width))

        hw_pixels_sum = np.zeros((self.height, self.width))
        hw_pixels_count = np.zeros((self.height, self.width))

        np.add.at(hw_pixels_sum, (y_bins, x_bins), z)
        np.add.at(hw_pixels_count, (y_bins, x_bins), 1)

        # Calculate the mean where count is not zero
        nonzero_mask = hw_pixels_count > 0
        depth_map[nonzero_mask] = (
            hw_pixels_sum[nonzero_mask] / hw_pixels_count[nonzero_mask]
        )

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
        pc = self._scale_point_cloud(pc)
        data_id["depth_map"] = np.flip(self._compute_depth_map(pc), axis=0)
        return data_id
