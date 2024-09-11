from collections import defaultdict
from typing import Optional

import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch
import torch.nn as nn
from scipy.interpolate import splev, splprep
from scipy.ndimage import grey_closing
from scipy.spatial.distance import euclidean

from spinediffusion.data.dataclass import SpineSample


class RandomRotationAugmentation(nn.Module):
    """Randomly rotates the data by a random angle within a user-defined range."""

    def __init__(self, theta_range: (list, tuple), num_aug: int, **kwargs):
        """Initializes the RandomRotationAugmentation class.

        Args:
            theta_range ((list, tuple)): The range of angles to rotate the data by.
                The range is defined as [theta_min, theta_max] and should be provided
                in radians.
        """
        super().__init__()
        self.theta_range = theta_range
        self.num_aug = num_aug

    def forward(self, data_id: dict) -> dict:
        """Rotates the data by a random angle within the user-defined range.

        Args:
            data_id (dict): A single sample of the data to augment.
                Must have the key 'backscan' and optionally 'esl' and 'isl'
                which store geometric data corresponding to point clouds (or lines).

        Returns:
            data_id (dict): The augmented data sample.
        """

        data_id_aug = SpineSample()

        axis = np.random.randint(0, 3)
        theta = (
            self.theta_range[1] - self.theta_range[0]
        ) * np.random.random() + self.theta_range[0]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        if axis == 0:
            R = np.array(
                [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
            )
        elif axis == 1:
            R = np.array(
                [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
            )
        elif axis == 2:
            R = np.array(
                [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
            )

        aug_backscan = o3d.geometry.PointCloud()
        aug_backscan.points = o3d.utility.Vector3dVector(
            (R @ np.asarray(data_id.backscan.points).T).T
        )

        data_id_aug.backscan = aug_backscan
        if data_id.esl is not None:
            data_id_aug.esl = (R @ data_id.esl.T).T
        if data_id.isl is not None:
            data_id_aug.isl = (R @ data_id.isl.T).T
        if data_id.special_points is not None:
            data_id_aug.special_points = {}
            for point_key, point in data_id.special_points.items():
                # check whether point is empty
                if len(point) == 0:
                    data_id_aug.special_points[point_key] = point
                else:
                    data_id_aug.special_points[point_key] = (R @ point.T).T

        return data_id_aug


class Closing(nn.Module):
    """Close the holes in a depth_map through a morphological closing
    operation."""

    def __init__(self, se_size: (tuple, list), **kwargs):
        """Initialize the Closing class.

        Args:
            se_size (tuple, list): The size of the structuring element
            to use for the closing operation.
        """
        super().__init__()
        self.se_size = se_size

    def forward(self, data_id: dict) -> dict:
        """Close the holes in the depth map.

        Args:
            data_id (dict): A single sample of the data to close.

        Returns:
            data_id (dict): The data sample with the holes closed.
        """
        assert (
            data_id.backscan_proj is not None
        ), "The backscan projection is missing from the data sample."

        structuring_element = np.ones(self.se_size)
        data_id.backscan_proj = grey_closing(
            data_id.backscan_proj, structure=structuring_element
        )

        return data_id


class ConstantNormalization(nn.Module):
    """Normalizes all geometric data by a user-defined constant."""

    def __init__(self, norm: float, **kwargs):
        """Initializes the ConstantNormalization class.

        Args:
            norm (float): The constant by which to normalize the data.
        """
        super().__init__()
        self.norm = norm

    def forward(self, data_id: dict) -> dict:
        """Normalizes the data by the user-defined constant.

        Args:
            data_id (dict): A single sample of the data to normalize.
            Must have the keys 'backscan', 'esl', 'isl', and 'fix_points'
            which store geometric data corresponding to point clouds (or lines).

        Returns:
            data_id (dict): The normalized data sample.
        """
        data_id.backscan.points = o3d.utility.Vector3dVector(
            np.asarray(data_id.backscan.points) / self.norm
        )
        data_id.esl = data_id.esl / self.norm
        data_id.isl = data_id.isl / self.norm
        for point_id, point in data_id.special_points.items():
            data_id.special_points[point_id] = point / self.norm

        return data_id


class SpineLengthNormalization(nn.Module):
    """Normalizes all geometric data per sample so that the spine length defined
    as the distance from C7 (7th cervical vertebrae) to DM (the dimple middle) is
    equal to some length (usually 1). DM is defined as the middle point between the
    two posterior superior iliac spines (PSIS) which are called DR and DL.
    """

    def __init__(self, norm_length: float, **kwargs):
        """Initializes the SpineLengthNormalization class.

        Args:
            norm_length (float): The length to normalize the spine to.
        """
        super().__init__()
        self.norm_length = norm_length

    def forward(self, data_id: dict) -> dict:
        """Normalizes the data so that the length of the spine is equal
        to the user-defined norm_length.

        Args:
            data_id (dict): A single sample of the data to normalize.
            Must have the keys 'backscan', 'esl', 'isl', and 'fix_points'
            which store geometric data corresponding to point clouds (or lines).

        Returns:
            data_id (dict): The normalized data sample.
        """
        # Compute DM
        C7 = data_id.special_points["C7"]
        DR = data_id.special_points["DR"]
        DL = data_id.special_points["DL"]
        DM = (DR + DL) / 2

        # Compute normalization constant
        spine_length = euclidean(C7, DM)
        norm = spine_length / self.norm_length

        # Normalize the data
        data_id.backscan.points = o3d.utility.Vector3dVector(
            np.asarray(data_id.backscan.points) / norm
        )
        data_id.esl = data_id.esl / norm
        data_id.isl = data_id.isl / norm
        for point_id, point in data_id.special_points.items():
            data_id.special_points[point_id] = point / norm

        return data_id


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
            spine_factor (float): The factor to scale the point cloud along the spine.
            intensity (float): The maximum intensity value to use for the depth map.
            method (str): The method to use for aggregating the z values of the points
                that fall into the same pixel. Should be either 'mean' or 'median'.
            z_lims (list, optional): The minimum and maximum z values to use for scaling
                the z values of the point cloud. If None, the minimum and maximum z values
                of the point cloud will be used. Defaults to None.
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
        backscan_pc = np.asarray(data_id.backscan.points)
        special_points = data_id.special_points

        backscan_pc = self._scale_point_cloud(backscan_pc, special_points)

        data_id.backscan_proj = np.flip(self._compute_depth_map(backscan_pc), axis=0)

        if data_id.esl is not None:
            esl_pc = self._scale_point_cloud(data_id.esl, special_points)
            data_id.esl_proj = np.flip(self._compute_depth_map(esl_pc), axis=0)

        if data_id.isl is not None:
            isl_pc = self._scale_point_cloud(data_id.isl, special_points)
            data_id.isl_proj = np.flip(self._compute_depth_map(isl_pc), axis=0)

        return data_id


class Resample3DCurve(nn.Module):
    """Resample a 3D curve using linear interpolation."""

    def __init__(self, n_points: int, **kwargs):
        """
        Initialize the Resample3DCurve class.

        Args:
            n_points (int) : The number of points to resample the curve to.
        """
        super().__init__()
        self.n_points = n_points

    def _resample_curve(self, curve: npt.NDArray) -> npt.NDArray:
        """
        Resample the 3D curve to a specified number of points.

        Args:
            curve (np.ndarray) : The input 3D curve of shape (N, 3).

        Returns:
            resampled_curve (np.ndarray) : The resampled 3D curve of shape (n_points, 3).
        """
        # Number of points in the original curve
        N = curve.shape[0]

        # Calculate the cumulative distance along the curve
        cum_dists = np.zeros(N)
        for i in range(1, N):
            cum_dists[i] = cum_dists[i - 1] + np.linalg.norm(curve[i] - curve[i - 1])

        # Normalize the distances
        cum_dists /= cum_dists[-1]

        # Generate the new set of points along the normalized distance
        new_distances = np.linspace(0, 1, self.n_points)

        # Interpolate the curve
        resampled_curve = np.zeros((self.n_points, 3))
        for i in range(3):  # Interpolate for each dimension
            resampled_curve[:, i] = np.interp(new_distances, cum_dists, curve[:, i])

        return resampled_curve

    def forward(self, data_id: dict) -> dict:
        """
        Resample the "esl" and "isl" curves in the data_id dictionary.

        Args:
            data_id (dict): A single sample of the data to resample.

        Returns:
            data_id (dict): The resampled data sample.
        """
        data_id.isl = self._resample_curve(data_id.isl)
        data_id.esl = self._resample_curve(data_id.esl)
        return data_id


class ResamplePointCloud(nn.Module):
    """Resample a point cloud using Poisson disk sampling or uniform sampling."""

    def __init__(self, n_points: int, method: str, **kwargs):
        """Initialize the ResamplePointCloud class.

        Args:
            n_points (int): The number of points to resample the point cloud to.
            method (str): The method to use for resampling. Must be either
            'poisson' or 'uniform'.
        """
        super().__init__()
        self.n_points = n_points
        self.method = method

    def forward(self, data_id: dict) -> dict:
        """Resample the "backscan" point cloud in the data_id dictionary.

        Args:
            data_id (dict): A single sample of the data to resample.

        Raises:
            ValueError: If the method is not 'poisson' or 'uniform'.

        Returns:
            dict: The resampled data sample.
        """
        pcd = data_id.backscan

        pcd.estimate_normals()

        # estimate radius for rolling ball
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 1.5])
        )

        # resample mesh, output is a point cloud
        if self.method == "poisson":
            pcd_sampled = mesh.sample_points_poisson_disk(
                number_of_points=self.n_points
            )
        elif self.method == "uniform":
            pcd_sampled = mesh.sample_points_uniformly(number_of_points=self.n_points)
        else:
            msg = (
                f"Method {self.method} is not supported."
                " Please use 'poisson' or 'uniform'."
            )
            raise ValueError(msg)

        data_id.backscan = pcd_sampled

        return data_id


class Tensorize(nn.Module):
    """Converts data to tensors."""

    def __init__(self, **kwargs):
        """Initializes the Tensorize class."""
        super().__init__()

    def forward(self, data_id: dict) -> dict:
        """Converts the data to tensors.

        Args:
            data_id (dict): A single sample of the data to tensorize.

        Returns:
            data_id (dict): The tensorized data sample.
        """
        data_id.backscan = torch.tensor(np.asarray(data_id.backscan.points))
        if data_id.esl is not None:
            data_id.esl = torch.tensor(data_id.esl)
        if data_id.isl is not None:
            data_id.isl = torch.tensor(data_id.isl)
        if data_id.backscan_proj is not None:
            data_id.backscan_proj = torch.tensor(
                data_id.backscan_proj.copy()
            ).unsqueeze(0)
        if data_id.esl_proj is not None:
            data_id.esl_proj = torch.tensor(data_id.esl_proj.copy()).unsqueeze(0)
        if data_id.isl_proj is not None:
            data_id.isl_proj = torch.tensor(data_id.isl_proj.copy()).unsqueeze(0)

        return data_id
