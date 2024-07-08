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


class Resample3DCurve(nn.Module):
    """Resample a 3D curve using cubic spline interpolation."""

    def __init__(self, n_points: float, transform_number: int):
        """Initialize the Resample3DCurve class.

        Args:
            n_points (float): The number of points to resample the curve to.
            transform_number (int): The ordinal position in which the transform
            will be applied to the data during preprocessing.
        """
        super().__init__()
        self.n_points = n_points
        self.transform_number = transform_number

    def _remove_duplicate_points(self, curve: npt.NDArray) -> npt.NDArray:
        """Remove duplicate points from a 3D curve.

        Args:
            curve (npt.NDArray): A 3D curve.

        Returns:
            npt.NDArray: The 3D curve with duplicate points removed.
        """
        curve = np.round(curve, 5)
        curve = np.unique(curve, axis=0)
        return curve

    def resample_curve(self, curve: npt.NDArray) -> npt.NDArray:
        """Resample a 3D curve using cubic spline interpolation.

        Args:
            curve (npt.NDArray): A 3D curve to resample.

        Returns:
            npt.NDArray: The resampled 3D curve.
        """
        curve = self._remove_duplicate_points(curve)
        tck, u = splprep(curve.T, s=0)
        u_new = np.linspace(0, 1, self.n_points)
        curve_new = splev(u_new, tck)
        return np.vstack(curve_new).T

    def forward(self, data_id: dict) -> dict:
        """Resample the "esl" and "isl" curves in the
        data_id dictionary.

        Args:
            data_id (dict): A single sample of the data to resample.

        Returns:
            data_id (dict): The resampled data sample.
        """
        data_id["isl"] = self.resample_curve(data_id["isl"])
        data_id["esl"] = self.resample_curve(data_id["esl"])
        return data_id


class ResamplePointCloud(nn.Module):
    """Resample a point cloud using Poisson disk sampling or uniform sampling."""

    def __init__(self, n_points: int, method: str, transform_number: int):
        """Initialize the ResamplePointCloud class.

        Args:
            n_points (int): The number of points to resample the point cloud to.
            method (str): The method to use for resampling. Must be either
            'poisson' or 'uniform'.
            transform_number (int): The ordinal position in which the transform
            will be applied to the data during preprocessing.
        """
        super().__init__()
        self.n_points = n_points
        self.method = method
        self.transform_number = transform_number

    def forward(self, data_id: dict) -> dict:
        """Resample the "backscan" point cloud in the data_id dictionary.

        Args:
            data_id (dict): A single sample of the data to resample.

        Raises:
            ValueError: If the method is not 'poisson' or 'uniform'.

        Returns:
            dict: The resampled data sample.
        """
        pcd = data_id["backscan"]

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

        data_id["backscan"] = pcd_sampled

        return data_id


class ConstantNormalization(nn.Module):
    """Normalizes all geometric data by a user-defined constant."""

    def __init__(self, norm: float, transform_number: int):
        """Initializes the ConstantNormalization class.

        Args:
            norm (float): The constant by which to normalize the data.
            transform_number (int): The ordinal position in which the transform
            will be applied to the data during preprocessing.
        """
        super().__init__()
        self.norm = norm
        self.transform_number = transform_number

    def forward(self, data_id: dict) -> dict:
        """Normalizes the data by the user-defined constant.

        Args:
            data_id (dict): A single sample of the data to normalize.
            Must have the keys 'backscan', 'esl', 'isl', and 'fix_points'
            which store geometric data corresponding to point clouds (or lines).

        Returns:
            data_id (dict): The normalized data sample.
        """
        data_id["backscan"].points = o3d.utility.Vector3dVector(
            np.asarray(data_id["backscan"].points) / self.norm
        )
        data_id["esl"] = data_id["esl"] / self.norm
        data_id["isl"] = data_id["isl"] / self.norm
        for point_id, point in data_id["special_points"].items():
            data_id["special_points"][point_id] = point / self.norm

        return data_id


class SpineLengthNormalization(nn.Module):
    """Normalizes all geometric data per sample so that the spine length defined
    as the distance from C7 (7th cervical vertebrae) to DM (the dimple middle) is
    equal to some length (usually 1). DM is defined as the middle point between the
    two posterior superior iliac spines (PSIS) which are called DR and DL.
    """

    def __init__(self, norm_length: float, transform_number: int):
        """Initializes the SpineLengthNormalization class.

        Args:
            norm_length (float): The length to normalize the spine to.
            transform_number (int): The ordinal position in which the transform
            will be applied to the data during preprocessing.
        """
        super().__init__()
        self.norm_length = norm_length
        self.transform_number = transform_number

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
        transform_number: int,
        z_lims: Optional[list] = None,
        **kwargs,
    ):
        """Initialize the ProjectToPlane class.

        Args:
            height (int): The height of the depth map.
            width (int): The width of the depth map.
            intensity (float): The maximum intensity value to use for the depth map.
            method (str): The method to use for aggregating the z values of the points
            assigned to the same pixel. Should be either 'mean' or 'median'.
            transform_number (int): The ordinal position in which the transform
            will be applied to the data during preprocessing.
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
        self.transform_number = transform_number
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

            for start, count in zip(idx_start, counts):  # noqa: B905
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


class Closing(nn.Module):
    """Close the holes in a depth_map through a morphological closing
    operation."""

    def __init__(self, se_size: (tuple, list), transform_number: int):
        """Initialize the Closing class.

        Args:
            se_size (tuple, list): The size of the structuring element
            to use for the closing operation.
            transform_number (int): The ordinal position in which the transform
            will be applied to the data during preprocessing.
        """
        super().__init__()
        self.se_size = se_size
        self.transform_number = transform_number

    def forward(self, data_id: dict) -> dict:
        """Close the holes in the depth map.

        Args:
            data_id (dict): A single sample of the data to close.

        Returns:
            data_id (dict): The data sample with the holes closed.
        """
        assert "depth_map" in data_id, "The depth map is missing from the data sample."

        structuring_element = np.ones(self.se_size)
        depth_map = data_id["depth_map"]

        data_id["depth_map"] = grey_closing(depth_map, structure=structuring_element)

        return data_id


class Tensorize(nn.Module):
    """Converts data to tensors."""

    def __init__(self, transform_number: int):
        """Initializes the Tensorize class.

        Args:
            transform_number (int): The ordinal position in which the transform
            will be applied to the data during preprocessing.
        """
        super().__init__()
        self.transform_number = transform_number

    def forward(self, data_id: dict) -> dict:
        """Converts the data to tensors.

        Args:
            data_id (dict): A single sample of the data to tensorize.

        Returns:
            data_id (dict): The tensorized data sample.
        """
        data_id["backscan"] = torch.tensor(np.asarray(data_id["backscan"].points))
        data_id["esl"] = torch.tensor(data_id["esl"])
        data_id["isl"] = torch.tensor(data_id["isl"])
        if "depth_map" in data_id:
            data_id["depth_map"] = torch.tensor(data_id["depth_map"].copy()).unsqueeze(
                0
            )

        return data_id
