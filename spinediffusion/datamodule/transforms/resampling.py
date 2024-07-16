import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch.nn as nn
from scipy.interpolate import splev, splprep

# class Resample3DCurve(nn.Module):
#     """Resample a 3D curve using cubic spline interpolation."""

#     def __init__(self, n_points: float, **kwargs):
#         """Initialize the Resample3DCurve class.

#         Args:
#             n_points (float): The number of points to resample the curve to.
#         """
#         super().__init__()
#         self.n_points = n_points

#     def _remove_duplicate_points(self, curve: npt.NDArray) -> npt.NDArray:
#         """Remove duplicate points from a 3D curve.

#         Args:
#             curve (npt.NDArray): A 3D curve.

#         Returns:
#             npt.NDArray: The 3D curve with duplicate points removed.
#         """
#         curve = np.round(curve, 5)
#         curve = np.unique(curve, axis=0)
#         return curve

#     def resample_curve(self, curve: npt.NDArray) -> npt.NDArray:
#         """Resample a 3D curve using cubic spline interpolation.

#         Args:
#             curve (npt.NDArray): A 3D curve to resample.

#         Returns:
#             npt.NDArray: The resampled 3D curve.
#         """
#         curve = self._remove_duplicate_points(curve)

#         distances = np.linalg.norm(np.diff(curve, axis=0), axis=1)
#         s = 0.1 * np.mean(distances)

#         tck, u = splprep(curve.T, s=s)
#         u_new = np.linspace(0, 1, self.n_points)
#         curve_new = splev(u_new, tck)
#         curve_resampled = np.vstack(curve_new).T

#         return curve_resampled

#     def forward(self, data_id: dict) -> dict:
#         """Resample the "esl" and "isl" curves in the
#         data_id dictionary.

#         Args:
#             data_id (dict): A single sample of the data to resample.

#         Returns:
#             data_id (dict): The resampled data sample.
#         """
#         data_id["isl"] = self.resample_curve(data_id["isl"])
#         data_id["esl"] = self.resample_curve(data_id["esl"])
#         return data_id


class Resample3DCurve(nn.Module):
    def __init__(self, n_points):
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
        data_id["isl"] = self._resample_curve(data_id["isl"])
        data_id["esl"] = self._resample_curve(data_id["esl"])
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
