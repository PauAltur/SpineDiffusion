import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch.nn as nn
from scipy.interpolate import splev, splprep


class Resample3DCurve(nn.Module):
    def __init__(self, n_points: float, **kwargs):
        super().__init__()
        self.n_points = n_points

    def resample_curve(self, curve: npt.NDArray) -> npt.NDArray:
        """Resample a 3D curve using cubic spline interpolation.

        Args:
            curve (npt.NDArray): A 3D curve to resample.

        Returns:
            npt.NDArray: The resampled 3D curve.
        """
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
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, n_points: int, method: str, **kwargs):
        """_summary_

        Args:
            n_points (int): _description_
            method (str): _description_
        """
        super().__init__()
        self.n_points = n_points
        self.method = method

    def forward(self, data_id: dict) -> dict:
        """_summary_

        Args:
            data_id (dict): _description_

        Raises:
            ValueError: _description_

        Returns:
            dict: _description_
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
            raise ValueError(
                "Invalid method for resampling point cloud. Must be either 'poisson' or 'uniform'."
            )

        data_id["backscan"] = pcd_sampled

        return data_id
