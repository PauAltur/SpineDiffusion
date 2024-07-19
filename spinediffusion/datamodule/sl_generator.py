from typing import Optional

import numpy as np
import numpy.typing as npt
import open3d as o3d
from scipy.interpolate import splev, splprep
from scipy.stats import norm, uniform

from spinediffusion.datamodule.transforms.projecting import ProjectToPlane

GENERATOR_METHODS = {
    "normal": norm,
    "uniform": uniform,
}


class SLGenerator:
    """Generates a set of splines based on the provided parameters."""

    def __init__(
        self,
        sl_mean: npt.NDArray,
        sl_std: npt.NDArray,
        length: float,
        sample_method: str,
        num_spl_points: int,
        project_args: Optional[dict] = None,
    ):
        """Initialize the SLGenerator with the provided parameters.

        Args:
            sl_mean (npt.NDArray): Mean values for the spline control points.
                It has shape (3,) and contains the mean values for the x, y,
                and z dimensions.
            sl_std (npt.NDArray): Standard deviation values for the spline
                control points. It has shape (3,) and contains the standard
                deviation values for the x, y, and z dimensions.
            length (float): The length of the spine in the y dimension.
            sample_method (str): The method used to sample the control points.
                It can be either 'normal' or 'uniform'.
            num_spl_points (int): The number of points to sample along the spline.
            project_args (Optional[dict]): Arguments to pass to the ProjectToPlane
                class. Defaults to None.
        """
        self.num_spl_points = num_spl_points
        self._setup(sl_mean, sl_std, length, sample_method, project_args)

    def _setup(
        self,
        sl_mean: npt.NDArray,
        sl_std: npt.NDArray,
        length: float,
        sample_method: str,
        project_args: Optional[dict],
    ):
        """Set up the SLGenerator with the provided parameters.

        Args:
            sl_mean (npt.NDArray): Mean values for the spline control points.
                It has shape (num_control_points, 3) and contains the mean
                values for the x, y, and z dimensions. The number of control
                points is determined by the first dimension.
            sl_std (npt.NDArray): Standard deviation values for the spline
                control points. It has shape (num_control_points, 3) and contains
                the standard deviation values for the x, y, and z dimensions. The
                number of control points is determined by the first dimension.
            length (float): The length of the spine in the y dimension.
            sample_method (str): The method used to sample the control points.
                It can be either 'normal' or 'uniform' (for now).
            project_args (Optional[dict]): Arguments to pass to the ProjectToPlane
                class. Defaults to None.
        """
        if isinstance(sl_mean, list):
            sl_mean = np.array(sl_mean)
        if isinstance(sl_std, list):
            sl_std = np.array(sl_std)

        assert sl_mean.shape == sl_std.shape, "Mean and std must have the same shape"

        self.num_control_points = sl_mean.shape[0]
        assert (
            self.num_control_points > 0
        ), "Number of control points must be greater than 0"

        assert sample_method in GENERATOR_METHODS, (
            f"Invalid sample method: {sample_method}. Must be one of "
            f"{list(GENERATOR_METHODS.keys())}"
        )
        dist = GENERATOR_METHODS[sample_method]

        length_std = length * 0.1

        self.x_dist = dist(loc=sl_mean[:, 0], scale=sl_std[:, 0])
        self.z_dist = dist(loc=sl_mean[:, 2], scale=sl_std[:, 2])
        self.y_length_dist = dist(loc=length, scale=length_std)

        if project_args:
            self.projector = ProjectToPlane(**project_args)
        else:
            self.projector = None

    def __next__(self) -> npt.NDArray:
        """Yield a spline every time it is called, based on the provided parameters.

        Returns:
            spline (npt.NDArray): A spline generated based on the provided parameters.
        """
        x_control = self.x_dist.rvs()
        new_length = self.y_length_dist.rvs()
        y_control = np.linspace(-new_length, new_length, self.num_control_points)
        z_control = self.z_dist.rvs()

        control_points = np.stack([x_control, y_control, z_control], axis=1)

        tck, u = splprep(control_points.T, s=0, k=3)
        u_new = np.linspace(0, 1, self.num_spl_points)
        spline = np.array(splev(u_new, tck)).T

        if self.projector:
            # TODO: Fix the transform so I do not need to create a dummy dictionary
            data_id = {}
            data_id["backscan"] = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(np.random.rand(100, 3))
            )
            data_id["esl"] = np.random.rand(100, 3)
            data_id["isl"] = spline
            # find the points that are on top of the spline at 10 and 90% of the length
            # This will be C7 and DM
            C7 = spline[np.argmin(np.abs(u_new - 0.9))]
            DM = spline[np.argmin(np.abs(u_new - 0.1))]
            # find values for DR and DL from DM so that DR + DL / 2 = DM
            DR = DM + (DM - C7) / 2
            DL = DM - (DM - C7) / 2
            data_id["special_points"] = {"C7": C7, "DM": DM, "DR": DR, "DL": DL}

            depthmap = self.projector(data_id)["isl_depth_map"]
            depthmap = np.expand_dims(depthmap, axis=0)
            return depthmap

        else:
            return spline

    def __iter__(self):
        return self
