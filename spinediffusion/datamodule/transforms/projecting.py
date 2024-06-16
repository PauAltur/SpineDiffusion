import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch.nn as nn


class ProjectToPlane(nn.Module):
    """Project a point cloud to a plane and create a depth map."""

    def __init__(
        self, height: int, width: int, z_lims: list, m_factor: float, **kwargs
    ):
        """Initialize the ProjectToPlane class.

        Args:
            height (int): The height of the depth map.
            width (int): The width of the depth map.
            z_lims (list): The limits of the z values to consider.
            m_factor (float): The multiplication factor for the average distance.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.z_lims = z_lims
        self.m_factor = m_factor

    def find_points_in_roi(self, pcd: npt.NDArray, roi: npt.NDArray) -> npt.NDArray:
        """Find the points in the point cloud that are within the region of interest.

        Args:
            pcd (np.ndarray): The point cloud to search.
            roi (np.ndarray): The region of interest to search for points.

        Returns:
            _type_: _description_
        """
        length = len(roi[0])
        inds = np.empty(length, dtype=object)

        for i in range(length):
            x_filter = np.logical_and(pcd[:, 0] >= roi[0][i], pcd[:, 0] <= roi[1][i])
            y_filter = np.logical_and(pcd[:, 1] >= roi[2][i], pcd[:, 1] <= roi[3][i])
            z_filter = np.logical_and(pcd[:, 2] >= roi[4][i], pcd[:, 2] <= roi[5][i])
            ind = np.where(
                np.logical_and(np.logical_and(x_filter, y_filter), z_filter)
            )[0]
            inds[i] = ind

        return inds

    def forward(self, data_id: dict) -> dict:
        """Project the point cloud to a plane and create a depth map.

        Args:
            data_id (dict): A single sample of the data to project.
            Must have the key 'backscan' which stores the point cloud.

        Returns:
            data_id (dict): The projected data sample. The key 'depth_map' will
            store the depth map.
        """
        pc = data_id["backscan"]

        # Scale to reverse normalization
        pcd = np.asarray(pc.points) * self.height

        # Create a KDTree for the point cloud
        pcd_ply = o3d.geometry.PointCloud()
        pcd_ply.points = o3d.utility.Vector3dVector(pcd)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_ply)

        # Compute the average distance to the nearest neighbor
        dist_sum = 0
        for i in range(pcd.shape[0]):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd[i], 2)
            dist = np.linalg.norm(pcd[i] - pcd[idx[1]])
            dist_sum += dist
        avg_dist = dist_sum / pcd.shape[0]
        avg_dist = self.m_factor * avg_dist

        # Create a grid for the pixel coordinates
        pc_xgrid = np.arange(-int(self.width // 2), int(self.width // 2), 1)
        pc_ygrid = np.arange(-int(self.height // 2), int(self.height // 2), 1)
        X, Y = np.meshgrid(pc_xgrid, pc_ygrid)

        # Calculate the scaling factor to map the z values to pixel values (0-1)
        scale_factor = 1 / (self.z_lims[1] - self.z_lims[0])

        delta2 = self.m_factor * avg_dist
        length = self.height * self.width
        inf = np.ones(length) * np.inf
        minus_inf = np.ones(length) * (-np.inf)

        roi_inds = self.find_points_in_roi(
            pcd,
            [
                np.ravel(X) - delta2,
                np.ravel(Y) + delta2,
                np.ravel(Y) - delta2,
                np.ravel(Y) + delta2,
                minus_inf,
                inf,
            ],
        )

        roi_pts = np.array([pcd[x, :] for x in roi_inds], dtype=object)
        z_median = np.array(
            [
                (
                    (np.median(x[:, 2]) - self.z_lims[0]) * scale_factor
                    if len(x) != 0
                    else 0.0
                )
                for x in roi_pts
            ],
            dtype=object,
        )
        Z = z_median.reshape(X.shape)
        Z = Z.astype("float64")
        data_id["depth_map"] = Z

        return data_id
