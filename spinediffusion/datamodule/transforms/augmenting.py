import numpy as np
import open3d as o3d
import torch.nn as nn


class RandomRotationAugmentation(nn.Module):

    def __init__(self, theta_range: (list, tuple), num_aug: int, **kwargs):
        """Initializes the RandomRotationAugmentation class.

        Args:
            theta_range ((list, tuple)): The range of angles to rotate the data by.
                The range is defined as [theta_min, theta_max] and should be provided
                in radians.
            num_aug (int): The number of augmented samples to generate.
        """
        super().__init__()
        self.theta_range = theta_range
        self.num_aug = num_aug

    def forward(self, data_id: dict, key: str) -> dict:
        """Rotates the data by a random angle within the user-defined range.

        Args:
            data_id (dict): A single sample of the data to augment.
                Must have the key 'backscan' and optionally 'esl' and 'isl'
                which store geometric data corresponding to point clouds (or lines).
            key (str): The key of the sample.

        Returns:
            data_id (dict): The augmented data sample.
        """

        data_id_aug = {}

        for i in range(self.num_aug):
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
                (R @ np.asarray(data_id["backscan"].points).T).T
            )

            data_id_aug[f"{key}_{i}"] = {}
            data_id_aug[f"{key}_{i}"]["backscan"] = aug_backscan
            if "esl" in data_id:
                data_id_aug[f"{key}_{i}"]["esl"] = (R @ data_id["esl"].T).T
            if "isl" in data_id:
                data_id_aug[f"{key}_{i}"]["isl"] = (R @ data_id["isl"].T).T
            if "special_points" in data_id:
                data_id_aug[f"{key}_{i}"]["special_points"] = (
                    R @ data_id["special_points"].T
                ).T
                for key, point in data_id["special_points"].items():
                    # check whether point is empty
                    if len(point) == 0:
                        data_id_aug[f"{key}_{i}"]["special_points"][key] = point
                    else:
                        data_id_aug[f"{key}_{i}"]["special_points"][key] = (
                            R @ point.T
                        ).T
        return data_id_aug
