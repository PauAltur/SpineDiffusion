import numpy as np
import numpy.typing as npt
import torch.nn as nn
from scipy.ndimage import grey_closing


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
        assert "depth_map" in data_id, "The depth map is missing from the data sample."

        structuring_element = np.ones(self.se_size)
        depth_map = data_id["depth_map"]

        data_id["depth_map"] = grey_closing(depth_map, structure=structuring_element)

        return data_id
