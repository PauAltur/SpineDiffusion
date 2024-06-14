import numpy as np
import torch
import torch.nn as nn


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
        data_id["backscan"] = torch.tensor(np.asarray(data_id["backscan"].points))
        data_id["esl"] = torch.tensor(data_id["esl"])
        data_id["isl"] = torch.tensor(data_id["isl"])
        return data_id
