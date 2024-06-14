import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SpineDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, data: list):
        """_summary_

        Args:
            data (list): _description_
        """
        self.data = data
        self.keys = list(data.keys())

    def __len__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return len(self.data)

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.data[self.keys[idx]]
