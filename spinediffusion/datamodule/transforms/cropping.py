import torch.nn as nn


class Crop(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, data):
        return data
