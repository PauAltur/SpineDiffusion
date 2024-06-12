import torch.nn as nn

class Resample3DCurve(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, data):
        return data