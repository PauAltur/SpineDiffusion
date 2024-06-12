import torch.nn as nn

class Center(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, data):
        return data