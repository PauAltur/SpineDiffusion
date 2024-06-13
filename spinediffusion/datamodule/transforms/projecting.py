import torch.nn as nn


class ProjectToPlane(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, data) -> dict:
        print(f"ProjectToPlane not implemented yet")
        return data
