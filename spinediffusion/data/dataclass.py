from dataclasses import dataclass
from typing import List, Optional, Union

import numpy.typing as npt
import open3d as o3d
import torch


@dataclass
class SpineSample:
    """Dataclass for storing spine data samples."""

    backscan: o3d.geometry.PointCloud
    special_points: List[npt.NDArray]
    backscan_proj: Optional[npt.NDArray] = None
    esl: Optional[npt.NDArray] = None
    esl_proj: Optional[npt.NDArray] = None
    isl: Optional[npt.NDArray] = None
    isl_proj: Optional[npt.NDArray] = None
    noise: Optional[Union[npt.NDArray, torch.Tensor]] = None
