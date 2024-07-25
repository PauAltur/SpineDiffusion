from dataclasses import dataclass
from typing import Optional

import numpy.typing as npt
import open3d as o3d


@dataclass
class SpineSample:
    """Class to represent a spine sample."""

    unique_id: str
    backscan: o3d.geometry.PointCloud
    special_pts: dict
    isl: Optional[npt.NDArray] = None
    esl: Optional[npt.NDArray] = None
