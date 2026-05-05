import enum
from typing import Any, Dict
from dataclasses import dataclass

Hyperparams = Dict[str, Any]

class Plans(enum.Enum):
    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITAL = "sagital"

@dataclass
class SliceConfig:
    plan: Plans = Plans.AXIAL
    num_slices: int = 10
    overlap: float = 0.0

class AvailableMetrics(enum.Enum):
    PSNR = "psnr"
    SSIM = "ssim"
    LPIPS = "lpips"
    FID = "fid"

Metrics = Dict[str, AvailableMetrics]
