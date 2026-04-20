import enum
from typing import Any, Dict

Hyperparams = Dict[str, Any]


class AvailableMetrics(enum.Enum):
    PSNR = "psnr"
    SSIM = "ssim"
    LPIPS = "lpips"
    FID = "fid"


Metrics = Dict[str, AvailableMetrics]
