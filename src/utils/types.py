import enum
from typing import Any, Dict, TypeAlias, Tuple
from dataclasses import dataclass

Hyperparams: TypeAlias = Dict[str, Any]


class AvailableMetrics(enum.Enum):
    PSNR = "psnr"
    SSIM = "ssim"
    LPIPS = "lpips"
    FID = "fid"
    WASSERSTEIN_DISTANCE = "wasserstein_distance"


Metrics: TypeAlias = Dict[AvailableMetrics | str, float]


class Plans(enum.Enum):
    AXIAL = "axial"
    SAGITAL = "sagital"
    CORONAL = "coronal"


class RenderMode(enum.Enum):
    ORTHOGRAPHIC = "orthographic"
    PERSPECTIVE = "perspective"


@dataclass
class GenerateCameraConfig:
    """
    Configurações da câmera ortográfica para extração de fatias volumétricas (NeRF).
    """

    # (center_x, center_y): Deslocamento lateral para enquadrar o tecido
    center: Tuple[float, float] = (0.0, 0.0)
    # (extent_x, extent_y): Campo de visão físico no mundo 3D (Zoom in/out)
    extent: Tuple[float, float] = (2.0, 2.0)
    # (width, height): Resolução em pixels do bitmap de saída
    resolution: Tuple[int, int] = (512, 512)
    # Espessura do plano de corte (nears/fars).
    # Mantido em 0.5 para garantir acúmulo suficiente das cores HE.
    thickness: float = 0.5
