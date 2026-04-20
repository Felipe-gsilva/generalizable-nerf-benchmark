import logging
import os
from typing import List

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAX_EPOCHS = 256
DEFAULT_MAX_EPOCHS_CLASSIFIER = 20


class Config:
    epochs: int
    img_size: int
    batch_size: int
    img_channels: int
    env: str
    logger: logging.Logger
    root_dir: str
    stages: List[str]
    datasets_name_list: List[str]
    nerf_models_to_run: List[str]

    def __init__(
        self,
        epochs: int = DEFAULT_MAX_EPOCHS,
        img_size: int = 256,
        batch_size: int = 16,
    ):
        self.env = "dev"
        self.batch_size = batch_size
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger()
        self.root_dir = ROOT_DIR
        self.stages = [
            "baseline",
            "gan_aug",
            "nerf_aug",
            "gan_nerf_aug",
        ]
        self.datasets_name_list = [
            "3D_virtual_HE_staining",
            "nerf_llff_data",
            # "UT-EndoMRI",
            # "ucsb"
        ]
        self.epochs = epochs
        self.img_size = img_size
        self.nerf_models_to_run = ["nerfacto", "instant-ngp", "merf-ns"]


config = Config()
