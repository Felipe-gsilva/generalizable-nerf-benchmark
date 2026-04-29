import subprocess
import sys
import time
import torch
import numpy as np
from PIL import Image

from pathlib import Path
from typing import Dict, Optional, List
from nerf.NeRFModel import NerfModel
from src.utils.config import config
from src.utils.types import Plans
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup


class CLINeRFWrapper(NerfModel):
    """
    A NeRF model wrapper utilizing the Nerfstudio CLI API.

    This class implements the BaseGenerativeModel (NerfModel) contract by
    translating lifecycle methods into subprocess commands (ns-train, ns-export)
    instead of relying on PyTorch optimization loops.
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        image_priority: str = "highest",
        output_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(model_name, dataset_name, image_priority, output_dir)
        self.model_state = None
        self.pipeline = None
        self.model = None

    # ------------------------------------------------------------------
    # Contract Implementations
    # ------------------------------------------------------------------

    def run(self, data_path: Path) -> None:
        """
        Implements the training contract using ns-train.
        Delegates data preparation to the base class and monitors the subprocess.
        """
        # 1. Prepare data using the inherited base class method
        if not self.process_data(data_path):
            config.logger.error(
                f"❌ Aborting training for {data_path}: data preparation failed."
            )
            return

        # 2. Output and Resume detection
        output_path = self.get_output_path(data_path)
        load_dir_arg = []
        latest_checkpoint = self._get_latest_checkpoint(output_path)

        if latest_checkpoint:
            config.logger.info(f"🔄 Checkpoint found: {latest_checkpoint}. Resuming.")
            load_dir_arg = ["--load-dir", str(latest_checkpoint)]
        else:
            config.logger.info("🆕 No checkpoint found. Starting fresh training.")

        config.logger.info(f"🚀 Training {self.model_name} on {data_path}...")

        # 3. Build training command
        cmd = [
            "ns-train",
            self.model_name,
            "--data",
            str(data_path),
            "--output-dir",
            str(output_path),
            "--vis",
            "tensorboard",
        ] + load_dir_arg

        # Add model-specific hyperparameters
        if self.model_name == "nerfacto":
            cmd += [
                "--pipeline.model.eval-num-rays-per-chunk",
                "1024",
                "--pipeline.datamanager.train-num-rays-per-batch",
                "1024",
                "--pipeline.model.log2-hashmap-size",
                "16",
                "--pipeline.model.camera-optimizer.mode",
                "off",
                "--pipeline.model.predict-normals",
                "True",
            ]
        elif self.model_name == "instant-ngp":
            cmd += [
                "--pipeline.model.eval-num-rays-per-chunk",
                "1024",
                "--pipeline.datamanager.train-num-rays-per-batch",
                "1024",
            ]

        # Use generic nerfstudio-data backend
        cmd += [
            "nerfstudio-data",
            "--downscale-factor",
            "1",
        ]

        # 4. Execute and log
        log_dir = Path("assets/logs") / self.model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.dataset_name}_{data_path.name}_{timestamp}.log"

        try:
            with open(log_file, "wb") as f:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
                for line in iter(process.stdout.readline, b""):
                    sys.stdout.write(line.decode("utf-8", errors="replace"))
                    f.write(line)

                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)

            config.logger.info(f"✅ Success training {self.model_name} on {data_path}")

        except subprocess.CalledProcessError as e:
            config.logger.error(f"❌ Training failed. Exit code: {e.returncode}")
        except KeyboardInterrupt:
            config.logger.info("\n🛑 User Interruption.")
            raise KeyboardInterrupt

    def load_checkpoint(self, path: Path) -> None:
        """
        Loads the checkpoint metadata into memory. Also loads the full
        nerfstudio pipeline for rendering.
        """
        config.logger.info(f"Loading NeRF model metadata and pipeline from {path}...")
        c_path = self._get_latest_checkpoint(path) if path.is_dir() else path
        c_path_obj = Path(c_path) if c_path else None

        if c_path_obj and c_path_obj.exists():
            try:
                checkpoint = torch.load(c_path_obj, map_location="cpu")
                self.model_state = checkpoint.get("model_state", None)

                # Load nerfstudio pipeline
                config_path = c_path_obj.parent / "config.yml"
                if config_path.exists():
                    _, pipeline, _, _ = eval_setup(config_path)
                    self.pipeline = pipeline
                    self.model = pipeline.model
                    self.model.eval()
                    config.logger.info(f"✅ Model metadata and pipeline loaded successfully from {path}")
                else:
                    config.logger.warning(f"⚠️ config.yml not found at {config_path}. Pipeline not loaded.")

            except Exception as e:
                config.logger.error(f"❌ Failed to load model from {path}: {e}")

    def render_orthographic_slice(
        self,
        plan: Plans,
        position: float,
        width: int = 512,
        height: int = 512,
        extent_x: float = 2.0,
        extent_y: float = 2.0,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not hasattr(self, "pipeline") or self.pipeline is None:
            config.logger.error("❌ Pipeline not loaded. Ensure load_checkpoint is called.")
            return None

        c2w = torch.eye(4)[:3, :4].float()

        if plan == Plans.AXIAL:
            c2w[2, 3] = position
        elif plan == Plans.SAGITAL:
            c2w = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, position], [0.0, 1.0, 0.0, 0.0]]
            ).float()
        elif plan == Plans.CORONAL:
            c2w = torch.tensor(
                [[0.0, 0.0, 1.0, position], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]
            ).float()

        fx = width / extent_x
        fy = height / extent_y
        cx = width / 2.0
        cy = height / 2.0

        camera = Cameras(
            camera_to_worlds=c2w.unsqueeze(0),
            fx=torch.Tensor([fx]),
            fy=torch.Tensor([fy]),
            cx=torch.Tensor([cx]),
            cy=torch.Tensor([cy]),
            width=torch.Tensor([width]),
            height=torch.Tensor([height]),
            camera_type=CameraType.ORTHOPHOTO,
        ).to(self.pipeline.device)

        ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)

        THICKNESS = 0.005
        if ray_bundle.nears is None and ray_bundle.fars is None:
            raise RuntimeError("Expected ray_bundle.nears and fars to be not None.")

        ray_bundle.nears = torch.zeros_like(ray_bundle.nears)
        ray_bundle.fars = torch.zeros_like(ray_bundle.fars) + THICKNESS

        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)

        return outputs

    def render_orthographic_slices(
        self, save: bool = False, num_slices: int = 10, plan: Plans = Plans.AXIAL
    ) -> Optional[List[np.ndarray]]:
        if not hasattr(self, "pipeline") or self.pipeline is None:
            config.logger.error("❌ No pipeline available for rendering.")
            return None

        # Just use output_dir as a base
        output_dir = self.output_dir / "slices_nerf"
        output_dir.mkdir(parents=True, exist_ok=True)

        aabb = self.pipeline.model.scene_box.aabb.cpu().numpy()
        min_bound, max_bound = aabb[0], aabb[1]

        if plan == Plans.AXIAL:
            min_val, max_val = min_bound[2], max_bound[2]
        elif plan == Plans.SAGITAL:
            min_val, max_val = min_bound[0], max_bound[0]
        elif plan == Plans.CORONAL:
            min_val, max_val = min_bound[1], max_bound[1]

        slab_size = (max_val - min_val) / num_slices
        output = []

        for i in range(num_slices):
            slice_center = min_val + (i + 0.5) * slab_size
            render_dict = self.render_orthographic_slice(
                plan=plan,
                position=slice_center,
                width=config.img_size if hasattr(config, "img_size") else 512,
                height=config.img_size if hasattr(config, "img_size") else 512,
                extent_x=max_bound[0] - min_bound[0],
                extent_y=max_bound[1] - min_bound[1],
            )

            if render_dict and "rgb" in render_dict:
                rgb_img = render_dict["rgb"].cpu().numpy()
                if rgb_img.ndim == 2:
                    dim = config.img_size if hasattr(config, "img_size") else 512
                    rgb_img = rgb_img.reshape(dim, dim, 3)
                elif rgb_img.ndim == 4:
                    rgb_img = rgb_img.squeeze(0)
                
                rgb_img = np.clip(rgb_img, 0.0, 1.0)
                rgb_uint8 = (rgb_img * 255.0).astype(np.uint8)

                if save:
                    png_out = output_dir / f"slice_nerf_{i:03d}.png"
                    Image.fromarray(rgb_uint8).save(png_out)
                    config.logger.info(f"✅ Saved NeRF slice {i} at z={slice_center:.3f} → {png_out}")

                output.append(rgb_uint8)
        return output

    def evaluate(self, rendered_images, ground_truth_images) -> Dict[str, float]:
        """
        Executes standard image evaluation using the inherited base class method.
        """
        return self.compare_images_quality(rendered_images, ground_truth_images)

    def get_memory_footprint(self) -> int:
        """
        Calculates memory footprint based on the loaded checkpoint state dictionary.
        """
        if self.model_state is None:
            return 0
        return sum(
            param.numel() * param.element_size()
            for param in self.model_state.values()
            if isinstance(param, torch.Tensor)
        )

    # ------------------------------------------------------------------
    # CLI-Specific Utilities
    # ------------------------------------------------------------------

    def export(self, export_type: str, data_path: Path):
        """
        Exports the trained NeRF model geometry (e.g., pointcloud).
        Specific to CLI wrappers interacting with ns-export.
        """
        output_path = self.get_output_path(data_path)
        latest_checkpoint = self._get_latest_checkpoint(output_path)

        if not latest_checkpoint:
            config.logger.error(
                f"❌ Cannot export, no checkpoint found in {output_path}"
            )
            return

        config_path = Path(latest_checkpoint).parent / "config.yml"
        target_export_path = (
            Path("assets/data/nerf_exports")
            / self.dataset_name
            / self.model_name
            / data_path.name
            / export_type
        )
        target_export_path.mkdir(parents=True, exist_ok=True)

        config.logger.info(f"--- Exporting {self.model_name} to {export_type} ---")
        cmd = [
            "ns-export",
            export_type,
            "--load-config",
            str(config_path),
            "--output-dir",
            str(target_export_path),
            "--num-points",
            "1000000",
            "--remove-outliers",
            "False",
            "--normal-method",
            "open3d",
        ]

        try:
            subprocess.run(cmd, check=True)
            config.logger.info(f"✅ Export successful")
        except subprocess.CalledProcessError as e:
            config.logger.error(f"❌ Export failed. Code: {e.returncode}")
