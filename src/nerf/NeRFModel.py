from typing import Dict, List, Literal, Optional
from PIL import Image
from nerfstudio.pipelines.base_pipeline import Pipeline
from validation.eval_images import calculate_fid, calculate_psnr_ssim_lpips
from dataset.ImageDataset import ImageDataset
from utils.config import config
from utils.types import AvailableMetrics, Plans
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import profiler
from pathlib import Path
from utils.config import config

import torch
import subprocess
import glob
import os
import json
import time
import sys
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_IMAGE_SEARCH_DEPTH = 3
CUSTOM_TRANSFORMS_DATASETS = {"3D_virtual_HE_staining"}
plan_map = {Plans.AXIAL: 1, Plans.CORONAL: 0, Plans.SAGITAL: 2}
logger = config.logger


class NerfModel:
    """
    A NeRF model using nerfstudio python API. This class is responsible for
    creating the NeRF model, training it, and rendering images from it,
    such as novel views. It also provides methods for saving and loading
    the model, as well as for evaluating the quality of the rendered images.
    """

    model_name: str
    pipeline: Optional[Pipeline]
    model: Optional[torch.nn.Module]
    best_model: Optional[dict]
    training_history: list[tuple[int, dict]]
    target_id: Optional[str]
    image_priority: str
    images: Optional[ImageDataset]

    def __init__(self, model_name: str, images: Optional[ImageDataset] = None) -> None:
        self.model_name = model_name
        self.pipeline = None
        self.model = None
        self.best_model = None
        self.training_history = []
        self.target_id = None
        self.image_priority = "highest"
        # Dynamically set dataset name inferred from image path
        self.dataset_name = images.name if images else "unknown"
        self.images = images
        self.dataset_path = images.dataset_path if images else None

    def _get_latest_checkpoint(self, output_path: Path):
        """Finds the most recent checkpoint recursively in the output directory."""
        patterns = [
            str(output_path / self.model_name / "**" / "nerfstudio_models"),
            str(output_path / "**" / self.model_name / "**" / "nerfstudio_models"),
        ]
        candidates = []
        for pattern in patterns:
            candidates.extend(glob.glob(pattern, recursive=True))

        if not candidates:
            return None
        try:
            latest = max(candidates, key=lambda p: Path(p).parent.name)
            return latest
        except Exception:
            return None

    def load_from_disk(
        self,
        path: Path,
        test_mode: Literal["test", "val", "inference"] = "test",
        eval_num_rays_per_chunk: Optional[int] = None,
    ) -> bool:
        """Loads a NeRF model from disk using eval_setup."""
        config.logger.info(f"Loading NeRF model from {path}...")
        c_path = self._get_latest_checkpoint(path) if path.is_dir() else path
        if not c_path or not Path(c_path).exists():
            config.logger.error(f"❌ No checkpoint found to load from {path}")
            return False

        try:
            config_path = Path(c_path).parent / "config.yml"
            if not config_path.exists():
                raise FileNotFoundError(f"config.yml not found at {config_path}")

            _, pipeline, _, _ = eval_setup(
                config_path=config_path,
                test_mode=test_mode,
                eval_num_rays_per_chunk=eval_num_rays_per_chunk,
            )
            self.pipeline = pipeline
            self.model = pipeline._model
            self.model.eval()
            config.logger.info(f"✅ Pipeline loaded successfully from {path}")
            return True
        except Exception as e:
            config.logger.error(
                f"❌ Failed to load model pipeline from {path} "
                f"(test_mode={test_mode}, eval_num_rays_per_chunk={eval_num_rays_per_chunk}): {e}"
            )
            return False

    def _downscale_factor_from_name(self, dir_name: str) -> int:
        if dir_name == "images":
            return 1
        if dir_name.startswith("images_"):
            suffix = dir_name.split("_", 1)[1]
            if suffix.isdigit():
                return int(suffix)
        return 1

    def _sort_image_dir_names(self, dir_names: list[str], priority: str) -> List[str]:
        is_highest = priority == "highest"
        unique_names = sorted(set(dir_names))
        return sorted(
            unique_names,
            key=lambda name: (
                self._downscale_factor_from_name(name)
                if is_highest
                else -self._downscale_factor_from_name(name),
                name,
            ),
        )

    def _dir_has_images(self, directory: Path) -> bool:
        return any(
            f.suffix.lower() in IMAGE_EXTENSIONS
            for f in directory.iterdir()
            if f.is_file()
        )

    def _find_images_dir(
        self, data_path: Path, max_depth: int = MAX_IMAGE_SEARCH_DEPTH
    ) -> Path:
        """Finds the best image directory for ns-process-data, including nested LLFF layouts."""
        priority = (
            self.image_priority
            if self.image_priority in {"highest", "lowest"}
            else "highest"
        )

        direct_named_dirs = self._sort_image_dir_names(
            [
                d.name
                for d in data_path.iterdir()
                if d.is_dir() and d.name.startswith("images")
            ],
            priority=priority,
        )
        candidate_dirs = direct_named_dirs

        for dir_name in candidate_dirs:
            candidate = data_path / dir_name
            if candidate.exists() and self._dir_has_images(candidate):
                return candidate

        if self._dir_has_images(data_path):
            return data_path

        root_depth = len(data_path.parts)
        nested_named_dirs = sorted(
            [
                d
                for d in data_path.rglob("images*")
                if d.is_dir()
                and (len(d.parts) - root_depth) <= max_depth
                and self._dir_has_images(d)
            ],
            key=lambda directory: (
                self._downscale_factor_from_name(directory.name)
                if priority == "highest"
                else -self._downscale_factor_from_name(directory.name),
                str(directory),
            ),
        )
        if nested_named_dirs:
            return nested_named_dirs[0]

        raise FileNotFoundError(
            f"""
            No usable images found in {data_path}.
            Checked: {candidate_dirs}, dataset root and nested images* directories.
            """
        )

    def _get_target_id(self, data_path: Path) -> str:
        if self.target_id:
            return self.target_id
        return data_path.name

    def _uses_custom_transforms(self) -> bool:
        return any(
            self.dataset_name == dataset_name
            or self.dataset_name.startswith(f"{dataset_name}_")
            for dataset_name in CUSTOM_TRANSFORMS_DATASETS
        )

    def _load_transforms_json(self, transforms_path: Path) -> Optional[dict]:
        try:
            with transforms_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            config.logger.warning(
                f"⚠️ Invalid transforms.json at {transforms_path}: {e}. Reprocessing."
            )
            return None

    def _is_custom_transforms_format_valid(self, transforms_data: dict) -> bool:
        frames = transforms_data.get("frames")
        if not isinstance(frames, list) or not frames:
            return False
        return all(
            isinstance(frame, dict)
            and isinstance(frame.get("row"), int)
            and isinstance(frame.get("col"), int)
            for frame in frames
        )

    def _get_missing_transform_files(
        self, data_path: Path, transforms_data: dict
    ) -> list[Path]:
        frames = transforms_data.get("frames")
        if not isinstance(frames, list):
            return [data_path / "transforms.json"]

        missing_files: list[Path] = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            file_path = frame.get("file_path")
            if not isinstance(file_path, str) or not file_path.strip():
                continue

            candidate = Path(file_path)
            if not candidate.is_absolute():
                candidate = data_path / candidate
            if candidate.suffix == "":
                for ext in IMAGE_EXTENSIONS:
                    candidate_with_ext = candidate.with_suffix(ext)
                    if candidate_with_ext.exists():
                        break
                else:
                    missing_files.append(candidate)
                    continue

            if not candidate.exists():
                missing_files.append(candidate)

        return missing_files

    def _run_colmap_transforms_generation(
        self, data_path: Path, images_dir: Path
    ) -> bool:
        config.logger.info(
            "🧭 Using COLMAP-based transforms generation via ns-process-data."
        )

        cmd = []
        if shutil.which("xvfb-run"):
            cmd += ["xvfb-run", "-a"]
        cmd += [
            "ns-process-data",
            "images",
            "--data",
            str(images_dir),
            "--output-dir",
            str(data_path),
            "--gpu",
        ]
        process_env = os.environ.copy()
        process_env["QT_QPA_PLATFORM"] = "offscreen"

        try:
            subprocess.run(cmd, check=True, env=process_env)
            config.logger.info("✅ ns-process-data finished successfully.")
            return True
        except subprocess.CalledProcessError as e:
            config.logger.error(f"❌ ns-process-data failed with code: {e.returncode}")
            return False

    @profiler.time_function
    def process_data(self, data_path: Path) -> bool:
        """
        Prepares nerfstudio transforms:
        - 3D_virtual_HE_staining: custom grid transforms.
        - Other datasets: COLMAP via ns-process-data.
        """
        transforms_path = data_path / "transforms.json"
        if transforms_path.exists():
            transforms_data = self._load_transforms_json(transforms_path)
            if transforms_data is not None:
                missing_files = self._get_missing_transform_files(
                    data_path, transforms_data
                )
                if not missing_files:
                    if (
                        self._uses_custom_transforms()
                        and not self._is_custom_transforms_format_valid(transforms_data)
                    ):
                        config.logger.warning(
                            f"⚠️ transforms.json in {data_path} does not match custom grid format. Regenerating."
                        )
                    else:
                        config.logger.info(
                            f"transforms.json already exists in {data_path}. Skipping regeneration."
                        )
                        return True
                else:
                    missing_preview = ", ".join(str(path) for path in missing_files[:3])
                    config.logger.warning(
                        f"⚠️ Found {len(missing_files)} missing image(s) referenced by transforms.json "
                        f"in {data_path}. Example(s): {missing_preview}. Regenerating transforms."
                    )

        config.logger.info(
            f"""
            Preparing nerfstudio inputs for {data_path}...
            """
        )

        try:
            images_dir = self._find_images_dir(data_path)
        except FileNotFoundError as e:
            config.logger.error(f"❌ {e}")
            return False

        return self._run_colmap_transforms_generation(data_path, images_dir)

    def get_output_path(self, data_path: Path) -> Path:
        # if it is a llff instance, we shall save it differently
        if self.dataset_name == "nerf_llff_data":
            output_path = Path("assets/data/nerf_checkpoints") / self.dataset_name
        else:
            output_path = (
                Path("assets/data/nerf_checkpoints")
                / self.dataset_name
                / self._get_target_id(data_path)
            )

        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def get_augmentation_output_path(self, data_path: Path) -> Path:
        """Returns where rendered NeRF slices are stored for augmentation."""
        output_path = (
            Path("assets/data/nerf_aug")
            / self.dataset_name
            / self.model_name
            / self._get_target_id(data_path)
        )
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    @profiler.time_function
    def train(self, data_path: Path):
        """Train the NeRF model using the provided data and configuration via CLI subprocess."""
        if not self.process_data(data_path):
            config.logger.error(
                f"❌ Aborting training for {data_path}: data preparation failed."
            )
            return

        output_path = self.get_output_path(data_path)
        load_dir_arg = []
        latest_checkpoint = self._get_latest_checkpoint(output_path)
        if latest_checkpoint:
            config.logger.info(f"🔄 Checkpoint found: {latest_checkpoint}. Resuming.")
            load_dir_arg = ["--load-dir", str(latest_checkpoint)]
        else:
            config.logger.info("🆕 No checkpoint found. Starting fresh training.")

        config.logger.info(f"🚀 Training {self.model_name} on {data_path}...")

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

        else:
            cmd += [
                "--pipeline.model.eval-num-rays-per-chunk",
                "1024",
                "--pipeline.datamanager.train-num-rays-per-batch",
                "1024",
                "--pipeline.model.log2-hashmap-size",
                "16",
            ]

        # Use generic nerfstudio-data as the backend.
        # It will read what process-data created.
        cmd += [
            "nerfstudio-data",
            "--downscale-factor",
            "1",
        ]

        log_dir = Path("assets/logs") / self.model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = (
            log_dir
            / f"{self.dataset_name}_{self._get_target_id(data_path)}_{timestamp}.log"
        )

        try:
            with open(log_file, "wb") as f:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
                if process.stdout is None:
                    raise RuntimeError("Failed to capture training output.")

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

    def save_best_model(self, path):
        """Save the best performing NeRF model to the specified path."""
        for epoch, metrics in self.training_history:
            if self.best_model is None or metrics["fid"] < self.best_model["fid"]:
                self.best_model = {
                    "epoch": epoch,
                    "model_state": self.model.state_dict() if self.model else None,
                    "metrics": metrics,
                }
        if self.best_model is not None and self.best_model.get("model_state"):
            torch.save(self.best_model["model_state"], path)

    def compare_images_quality(
        self, rendered_images, ground_truth_images, metrics: List[AvailableMetrics]
    ):
        """
        Evaluate the quality of the rendered images against the ground truth
        images using metrics such as FID, PSNR, SSIM, and LPIPS.
        """
        results = {}
        if AvailableMetrics.FID in metrics:
            fid_score = calculate_fid(rendered_images, ground_truth_images)
            results["fid"] = fid_score
        if (
            AvailableMetrics.PSNR in metrics
            or AvailableMetrics.SSIM in metrics
            or AvailableMetrics.LPIPS in metrics
        ):
            psnr_score, ssim_score, lpips_score = calculate_psnr_ssim_lpips(
                rendered_images, ground_truth_images
            )
            if AvailableMetrics.PSNR in metrics:
                results["psnr"] = psnr_score
            if AvailableMetrics.SSIM in metrics:
                results["ssim"] = ssim_score
            if AvailableMetrics.LPIPS in metrics:
                results["lpips"] = lpips_score

        return results

    def get_memory_footprint(self) -> int:
        """Returns the memory footprint of the NeRF model."""
        if self.model is None:
            return 0
        return sum(
            param.numel() * param.element_size() for param in self.model.parameters()
        )

    def render_image(
        self,
        plan: Plans,
        position: float,
        width: int = 512,
        height: int = 512,
        extent_x: float = 2.0,
        extent_y: float = 2.0,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extracts a single orthographic slice from the trained NeRF model.
        Uses a squashed ray bundle to sample a thin plane in the 3D volume.

        Args:
            plan: slicing plane (axial, sagital, coronal)
            position: coordinate along the chosen axis where the slice is taken
            width, height: output resolution
            extent_x, extent_y: Physical coverage of the camera view in NeRF coordinates.
        """
        if not hasattr(self, "pipeline") or self.pipeline is None:
            config.logger.error("❌ Pipeline not loaded. Call load_from_disk first.")
            return None

        position = float(position)
        c2w = torch.eye(4)[:3, :4].float()
        world_size = self.pipeline.world_size

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

        # The camera parameters (fx, fy, cx, cy) in an orthographic camera
        # map pixel coordinates to metric space.
        fx = width / extent_x
        fy = height / extent_y
        cx = width / 2.0
        cy = height / 2.0

        camera = Cameras(
            camera_to_worlds=c2w.unsqueeze(0),
            fx=torch.tensor([fx], dtype=torch.float32),
            fy=torch.tensor([fy], dtype=torch.float32),
            cx=torch.tensor([cx], dtype=torch.float32),
            cy=torch.tensor([cy], dtype=torch.float32),
            width=torch.tensor([width], dtype=torch.int64),
            height=torch.tensor([height], dtype=torch.int64),
            camera_type=CameraType.ORTHOPHOTO,
        ).to(self.pipeline.device)
        # Generate RayBundle
        ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)
        # Force the ray near/far clipping planes to be extremely close together,
        # effectively capturing a single very thin slice.
        THICKNESS = 0.005

        if ray_bundle.nears is None or ray_bundle.fars is None:
            base = ray_bundle.origins[..., :1]
            ray_bundle.nears = torch.zeros_like(base)
            ray_bundle.fars = torch.zeros_like(base) + THICKNESS
        else:
            ray_bundle.nears = torch.zeros_like(ray_bundle.nears)
            ray_bundle.fars = torch.zeros_like(ray_bundle.fars) + THICKNESS

        ray_bundle.nears = torch.zeros_like(ray_bundle.nears)
        ray_bundle.fars = torch.zeros_like(ray_bundle.fars) + THICKNESS
        # Render outputs
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)

        return outputs

    def render(
        self, save: bool = False, num_slices: int = 10, plan: Plans = Plans.AXIAL
    ) -> Optional[List[np.ndarray]]:
        if not self.images or not self.images.dataset_path:
            logger.error("❌ No dataset path available for slicing.")
            return
        path = self.get_output_path(self.images.dataset_path)
        if self.pipeline is None:
            loaded = self.load_from_disk(
                path, test_mode="inference", eval_num_rays_per_chunk=1024
            )
            if not loaded:
                logger.error(f"❌ Failed to load pipeline for NeRF slicing at {path}")
                return

        if self.pipeline is None:
            logger.error(f"❌ Failed to load pipeline for NeRF slicing at {path}")
            return

        output_dir = self.get_augmentation_output_path(self.images.dataset_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Determine scaling and bounds for the slicing based on the pipeline's datamanager
        # Typically the scene box provides the bounds.
        aabb = self.pipeline.model.scene_box.aabb.cpu().numpy()
        min_bound = aabb[0]
        max_bound = aabb[1]
        if plan == Plans.AXIAL:
            min, max = min_bound[2], max_bound[2]
        elif plan == Plans.SAGITAL:
            min, max = min_bound[0], max_bound[0]
        elif plan == Plans.CORONAL:
            min, max = min_bound[1], max_bound[1]

        slab_size = (max - min) / num_slices
        output = []

        for i in range(num_slices):
            slice_center = min + (i + 0.5) * slab_size
            # extract image via orthographic camera
            render_dict = self.render_image(
                plan=plan,
                position=slice_center,
                width=config.img_size,
                height=config.img_size,
                extent_x=max_bound[0] - min_bound[0],
                extent_y=max_bound[1] - min_bound[1],
            )

            if render_dict and "rgb" in render_dict:
                rgb_img = render_dict["rgb"].cpu().numpy()
                if rgb_img.ndim == 2:
                    rgb_img = rgb_img.reshape(config.img_size, config.img_size, 3)
                elif rgb_img.ndim == 4:
                    rgb_img = rgb_img.squeeze(0)
                rgb_img = np.clip(rgb_img, 0.0, 1.0)
                rgb_uint8 = (rgb_img * 255.0).astype(np.uint8)

                if save:
                    png_out = output_dir / f"slice_nerf_{i:03d}.png"
                    Image.fromarray(rgb_uint8).save(png_out)

                    logger.info(
                        f"✅ Saved NeRF slice {i} at z={slice_center:.3f} → {png_out}"
                    )

                output.append(rgb_uint8)
        return output
