import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import yaml
import torch
import subprocess
import multiprocessing
import os
import json
import time
import sys
import shutil
import numpy as np
import matplotlib
import tempfile
import re

from typing import Dict, List, Literal, Optional, Tuple
from PIL import Image
from nerfstudio.pipelines.base_pipeline import Pipeline
from src.validation.eval_images import calculate_fid, calculate_psnr_ssim_lpips
from src.dataset.ImageDataset import ImageDataset, _llffhold_indices
from src.utils.config import config
from src.utils.types import AvailableMetrics, Plans, GenerateCameraConfig, RenderMode
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import profiler
from pathlib import Path
from nerfstudio.cameras.rays import RayBundle

# Universal Monkey-Patch for RayBundle to fix Shape Mismatch during get_outputs_for_camera_ray_bundle
_original_sliced = RayBundle.get_row_major_sliced_ray_bundle

def _safe_sliced(self, start_idx, end_idx):
    popped = {}
    if self.metadata is not None:
        batch_shape = self.origins.shape[:-1]
        batch_ndim = len(batch_shape)
        for k in list(self.metadata.keys()):
            v = self.metadata[k]
            should_pop = True
            if isinstance(v, torch.Tensor):
                if v.ndim == batch_ndim + 1 and v.shape[:batch_ndim] == batch_shape:
                    should_pop = False
            elif isinstance(v, dict):
                dict_safe = True
                for kk, vv in v.items():
                    if not isinstance(vv, torch.Tensor):
                        dict_safe = False
                        break
                    if vv.ndim != batch_ndim + 1 or vv.shape[:batch_ndim] != batch_shape:
                        dict_safe = False
                        break
                if dict_safe:
                    should_pop = False
            
            if should_pop:
                popped[k] = self.metadata.pop(k)
    
    sliced = _original_sliced(self, start_idx, end_idx)
    
    if popped:
        if self.metadata is None:
            self.metadata = {}
        if sliced.metadata is None:
            sliced.metadata = {}
        for k, v in popped.items():
            self.metadata[k] = v
            sliced.metadata[k] = v
            
    return sliced

RayBundle.get_row_major_sliced_ray_bundle = _safe_sliced
matplotlib.use("Agg")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_IMAGE_SEARCH_DEPTH = 3
CUSTOM_TRANSFORMS_DATASETS = {"3D_virtual_HE_staining"}
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

    def __init__(
        self,
        model_name: str,
        images: Optional[ImageDataset] = None,
        num_views: int = 0,  # 0 = Usa o dataset todo. >0 = Few-shot.
        split: List[float] = [0.7, 0.15, 0.15],
        split_strategy: str = "uniform",  # "uniform" ou "random"
        llffhold: Optional[int] = None,
        regime: Literal["per-scene", "zero-shot", "tta"] = "per-scene",
        tta_steps: int = 500,
    ) -> None:
        self.model_name = model_name
        self.pipeline = None
        self.model = None
        self.best_model = None
        self.training_history = []
        self.target_id = None
        self.image_priority = "highest"

        self.dataset_name = images.name if images else "unknown"
        self.images = images
        self.dataset_path = images.dataset_path if images else None
        self.regime = regime
        self.tta_steps = tta_steps

        # Parâmetros do Paper
        self.num_views = num_views
        if sum(split) != 1.0:
            raise ValueError("Split ratios must sum to 1.0")
        self.val_split = split[1]
        self.test_split = split[2]
        self.split_strategy = split_strategy
        if llffhold is not None and llffhold <= 0:
            raise ValueError("llffhold must be > 0 when provided.")
        self.llffhold = llffhold

    def _apply_dataset_splits(self, data_path: Path) -> Path:
        """
        Split train/val/test based on the class parameters and save a new JSON file for Nerfstudio to read.
        """
        json_path = data_path / "transforms.json"

        if self.llffhold is not None and self.num_views > 0:
            out_json_path = (
                data_path
                / f"transforms_llffhold{self.llffhold}_{self.num_views}views_{self.split_strategy}.json"
            )
        elif self.llffhold is not None:
            out_json_path = data_path / f"transforms_llffhold{self.llffhold}.json"
        elif self.num_views > 0:
            out_json_path = (
                data_path
                / f"transforms_{self.num_views}views_{self.split_strategy}.json"
            )
        else:
            out_json_path = data_path / "transforms_standard_split.json"

        with open(json_path, "r") as f:
            data = json.load(f)

        frames = data.get("frames", [])
        total_images = len(frames)

        if total_images == 0:
            return json_path

        if self.llffhold is not None:
            test_indices = _llffhold_indices(
                total=total_images, hold=self.llffhold, split="test"
            )
            train_pool = _llffhold_indices(
                total=total_images, hold=self.llffhold, split="train"
            )

            if self.num_views > 0 and self.num_views < len(train_pool):
                if self.split_strategy == "uniform":
                    picked_positions = np.linspace(
                        0, len(train_pool) - 1, self.num_views, dtype=int
                    ).tolist()
                    train_indices = [train_pool[pos] for pos in picked_positions]
                elif self.split_strategy == "random":
                    np.random.seed(42)
                    train_indices = np.random.choice(
                        train_pool, self.num_views, replace=False
                    ).tolist()
                else:
                    raise ValueError(f"Strategy {self.split_strategy} not supported.")
            else:
                train_indices = train_pool
            val_set = set(i for i in train_pool if i not in set(train_indices))
            test_set = set(test_indices)
            train_set = set(train_indices)
        else:
            # if num_views is set to be less than total images, we take that many for training and split the rest for val/test
            if self.num_views > 0 and self.num_views < total_images:
                train_count = self.num_views
                remaining = total_images - train_count

                pool_ratio = self.val_split + self.test_split
                if pool_ratio > 0:
                    val_count = int(remaining * (self.val_split / pool_ratio))
                else:
                    val_count = remaining // 2

            else:
                # default split
                train_count = int(
                    total_images * (1.0 - self.val_split - self.test_split)
                )
                val_count = int(total_images * self.val_split)

            all_indices = np.arange(total_images)
            # this is not completely spatial coherent. Since the cameras do not need to be in order, it means that this "uniform" selection is not 3D uniform.
            # the correct approach is to get all the values from the cameras to the c2w and them interpolate them in 3D space, but this is a good approximation and much faster to implement.
            if self.split_strategy == "uniform":
                train_indices = np.linspace(
                    0, total_images - 1, train_count, dtype=int
                ).tolist()
            elif self.split_strategy == "random":
                np.random.seed(42)
                train_indices = np.random.choice(
                    total_images, train_count, replace=False
                ).tolist()
            else:
                raise ValueError(f"Strategy {self.split_strategy} not supported.")

            # randomly shuffle the remaining indices for val/test split
            remaining_indices = [idx for idx in all_indices if idx not in train_indices]
            np.random.seed(42)  # fixed seed
            np.random.shuffle(remaining_indices)

            val_set = set(remaining_indices[:val_count])
            test_set = set(remaining_indices[val_count:])
            train_set = set(train_indices)

        # Updates the frames in the JSON with the new split information
        for i, frame in enumerate(frames):
            if i in train_set:
                frame["split"] = "train"
            elif i in val_set:
                frame["split"] = "val"
            else:
                frame["split"] = "test"

        with open(out_json_path, "w") as f:
            json.dump(data, f, indent=4)

        if self.llffhold is not None:
            config.logger.info(
                f"📊 Dataset LLFF hold-{self.llffhold} [{out_json_path.name}]: "
                f"{len(train_set)} Train | {len(val_set)} Val | {len(test_set)} Test"
            )
        else:
            config.logger.info(
                f"📊 Dataset Split [{out_json_path.name}]: {len(train_set)} Train | {len(val_set)} Val | {len(test_set)} Test"
            )
        return out_json_path

    def _get_eval_dataloader(self):
        """Helper to get evaluation dataloader / datalist across different datamanagers."""
        if hasattr(self.pipeline.datamanager, "fixed_indices_eval_dataloader"):
            return self.pipeline.datamanager.fixed_indices_eval_dataloader
        
        # Fallback for VanillaDataManager / ParallelDataManager
        eval_dataset = self.pipeline.datamanager.eval_dataset
        cameras = eval_dataset.cameras
        eval_dataloader = []
        for i in range(len(eval_dataset)):
            camera = cameras[i : i + 1]
            data = eval_dataset[i]
            eval_dataloader.append((camera, data))
        return eval_dataloader

    def _get_latest_checkpoint(self, output_path: Path) -> Optional[Path]:
        """Finds the most recent checkpoint recursively in the output directory."""
        candidates = list(output_path.rglob("nerfstudio_models"))
        valid_candidates = [c for c in candidates if self.model_name in c.parts]

        if not valid_candidates:
            return None

        try:
            latest = max(valid_candidates, key=lambda p: p.parent.name)
            return latest
        except Exception:
            return None

    def _patch_config_yaml(self, config_path: Path, checkpoint_dir: Path, downscale_factor: Optional[int] = None) -> Path:
        """
        Patches the config to set an explicit 'load_dir', bypassing folder crawling logic.
        """
        try:
            with config_path.open("r") as f:
                content = f.read()

            content = content.replace("/workspace/", "/home/felipe-gsilva/dev/cs/nerf-ann-paper/")

            if checkpoint_dir.name == "nerfstudio_models":
                load_target = checkpoint_dir.parent
            else:
                load_target = checkpoint_dir

            load_dir_parts = load_target.resolve().parts
            replacement_parts = "\n".join([f"- {p}" for p in load_dir_parts])
            load_dir_yaml = f"load_dir: !!python/object/apply:pathlib.PosixPath\n{replacement_parts}"
            load_dir_pattern = r"load_dir:\s*(?:null|!!python/object/apply:pathlib\.PosixPath\n(?:\s*- [^\n]+\n?)*)"
            content = re.sub(load_dir_pattern, load_dir_yaml + "\n", content)
            parts = checkpoint_dir.parts
            try:
                if "nerf_checkpoints" in parts:
                    idx = parts.index("nerf_checkpoints")
                    new_output_dir_parts = parts[: idx + 2]
                    new_experiment_name = parts[idx + 2]

                    content = re.sub(
                        r"experiment_name: .*",
                        f"experiment_name: {new_experiment_name}",
                        content,
                    )
                    output_dir_pattern = r"(output_dir: !!python/object/apply:pathlib\.PosixPath\n)(?:\s*- .*\n?)*"
                    out_parts = "\n".join([f"- {p}" for p in new_output_dir_parts])
                    content = re.sub(
                        output_dir_pattern, r"\1" + out_parts + "\n", content
                    )
            except Exception:
                pass

            if downscale_factor is not None:
                # Force the config to use the requested downscale factor
                content = re.sub(
                    r"downscale_factor:\s*\d+",
                    f"downscale_factor: {downscale_factor}",
                    content,
                )

            temp_config = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yml", delete=False
            )
            temp_config.write(content)
            temp_config.close()
            return Path(temp_config.name)

        except Exception as e:
            config.logger.warning(f"⚠️ Failed to patch config.yml: {e}")
            return config_path

    def load_from_disk(
        self,
        path: Path,
        mode: Literal["test", "val", "inference"] = "test",
        downscale_factor: Optional[int] = None,
    ) -> bool:
        """Loads a NeRF model from disk using eval_setup."""
        config.logger.info(f"Loading NeRF model from {path}...")
        c_path = self._get_latest_checkpoint(path) if path.is_dir() else path
        if not c_path or not Path(c_path).exists():
            config.logger.error(f"❌ No checkpoint found to load from {path}")
            return False

        temp_config_path = None
        try:
            multiprocessing.set_start_method("spawn", force=True)
            config_path = Path(c_path).parent / "config.yml"
            if not config_path.exists():
                raise FileNotFoundError(f"config.yml not found at {config_path}")

            # Patch the config to handle moved checkpoints or "dirty" names
            temp_config_path = self._patch_config_yaml(config_path, Path(c_path), downscale_factor=downscale_factor)

            _, pipeline, _, _ = eval_setup(
                config_path=temp_config_path,
                test_mode=mode,
            )

            self.pipeline = pipeline
            self.model = pipeline._model
            self.model.eval()
            config.logger.info(f"✅ Pipeline loaded successfully from {path}")
            return True

        except Exception as e:
            import traceback  # Adicione o import aqui se não estiver no topo do arquivo

            config.logger.error(
                f"❌ Failed to load model pipeline from {path} (mode={mode}):\n{traceback.format_exc()} | {e}"
            )
            return False
        finally:
            if (
                temp_config_path
                and temp_config_path != config_path
                and temp_config_path.exists()
            ):
                try:
                    os.unlink(temp_config_path)
                except Exception:
                    pass

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
        has_ext = any(
            f.suffix.lower() in IMAGE_EXTENSIONS
            for f in directory.iterdir()
            if f.is_file()
        )
        if not has_ext:
            return False

        return True

    def _find_images_dir(
        self, data_path: Path, max_depth: int = MAX_IMAGE_SEARCH_DEPTH
    ) -> Optional[Path]:
        """Finds the base image directory without downscale sorting."""
        images_dir = data_path / "images"
        if images_dir.exists() and self._dir_has_images(images_dir):
            return images_dir

        if self._dir_has_images(data_path):
            return data_path

        root_depth = len(data_path.parts)
        for d in data_path.rglob("images*"):
            if (
                d.is_dir()
                and (len(d.parts) - root_depth) <= max_depth
                and self._dir_has_images(d)
            ):
                return d

    def _find_scene_directories(self, data_path: Path) -> list[tuple[Path, Path]]:
        if not data_path.exists() or not data_path.is_dir():
            return []

        scene_dirs: list[tuple[Path, Path]] = []
        for entry in sorted(
            (path for path in data_path.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            images_dir = entry / "images"
            if images_dir.is_dir() and self._dir_has_images(images_dir):
                scene_dirs.append((entry, images_dir))

        return scene_dirs

    def _process_scene_data(
        self,
        data_path: Path,
        num_imgs: int = 0,
        images_dir: Optional[Path] = None,
    ) -> bool:
        transforms_path = data_path / "transforms.json"
        if transforms_path.exists() and num_imgs == 0:
            transforms_data = self._load_transforms_json(transforms_path)
            if transforms_data is not None:
                missing_files = self._get_missing_transform_files(
                    data_path, transforms_data
                )
                if not missing_files:
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

        if images_dir is None:
            try:
                images_dir = self._find_images_dir(data_path)
            except FileNotFoundError as e:
                config.logger.error(f"❌ {e}")
                return False

        if not images_dir:
            config.logger.error(
                f"❌ No valid images directory found in {data_path} for nerfstudio processing."
            )
            return False

        return self._run_colmap_transforms_generation(data_path, images_dir)

    def _get_target_id(self, data_path: Path) -> str:
        if self.target_id:
            return self.target_id
        return data_path.name

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
            "🧭 Running optimized COLMAP pipeline (Feature Extraction -> Matching -> Mapping)."
        )

        db_path = data_path / "colmap.db"
        sparse_path = data_path / "sparse" / "0"
        sparse_path.mkdir(parents=True, exist_ok=True)

        # 1. CRITICAL: Stop CPU thread starvation
        process_env = os.environ.copy()
        process_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        process_env["OMP_NUM_THREADS"] = "4"
        process_env["OPENBLAS_NUM_THREADS"] = "4"
        process_env["MKL_NUM_THREADS"] = "4"
        process_env["PATH"] = (
            f"{os.path.expanduser('~/.local/bin')}:{os.environ.get('PATH', '')}"
        )

        has_xvfb = shutil.which("xvfb-run") is not None
        if has_xvfb:
            process_env.pop("QT_QPA_PLATFORM", None)
            xvfb_prefix = ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24"]
        else:
            if not process_env.get("DISPLAY"):
                process_env["QT_QPA_PLATFORM"] = "offscreen"
                config.logger.warning(
                    "No DISPLAY detected. COLMAP GPU mode may fail without X server/xvfb."
                )
            xvfb_prefix = []

        # 2. Define the explicit COLMAP stages
        commands = [
            {
                "name": "Feature Extraction",
                "cmd": [
                    "colmap",
                    "feature_extractor",
                    "--database_path",
                    str(db_path),
                    "--image_path",
                    str(images_dir),
                    "--ImageReader.single_camera",
                    "1",
                    "--SiftExtraction.use_gpu",
                    "1",
                    "--SiftExtraction.gpu_index",
                    "0",
                    "--SiftExtraction.num_threads",
                    "4",
                    "--SiftExtraction.max_image_size",
                    "1600",
                    "--SiftExtraction.max_num_features",
                    "8192",
                ],
            },
            {
                "name": "Exhaustive Matching",
                "cmd": [
                    "colmap",
                    "exhaustive_matcher",
                    "--database_path",
                    str(db_path),
                    "--SiftMatching.use_gpu",
                    "1",
                    "--SiftMatching.gpu_index",
                    "0",
                    "--SiftMatching.num_threads",
                    "4",
                    "--SiftMatching.guided_matching",
                    "1",
                ],
            },
            {
                "name": "Mapping (Pose Estimation)",
                "cmd": [
                    "colmap",
                    "mapper",
                    "--database_path",
                    str(db_path),
                    "--image_path",
                    str(images_dir),
                    "--output_path",
                    str(sparse_path),
                    "--Mapper.num_threads",
                    "4",
                ],
            },
            {
                "name": "Nerfstudio Conversion",
                "cmd": [
                    "ns-process-data",
                    "images",
                    "--data",
                    str(images_dir.resolve()),
                    "--output-dir",
                    str(data_path.resolve()),
                    "--skip-colmap",
                    "--colmap-model-path",
                    str(sparse_path.resolve()),
                ],
            },
        ]

        for step in commands:
            try:
                config.logger.info(f"⏳ Running: {step['name']}...")
                final_cmd = xvfb_prefix + step["cmd"] if has_xvfb else step["cmd"]

                subprocess.run(final_cmd, check=True, env=process_env)
                config.logger.info(f"✅ Finished {step['name']}")

            except subprocess.CalledProcessError as e:
                config.logger.error(
                    f"❌ Pipeline failed at {step['name']} with code: {e.returncode}"
                )
                return False

        config.logger.info("🎉 COLMAP transforms generated successfully.")
        return True

    @profiler.time_function
    def process_data(self, data_path: Path, num_imgs: int = 0) -> bool:
        """
            Prepares nerfstudio transforms:

            Args:
            data_path:
            - 3D_virtual_HE_staining: custom grid transforms.
            - Other datasets: COLMAP via ns-process-data.

        num_imgs:
        - if > 0, limits the number of images in transforms.json (for quick validation).
        """
        scene_dirs = []
        if not self._dir_has_images(data_path) and not (data_path / "images").exists():
            scene_dirs = self._find_scene_directories(data_path)

        if scene_dirs:
            config.logger.info(
                f"Detected {len(scene_dirs)} scene(s) under {data_path}. "
                "Running COLMAP sequentially per scene."
            )
            all_success = True
            for scene_path, images_dir in scene_dirs:
                success = self._process_scene_data(
                    scene_path, num_imgs=num_imgs, images_dir=images_dir
                )
                if not success:
                    config.logger.error(
                        f"❌ Scene preprocessing failed for {scene_path}. Continuing."
                    )
                    all_success = False
            return all_success

        return self._process_scene_data(data_path, num_imgs=num_imgs)

    def get_output_path(self, data_path: Path) -> Path:
        """Generates a clean base path for checkpoints."""
        # Clean the dataset name (removes _images__Gastric if appended)
        clean_dataset = (
            self.dataset_name.split("_images")[0] if self.dataset_name else "unknown"
        )

        # Clean the target ID (converts images__Gastric to just Gastric)
        clean_target = self._get_target_id(data_path)
        if "__" in clean_target:
            clean_target = clean_target.split("__")[-1]

        # LLFF override
        if clean_dataset == "nerf_llff_data":
            output_path = Path("assets/data/nerf_checkpoints") / clean_dataset
        else:
            output_path = (
                Path("assets/data/nerf_checkpoints") / clean_dataset / clean_target
            )

        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def get_augmentation_output_path(self, data_path: Path) -> Path:
        """Returns where rendered NeRF slices are stored for augmentation."""
        clean_dataset = (
            self.dataset_name.split("_images")[0] if self.dataset_name else "unknown"
        )
        clean_target = self._get_target_id(data_path)
        if "__" in clean_target:
            clean_target = clean_target.split("__")[-1]

        output_path = (
            Path("assets/data/nerf_aug")
            / clean_dataset
            / self.model_name
            / clean_target
        )
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def _get_checkpoint_config_overrides(
        self, checkpoint_dir: str
    ) -> tuple[list[str], Optional[int]]:
        """
        When resuming a checkpoint, read model hyper-params from the saved config.yml.
        Returns: (overrides_list, original_downscale_factor)
        """
        config_path = Path(checkpoint_dir).parent / "config.yml"
        if not config_path.exists():
            return [], None
        try:
            with config_path.open("r") as f:
                cfg = yaml.safe_load(f)
            overrides = []
            original_downscale = None

            # Walk the nested dict to find init_resolution under pipeline.model
            model_cfg = cfg.get("pipeline", {}).get("model", {})
            init_res = model_cfg.get("init_resolution")
            if init_res is not None:
                overrides += ["--pipeline.model.init-resolution", str(init_res)]

            # Extract the original downscale factor
            dataparser_cfg = (
                cfg.get("pipeline", {}).get("datamanager", {}).get("dataparser", {})
            )
            if "downscale_factor" in dataparser_cfg:
                original_downscale = dataparser_cfg["downscale_factor"]

            return overrides, original_downscale
        except Exception as e:
            config.logger.warning(f"⚠️ Could not read checkpoint config.yml: {e}")
            return [], None

    @profiler.time_function
    def train(self, data_path: Path, downscale_factor: int = 1, num_imgs: int = 0, extra_cmd_args: list = []):
        """Prepares data and handles training/loading according to the selected regime."""
        if not self.process_data(data_path, num_imgs=num_imgs):
            config.logger.error(
                f"❌ Aborting training for {data_path}: data preparation failed."
            )
            return

        split_json_path = self._apply_dataset_splits(data_path)
        output_path = self.get_output_path(data_path)

        clean_dataset = (
            self.dataset_name.split("_images")[0] if self.dataset_name else "unknown"
        )
        clean_target = self._get_target_id(data_path)
        if "__" in clean_target:
            clean_target = clean_target.split("__")[-1]

        base_output_dir = Path("assets/data/nerf_checkpoints")
        experiment_name = f"{clean_dataset}/{clean_target}"

        cmd = [
            "ns-train",
            self.model_name,
            "--output-dir",
            str(base_output_dir),
            "--experiment-name",
            experiment_name,
            "--vis",
            "tensorboard",
            "--data",
            str(split_json_path),
        ]

        if self.regime == "tta":
            config.logger.info(
                f"🔧 TTA regime enabled. Limiting adaptation to {self.tta_steps} steps."
            )
            cmd += [
                "--max-num-iterations",
                str(self.tta_steps),
                "--steps-per-save",
                str(self.tta_steps),
                "--steps-per-eval-all-images",
                str(self.tta_steps),
            ]
            config.logger.info(
                "ℹ️ Delegating pretrained weight resolution to Nerfstudio/model implementation."
            )
        elif self.regime == "zero-shot":
            config.logger.info(
                f"❄️ Zero-shot regime enabled. Running 0 iterations to trigger internal model loading."
            )
            cmd += [
                "--max-num-iterations",
                "0",
                "--steps-per-save",
                "0",
                "--steps-per-eval-all-images",
                "0",
            ]
        elif self.regime == "per-scene":
            config.logger.info(
                "🆕 Per-scene regime enabled. Running full optimization for the target scene."
            )
            latest_checkpoint = self._get_latest_checkpoint(output_path)
            if latest_checkpoint:
                cmd += ["--load-dir", str(latest_checkpoint)]
                checkpoint_overrides, saved_downscale = (
                    self._get_checkpoint_config_overrides(latest_checkpoint)
                )
                cmd += checkpoint_overrides
                if saved_downscale is not None and saved_downscale != downscale_factor:
                    config.logger.warning(
                        f"⚠️ Checkpoint trained with downscale_factor={saved_downscale}. "
                        f"Overriding requested {downscale_factor} to prevent IndexError."
                    )
                    downscale_factor = saved_downscale
        else:
            raise ValueError(
                f"Unsupported regime '{self.regime}'. Expected one of: per-scene, zero-shot, tta."
            )

        if self.model_name == "pixel-nerf":
            cmd += [
                "--mixed-precision",
                "True",
                "--pipeline.model.eval-num-rays-per-chunk",
                "128",
                "--pipeline.datamanager.train-num-rays-per-batch",
                "128",
                "--pipeline.datamanager.cache-images-type",
                "uint8",
            ]
        if self.model_name in ["gnt"]:
            cmd += [
                "--pipeline.model.N-samples",
                "48",
                "--pipeline.model.N-importance",
                "48",
                "--mixed-precision",
                "True",
                "--pipeline.model.transdepth",
                "8",
                "--pipeline.model.netwidth",
                "64",
                "--pipeline.model.eval-num-rays-per-chunk",
                "64",
                "--pipeline.datamanager.train-num-rays-per-batch",
                "64",
                "--pipeline.datamanager.cache-images-type",
                "uint8",
            ]
        elif self.model_name in ["nerfacto"]:
            cmd += [
                "--pipeline.model.eval-num-rays-per-chunk",
                "512",
                "--pipeline.datamanager.train-num-rays-per-batch",
                "512",
                "--pipeline.model.log2-hashmap-size",
                "16",
                "--pipeline.model.camera-optimizer.mode",
                "off",
                "--pipeline.model.background-color",
                "white",
            ]
        elif self.model_name in ["instant-ngp"]:
            cmd += [
                "--pipeline.model.eval-num-rays-per-chunk",
                "128",
                "--pipeline.datamanager.train-num-rays-per-batch",
                "512",
                "--pipeline.model.log2-hashmap-size",
                "16",
                "--pipeline.model.background-color",
                "white",
                "--pipeline.datamanager.cache-images-type",
                "uint8",
                "--mixed-precision",
                "True",
            ]

        if self.regime in ["tta", "zero-shot"] and self.model_name in ["pixel-nerf", "gnt"]:
            cmd += [
                "--pipeline.model.transfer-learning",
                "True",
            ]
            if self.model_name == "gnt":
                gnt_pretrained_path = Path("assets/pretrained/gnt_pretrained.pth")
                download_pretrained_gnt_model(gnt_pretrained_path)

                cmd += [
                    "--pipeline.model.pretrained-ckpt-path",
                    str(gnt_pretrained_path.resolve()),
                ]

            if self.model_name == "pixel-nerf":
                pixelnerf_pretrained_zip = Path(
                    "assets/pretrained/pixelnerf_pretrained.zip"
                )
                unzipped_path = download_pretrained_pixelnerf_weights(pixelnerf_pretrained_zip)

                cmd += [
                    "--pipeline.model.transfer-learning",
                    "True",
                    "--pipeline.model.pretrained-ckpt-path",
                    str(unzipped_path.resolve()),
                ]

        if extra_cmd_args:
            cmd += extra_cmd_args
            config.logger.info(f"➕ Appending extra CLI args: {extra_cmd_args}")

        cmd += [
            "nerfstudio-data",
            "--downscale-factor",
            str(downscale_factor),
            "--eval-mode",
            "fraction",
        ]

        process_env = os.environ.copy()
        process_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        process_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        process_env["PYTHONUNBUFFERED"] = "1"
        process_env["TERM"] = "dumb"
        process_env["MAX_JOBS"] = "1"
        process_env["OMP_NUM_THREADS"] = "1"

        log_dir = Path("assets/logs") / self.model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = (
            log_dir / f"{clean_dataset}_{clean_target}_{self.regime}_{timestamp}.log"
        )

        try:
            with open(log_file, "wb") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=process_env,
                )
                if process.stdout is None:
                    raise RuntimeError("Failed to capture training output.")

                for chunk in iter(lambda: process.stdout.read(1024), b""):
                    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
                    f.write(chunk)

                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)

            config.logger.info(
                f"✅ Success training/adapting {self.model_name} on {data_path} via {self.regime}"
            )

        except subprocess.CalledProcessError as e:
            config.logger.error(f"❌ Subprocess failed. Exit code: {e.returncode}")
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

    def _rendered_images_to_tensor(
        self, rendered_images: List[np.ndarray]
    ) -> Optional[torch.Tensor]:
        if not rendered_images:
            return None
        try:
            rendered_array = np.stack(rendered_images, axis=0).astype(
                np.uint8, copy=False
            )
        except ValueError as e:
            logger.error(f"❌ Failed to stack rendered images for metrics: {e}")
            return None
        return torch.from_numpy(rendered_array).permute(0, 3, 1, 2).contiguous()

    def _ground_truth_images_to_tensor(
        self, image_paths: List[Path], image_size: int, limit: int
    ) -> Optional[torch.Tensor]:
        if limit <= 0:
            return None

        image_tensors: List[torch.Tensor] = []
        for image_path in image_paths[:limit]:
            try:
                with Image.open(image_path) as image:
                    image_rgb = image.convert("RGB").resize((image_size, image_size))
                    image_np = np.asarray(image_rgb, dtype=np.uint8)
                image_tensors.append(
                    torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
                )
            except OSError as e:
                logger.warning(
                    f"⚠️ Failed to read ground truth image '{image_path}' for metrics: {e}"
                )

        if not image_tensors:
            return None
        return torch.stack(image_tensors, dim=0)

    def measure_inference_fps(
        self, mode: RenderMode, num_warmup: int = 3, num_test: int = 10
    ) -> float:
        """
        Measures the inference speed (FPS) of the NeRF model in either Orthographic or Perspective mode.
         - Orthographic: Uses a synthetic ray bundle for a fixed view, injected with valid eval context.
         - Perspective: Uses a real validation camera ray bundle populated with real source contexts.
        """
        if self.pipeline is None or self.model is None:
            return 0.0

        self.model.eval()
        device = self.pipeline.device

        try:
            model_metadata = None
            if self.model_name in ["gnt", "pixel-nerf"]:
                eval_ray_bundle, eval_batch = self.pipeline.datamanager.next_eval(step=0)
                if self.model_name == "gnt":
                    eval_batch = self.pipeline._inject_gnt_metadata(
                        eval_ray_bundle, eval_batch, split="eval"
                    )
                model_metadata = eval_ray_bundle.metadata

        except Exception as e:
            config.logger.error(
                f"Failed to fetch evaluation source metadata for {self.model_name}: {e}"
            )
            return 0.0

        with torch.no_grad():
            if mode == RenderMode.ORTHOGRAPHIC:
                cam = GenerateCameraConfig(
                    center=(0.0, 0.0),
                    extent=(1.0, 1.0),
                    resolution=(config.img_size, config.img_size),
                    thickness=0.1,
                )
                ray_bundle = self._generate_orthographic_rays(Plans.AXIAL, 0.0, cam)
            else:
                eval_dataloader = self._get_eval_dataloader()
                outputs = next(iter(eval_dataloader))
                camera = outputs[0]
                camera = camera.to(device)
                ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
            ray_bundle = ray_bundle.to(device)
            if model_metadata is not None:
                # Some models (e.g., PixelNeRF) require additional per-ray-bundle metadata
                # such as focal length and source-view conditioning tensors.
                injected = 0
                for k, v in model_metadata.items():
                    if isinstance(v, torch.Tensor):
                        v = v.to(device)
                    elif isinstance(v, dict):
                        # Best-effort device move for nested tensor dicts.
                        v = {
                            kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                            for kk, vv in v.items()
                        }
                    ray_bundle.metadata[k] = v
                    injected += 1
                if injected == 0:
                    config.logger.warning(
                        f"⚠️ No metadata keys injected for {self.model_name}; FPS measurement may be invalid."
                    )

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            try:
                # Warmup
                for _ in range(num_warmup):
                    self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)

                torch.cuda.synchronize()
                start_event.record()

                # Benchmark
                for _ in range(num_test):
                    self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
            except KeyError as e:
                config.logger.warning(
                    f"⚠️ Skipping FPS measurement for {self.model_name}: {e}"
                )
                return 0.0
            except Exception as e:
                config.logger.warning(
                    f"⚠️ Skipping FPS measurement for {self.model_name} due to runtime error: {e}"
                )
                return 0.0

            end_event.record()
            torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        time_per_frame_s = (total_time_ms / 1000.0) / num_test
        fps = 1.0 / time_per_frame_s

        config.logger.info(
            f"⚡ Inferência ({mode.value}): {fps:.2f} FPS ({time_per_frame_s * 1000:.2f} ms/frame)"
        )
        return fps

    def get_perspective_test_metrics(
        self, metrics: List[AvailableMetrics], downscale_factor: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Review rendered images from the perspective test set and compute quality metrics against ground truth.
        The Metrics are PSNR, SSIM, LPIPS, and FID, which are standard for image quality assessment in NeRF papers.
        """
        assert self.pipeline is not None and self.model is not None, (
            "Pipeline and model must be loaded for evaluation."
        )

        self.model.eval()
        rendered_images = []
        gt_images = []
        eval_dataloader = self._get_eval_dataloader()

        config.logger.info(
            "🔍 Evaluating perspective test metrics against ground truth views... (This may take a while depending on the number of test views and image resolution.)"
        )

        def ensure_3d_rgb(img: torch.Tensor, th: int, tw: int) -> torch.Tensor:
            if img.ndim == 3:
                return img
            elif img.ndim == 4:
                return img.squeeze(0)
            elif img.ndim == 2:
                C = img.shape[1]
                try:
                    return img.reshape(th, tw, C)
                except Exception:
                    H = int(np.sqrt(img.shape[0]))
                    W = img.shape[0] // H
                    return img.reshape(H, W, C)
            return img

        target_h, target_w = None, None
        with torch.no_grad():
            model_metadata = None
            if self.model_name in ["gnt", "pixel-nerf"]:
                try:
                    eval_ray_bundle, eval_batch = self.pipeline.datamanager.next_eval(step=0)
                    if self.model_name == "gnt":
                        self.pipeline._inject_gnt_metadata(eval_ray_bundle, eval_batch, split="eval")
                    model_metadata = eval_ray_bundle.metadata
                except Exception as e:
                    config.logger.warning(f"Failed to fetch metadata in perspective eval for {self.model_name}: {e}")

            for outputs in eval_dataloader:
                camera = outputs[0].to(self.pipeline.device)
                if downscale_factor is not None and downscale_factor != 1:
                    camera.rescale_output_resolution(1.0 / downscale_factor)
                batch = outputs[1]
                if model_metadata is not None:
                    ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
                    ray_bundle = ray_bundle.to(self.pipeline.device)
                    for k, v in model_metadata.items():
                        if isinstance(v, torch.Tensor):
                            v = v.to(self.pipeline.device)
                        elif isinstance(v, dict):
                            v = {
                                kk: (vv.to(self.pipeline.device) if isinstance(vv, torch.Tensor) else vv)
                                for kk, vv in v.items()
                            }
                        ray_bundle.metadata[k] = v
                    outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
                else:
                    outputs = self.pipeline.model.get_outputs_for_camera(camera)
                # Extract RGB and GT images, move to CPU for metric calculations
                rendered_rgb = outputs["rgb"].cpu()
                gt_rgb = batch["image"].cpu()

                if target_h is None or target_w is None:
                    target_h = int(camera.height.item())
                    target_w = int(camera.width.item())

                rendered_rgb = ensure_3d_rgb(rendered_rgb, target_h, target_w)
                gt_rgb = ensure_3d_rgb(gt_rgb, target_h, target_w)

                # Resize gt_rgb if it doesn't match target shape
                if gt_rgb.shape[0] != target_h or gt_rgb.shape[1] != target_w:
                    config.logger.warning(
                        f"⚠️ GT image shape mismatch: {gt_rgb.shape} vs target ({target_h}, {target_w}). Resizing."
                    )
                    gt_rgb_t = gt_rgb.permute(2, 0, 1).unsqueeze(0)
                    gt_rgb = torch.nn.functional.interpolate(
                        gt_rgb_t,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)

                # Resize rendered_rgb if it doesn't match target shape
                if rendered_rgb.shape[0] != target_h or rendered_rgb.shape[1] != target_w:
                    config.logger.warning(
                        f"⚠️ Rendered image shape mismatch: {rendered_rgb.shape} vs target ({target_h}, {target_w}). Resizing."
                    )
                    rendered_rgb_t = rendered_rgb.permute(2, 0, 1).unsqueeze(0)
                    rendered_rgb = torch.nn.functional.interpolate(
                        rendered_rgb_t,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)

                rendered_images.append(rendered_rgb)
                gt_images.append(gt_rgb)

        rendered_tensor = (torch.stack(rendered_images).permute(0, 3, 1, 2) * 255).to(
            torch.uint8
        )
        gt_tensor = (torch.stack(gt_images).permute(0, 3, 1, 2) * 255).to(torch.uint8)
        return self.compare_images_quality(rendered_tensor, gt_tensor, metrics)

    def evaluate_all_metrics(
        self,
        mode: RenderMode,
        metrics: List[AvailableMetrics],
        already_rendered_slices: Optional[List[np.ndarray]] = None,
        downscale_factor: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Returns a comprehensive dictionary of evaluation results, including:
            - Timestamp of evaluation
            - Memory footprint of the model in bytes
            - Inference FPS in the specified rendering mode
            - Visual quality metrics (FID, PSNR, SSIM, LPIPS) if applicable
        The method adapts to the rendering mode:
            - Orthographic: Evaluates quality against ground truth slices if available, otherwise renders new slices for evaluation.
            - Perspective: Evaluates quality against the test set views using the pipeline's dataloader
        """
        if self.pipeline is None:
            if self.images and self.images.dataset_path:
                path = self.get_output_path(self.images.dataset_path)
                loaded = self.load_from_disk(path, mode="test", downscale_factor=downscale_factor)
                if not loaded:
                    logger.error(
                        f"❌ Failed to load pipeline for metrics evaluation at {path}"
                    )
                    return {}


        results = {}
        results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        param_size, cuda_mem = self.get_memory_footprint()
        results["memory_footprint_bytes"] = param_size
        results["cuda_memory_usage_bytes"] = cuda_mem
        results["inference_fps"] = self.measure_inference_fps(mode)

        # visual quality metrics depends on the rendering mode and available data
        if mode == RenderMode.ORTHOGRAPHIC:
            if already_rendered_slices is not None:
                rendered_slices = already_rendered_slices
                logger.info(
                    "✅ Using pre-rendered orthographic slices for quality evaluation."
                )
            else:
                rendered_slices = self.render(num_slices=10, plan=Plans.AXIAL)
                if not rendered_slices:
                    logger.warning(
                        "⚠️ No rendered slices available for quality evaluation."
                    )
                    return results
            quality_metrics = self.evaluate_rendered_images_quality(
                rendered_slices, metrics
            )

        else:
            quality_metrics = self.get_perspective_test_metrics(metrics, downscale_factor=downscale_factor)

        if quality_metrics:
            results.update(quality_metrics)

        return results

    def evaluate_rendered_images_quality(
        self,
        rendered_images: List[np.ndarray],
        metrics: Optional[List[AvailableMetrics]] = None,
    ) -> Optional[Dict[str, float]]:
        if self.images is None:
            logger.warning(
                "⚠️ No dataset attached to NeRF model. Metrics evaluation skipped."
            )
            return None

        image_paths = self.images.get_image_paths()
        if not image_paths:
            logger.warning(
                "⚠️ No ground truth images available for NeRF metric evaluation."
            )
            return None
        if not rendered_images:
            logger.warning("⚠️ No rendered images available for NeRF metric evaluation.")
            return None

        limit = min(len(rendered_images), len(image_paths))
        if limit == 0:
            logger.warning("⚠️ Empty evaluation set for NeRF metrics.")
            return None

        rendered_tensor = self._rendered_images_to_tensor(rendered_images[:limit])
        ground_truth_tensor = self._ground_truth_images_to_tensor(
            image_paths=image_paths,
            image_size=config.img_size,
            limit=limit,
        )
        if rendered_tensor is None or ground_truth_tensor is None:
            return None

        effective_count = min(rendered_tensor.shape[0], ground_truth_tensor.shape[0])
        if effective_count == 0:
            logger.warning("⚠️ Empty tensors generated for NeRF metrics evaluation.")
            return None

        # Resize rendered_tensor to match ground_truth_tensor shape if they differ
        if rendered_tensor.shape[2:] != ground_truth_tensor.shape[2:]:
            logger.warning(
                f"⚠️ Shape mismatch in ORTHOGRAPHIC metrics: rendered {rendered_tensor.shape[2:]} vs GT {ground_truth_tensor.shape[2:]}. Resizing rendered to match GT."
            )
            rendered_tensor = torch.nn.functional.interpolate(
                rendered_tensor.float(),
                size=ground_truth_tensor.shape[2:],
                mode="bilinear",
                align_corners=False
            ).to(torch.uint8)

        selected_metrics = metrics or [
            AvailableMetrics.FID,
            AvailableMetrics.PSNR,
            AvailableMetrics.SSIM,
            AvailableMetrics.LPIPS,
        ]
        return self.compare_images_quality(
            rendered_images=rendered_tensor[:effective_count].to(torch.uint8),
            ground_truth_images=ground_truth_tensor[:effective_count].to(torch.uint8),
            metrics=selected_metrics,
        )

    def get_memory_footprint(self) -> Tuple[int, int]:
        """Returns the memory footprint of the NeRF model."""
        if self.model is None:
            return 0

        cuda_mem_usage = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        param_size = sum(
            param.numel() * param.element_size() for param in self.model.parameters()
        )
        if cuda_mem_usage - param_size > 0:
            config.logger.warning(
                f"⚠️ Detected CUDA memory usage ({cuda_mem_usage} bytes) exceeds model parameter size ({param_size} bytes). "
                "This may indicate additional memory usage from activations, buffers, or other components. "
                "Reported memory footprint will reflect parameter size only."
            )
        return param_size, cuda_mem_usage

    def _generate_orthographic_rays(
        self, plan: Plans, position: float, cam_config: GenerateCameraConfig
    ):
        """
        Mounts the extrinsic and intrinsic parameters for an orthographic camera based on the specified plane and position.
        """
        position = float(position)

        # 1. Extraindo os parâmetros do Dataclass
        cx, cy = cam_config.center
        ext_x, ext_y = cam_config.extent
        width, height = cam_config.resolution

        # 2. Configurando a Translação da Câmera (Pan) dependendo do plano
        if plan == Plans.AXIAL:
            c2w = torch.tensor(
                [[1.0, 0.0, 0.0, cx], [0.0, 1.0, 0.0, cy], [0.0, 0.0, 1.0, position]]
            ).float()
        elif plan == Plans.SAGITAL:
            c2w = torch.tensor(
                [[0.0, 0.0, -1.0, position], [0.0, 1.0, 0.0, cy], [1.0, 0.0, 0.0, cx]]
            ).float()
        elif plan == Plans.CORONAL:
            c2w = torch.tensor(
                [[1.0, 0.0, 0.0, cx], [0.0, 0.0, -1.0, position], [0.0, 1.0, 0.0, cy]]
            ).float()
        else:
            c2w = torch.eye(4)[:3, :4].float()

        fx = width / ext_x
        fy = height / ext_y
        center_x_pixel = width / 2.0
        center_y_pixel = height / 2.0

        camera = Cameras(
            camera_to_worlds=c2w.unsqueeze(0),
            fx=torch.tensor([fx], dtype=torch.float32),
            fy=torch.tensor([fy], dtype=torch.float32),
            cx=torch.tensor([center_x_pixel], dtype=torch.float32),
            cy=torch.tensor([center_y_pixel], dtype=torch.float32),
            width=torch.tensor([width], dtype=torch.int64),
            height=torch.tensor([height], dtype=torch.int64),
            camera_type=CameraType.ORTHOPHOTO,
        ).to(self.pipeline.device)

        ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)
        base = ray_bundle.origins[..., :1]
        ray_bundle.nears = torch.zeros_like(base)
        ray_bundle.fars = torch.zeros_like(base) + cam_config.thickness

        return ray_bundle

    def render_image(
        self,
        plan: Plans,
        position: float,
        cam_config: GenerateCameraConfig = GenerateCameraConfig(),
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extracts a single orthographic slice from the trained NeRF model.
        """
        if not hasattr(self, "pipeline") or self.pipeline is None:
            config.logger.error("❌ Pipeline not loaded. Call load_from_disk first.")
            return None

        with torch.no_grad():
            ray_bundle = self._generate_orthographic_rays(plan, position, cam_config)
            ray_bundle = ray_bundle.to(self.pipeline.device)
            
            if self.model_name in ["gnt", "pixel-nerf"]:
                try:
                    eval_ray_bundle, eval_batch = self.pipeline.datamanager.next_eval(step=0)
                    if self.model_name == "gnt":
                        self.pipeline._inject_gnt_metadata(eval_ray_bundle, eval_batch, split="eval")
                    
                    for k, v in eval_ray_bundle.metadata.items():
                        if isinstance(v, torch.Tensor):
                            eval_ray_bundle.metadata[k] = v.to(self.pipeline.device)
                    ray_bundle.metadata.update(eval_ray_bundle.metadata)
                except Exception as e:
                    config.logger.warning(f"Failed to fetch metadata in render_image for {self.model_name}: {e}")

            outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)

        return outputs

    def render_ortographic(
        self, plan, slice_axis, num_slices, save_path: Optional[Path]
    ) -> List[np.ndarray]:
        assert self.pipeline is not None, (
            "Pipeline must be loaded to render orthographic slices."
        )

        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

        min_bound = np.array([0.0, 0.0, 0.0])
        max_bound = np.array([1.0, 1.0, 1.0])

        SAFE_MARGIN = 0.0
        if plan == Plans.AXIAL:
            slice_axis = 0
            img_axis_1, img_axis_2 = 1, 2
        elif plan == Plans.SAGITAL:
            slice_axis = 1
            img_axis_1, img_axis_2 = 0, 2
        elif plan == Plans.CORONAL:
            slice_axis = 2
            img_axis_1, img_axis_2 = 0, 1

        min_val = min_bound[slice_axis] - SAFE_MARGIN
        max_val = max_bound[slice_axis] - SAFE_MARGIN
        slab_size = (max_val - min_val) / num_slices
        output = []

        # Centro dinâmico baseado no plano da imagem
        center_1 = (min_bound[img_axis_1] + max_bound[img_axis_1]) / 2.0
        center_2 = (min_bound[img_axis_2] + max_bound[img_axis_2]) / 2.0
        CAMERA_POS = (center_1, center_2)

        SAFE_EXTENT_CONSTANT = 0.0
        THICKNESS_MULTIPLIER = 1.25

        logger.info(
            f"Calculated camera center: {CAMERA_POS}, slab size: {slab_size:.3f}"
        )
        logger.info(f"Raw AABB: min={min_bound}, max={max_bound}")
        logger.info(
            f"Extent calculated: x={max_bound[0] - min_bound[0] - SAFE_EXTENT_CONSTANT}, y={max_bound[1] - min_bound[1] - SAFE_EXTENT_CONSTANT}"
        )
        cam = GenerateCameraConfig(
            center=(CAMERA_POS),
            extent=(
                max_bound[0] - min_bound[0] - SAFE_EXTENT_CONSTANT,
                max_bound[1] - min_bound[1] - SAFE_EXTENT_CONSTANT,
            ),
            resolution=(config.img_size, config.img_size),
            thickness=slab_size * THICKNESS_MULTIPLIER,
        )

        for i in range(num_slices):
            slice_center = min_val + (i + 0.5) * slab_size
            # extract image via orthographic camera
            render_dict = self.render_image(
                plan=plan, position=slice_center, cam_config=cam
            )

            if render_dict and "rgb" in render_dict:
                rgb_img = render_dict["rgb"].cpu().numpy()
                if rgb_img.ndim == 2:
                    rgb_img = rgb_img.reshape(config.img_size, config.img_size, 3)
                elif rgb_img.ndim == 4:
                    rgb_img = rgb_img.squeeze(0)
                rgb_img = np.clip(rgb_img, 0.0, 1.0)
                rgb_uint8 = (rgb_img * 255.0).astype(np.uint8)

                if save_path is not None:
                    png_out = save_path / f"slice_nerf_{i:03d}.png"
                    Image.fromarray(rgb_uint8).save(png_out)

                    logger.info(
                        f"✅ Saved NeRF slice {i} at z={slice_center:.3f} → {png_out}"
                    )

                output.append(rgb_uint8)
        return output

    def render_perspective(self, save_path: Optional[Path], downscale_factor: Optional[int] = None) -> List[np.ndarray]:
        assert self.pipeline is not None, (
            "Pipeline must be loaded to render perspective views."
        )

        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

        rendered_images = []
        eval_dataloader = self._get_eval_dataloader()
        with torch.no_grad():
            model_metadata = None
            if self.model_name in ["gnt", "pixel-nerf"]:
                try:
                    eval_ray_bundle, eval_batch = self.pipeline.datamanager.next_eval(step=0)
                    if self.model_name == "gnt":
                        self.pipeline._inject_gnt_metadata(eval_ray_bundle, eval_batch, split="eval")
                    model_metadata = eval_ray_bundle.metadata
                except Exception as e:
                    config.logger.warning(f"Failed to fetch metadata in perspective render for {self.model_name}: {e}")

            for cam_idx, outputs in enumerate(eval_dataloader):
                camera = outputs[0].to(self.pipeline.device)
                if downscale_factor is not None and downscale_factor != 1:
                    camera.rescale_output_resolution(1.0 / downscale_factor)
                if model_metadata is not None:
                    ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
                    ray_bundle = ray_bundle.to(self.pipeline.device)
                    for k, v in model_metadata.items():
                        if isinstance(v, torch.Tensor):
                            model_metadata[k] = v.to(self.pipeline.device)
                    ray_bundle.metadata.update(model_metadata)
                    outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
                else:
                    outputs = self.pipeline.model.get_outputs_for_camera(camera)
                rendered_rgb = outputs["rgb"].cpu().numpy()
                rendered_images.append(rendered_rgb)

                if save_path is not None:
                    img_uint8 = np.clip(rendered_rgb, 0.0, 1.0)
                    img_uint8 = (img_uint8 * 255.0).astype(np.uint8)
                    png_out = save_path / f"perspective_view_{cam_idx:03d}.png"
                    Image.fromarray(img_uint8).save(png_out)
                    logger.info(
                        f"✅ Saved perspective rendered view {cam_idx} → {png_out}"
                    )
        return rendered_images

    def render(
        self,
        mode=RenderMode.ORTHOGRAPHIC,
        save_path: Optional[Path] = None,
        num_slices: int = 10,
        plan: Optional[Plans] = None,
        downscale_factor: Optional[int] = None,
    ) -> Optional[List[np.ndarray]]:
        """
        Renders either orthographic slices or perspective views from the NeRF model based on the specified mode.
         - Orthographic: Renders a series of slices along the specified plane and saves them if requested.
         - Perspective: Renders views from the evaluation camera poses and saves them if requested.
         The method ensures the pipeline is loaded before rendering and handles output organization for both modes.
         Returns a list of rendered images as numpy arrays, or None if rendering fails.
         Note: Rendering can be time-consuming depending on the number of slices/views and image resolution.
        """
        if not self.images or not self.images.dataset_path:
            logger.error("❌ No dataset path available for slicing.")
            return
        path = self.get_output_path(self.images.dataset_path)
        if self.pipeline is None:
            loaded = self.load_from_disk(path, mode="inference", downscale_factor=downscale_factor)
            if not loaded:
                logger.error(f"❌ Failed to load pipeline for NeRF slicing at {path}")
                return

        if self.pipeline is None:
            logger.error(f"❌ Failed to load pipeline for NeRF slicing at {path}")
            return

        output_dir = self.get_augmentation_output_path(self.images.dataset_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        if mode == RenderMode.ORTHOGRAPHIC:
            if not plan:
                logger.error("❌ Plan must be specified for orthographic rendering.")
                return
            return self.render_ortographic(
                plan, slice_axis=0, num_slices=num_slices, save_path=save_path
            )
        elif mode == RenderMode.PERSPECTIVE:
            return self.render_perspective(save_path=save_path, downscale_factor=downscale_factor)


def download_pretrained_pixelnerf_weights(path: Path) -> Path:
    import gdown

    output = path
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(output):
        print("Downloading pretrained PixelNeRF weights...")
        gdown.download(id="1UO_rL201guN6euoWkCOn-XpqR2e8o6ju", output=str(output), quiet=False)
    else:
        print("Pretrained PixelNeRF weights already downloaded.")

    unzipped_path = path.parent / "pixelnerf_pretrained"
    if not os.path.exists(unzipped_path):
        print("Unzipping pretrained weights...")
        import zipfile
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(unzipped_path)
        print(f"Pretrained weights downloaded and unzipped to {unzipped_path}")
    
    # Check if there is a nested folder with the same name
    nested_path = unzipped_path / "pixelnerf_pretrained"
    if nested_path.exists() and nested_path.is_dir():
        return nested_path
    return unzipped_path


def download_pretrained_gnt_model(path):
    from gdown import download

    output = path
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(output):
        download(id="1YvOJXa5eGpKgoMYcxC2ma7prB1n5UwRn", output=str(output.resolve()), quiet=False)
        print(f"Pretrained GNT model downloaded to {output}")
