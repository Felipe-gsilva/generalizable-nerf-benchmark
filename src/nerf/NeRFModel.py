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

from typing import Dict, List, Literal, Optional
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

    def _patch_config_yaml(self, config_path: Path, checkpoint_dir: Path) -> Path:
        """
        Patches the config to set an explicit 'load_dir', bypassing folder crawling logic.
        """
        try:
            with config_path.open("r") as f:
                content = f.read()

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
            temp_config_path = self._patch_config_yaml(config_path, Path(c_path))

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
            "--colmap-cmd",
            "colmap",
        ]
        process_env = os.environ.copy()
        process_env["QT_QPA_PLATFORM"] = "offscreen"
        process_env["PATH"] = (
            f"{os.path.expanduser('~/.local/bin')}:{os.environ['PATH']}"
        )
        process_env.pop("DISPLAY", None)
        process_env.pop("XAUTHORITY", None)

        try:
            subprocess.run(cmd, check=True, env=process_env)
            config.logger.info("✅ ns-process-data finished successfully.")
            return True
        except subprocess.CalledProcessError as e:
            config.logger.error(f"❌ ns-process-data failed with code: {e.returncode}")
            return False

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
    def train(self, data_path: Path, downscale_factor: int = 1, num_imgs: int = 0):
        """Prepares data and handles training/loading according to the selected regime."""
        if not self.process_data(data_path, num_imgs=num_imgs):
            config.logger.error(
                f"❌ Aborting training for {data_path}: data preparation failed."
            )
            return

        split_json_path = self._apply_dataset_splits(data_path)
        output_path = self.get_output_path(data_path)

        if self.regime == "zero-shot":
            config.logger.info(
                f"❄️ Zero-shot regime enabled for {self.model_name}. Skipping local optimization."
            )
            global_weights_dir = Path("assets/data/pretrained") / self.model_name
            success = self.load_from_disk(global_weights_dir, mode="inference")
            if not success:
                raise RuntimeError(
                    "Failed to load pretrained weights for zero-shot from: "
                    f"{global_weights_dir}"
                )
            return

        clean_dataset = (
            self.dataset_name.split("_images")[0] if self.dataset_name else "unknown"
        )
        clean_target = self._get_target_id(data_path)
        if "__" in clean_target:
            clean_target = clean_target.split("__")[-1]

        base_output_dir = Path("assets/data/nerf_checkpoints")
        experiment_name = f"{clean_dataset}/{clean_target}"

        cmd = [
            "python",
            "-c",
            "import torch; torch.backends.cudnn.enabled=False; torch.backends.cudnn.benchmark=False; import sys; sys.argv.pop(0); from nerfstudio.scripts.train import entrypoint; entrypoint()",
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
                "--pipeline.model.N-samples",
                "32",
                "--pipeline.model.N-importance",
                "32",
                "--mixed-precision",
                "True",
                "--pipeline.model.eval-num-rays-per-chunk",
                "512",
                "--pipeline.datamanager.train-num-rays-per-batch",
                "512",
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
                "2",
                "--pipeline.model.netwidth",
                "128",
                "--pipeline.model.eval-num-rays-per-chunk",
                "512",
                "--pipeline.datamanager.train-num-rays-per-batch",
                "512",
                "--pipeline.datamanager.cache-images-type",
                "uint8",
            ]
        elif self.model_name == "nerfacto":
            cmd += [
                "--pipeline.model.eval-num-rays-per-chunk",
                "1024",
                "--pipeline.datamanager.train-num-rays-per-batch",
                "1024",
                "--pipeline.model.log2-hashmap-size",
                "16",
                "--pipeline.model.camera-optimizer.mode",
                "off",
                "--pipeline.model.background-color",
                "white",
            ]
        elif self.model_name == "splatfacto":
            cmd += []
        elif self.model_name == "merf-ns":
            cmd += [
                "--pipeline.model.s3im-loss-mult",
                "0.0",
                # "--pipeline.datamanager.train-num-images-to-sample-from",
                # "500",
                # "--pipeline.model.eval-num-rays-per-chunk",
                # "256",
                # "--pipeline.datamanager.train-num-rays-per-batch",
                # "256",
                # "--auto-scale-poses", "False",
            ]
        if not self.model_name == "splatfacto":
            cmd += [
                # "--pipeline.datamanager.cache-images-type",
                # "uint8",
                # "--pipeline.datamanager.train-num-images-to-sample-from",
                # "500",
                # "--pipeline.model.eval-num-rays-per-chunk",
                # "256",
                # "--pipeline.datamanager.train-num-rays-per-batch",
                # "256",
                # "--auto-scale-poses", "False",
            ]

        if self.regime == "tta" and self.model_name in ["pixel-nerf", "gnt"]:
            cmd += [
                "--pipeline.model.transfer_learning",
                "True",
            ]

        cmd += [
            "nerfstudio-data",
            "--downscale-factor",
            str(downscale_factor),
            "--eval-mode",
            "fraction",
        ]

        process_env = os.environ.copy()
        process_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

                for line in iter(process.stdout.readline, b""):
                    sys.stdout.write(line.decode("utf-8", errors="replace"))
                    f.write(line)
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
            config.logger.info(
                f"✅ Success training/adapting {self.model_name} on {data_path} via {self.regime}"
            )
            if self.regime == "tta":
                if not self.load_from_disk(output_path, mode="test"):
                    raise RuntimeError(
                        f"Failed to load adapted TTA weights from: {output_path}"
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
            gnt_metadata = None
            eval_ray_bundle, eval_batch = self.pipeline.datamanager.next_eval(step=0)
            if self.model_name == "gnt":
                eval_batch = self.pipeline._inject_gnt_metadata(
                    eval_ray_bundle, eval_batch, split="eval"
                )
                gnt_metadata = eval_ray_bundle.metadata

        except Exception as e:
            config.logger.error(
                f"Failed to fetch evaluation source metadata for GNT: {e}"
            )
            return 0.0

        if mode == RenderMode.ORTHOGRAPHIC:
            cam = GenerateCameraConfig(
                center=(0.0, 0.0),
                extent=(1.0, 1.0),
                resolution=(config.img_size, config.img_size),
                thickness=0.1,
            )
            ray_bundle = self._generate_orthographic_rays(Plans.AXIAL, 0.0, cam)
            # hot fix for gnt (I will try to unify this later but gnt's metadata handling is currently very coupled to the dataloader and eval batch)
            if self.model_name == "gnt" and gnt_metadata is not None:
                ray_bundle.metadata.update(gnt_metadata)
        else:
            ray_bundle = eval_ray_bundle

        ray_bundle = ray_bundle.to(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)

            torch.cuda.synchronize()
            start_event.record()

            # Benchmark
            for _ in range(num_test):
                self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)

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
        self, metrics: List[AvailableMetrics]
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

        eval_dataloader = self.pipeline.datamanager.fixed_indices_eval_dataloader

        config.logger.info(
            "🔍 Evaluating perspective test metrics against ground truth views... (This may take a while depending on the number of test views and image resolution.)"
        )
        with torch.no_grad():
            for camera, batch, _ in eval_dataloader:
                camera = camera.to(self.pipeline.device)
                outputs = self.pipeline.model.get_outputs_for_camera(camera)
                # Extract RGB and GT images, move to CPU for metric calculations
                rendered_rgb = outputs["rgb"].cpu()
                gt_rgb = batch["image"].cpu()
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
        results = {}
        results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        results["memory_footprint_bytes"] = self.get_memory_footprint()
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
            quality_metrics = self.get_perspective_test_metrics(metrics)

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

    def get_memory_footprint(self) -> int:
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
        return param_size

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

        ray_bundle = self._generate_orthographic_rays(plan, position, cam_config)

        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)

        return outputs

    def render(
        self,
        save: bool = False,
        num_slices: int = 10,
        plan: Plans = Plans.AXIAL,
    ) -> Optional[List[np.ndarray]]:
        if not self.images or not self.images.dataset_path:
            logger.error("❌ No dataset path available for slicing.")
            return
        path = self.get_output_path(self.images.dataset_path)
        if self.pipeline is None:
            loaded = self.load_from_disk(path, mode="inference")
            if not loaded:
                logger.error(f"❌ Failed to load pipeline for NeRF slicing at {path}")
                return

        if self.pipeline is None:
            logger.error(f"❌ Failed to load pipeline for NeRF slicing at {path}")
            return

        output_dir = self.get_augmentation_output_path(self.images.dataset_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Determine scaling and bounds for the slicing based on the pipeline's datamanager
        # aabb = self.pipeline.model.scene_box.aabb.cpu().numpy()
        # Im fixing the bounds to [0,1] for a test
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

                if save:
                    png_out = output_dir / f"slice_nerf_{i:03d}.png"
                    Image.fromarray(rgb_uint8).save(png_out)

                    logger.info(
                        f"✅ Saved NeRF slice {i} at z={slice_center:.3f} → {png_out}"
                    )

                output.append(rgb_uint8)
        return output
