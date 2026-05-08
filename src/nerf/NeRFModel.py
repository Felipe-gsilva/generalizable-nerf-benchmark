from typing import Dict, List, Literal, Optional
from PIL import Image
from nerfstudio.pipelines.base_pipeline import Pipeline
from src.validation.eval_images import calculate_fid, calculate_psnr_ssim_lpips
from dataset.ImageDataset import ImageDataset
from utils.config import config
from utils.types import AvailableMetrics, Plans, GenerateCameraConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import profiler
from pathlib import Path

import yaml
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
import tempfile
import re

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

    def __init__(
        self,
        model_name: str,
        images: Optional[ImageDataset] = None,
        num_views: int = 0,  # 0 = Usa o dataset todo. >0 = Few-shot.
        split: List[float] = [0.7, 0.15, 0.15],
        split_strategy: str = "uniform",  # "uniform" ou "random"
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

        # Parâmetros do Paper
        self.num_views = num_views
        if sum(split) != 1.0:
            raise ValueError("Split ratios must sum to 1.0")
        self.val_split = split[1]
        self.test_split = split[2]
        self.split_strategy = split_strategy

    def _apply_dataset_splits(self, data_path: Path) -> Path:
        """
        Calcula os splits de train/val/test baseados nos parâmetros da classe
        e salva um novo arquivo JSON para o Nerfstudio ler.
        """
        json_path = data_path / "transforms.json"

        if self.num_views > 0:
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

        # 1. Definindo as quantidades absolutas de cada split
        if self.num_views > 0 and self.num_views < total_images:
            train_count = self.num_views
            remaining = total_images - train_count

            # Divide o restante proporcionalmente entre val e test
            pool_ratio = self.val_split + self.test_split
            if pool_ratio > 0:
                val_count = int(remaining * (self.val_split / pool_ratio))
            else:
                val_count = remaining // 2
            test_count = remaining - val_count
        else:
            # Split padrão percentual (caso num_views = 0)
            train_count = int(total_images * (1.0 - self.val_split - self.test_split))
            val_count = int(total_images * self.val_split)
            test_count = total_images - train_count - val_count

        # 2. Selecionando os Índices de Treino (Estratégia Few-Shot)
        all_indices = np.arange(total_images)
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

        # 3. Selecionando Val e Test aleatoriamente do que sobrou
        remaining_indices = [idx for idx in all_indices if idx not in train_indices]
        np.random.seed(
            42
        )  # Seed fixa garante que testes diferentes avaliem nas mesmas imagens
        np.random.shuffle(remaining_indices)

        val_set = set(remaining_indices[:val_count])
        test_set = set(remaining_indices[val_count:])
        train_set = set(train_indices)

        # 4. Marcando o arquivo
        for i, frame in enumerate(frames):
            if i in train_set:
                frame["split"] = "train"
            elif i in val_set:
                frame["split"] = "val"
            else:
                frame["split"] = "test"

        # 5. Salvando o arquivo de experimento
        with open(out_json_path, "w") as f:
            json.dump(data, f, indent=4)

        config.logger.info(
            f"📊 Dataset Split [{out_json_path.name}]: {len(train_set)} Train | {len(val_set)} Val | {len(test_set)} Test"
        )
        return out_json_path

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

    def _patch_config_yaml(self, config_path: Path, checkpoint_dir: Path) -> Path:
        """
        Nerfstudio's config.yml contains 'output_dir' and 'experiment_name' which it uses
        to find checkpoints. If these mismatch the current location, loading fails.
        This method patches the config to set an explicit 'load_dir', which bypasses
        the folder crawling logic and makes loading robust.
        """
        try:
            with config_path.open("r") as f:
                content = f.read()

            # Set load_dir to the absolute path of the checkpoint_dir
            load_dir_parts = checkpoint_dir.resolve().parts
            replacement_parts = "\n".join([f"- {p}" for p in load_dir_parts])
            load_dir_yaml = f"load_dir: !!python/object/apply:pathlib.PosixPath\n{replacement_parts}"

            # Replace load_dir: null with the actual path
            content = re.sub(r"load_dir: null", load_dir_yaml, content)

            # Also update experiment_name and output_dir to match current structure
            parts = checkpoint_dir.parts
            try:
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
                content = re.sub(output_dir_pattern, r"\1" + out_parts + "\n", content)
            except Exception:
                pass

            # Save to a temporary file
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
        eval_num_rays_per_chunk: Optional[int] = None,
    ) -> bool:
        """Loads a NeRF model from disk using eval_setup."""
        config.logger.info(f"Loading NeRF model from {path}...")
        c_path = self._get_latest_checkpoint(path) if path.is_dir() else path
        if not c_path or not Path(c_path).exists():
            config.logger.error(f"❌ No checkpoint found to load from {path}")
            return False

        temp_config_path = None
        try:
            config_path = Path(c_path).parent / "config.yml"
            if not config_path.exists():
                raise FileNotFoundError(f"config.yml not found at {config_path}")

            # Patch the config to handle moved checkpoints or "dirty" names
            temp_config_path = self._patch_config_yaml(config_path, Path(c_path))

            _, pipeline, _, _ = eval_setup(
                config_path=temp_config_path,
                test_mode=mode,
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
                f"(mode={mode}, eval_num_rays_per_chunk={eval_num_rays_per_chunk}): {e}"
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

        if self._uses_custom_transforms():
            return any(
                "registered_" in f.name for f in directory.iterdir() if f.is_file()
            )

        return True

    def _find_images_dir(
        self, data_path: Path, max_depth: int = MAX_IMAGE_SEARCH_DEPTH
    ) -> Path:
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
        - 3D_virtual_HE_staining: custom grid transforms.
        - Other datasets: COLMAP via ns-process-data.
        num_imgs: if > 0, limits the number of images in transforms.json (for quick validation).
        """
        transforms_path = data_path / "transforms.json"
        if transforms_path.exists() and num_imgs == 0:
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
        """Train the NeRF model using the provided data and configuration via CLI subprocess."""
        if not self.process_data(data_path, num_imgs=num_imgs):
            config.logger.error(
                f"❌ Aborting training for {data_path}: data preparation failed."
            )
            return

        # 1. Setup clean names for Nerfstudio args
        clean_dataset = (
            self.dataset_name.split("_images")[0] if self.dataset_name else "unknown"
        )
        clean_target = self._get_target_id(data_path)
        if "__" in clean_target:
            clean_target = clean_target.split("__")[-1]

        base_output_dir = Path("assets/data/nerf_checkpoints")
        experiment_name = f"{clean_dataset}/{clean_target}"

        # 2. Check for resumes
        output_path = self.get_output_path(data_path)
        load_dir_arg = []
        checkpoint_overrides = []
        latest_checkpoint = self._get_latest_checkpoint(output_path)

        if latest_checkpoint:
            config.logger.info(f"🔄 Checkpoint found: {latest_checkpoint}. Resuming.")
            load_dir_arg = ["--load-dir", str(latest_checkpoint)]
            checkpoint_overrides, saved_downscale = (
                self._get_checkpoint_config_overrides(latest_checkpoint)
            )

            # 🔥 Safely override the downscale factor if resuming an older run
            if saved_downscale is not None and saved_downscale != downscale_factor:
                config.logger.warning(
                    f"⚠️ Checkpoint trained with downscale_factor={saved_downscale}. "
                    f"Overriding requested {downscale_factor} to prevent IndexError."
                )
                downscale_factor = saved_downscale
        else:
            config.logger.info("🆕 No checkpoint found. Starting fresh training.")

        config.logger.info(
            f"🚀 Training {self.model_name} on {data_path} (Downscale: {downscale_factor}x)..."
        )

        split_json_path = self._apply_dataset_splits(data_path)

        cmd = (
            [
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
            + load_dir_arg
            + checkpoint_overrides
        )

        if self.model_name == "gnt":
            cmd += [
                "--pipeline.model.N-samples",
                "48",
                "--pipeline.model.N-importance",
                "48",
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
            ]
        elif self.model_name == "splatfacto":
            cmd += []
        elif self.model_name == "merf-ns":
            cmd += [
                "--pipeline.model.eval-num-rays-per-chunk",
                "1024",
                "--pipeline.model.s3im-loss-mult",
                "0.0",
            ]
        cmd += [
            "--pipeline.datamanager.cache-images-type",
            "uint8",
            "--pipeline.datamanager.train-num-images-to-sample-from",
            "500",
            "--pipeline.model.eval-num-rays-per-chunk",
            "1024",
            "--pipeline.datamanager.train-num-rays-per-batch",
            "1024",
            # "--auto-scale-poses", "False",
        ]

        cmd += [
            "nerfstudio-data",
            "--downscale-factor",
            str(downscale_factor),
            "--eval-mode",
            "filename",
        ]

        process_env = os.environ.copy()
        process_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        log_dir = Path("assets/logs") / self.model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{clean_dataset}_{clean_target}_{timestamp}.log"

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
        return sum(
            param.numel() * param.element_size() for param in self.model.parameters()
        )

    def print_scene_bounds(self):
        """Imprime os limites do modelo para ajudar no alinhamento da câmera."""
        if not hasattr(self, "pipeline") or self.pipeline is None:
            print("Pipeline não carregado.")
            return

        # O AABB (Axis-Aligned Bounding Box) dita os limites de renderização
        aabb = self.pipeline.model.scene_box.aabb
        print(f"Limites da Cena (Min X,Y,Z): {aabb[0].tolist()}")
        print(f"Limites da Cena (Max X,Y,Z): {aabb[1].tolist()}")

        # O centro ideal para posicionar a câmera
        centro = (aabb[0] + aabb[1]) / 2.0
        print(f"Centro Geométrico Estimado: {centro.tolist()}")

        # O extent (tamanho) recomendado
        tamanho = aabb[1] - aabb[0]
        print(f"Tamanho do Objeto em X, Y, Z: {tamanho.tolist()}")

    def render_image(
        self,
        plan: Plans,
        position: float,
        cam_config: GenerateCameraConfig = GenerateCameraConfig(),
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extracts a single orthographic slice from the trained NeRF model.
        Uses a squashed ray bundle to sample a thin plane in the 3D volume.

        Args:
            plan: slicing plane (axial, sagital, coronal)
            position: coordinate along the chosen axis where the slice is taken
            width, height: output resolution
            center_x, center_y: center of the camera view in NeRF coordinates (default 0,0)
            extent_x, extent_y: Physical coverage of the camera view in NeRF coordinates.
        """

        if not hasattr(self, "pipeline") or self.pipeline is None:
            config.logger.error("❌ Pipeline not loaded. Call load_from_disk first.")
            return None

        self.print_scene_bounds()

        if not hasattr(self, "pipeline") or self.pipeline is None:
            config.logger.error("❌ Pipeline not loaded. Call load_from_disk first.")
            return None

        position = float(position)
        c2w = torch.eye(4)[:3, :4].float()

        # 1. Extraindo os parâmetros do Dataclass
        cx, cy = cam_config.center
        ext_x, ext_y = cam_config.extent
        width, height = cam_config.resolution

        # 2. Configurando a Translação da Câmera (Pan)
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

        # 3. Configurando a Escala (Zoom)
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

        # 4. Aplicando a Espessura Dinâmica
        base = ray_bundle.origins[..., :1]
        ray_bundle.nears = torch.zeros_like(base)
        ray_bundle.fars = torch.zeros_like(base) + cam_config.thickness

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
            loaded = self.load_from_disk(
                path, mode="inference", eval_num_rays_per_chunk=1024
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

        cam = GenerateCameraConfig(
            center=(0.0, 0.0),
            extent=(max_bound[0] - min_bound[0], max_bound[1] - min_bound[1]),
            resolution=(config.img_size, config.img_size),
            thickness=slab_size * 1.25,
        )

        for i in range(num_slices):
            slice_center = min + (i + 0.5) * slab_size
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
