import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional
from abc import ABC, abstractmethod

from src.validation.eval_images import calculate_fid, calculate_psnr_ssim_lpips
from src.utils.config import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_IMAGE_SEARCH_DEPTH = 3


class NerfModel(ABC):
    """
    The universal Base Class for all Generative and NeRF models in the pipeline.

    It strictly handles paths, data preprocessing, shared evaluation metrics,
    and enforces the core lifecycle contract (run, load, evaluate) without
    dictating HOW the underlying model is optimized.
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        image_priority: str = "highest",
        output_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.image_priority = image_priority

        # Shared state for tracking
        self._best_metric_value: float = float("inf")
        self.output_dir: Path = output_dir or Path("assets/data/nerf_checkpoints")

    # ------------------------------------------------------------------
    # (Subclasses MUST implement these)
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, data_path: Path) -> None:
        """
        The main entry point for training.
        - PyTorch wrappers will implement an epoch loop (for GANs).
        - CLI wrappers will trigger a subprocess (ns-train).
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Loads the model state from a given path."""
        pass

    @abstractmethod
    def evaluate(self, rendered_images, ground_truth_images) -> Dict[str, float]:
        """
        Evaluate the model. Subclasses must decide how to generate the
        rendered images, but they can use the base class's `compare_images_quality`
        to calculate the actual metrics.
        """
        pass

    @abstractmethod
    def get_memory_footprint(self) -> int:
        """Returns memory footprint in bytes."""
        pass

    # ------------------------------------------------------------------
    # 2. SHARED CONCRETE UTILITIES (Inherited by all subclasses)
    # ------------------------------------------------------------------

    def compare_images_quality(
        self, rendered_images, ground_truth_images
    ) -> Dict[str, float]:
        """
        Shared evaluation logic. Subclasses call this inside their `evaluate()` methods.
        """
        fid_score = calculate_fid(rendered_images, ground_truth_images)
        psnr_score, ssim_score, lpips_score = calculate_psnr_ssim_lpips(
            rendered_images, ground_truth_images
        )
        return {
            "fid": fid_score,
            "psnr": psnr_score,
            "ssim": ssim_score,
            "lpips": lpips_score,
        }

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

    def _downscale_factor_from_name(self, dir_name: str) -> int:
        if dir_name == "images":
            return 1
        if dir_name.startswith("images_"):
            suffix = dir_name.split("_", 1)[1]
            if suffix.isdigit():
                return int(suffix)
        return 1

    def _sort_image_dir_names(self, dir_names: list[str], priority: str) -> list[str]:
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
        path = self.target_id if self.target_id else data_path.name
        self.target_id = path
        return path

    def process_data(self, data_path: Path) -> bool:
        """
        Processes raw images into generic nerf format using
        ns-process-data images if transforms.json is missing.
        """
        if (data_path / "transforms.json").exists():
            config.logger.info(
                f"transforms.json already exists in {data_path}. Skipping ns-process-data."
            )
            return True

        config.logger.info(
            f"""
            No transforms.json found.
            Running ns-process-data on {data_path}...
            """
        )

        try:
            images_dir = self._find_images_dir(data_path)
        except FileNotFoundError as e:
            config.logger.error(f"❌ {e}")
            return False

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
