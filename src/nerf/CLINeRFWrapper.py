import subprocess
import sys
import time
import torch

from pathlib import Path
from typing import Dict, Optional
from nerf.NeRFModel import NerfModel
from src.utils.config import config


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
            "--data", str(data_path),
            "--output-dir", str(output_path),
            "--vis", "tensorboard",
        ] + load_dir_arg

        # Add model-specific hyperparameters
        if self.model_name == "nerfacto":
            cmd += [
                "--pipeline.model.eval-num-rays-per-chunk", "1024",
                "--pipeline.datamanager.train-num-rays-per-batch", "1024",
                "--pipeline.model.log2-hashmap-size", "16",
                "--pipeline.model.camera-optimizer.mode", "off",
                "--pipeline.model.predict-normals", "True",
            ]
        elif self.model_name == "instant-ngp":
            cmd += [
                "--pipeline.model.eval-num-rays-per-chunk", "1024",
                "--pipeline.datamanager.train-num-rays-per-batch", "1024",
            ]

        # Use generic nerfstudio-data backend
        cmd += [
            "nerfstudio-data",
            "--downscale-factor", "1",
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
        Loads the checkpoint metadata into memory. For CLI execution, actual 
        resumption is handled via '--load-dir' in the run() method.
        """
        config.logger.info(f"Loading NeRF model metadata from {path}...")
        c_path = self._get_latest_checkpoint(path) if path.is_dir() else path
        
        if c_path and c_path.exists():
            try:
                checkpoint = torch.load(c_path, map_location="cpu")
                self.model_state = checkpoint.get("model_state", None)
                config.logger.info(f"✅ Model metadata loaded successfully from {path}")
            except Exception as e:
                config.logger.error(f"❌ Failed to load model from {path}: {e}")

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
            "--load-config", str(config_path),
            "--output-dir", str(target_export_path),
            "--num-points", "1000000",
            "--remove-outliers", "False",
            "--normal-method", "open3d",
        ]

        try:
            subprocess.run(cmd, check=True)
            config.logger.info(f"✅ Export successful")
        except subprocess.CalledProcessError as e:
            config.logger.error(f"❌ Export failed. Code: {e.returncode}")
