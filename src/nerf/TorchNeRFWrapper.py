import time
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from abc import ABC, abstractmethod
from nerf.NeRFModel import NerfModel
from src.utils.config import config
from utils.metrics import MetricsLogger
from utils.types import Hyperparams


class TorchNeRFWrapper(NerfModel, ABC):
    """
    An intermediate abstract base class for PyTorch-native NeRFs.
    
    This class handles standard PyTorch boilerplate (training loops, 
    checkpoint saving/loading, memory calculation) while remaining 
    strictly agnostic to the underlying network architecture. 
    Subclasses can implement GANs, Transformers, or standard MLPs.
    """

    DEFAULT_CONFIG: Hyperparams = {
        "NUM_EPOCHS": 200,
        "LEARNING_RATE": 0.0002,
        "BATCH_SIZE": 16,
        "CHECKPOINT_EVERY": 50,
    }

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        image_priority: str = "highest",
        output_dir: Optional[Path] = None,
        resume: bool = False,
    ) -> None:
        super().__init__(model_name, dataset_name, image_priority, output_dir)
        
        self.resume = resume
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.__class__.__name__}] Target Device: {self.device}")

        self.training_config: Hyperparams = {**self.DEFAULT_CONFIG, **config.__dict__}
        self.training_history = []
        
        # Dynamic storage for any number of networks and optimizers
        self.models = torch.nn.ModuleDict()
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}

    # ------------------------------------------------------------------
    # 1. The Agnostic PyTorch Contract
    # ------------------------------------------------------------------

    @abstractmethod
    def build_dataloaders(self, data_path: Path):
        """
        Instantiate and return the training and validation dataloaders.
        Returns: Tuple[DataLoader, DataLoader]
        """
        ...

    @abstractmethod
    def build_models(self) -> Dict[str, torch.nn.Module]:
        """
        Instantiate and return a dictionary of networks.
        Example: {"generator": Gen(), "discriminator": Disc()}
              or {"transformer": ViT(), "head": MLP()}
        """
        ... 

    @abstractmethod
    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """
        Instantiate and return a dictionary of optimizers mapped to the models.
        Example: {"g_opt": Adam(...), "d_opt": Adam(...)}
        """
        ... 

    @abstractmethod
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        Execute one full epoch using self.models and self.optimizers.
        Returns a dictionary of epoch metrics (e.g., {"loss": 0.5}).
        """
        ...

    @abstractmethod
    def _generate_eval_images(self, dataloader) -> Tuple[list, list]:
        """
        Helper function. Forward pass the current self.models to generate 
        images for the evaluate() method. Returns (rendered, ground_truth).
        """
        ... 

    # ------------------------------------------------------------------
    # 2. Boilerplate Implementations
    # ------------------------------------------------------------------

    def run(self, data_path: Path) -> None:
        """The main PyTorch training loop."""
        
        if not self.process_data(data_path):
            config.logger.error("❌ Data processing failed. Aborting training.")
            return

        output_path = self.get_output_path(data_path)
        
        # 1. Register dynamically generated models to the ModuleDict
        model_dict = self.build_models()
        for name, mod in model_dict.items():
            self.models[name] = mod.to(self.device)
            
        self.optimizers = self.build_optimizers()

        if self.resume:
            self.load_checkpoint(output_path)

        # 2. Setup Dataloader
        dataloader = self._build_dataloader(data_path)

        checkpoint_every = self.training_config.get("CHECKPOINT_EVERY", 50)
        num_epochs = self.training_config["NUM_EPOCHS"]

        metrics_logger = MetricsLogger(
            model_name=self.__class__.__name__,
            hyperparams=self.training_config,
        )

        pbar = tqdm(range(1, num_epochs + 1), desc=f"Training {self.__class__.__name__}")

        for epoch in pbar:
            start_time = time.time()
            
            # Subclass executes its specific architecture logic
            train_metrics = self.train_epoch(dataloader)
            
            # Shared evaluation
            rendered_images, gt_images = self._generate_eval_images(dataloader)
            eval_metrics = self.evaluate(rendered_images, gt_images)
            
            epoch_time = time.time() - start_time

            all_metrics = {
                "epoch": epoch,
                "training_time": epoch_time,
                "performance_metrics": {
                    "memory_footprint": self.get_memory_footprint(),
                },
                **train_metrics,
                **eval_metrics
            }
            metrics_logger.log(**all_metrics)
            self.training_history.append((epoch, all_metrics))

            current_fid = eval_metrics.get("fid", float("inf"))
            if current_fid < self._best_metric_value:
                self._best_metric_value = current_fid
                self.save_checkpoint(epoch, output_path, is_best=True)

            if epoch % checkpoint_every == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, output_path, is_best=False)

        metrics_logger.close()
        print(f"\n[{self.__class__.__name__}] Training Complete. Best FID: {self._best_metric_value:.4f}")

    def save_checkpoint(self, epoch: int, output_path: Path, is_best: bool = False) -> None:
        """Agnostic checkpoint saving mapping all models and optimizers."""
        state = {
            "epoch": epoch,
            "models_state": {name: mod.state_dict() for name, mod in self.models.items()},
            "optimizers_state": {name: opt.state_dict() for name, opt in self.optimizers.items()},
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"best_model.pt" if is_best else f"checkpoint_ep{epoch}_{timestamp}.pt"
        save_file = output_path / filename
        
        torch.save(state, save_file)
        config.logger.info(f"💾 Checkpoint saved: {save_file.name}")

    def load_checkpoint(self, path: Path) -> None:
        """Agnostic checkpoint loading."""
        c_path = self._get_latest_checkpoint(path) if path.is_dir() else path
        if not c_path or not c_path.exists():
            config.logger.warning("🆕 No checkpoint found. Starting fresh.")
            return

        try:
            checkpoint = torch.load(c_path, map_location=self.device)
            
            # Load all mapped models
            for name, state_dict in checkpoint.get("models_state", {}).items():
                if name in self.models:
                    self.models[name].load_state_dict(state_dict)
                    
            # Load all mapped optimizers
            for name, state_dict in checkpoint.get("optimizers_state", {}).items():
                if name in self.optimizers:
                    self.optimizers[name].load_state_dict(state_dict)
                    
            config.logger.info(f"✅ Resumed successfully from {c_path.name} (Epoch {checkpoint.get('epoch', 'N/A')})")
        except Exception as e:
            config.logger.error(f"❌ Checkpoint load failed: {e}")

    def evaluate(self, rendered_images, ground_truth_images) -> Dict[str, float]:
        """Utilizes the shared visual metrics defined in the NerfModel base class."""
        if not rendered_images or not ground_truth_images:
            return {"fid": float("inf"), "psnr": 0.0, "ssim": 0.0, "lpips": float("inf")}
            
        return self.compare_images_quality(rendered_images, ground_truth_images)

    def get_memory_footprint(self) -> int:
        """Returns the combined memory footprint of all models in the ModuleDict."""
        return sum(p.numel() * p.element_size() for p in self.models.parameters())
