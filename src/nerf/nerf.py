from src.validation.eval_images import calculate_fid, calculate_psnr_ssim_lpips
from src.dataset.ImageDataset import ImageDataset
import torch


class NerfModel:
    """
    A NeRF model using nerfstudio python API. This class is responsible for
    creating the NeRF model, training it, and rendering images from it,
    such as novel views. It also provides methods for saving and loading
    the model, as well as for evaluating the quality of the rendered images.
    """

    def __init__(self, model_name: str, images: ImageDataset) -> None:
        self.model_name = model_name
        self.model = None
        self.best_model = None
        self.training_history = []
        self.images = images

    def create_model(self, config):
        """Create the NeRF model using the provided configuration."""
        pass

    def train(self, data, config):
        """Train the NeRF model using the provided data and configuration."""
        pass

    def render(self, view):
        """Render an image from the NeRF model given a specific view."""
        pass

    def save_model(self, path):
        """Save the NeRF model to the specified path."""
        pass

    def load_model(self, path):
        """Load the NeRF model from the specified path."""
        pass

    def save_best_model(self, path):
        """Save the best performing NeRF model to the specified path."""
        for epoch, metrics in self.training_history:
            if self.best_model is None or metrics["fid"] < self.best_model["fid"]:
                self.best_model = {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "metrics": metrics,
                }
        if self.best_model is not None:
            torch.save(self.best_model["model_state"], path)

    def compare_images_quality(self, rendered_images, ground_truth_images):
        """Evaluate the quality of the rendered images against the ground truth images using metrics such as FID, PSNR, SSIM, and LPIPS."""
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
