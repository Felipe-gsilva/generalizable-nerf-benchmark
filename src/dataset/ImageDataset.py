from pathlib import Path
from typing import Dict, List, Optional

from torch import Tensor, clamp
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, functional

from src.dataset.SubsetWrapper import SubsetWrapper
from src.utils.config import config


class ImageDataset(Dataset):
    image_data: ImageFolder
    data_loader: DataLoader
    name: str
    step: str
    dataset_path: Path
    image_size: Dict[str, int]

    def __init__(
        self,
        name: str = "",
        step: str = "",
        image_width: int = config.img_size,
        image_height: int = config.img_size,
        dataset_path: Path = None,
        image_subset_wrapper: SubsetWrapper = None,
        transform: Optional[Compose] = None,
        concat_dataset: ConcatDataset = None,
        concat_classes: List[str] = None,
        concat_class_to_idx: dict = None,
    ):
        self.name = name
        self.step = step
        self.image_size = {"width": image_width, "height": image_height}

        # --- initialised from a SubsetWrapper (split result) ---
        if not dataset_path and image_subset_wrapper:
            self.image_data = image_subset_wrapper
            self.dataset_path = image_subset_wrapper.dataset_path
            self.name = image_subset_wrapper.name
            self.step = image_subset_wrapper.step
            return

        # --- initialised from a pre-built ConcatDataset ---
        if concat_dataset is not None:
            self.image_data = concat_dataset
            self._concat_classes = concat_classes or []
            self._concat_class_to_idx = concat_class_to_idx or {}
            self.dataset_path = dataset_path
            return

        # --- default: single ImageFolder ---
        self.dataset_path = dataset_path
        self.name = str(dataset_path.absolute()).split("/")[-1]

        if transform is None:
            transform = Compose(
                [
                    Resize((self.image_size["width"], self.image_size["height"])),
                    ToTensor(),
                ]
            )

        self.image_data = ImageFolder(root=dataset_path.absolute(), transform=transform)

    def load(
        self,
        batch_size: int = 256,
        shuffle: bool = True,
        num_workers: int = 4,
        prefetch_factor: int = 8,
        persistent_workers=True,
        pin_memory: bool = False,
    ):
        try:
            self.data_loader = DataLoader(
                self.image_data,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                pin_memory=pin_memory,
            )
        except Exception as e:
            config.logger.error(e)

    def get_stats(self):
        features, labels = next(iter(self.data_loader))
        config.logger.info(f"Feature batch shape: {features.size()}")
        config.logger.info(f"Labels batch shape: {labels.size()}")

    @property
    def classes(self) -> list:
        """Return class names regardless of how the dataset was initialised."""
        if hasattr(self, "_concat_classes") and self._concat_classes:
            return self._concat_classes
        return self.image_data.classes

    @property
    def class_to_idx(self) -> dict:
        """Return class-to-index mapping regardless of how the dataset was initialised."""
        if hasattr(self, "_concat_class_to_idx") and self._concat_class_to_idx:
            return self._concat_class_to_idx
        return self.image_data.class_to_idx

    def __getitem__(self, idx) -> tuple[Tensor, int]:
        return self.image_data[idx]

    def __len__(self) -> int:
        return len(self.image_data)

    def get_image_paths(self) -> list[Path]:
        """Return list of image file paths. Useful for NeRF/GAN pipelines."""
        if isinstance(self.image_data, ConcatDataset):
            paths = []
            for ds in self.image_data.datasets:
                paths.extend(Path(s[0]) for s in ds.samples)
            return paths
        return [Path(s[0]) for s in self.image_data.samples]

    def save_generated_images_to_disk(
        self, output_path: Path, generated_images: Tensor, labels: Tensor = None
    ):
        output_path.mkdir(parents=True, exist_ok=True)
        classes = self.image_data.classes

        for idx in range(generated_images.size(0)):
            image_tensor = (generated_images[idx] + 1) / 2
            image_tensor = clamp(image_tensor, 0, 1)

            image = functional.to_pil_image(image_tensor.cpu())

            if labels is not None:
                class_name = classes[labels[idx].item()]
                class_dir = output_path / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                image_filename = f"generated_{class_name}_{idx}.png"
                image.save(class_dir / image_filename)
            else:
                image_filename = f"generated_{idx}.png"
                image.save(output_path / image_filename)
