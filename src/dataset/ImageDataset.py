from pathlib import Path
from typing import Dict, List, Optional, Union, cast

from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, Resize, ToTensor, functional

from src.dataset.SubsetWrapper import SubsetWrapper
from src.utils.config import config


class _FlatImageFolder(Dataset):
    """Fallback dataset for directories that contain images directly (no class subfolders)."""

    classes: list[str]
    class_to_idx: dict[str, int]
    samples: list[tuple[str, int]]
    targets: list[int]

    def __init__(self, root: Path, transform: Optional[Compose] = None):
        self.root = root
        self.transform = transform
        self.classes = ["default"]
        self.class_to_idx = {"default": 0}

        valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self.samples = sorted(
            [
                (str(image_path), 0)
                for image_path in root.iterdir()
                if image_path.is_file() and image_path.suffix.lower() in valid_suffixes
            ]
        )
        self.targets = [0] * len(self.samples)

        if not self.samples:
            raise FileNotFoundError(
                f"No images found in flat dataset directory: {root}"
            )

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        sample_path, target = self.samples[idx]
        image = default_loader(sample_path)
        if self.transform is not None:
            image = self.transform(image)
        return cast(tuple[Tensor, int], (image, target))

    def __len__(self) -> int:
        return len(self.samples)


class ImageDataset(Dataset):
    image_data: Union[ImageFolder, _FlatImageFolder, ConcatDataset, SubsetWrapper]
    data_loader: DataLoader
    name: str
    step: str
    dataset_path: Optional[Path]
    image_size: Dict[str, int]

    @staticmethod
    def _drop_missing_samples(image_folder: ImageFolder, source_label: str) -> int:
        """Remove non-existent files from an ImageFolder to avoid runtime crashes."""
        valid_samples = []
        missing_count = 0
        for sample_path, target in image_folder.samples:
            if Path(sample_path).is_file():
                valid_samples.append((sample_path, target))
            else:
                missing_count += 1

        if missing_count > 0:
            image_folder.samples = valid_samples
            image_folder.targets = [target for _, target in valid_samples]
            if hasattr(image_folder, "imgs"):
                image_folder.imgs = image_folder.samples
            config.logger.warning(
                "Removed %d missing image(s) from dataset source: %s",
                missing_count,
                source_label,
            )

        return missing_count

    def __init__(
        self,
        name: str = "",
        step: str = "",
        image_width: int = config.img_size,
        image_height: int = config.img_size,
        image_subset_wrapper: Optional[SubsetWrapper] = None,
        dataset_path: Optional[Path] = None,
        transform: Optional[Compose] = None,
        concat_dataset: Optional[ConcatDataset] = None,
        concat_classes: Optional[List[str]] = None,
        concat_class_to_idx: Optional[dict] = None,
    ):
        sources = sum(
            x is not None for x in [dataset_path, image_subset_wrapper, concat_dataset]
        )
        if sources != 1:
            raise ValueError(
                f"Must provide exactly one of: dataset_path, image_subset_wrapper, or concat_dataset. (Provided {sources})"
            )

        self.name = name
        self.step = step
        self.image_size = {"width": image_width, "height": image_height}

        if image_subset_wrapper:
            self.image_data = image_subset_wrapper
            self.dataset_path = image_subset_wrapper.dataset_path
            self.name = self.name or image_subset_wrapper.name
            self.step = self.step or image_subset_wrapper.step

        elif concat_dataset:
            if hasattr(concat_dataset, "datasets"):
                for index, ds in enumerate(concat_dataset.datasets):
                    if isinstance(ds, ImageFolder):
                        self._drop_missing_samples(ds, f"concat[{index}]")
            self.image_data = concat_dataset
            self._concat_classes = concat_classes or []
            self._concat_class_to_idx = concat_class_to_idx or {}
            self.dataset_path = dataset_path

        elif dataset_path:
            self.dataset_path = dataset_path
            self.name = self.name or str(dataset_path.absolute()).split("/")[-1]

            if transform is None:
                transform = Compose(
                    [
                        Resize((self.image_size["width"], self.image_size["height"])),
                        ToTensor(),
                    ]
                )

            resolved_root = dataset_path.expanduser().resolve()
            has_class_dirs = any(p.is_dir() for p in resolved_root.iterdir())

            if has_class_dirs:
                self.image_data = ImageFolder(
                    root=str(resolved_root), transform=transform
                )
                self._drop_missing_samples(self.image_data, str(dataset_path))
            else:
                self.image_data = _FlatImageFolder(
                    root=resolved_root,
                    transform=transform,
                )

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
        return getattr(self.image_data, "classes", [])

    @property
    def class_to_idx(self) -> dict:
        """Return class-to-index mapping regardless of how the dataset was initialised."""
        if hasattr(self, "_concat_class_to_idx") and self._concat_class_to_idx:
            return self._concat_class_to_idx
        return getattr(self.image_data, "class_to_idx", {})

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return cast(tuple[Tensor, int], self.image_data[idx])

    def __len__(self) -> int:
        return len(self.image_data)

    def get_image_paths(self) -> list[Path]:
        """Return list of image file paths. Useful for NeRF/GAN pipelines."""
        if isinstance(self.image_data, ConcatDataset):
            paths = []
            for ds in self.image_data.datasets:
                if hasattr(ds, "samples"):
                    paths.extend(Path(s[0]) for s in ds.samples)
            return paths
        return [Path(s[0]) for s in self.image_data.samples]

    def save_generated_images_to_disk(
        self,
        output_path: Path,
        generated_images: Tensor,
        labels: Optional[Tensor] = None,
    ):
        output_path.mkdir(parents=True, exist_ok=True)
        classes = self.classes

        for idx in range(generated_images.size(0)):
            image_tensor = (generated_images[idx] + 1) / 2
            image_tensor = Tensor.clamp(image_tensor, 0, 1)

            image = functional.to_pil_image(image_tensor.cpu())

            if labels is not None:
                label_idx = int(labels[idx].item())
                class_name = classes[label_idx]
                class_dir = output_path / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                image_filename = f"generated_{class_name}_{idx}.png"
                image.save(class_dir / image_filename)
            else:
                image_filename = f"generated_{idx}.png"
                image.save(output_path / image_filename)
