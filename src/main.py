import os
from argparse import ArgumentParser
from pathlib import Path

from src.dataset.ImageDataset import ImageDataset
from src.utils.config import config
from src.nerf.NeRFModel import NerfModel
from src.nerf.CLINeRFWrapper import CLINeRFWrapper


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_DATASET_SEARCH_DEPTH = 3
MAX_EXPORT_SEARCH_DEPTH = 3


def has_usable_images(path: Path) -> bool:
    return any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in path.iterdir()
    )


def _resolve_target_id(dataset_path: Path, target_path: Path) -> str:
    return "__".join(target_path.relative_to(dataset_path).parts)


def _discover_export_targets(
    model_path: Path, methods: list[str], max_depth: int = MAX_EXPORT_SEARCH_DEPTH
) -> list[Path]:
    method_names = set(methods)
    root_depth = len(model_path.parts)
    targets = sorted(
        {
            method_path.parent
            for method_path in model_path.rglob("*")
            if method_path.is_dir()
            and method_path.name in method_names
            and (len(method_path.parts) - root_depth) <= max_depth
        }
    )
    return targets


def resolve_data_targets(
    base_root: Path, dataset_name: str, max_depth: int = MAX_DATASET_SEARCH_DEPTH
) -> list[Path]:
    """
    Inteligently resolves data paths based on the dataset structure:
    Format A (Baseline): assets/data/baseline/dataset_name/images/Class
    Format B (LLFF): assets/data/baseline/dataset_name/Scene
    """
    dataset_path = base_root / dataset_name

    if not dataset_path.exists():
        config.logger.error(f"❌ Dataset path not found: {dataset_path}")
        return []

    images_dir = dataset_path / "images"
    targets: list[Path] = []

    if images_dir.exists():
        subdirs = [d for d in images_dir.iterdir() if d.is_dir()]
        if subdirs:
            config.logger.info(
                f"🔀 Multi-Class format detected: {[d.name for d in subdirs]}"
            )
            targets.extend(subdirs)
        else:
            config.logger.info("Single Class format detected.")
            targets.append(images_dir)

    # Format B (LLFF: Scene directories exist and they contain images/)
    else:
        root_depth = len(dataset_path.parts)
        image_dirs = sorted(
            [
                d
                for d in dataset_path.rglob("images*")
                if d.is_dir()
                and (len(d.parts) - root_depth) <= max_depth
                and has_usable_images(d)
            ]
        )
        if image_dirs:
            targets.extend(d.parent for d in image_dirs)
            config.logger.info(
                f"🔀 LLFF format (nested scenes) detected: {[str(t.relative_to(dataset_path)) for t in targets]}"
            )
        else:
            nested_leaf_dirs = sorted(
                [
                    d
                    for d in dataset_path.rglob("*")
                    if d.is_dir()
                    and (len(d.parts) - root_depth) <= max_depth
                    and has_usable_images(d)
                ]
            )
            if nested_leaf_dirs:
                targets.extend(nested_leaf_dirs)
                config.logger.info(
                    f"🔀 Nested image directories detected: {[str(t.relative_to(dataset_path)) for t in targets]}"
                )
            else:
                targets.append(dataset_path)

    # Deduplicate while preserving order.
    deduped_targets = list(dict.fromkeys(targets))
    if deduped_targets and deduped_targets != targets:
        config.logger.info(
            f"🔀 Duplicate targets removed: {len(targets) - len(deduped_targets)}"
        )

    return deduped_targets


def train(dataset_name_list, model_name_list, image_priority: str = "highest"):
    base_root = Path("assets/data/baseline")

    for dataset_name in dataset_name_list:
        targets = resolve_data_targets(base_root, dataset_name)

        for target_dir in targets:
            target_id = _resolve_target_id(base_root / dataset_name, target_dir)
            config.logger.info(f"\n--- Processing: {dataset_name}/{target_id} ---")

            for model_name in model_name_list:
                try:
                    nerf_wrapper = CLINeRFWrapper(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        image_priority=image_priority,
                    )
                    nerf_wrapper.target_id = target_id
                    nerf_wrapper.image_priority = image_priority
                    nerf_wrapper.run(data_path=target_dir)
                except KeyboardInterrupt:
                    config.logger.info("\n🛑 User interruption in main loop.")
                    return


def generate(dataset_name_list, model_name_list):
    print("🔍 Starting NeRF slicing process...")
    base_path = Path("assets/data/nerf_exports")
    for dataset_name in dataset_name_list:
        for model in model_name_list:
            model_path = base_path / dataset_name / model
            if not model_path.exists():
                config.logger.error(f"❌ Model path not found: {model_path}")
                continue

            export_targets = _discover_export_targets(model_path, model_name_list)
            if not export_targets:
                config.logger.warning(f"⚠️ No export targets found for {model_path}")
                continue

            for target in export_targets:
                target_id = _resolve_target_id(model_path, target)
                config.logger.info(
                    f"\n--- Generating from: {dataset_name}/{model}/{target_id} ---"
                )

                try:
                    nerf_wrapper = CLINeRFWrapper(
                        model_name=model,
                        dataset_name=dataset_name,
                        image_priority="highest",
                    )
                    nerf_wrapper.target_id = target_id
                    nerf_wrapper.export("pointcloud", data_path=target)
                except KeyboardInterrupt:
                    config.logger.info("\n🛑 User interruption in generation loop.")
                    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train", action="store_true", help="Train NeRF models for each dataset"
    )
    parser.add_argument(
        "--export", action="store_true", help="Export NeRF models to another formats"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic data using trained NeRF models",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=config.datasets_name_list,
        help="List of datasets to process",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=config.nerf_models_to_run,
        help="List of NeRF models to process",
        choices=config.nerf_models_to_run,
    )
    
    args = parser.parse_args()

    if args.train:
        train(
            dataset_name_list=args.datasets,
            model_name_list=args.models,
            image_priority=args.image_priority,
        )
    if args.export:
        export(
            dataset_name_list=args.datasets,
            export_formats=args.formats,
            model_name_list=args.models,
        )
    if args.generate:
        generate(dataset_name_list=args.datasets, slice_methods=args.methods)
