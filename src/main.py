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


def export(dataset_name_list, export_formats, model_name_list):
    base_root = Path("assets/data/baseline")

    for dataset_name in dataset_name_list:
        targets = resolve_data_targets(base_root, dataset_name)

        for target_dir in targets:
            for model_name in model_name_list:
                target_id = _resolve_target_id(base_root / dataset_name, target_dir)
                dataset_path = base_root / dataset_name
                step = "train"

                dataset = ImageDataset(
                    name=dataset_name,
                    dataset_path=dataset_path,
                    step=step,
                )
                # Need to use the NerfModel class
                nerf_wrapper = NerfModel(model_name=model_name, images=dataset)
                nerf_wrapper.target_id = target_id
                for export_type in export_formats:
                    nerf_wrapper.export(export_type=export_type, data_path=target_dir)


def generate(dataset_name_list, slice_methods):
    print("🔍 Starting NeRF slicing process...")
    base_path = Path("assets/data/nerf_exports")
    for dataset_name in dataset_name_list:
        print(f"\n=============================================")
        dataset_path = base_path / dataset_name
        if not dataset_path.exists():
            print(f"❌ No exports found for {dataset_name}")
            continue

        model_name_list = os.listdir(dataset_path)

        for model_name in model_name_list:
            model_path = dataset_path / model_name

            if not model_path.exists():
                print(f"❌ No exports found for model '{model_name}' in {dataset_path}")
                continue

            target_paths = _discover_export_targets(model_path, slice_methods)
            if not target_paths:
                target_paths = [model_path]

            for class_path in target_paths:
                rel_target = (
                    class_path.relative_to(model_path).as_posix()
                    if class_path != model_path
                    else None
                )
                label = (
                    f"{dataset_name}/{model_name}/{rel_target}"
                    if rel_target
                    else f"{dataset_name}/{model_name}"
                )
                print(f"\n--- Processing {label} ---")
                for method in slice_methods:
                    method_path = class_path / method
                    print(f"🔍 Checking for {method} exports in {method_path}...")
                    if method_path.exists():
                        slice_nerf_model(method_path, method)
                    else:
                        print(
                            f"⚠️ No exports found for method '{method}' in {method_path}"
                        )


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
    parser.add_argument(
        "--methods",
        nargs="+",
        default=config.nerf_slice_methods,
        help="List of slicing methods to apply",
        choices=config.nerf_slice_methods,
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=config.nerf_export_formats,
        help="List of export formats to apply",
    )
    parser.add_argument(
        "--image-priority",
        choices=["highest", "lowest"],
        default="highest",
        help="Priority for selecting images/images_N folders during data processing.",
    )
    parser.add_argument(
        "--render-pointcloud",
        action="store_true",
        help="Whether to render point clouds during NeRF slicing (if supported by the method).",
    )

    args = parser.parse_args()
    config.render_pointcloud = args.render_pointcloud

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
