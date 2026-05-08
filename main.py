from pathlib import Path
from src.dataset.ImageDataset import ImageDataset
from src.nerf.NeRFModel import NerfModel

def main(
    dataset_path: str = "data/lego/train",

        ):
    print("Hello from nerf-ann-paper!")
    images = ImageDataset(dataset_path=Path(dataset_path))
    model = NerfModel("gnt", images=images)
    model.train(Path(dataset_path) , downscale_factor=1)


if __name__ == "__main__":
    main()
