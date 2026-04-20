# compare image quality with FID, PSNR, SSIM, and LPIPS metrics
import os
from typing import List
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
import argparse
from torchmetrics.image.fid import FrechetInceptionDistance


def calculate_fid(
    real_images_cat: torch.Tensor,
    fake_images_cat: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
) -> float:
    assert real_images_cat.shape[0] == fake_images_cat.shape[0]
    assert real_images_cat.dtype == torch.uint8 and fake_images_cat.dtype == torch.uint8
    # Initialize the metric (it will download InceptionV3 weights the first time) feature=2048 is standard for FID
    fid = FrechetInceptionDistance().to(device)

    for i in range(0, len(real_images_cat), batch_size):
        batch = real_images_cat[i : i + batch_size].to(device)
        fid.update(batch, real=True)
        del batch

    for i in range(0, len(fake_images_cat), batch_size):
        batch = fake_images_cat[i : i + batch_size].to(device)
        fid.update(batch, real=False)
        del batch

    torch.cuda.empty_cache()
    return fid.compute().item()


def calculate_psnr_ssim_lpips(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    lpips_model=lpips.LPIPS(net="alex"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
):
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    for i in range(0, len(real_images), batch_size):
        real = real_images[i : i + batch_size]  # [B, C, H, W]
        gen = generated_images[i : i + batch_size]

        real_np = real.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]
        gen_np = gen.permute(0, 2, 3, 1).cpu().numpy()

        for j in range(len(real_np)):
            psnr_scores.append(psnr(real_np[j], gen_np[j]))
            ssim_scores.append(ssim(real_np[j], gen_np[j], channel_axis=-1))

        # lpips: already [B, C, H, W], just convert to float
        real_lpips = (real.float() / 127.5 - 1).to(device)
        gen_lpips = (gen.float() / 127.5 - 1).to(device)
        lpips_scores.append(lpips_model(real_lpips, gen_lpips).mean().item())

    return (
        float(np.mean(psnr_scores)),
        float(np.mean(ssim_scores)),
        float(np.mean(lpips_scores)),
    )


# load into torch tensors
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert("RGB")
        if img is not None:
            images.append(np.array(img))
    return torch.tensor(np.stack(images)).permute(0, 3, 1, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate generated images against real images using FID, PSNR, SSIM, and LPIPS metrics."
    )
    parser.add_argument(
        "--real_folder",
        type=str,
        required=True,
        help="Path to the folder containing real images.",
    )
    parser.add_argument(
        "--generated_folder",
        type=str,
        required=True,
        help="Path to the folder containing generated images.",
    )
    args = parser.parse_args()

    fid_score = calculate_fid(
        load_images_from_folder(args.real_folder),
        load_images_from_folder(args.generated_folder),
    )
    psnr_score, ssim_score, lpips_score = calculate_psnr_ssim_lpips(
        load_images_from_folder(args.real_folder),
        load_images_from_folder(args.generated_folder),
    )

    print(f"FID Score: {fid_score}")
    print(f"PSNR Score: {psnr_score}")
    print(f"SSIM Score: {ssim_score}")
    print(f"LPIPS Score: {lpips_score}")
