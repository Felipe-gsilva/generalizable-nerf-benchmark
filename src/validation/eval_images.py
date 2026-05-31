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


def align_image_tensors(real: torch.Tensor, gen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ensures that real and gen image tensors have the exact same shape:
    [B, 3, H, W] where B, H, W are matched, and channel count is 3.
    """
    if real.ndim != 4:
        raise ValueError(f"real_images must be 4D [B, C, H, W], got shape {real.shape}")
    if gen.ndim != 4:
        raise ValueError(f"generated_images must be 4D [B, C, H, W], got shape {gen.shape}")

    B_real, C_real, H_real, W_real = real.shape
    B_gen, C_gen, H_gen, W_gen = gen.shape

    B = min(B_real, B_gen)
    real = real[:B]
    gen = gen[:B]

    if C_real != 3:
        if C_real > 3:
            real = real[:, :3, :, :]
        elif C_real == 1:
            real = real.repeat(1, 3, 1, 1)

    if C_gen != 3:
        if C_gen > 3:
            gen = gen[:, :3, :, :]
        elif C_gen == 1:
            gen = gen.repeat(1, 3, 1, 1)

    if H_gen != H_real or W_gen != W_real:
        orig_dtype = gen.dtype
        gen = torch.nn.functional.interpolate(
            gen.float(),
            size=(H_real, W_real),
            mode="bilinear",
            align_corners=False
        ).to(orig_dtype)

    return real, gen


def calculate_fid(
    real_images_cat: torch.Tensor,
    fake_images_cat: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
) -> float:
    real_images_cat, fake_images_cat = real_images_cat.to(device), fake_images_cat.to(device)
    real_images_cat, fake_images_cat = align_image_tensors(real_images_cat, fake_images_cat)
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
    lpips_model: "lpips.LPIPS | None" = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
):
    device = torch.device(device)

    real_images, generated_images = real_images.to(device), generated_images.to(device)
    real_images, generated_images = align_image_tensors(real_images, generated_images)

    # Ensure LPIPS model and inputs are always on the same device.
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net="vgg")
    lpips_model = lpips_model.to(device)
    lpips_model.eval()

    psnr_scores: list[float] = []
    ssim_scores: list[float] = []
    lpips_scores: list[float] = []

    with torch.no_grad():
        for i in range(0, len(real_images), batch_size):
            real = real_images[i : i + batch_size]  # [B, C, H, W]
            gen = generated_images[i : i + batch_size]

            real_np = real.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]
            gen_np = gen.permute(0, 2, 3, 1).cpu().numpy()

            for j in range(len(real_np)):
                psnr_scores.append(psnr(real_np[j], gen_np[j], data_range=255))
                ssim_scores.append(ssim(real_np[j], gen_np[j], channel_axis=-1))

            # LPIPS expects float in [-1, 1]
            real_lpips = (real.float() / 127.5 - 1)
            gen_lpips = (gen.float() / 127.5 - 1)
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
