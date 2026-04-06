# compare image quality with FID, PSNR, SSIM, and LPIPS metrics
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.linalg import sqrtm
import lpips
import torch
import argparse


def calculate_fid(real_images, generated_images):
    # Calculate the mean and covariance of real and generated images
    mu_real = np.mean(real_images, axis=0)
    sigma_real = np.cov(real_images, rowvar=False)
    mu_gen = np.mean(generated_images, axis=0)
    sigma_gen = np.cov(generated_images, rowvar=False)
    # Calculate the FID score
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real.dot(sigma_gen))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)

    return fid_score


def calculate_psnr_ssim_lpips(real_images, generated_images):
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    for real, gen in zip(real_images, generated_images):
        psnr_scores.append(psnr(real, gen))
        ssim_scores.append(ssim(real, gen, multichannel=True))

        # Calculate LPIPS score using the lpips library
        lpips_model = lpips.LPIPS(net="alex")
        lpips_score = lpips_model.forward(
            torch.tensor(real).unsqueeze(0), torch.tensor(gen).unsqueeze(0)
        )
        lpips_scores.append(lpips_score.item())

    return np.mean(psnr_scores), np.mean(ssim_scores), np.mean(lpips_scores)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert("RGB")
        if img is not None:
            images.append(np.array(img))
    return np.array(images)


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
