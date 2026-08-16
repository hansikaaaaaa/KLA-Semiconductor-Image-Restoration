import argparse
import time
from pathlib import Path

import numpy as np
import torch

from dataset import discover_test_images, load_npy
from model import RestorationNet


def main():
    parser = argparse.ArgumentParser(
        description="KLA Semiconductor Image Restoration - Standalone Inference"
    )

    parser.add_argument(
        "test_dir",
        help="Path to directory containing test .npy images"
    )

    parser.add_argument(
        "output_dir",
        help="Path to directory where restored .npy images will be saved"
    )

    parser.add_argument(
        "--checkpoint",
        default="weights/best.pt",
        help="Path to trained model checkpoint"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading checkpoint:", args.checkpoint)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------
    # Load exact architecture from checkpoint
    # --------------------------------------------------

    model_args = checkpoint.get(
        "model_args",
        {
            "base_channels": 64,
            "num_blocks": 10
        }
    )

    print("Model configuration:", model_args)

    model = RestorationNet(
        **model_args
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    print(
        "Validation PSNR:",
        checkpoint.get("val_psnr")
    )

    print(
        "Validation SSIM:",
        checkpoint.get("val_ssim")
    )

    scale = float(
        checkpoint.get("scale", 1.0)
    )

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Discover test images
    # --------------------------------------------------

    files = discover_test_images(
        args.test_dir
    )

    if len(files) == 0:
        raise RuntimeError(
            f"No test images found in: {args.test_dir}"
        )

    print("Test images:", len(files))

    times = []

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    with torch.inference_mode():

        for i, file in enumerate(files, 1):

            array = load_npy(str(file))

            x = torch.from_numpy(
                array / scale
            ).float()[None, None].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            y = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter() - start
            )

            times.append(elapsed)

            # --------------------------------------------------
            # Convert prediction to NumPy
            # --------------------------------------------------

            pred = (
                y[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # --------------------------------------------------
            # Safety clamp
            # --------------------------------------------------

            pred = np.clip(
                pred,
                0.0,
                1.0
            )

            # --------------------------------------------------
            # Save restored image
            # --------------------------------------------------

            np.save(
                output_dir / f"{file.stem}.npy",
                pred
            )

            if i % 50 == 0 or i == len(files):
                print(
                    f"Processed {i}/{len(files)}"
                )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Device:", device)
    print("Images:", len(files))
    print(
        "Average inference time:",
        f"{np.mean(times):.4f} s/image"
    )
    print(
        "Outputs:",
        output_dir.resolve()
    )
    print("=" * 60)


if __name__ == "__main__":
    main()