from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    extract_if_zip,
    discover_pairs,
    PairedRestorationDataset,
)
from losses import restoration_loss
from metrics import psnr, ssim
from model import RestorationNet


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def validate(model, loader, device):

    model.eval()

    psnr_values = []
    ssim_values = []

    for lr, gt, _ in loader:

        lr = lr.to(
            device,
            non_blocking=True
        )

        gt = gt.to(
            device,
            non_blocking=True
        )

        pred = model(lr)

        pred = torch.clamp(
            pred,
            0.0,
            1.0
        )

        pred_np = (
            pred[:, 0]
            .detach()
            .cpu()
            .numpy()
        )

        gt_np = (
            gt[:, 0]
            .detach()
            .cpu()
            .numpy()
        )

        for p, g in zip(
            pred_np,
            gt_np
        ):

            psnr_values.append(
                psnr(p, g, data_range=1.0)
            )

            ssim_values.append(
                ssim(p, g, data_range=1.0)
            )

    return (
        float(np.mean(psnr_values)),
        float(np.mean(ssim_values)),
    )


def parse_args():

    p = argparse.ArgumentParser(
        description=(
            "Train KLA semiconductor image "
            "restoration model."
        )
    )

    p.add_argument(
        "--train",
        default="train"
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1.5e-4
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5
    )

    p.add_argument(
        "--patch-size",
        type=int,
        default=96
    )

    p.add_argument(
        "--workers",
        type=int,
        default=2
    )

    p.add_argument(
        "--out-dir",
        default="runs_final"
    )

    p.add_argument(
        "--cpu",
        action="store_true"
    )

    return p.parse_args()


def main():

    args = parse_args()

    seed_everything(42)

    out = Path(args.out_dir)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    if (
        torch.cuda.is_available()
        and not args.cpu
    ):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("KLA Semiconductor Image Restoration")
    print("=" * 60)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    root = extract_if_zip(
        args.train
    )

    pairs = discover_pairs(root)

    print(
        "Total paired samples:",
        len(pairs)
    )

    # --------------------------------------------------
    # Deterministic train/validation split
    # --------------------------------------------------

    rng = np.random.default_rng(42)

    ids = np.arange(
        len(pairs)
    )

    rng.shuffle(ids)

    n_val = max(
        1,
        int(
            0.10 * len(ids)
        )
    )

    val_pairs = [
        pairs[i]
        for i in ids[:n_val]
    ]

    train_pairs = [
        pairs[i]
        for i in ids[n_val:]
    ]

    print(
        "Training samples:",
        len(train_pairs)
    )

    print(
        "Validation samples:",
        len(val_pairs)
    )

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    train_ds = PairedRestorationDataset(
        train_pairs,
        patch_size=args.patch_size,
        augment=True,
        seed=42,
        noise_aug=False,
    )

    val_ds = PairedRestorationDataset(
        val_pairs,
        patch_size=None,
        augment=False,
        seed=42,
        noise_aug=False,
    )

    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.workers > 0
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=min(
            args.workers,
            2
        ),
        pin_memory=pin_memory,
        persistent_workers=(
            args.workers > 0
        ),
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = RestorationNet().to(
        device
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{parameters:,}"
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=2e-5,
        )
    )

    # --------------------------------------------------
    # AMP
    # --------------------------------------------------

    amp_enabled = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    history_path = (
        out / "history.csv"
    )

    best_psnr = -float("inf")

    with history_path.open(
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_psnr",
                "val_ssim",
                "learning_rate",
                "seconds",
            ]
        )

        for epoch in range(
            1,
            args.epochs + 1
        ):

            start = time.perf_counter()

            model.train()

            total_loss = 0.0

            for lr, gt, _ in train_loader:

                lr = lr.to(
                    device,
                    non_blocking=True
                )

                gt = gt.to(
                    device,
                    non_blocking=True
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.autocast(
                    device_type="cuda",
                    enabled=amp_enabled,
                ):

                    pred = model(lr)

                    loss = restoration_loss(
                        pred,
                        gt
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                total_loss += float(
                    loss.detach()
                )

            scheduler.step()

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------

            val_psnr, val_ssim = validate(
                model,
                val_loader,
                device
            )

            epoch_loss = (
                total_loss
                / max(
                    1,
                    len(train_loader)
                )
            )

            seconds = (
                time.perf_counter()
                - start
            )

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            writer.writerow(
                [
                    epoch,
                    epoch_loss,
                    val_psnr,
                    val_ssim,
                    current_lr,
                    seconds,
                ]
            )

            f.flush()

            # --------------------------------------------------
            # Checkpoint
            # --------------------------------------------------

            state = {
                "model": model.state_dict(),

                "model_args": {
                    "base_channels": 48,
                    "num_blocks": 8,
                },

                "epoch": epoch,

                "val_psnr": val_psnr,

                "val_ssim": val_ssim,
            }

            torch.save(
                state,
                out / "last.pt"
            )

            if val_psnr > best_psnr:

                best_psnr = val_psnr

                torch.save(
                    state,
                    out / "best.pt"
                )

            print(
                f"Epoch "
                f"{epoch:03d}/{args.epochs} | "
                f"loss={epoch_loss:.5f} | "
                f"PSNR={val_psnr:.3f} | "
                f"SSIM={val_ssim:.4f} | "
                f"time={seconds:.1f}s"
            )

    print()
    print("=" * 60)
    print(
        f"Training complete. "
        f"Best PSNR = {best_psnr:.4f}"
    )
    print(
        "Checkpoint:",
        out / "best.pt"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()