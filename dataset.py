from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def extract_if_zip(path: str, cache_root: str = ".cache_data") -> str:
    """Extract a dataset ZIP once and return the extracted directory."""
    p = Path(path)

    if p.is_dir():
        return str(p)

    if not p.is_file() or p.suffix.lower() != ".zip":
        raise FileNotFoundError(
            f"Dataset path is not a directory or ZIP: {path}"
        )

    digest = hashlib.sha1(
        f"{p.resolve()}|{p.stat().st_size}|{p.stat().st_mtime_ns}".encode()
    ).hexdigest()[:12]

    cache = Path(cache_root) / f"{p.stem}_{digest}"
    marker = cache / ".extracted"

    if not marker.exists():
        cache.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(p, "r") as z:
            z.extractall(cache)

        marker.write_text("ok")

    return str(cache)


def _valid_npy(p: Path) -> bool:
    return (
        p.suffix.lower() == ".npy"
        and not p.name.startswith("._")
        and "__MACOSX" not in p.parts
    )


def find_npy_files(root: str):
    return sorted(
        p for p in Path(root).rglob("*.npy")
        if _valid_npy(p)
    )


def _find_named_folder(root: str, name: str):
    target = name.lower()

    for p in Path(root).rglob("*"):
        if p.is_dir() and p.name.lower() == target:
            return p

    return None


def discover_pairs(root: str) -> List[Tuple[str, str]]:
    """
    Find matching NoisyLR and GT files.

    Expected:
        train/
            NoisyLR/
                000000.npy
                ...
            GT/
                000000.npy
                ...
    """

    rootp = Path(root)

    gt_dir = _find_named_folder(str(rootp), "GT")
    lr_dir = _find_named_folder(str(rootp), "NoisyLR")

    if gt_dir is None or lr_dir is None:
        raise RuntimeError(
            f"Expected GT and NoisyLR folders. "
            f"Found GT={gt_dir}, NoisyLR={lr_dir}"
        )

    gt = {
        p.stem.lower(): str(p)
        for p in gt_dir.glob("*.npy")
        if _valid_npy(p)
    }

    lr = {
        p.stem.lower(): str(p)
        for p in lr_dir.glob("*.npy")
        if _valid_npy(p)
    }

    common = sorted(set(gt) & set(lr))

    if not common:
        raise RuntimeError(
            "No matching GT/NoisyLR pairs found."
        )

    return [(lr[k], gt[k]) for k in common]


def discover_test_images(root: str):
    files = find_npy_files(root)

    if not files:
        raise RuntimeError(
            "No .npy test images found."
        )

    return files


def load_npy(path: str) -> np.ndarray:
    """Load a finite 2-D grayscale float32 NPY image."""

    x = np.load(path, allow_pickle=False)
    x = np.asarray(x)

    if x.ndim == 3:
        if x.shape[0] == 1:
            x = x[0]
        elif x.shape[-1] == 1:
            x = x[..., 0]
        else:
            raise ValueError(
                f"Expected grayscale image, got {x.shape}: {path}"
            )

    if x.ndim != 2:
        raise ValueError(
            f"Expected 2-D grayscale image, got {x.shape}: {path}"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            f"NaN/Inf values found in {path}"
        )

    return x.astype(np.float32)


def robust_scale_from_gt(pairs, max_samples=256):
    """
    GT is already defined in [0,1].

    We therefore do NOT normalize GT using an LR-derived scale.

    A scale of 1.0 keeps GT in its original physical/image range.
    """

    return 1.0


class PairedRestorationDataset(Dataset):

    def __init__(
        self,
        pairs,
        patch_size=64,
        augment=False,
        seed=42,
        noise_aug=True,
    ):
        self.pairs = pairs
        self.patch_size = patch_size
        self.augment = augment
        self.seed = seed
        self.noise_aug = noise_aug

    def __len__(self):
        return len(self.pairs)

    def _crop(self, lr, gt, rng):

        if self.patch_size is None:
            return lr, gt

        ph = pw = self.patch_size

        if lr.shape[0] < ph or lr.shape[1] < pw:
            return lr, gt

        sy = gt.shape[0] / lr.shape[0]
        sx = gt.shape[1] / lr.shape[1]

        if abs(sy - 2.0) > 1e-6 or abs(sx - 2.0) > 1e-6:
            raise ValueError(
                f"Expected 2x LR->GT scaling. "
                f"LR={lr.shape}, GT={gt.shape}"
            )

        y = int(
            rng.integers(
                0,
                lr.shape[0] - ph + 1
            )
        )

        x = int(
            rng.integers(
                0,
                lr.shape[1] - pw + 1
            )
        )

        hy = y * 2
        hx = x * 2

        return (
            lr[y:y + ph, x:x + pw],
            gt[hy:hy + 2 * ph, hx:hx + 2 * pw],
        )

    def __getitem__(self, idx):

        lr_path, gt_path = self.pairs[idx]

        lr = load_npy(lr_path)
        gt = load_npy(gt_path)

        if (
            gt.shape[0] != 2 * lr.shape[0]
            or gt.shape[1] != 2 * lr.shape[1]
        ):
            raise ValueError(
                f"Expected GT to be exactly 2x LR. "
                f"LR={lr.shape}, GT={gt.shape}"
            )

        rng = np.random.default_rng(
            self.seed + idx
        )

        lr, gt = self._crop(
            lr,
            gt,
            rng
        )

        # --------------------------------------------------
        # Geometric augmentation
        # --------------------------------------------------

        if self.augment:

            if rng.random() < 0.5:
                lr = np.flip(
                    lr,
                    axis=1
                ).copy()

                gt = np.flip(
                    gt,
                    axis=1
                ).copy()

            if rng.random() < 0.5:
                lr = np.flip(
                    lr,
                    axis=0
                ).copy()

                gt = np.flip(
                    gt,
                    axis=0
                ).copy()

            if rng.random() < 0.25:

                k = int(
                    rng.integers(1, 4)
                )

                lr = np.rot90(
                    lr,
                    k
                ).copy()

                gt = np.rot90(
                    gt,
                    k
                ).copy()

        # --------------------------------------------------
        # IMPORTANT NORMALIZATION
        #
        # GT is already [0,1].
        #
        # LR may contain values outside [0,1].
        # We preserve those values instead of clipping.
        # --------------------------------------------------

        gt = np.clip(
            gt,
            0.0,
            1.0
        )

        # --------------------------------------------------
        # Additional training degradation augmentation
        #
        # Gaussian:
        #      x' = x + n
        #
        # Speckle:
        #      x' = x * (1+n)
        #
        # These augment the existing real degradation.
        # --------------------------------------------------

        if self.augment and self.noise_aug:

            # Additive Gaussian noise
            if rng.random() < 0.50:

                sigma = float(
                    rng.uniform(
                        0.002,
                        0.015
                    )
                )

                gaussian = rng.normal(
                    0.0,
                    sigma,
                    size=lr.shape
                ).astype(np.float32)

                lr = lr + gaussian

            # Multiplicative speckle noise
            if rng.random() < 0.50:

                strength = float(
                    rng.uniform(
                        0.01,
                        0.05
                    )
                )

                speckle = rng.normal(
                    0.0,
                    strength,
                    size=lr.shape
                ).astype(np.float32)

                lr = lr * (
                    1.0 + speckle
                )

        lr = torch.from_numpy(
            np.ascontiguousarray(lr)
        ).unsqueeze(0).float()

        gt = torch.from_numpy(
            np.ascontiguousarray(gt)
        ).unsqueeze(0).float()

        return (
            lr,
            gt,
            Path(lr_path).stem
        )