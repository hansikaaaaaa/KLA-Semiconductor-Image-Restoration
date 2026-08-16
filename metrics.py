from __future__ import annotations

import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


def psnr(
    pred,
    target,
    data_range=1.0,
):
    pred = np.asarray(
        pred,
        dtype=np.float32
    )

    target = np.asarray(
        target,
        dtype=np.float32
    )

    pred = np.clip(
        pred,
        0.0,
        1.0
    )

    target = np.clip(
        target,
        0.0,
        1.0
    )

    return float(
        peak_signal_noise_ratio(
            target,
            pred,
            data_range=data_range,
        )
    )


def ssim(
    pred,
    target,
    data_range=1.0,
):
    pred = np.asarray(
        pred,
        dtype=np.float32
    )

    target = np.asarray(
        target,
        dtype=np.float32
    )

    pred = np.clip(
        pred,
        0.0,
        1.0
    )

    target = np.clip(
        target,
        0.0,
        1.0
    )

    return float(
        structural_similarity(
            target,
            pred,
            data_range=data_range,
        )
    )