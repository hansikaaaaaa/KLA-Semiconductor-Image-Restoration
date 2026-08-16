import torch
import torch.nn.functional as F


def charbonnier(p, t, eps=1e-3):
    diff = p - t

    return torch.mean(
        torch.sqrt(
            diff * diff + eps * eps
        )
    )


def mse_loss(p, t):
    return F.mse_loss(p, t)


def gradient_loss(p, t):

    px = (
        p[:, :, :, 1:]
        - p[:, :, :, :-1]
    )

    tx = (
        t[:, :, :, 1:]
        - t[:, :, :, :-1]
    )

    py = (
        p[:, :, 1:, :]
        - p[:, :, :-1, :]
    )

    ty = (
        t[:, :, 1:, :]
        - t[:, :, :-1, :]
    )

    return (
        F.l1_loss(px, tx)
        +
        F.l1_loss(py, ty)
    )


def restoration_loss(p, t):

    mse = mse_loss(p, t)

    char = charbonnier(p, t)

    grad = gradient_loss(p, t)

    return (
        0.80 * mse
        +
        0.18 * char
        +
        0.02 * grad
    )