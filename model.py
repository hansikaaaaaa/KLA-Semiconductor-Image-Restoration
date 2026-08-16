import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()

        self.c1 = nn.Conv2d(
            c,
            c,
            3,
            padding=dilation,
            dilation=dilation
        )

        self.c2 = nn.Conv2d(
            c,
            c,
            3,
            padding=1
        )

    def forward(self, x):
        residual = self.c2(
            F.gelu(
                self.c1(x)
            )
        )

        return x + 0.2 * residual


class SEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()

        h = max(c // 8, 4)

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                c,
                h,
                1
            ),

            nn.GELU(),

            nn.Conv2d(
                h,
                c,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x)


class RestorationNet(nn.Module):
    """
    1-channel noisy LR -> 1-channel clean HR image at 2x.

    PSNR-focused lightweight restoration architecture:

    - 64 feature channels
    - 10 residual blocks
    - dilated convolutions
    - SE channel attention
    - PixelShuffle 2x
    - bicubic low-frequency skip
    - unconstrained learned residual
    - final [0,1] output constraint
    """

    def __init__(
        self,
        base_channels=64,
        num_blocks=10
    ):
        super().__init__()

        c = base_channels

        # --------------------------------------------------
        # Input feature extraction
        # --------------------------------------------------

        self.head = nn.Conv2d(
            1,
            c,
            3,
            padding=1
        )

        # --------------------------------------------------
        # Residual feature body
        # --------------------------------------------------

        blocks = []

        for i in range(num_blocks):

            # Dilated blocks at multiple depths
            dilation = 2 if i in (2, 5, 8) else 1

            blocks.append(
                ResBlock(
                    c,
                    dilation=dilation
                )
            )

            # Channel attention
            if i in (3, 6, 9):
                blocks.append(
                    SEBlock(c)
                )

        self.body = nn.Sequential(
            *blocks
        )

        # --------------------------------------------------
        # Global feature fusion
        # --------------------------------------------------

        self.fuse = nn.Conv2d(
            c,
            c,
            3,
            padding=1
        )

        # --------------------------------------------------
        # 2x upsampling
        # --------------------------------------------------

        self.up = nn.Sequential(

            nn.Conv2d(
                c,
                4 * c,
                3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.GELU(),

            nn.Conv2d(
                c,
                c,
                3,
                padding=1
            ),

            nn.GELU()
        )

        # --------------------------------------------------
        # Reconstruction head
        # --------------------------------------------------

        self.tail = nn.Sequential(

            nn.Conv2d(
                c,
                c // 2,
                3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                c // 2,
                1,
                3,
                padding=1
            )
        )

    def forward(self, x):

        # --------------------------------------------------
        # Feature extraction
        # --------------------------------------------------

        f = F.gelu(
            self.head(x)
        )

        # --------------------------------------------------
        # Deep residual feature extraction
        # --------------------------------------------------

        body = self.body(f)

        b = self.fuse(body) + f

        # --------------------------------------------------
        # 2x reconstruction
        # --------------------------------------------------

        u = self.up(b)

        # --------------------------------------------------
        # Bicubic base image
        # --------------------------------------------------

        base = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False
        )

        # --------------------------------------------------
        # Learned high-frequency correction
        #
        # IMPORTANT:
        # No 0.25 scaling and no tanh bottleneck.
        # The network can learn the required correction.
        # --------------------------------------------------

        residual = self.tail(u)

        # --------------------------------------------------
        # Final image
        # --------------------------------------------------

        return torch.clamp(
            base + residual,
            0.0,
            1.0
        )