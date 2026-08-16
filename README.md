# KLA Semiconductor Image Restoration

Deep learning based image restoration system developed for the KLA Semiconductor Image Restoration challenge.

The model takes noisy/low-quality grayscale semiconductor images in `.npy` format and produces restored images in `.npy` format.

## Model

The final trained model is a custom convolutional restoration network implemented in PyTorch.

- Base channels: 64
- Number of residual blocks: 10
- Input: single-channel grayscale `.npy`
- Output: single-channel restored `.npy`
- Framework: PyTorch
- GPU acceleration: CUDA

The trained weights are provided in:

```text
weights/best.pt