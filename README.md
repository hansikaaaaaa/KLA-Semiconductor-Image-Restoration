# KLA Semiconductor Image Restoration

Deep learning based image restoration system developed for the **KLA Semiconductor Image Restoration Challenge**.

The model takes noisy/low-quality grayscale semiconductor images in `.npy` format and produces restored images in `.npy` format.

## Model

The final trained model is a custom convolutional restoration network implemented in PyTorch.

* **Input:** Single-channel grayscale `.npy` image
* **Output:** Single-channel restored `.npy` image
* **Base channels:** 64
* **Residual blocks:** 10
* **Framework:** PyTorch
* **GPU acceleration:** CUDA
* **Trained model:** `weights/best.pt`

The trained checkpoint contains the model parameters and configuration required by the evaluation script.

## Repository Structure

```text
KLA-Semiconductor-Image-Restoration/
│
├── dataset.py
├── model.py
├── losses.py
├── metrics.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── weights/
│   └── best.pt
│
└── restored_test_outputs/
    ├── 000000.npy
    ├── 000001.npy
    ├── ...
    └── 000399.npy
```

## Requirements

The project uses Python and PyTorch.

The complete package environment used for the project is provided in:

```text
requirements.txt
```

Install all required packages using:

```bash
pip install -r requirements.txt
```

For GPU inference, a CUDA-enabled PyTorch installation is recommended.

## Setup

Clone the repository:

```bash
git clone https://github.com/hansikaaaaaa/KLA-Semiconductor-Image-Restoration.git
```

Enter the repository:

```bash
cd KLA-Semiconductor-Image-Restoration
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The trained model is already included in:

```text
weights/best.pt
```

No modification of the source code or model files is required for inference.

## Input Data Format

The evaluation script expects a directory containing the test images in `.npy` format.

Each input file represents a single-channel grayscale image.

Example:

```text
NoisyLR/
├── 000000.npy
├── 000001.npy
├── 000002.npy
├── ...
└── 000399.npy
```

The evaluation script automatically discovers the supported `.npy` test images in the specified directory.

## Running Inference

The evaluation script is the main benchmarking entry point for the trained model.

From the **repository root**, run:

```bash
python evaluate.py <test_images_directory> <output_directory>
```

### Example

```bash
python evaluate.py "C:\path\to\NoisyLR" "restored_outputs"
```

For example, if the test images are located at:

```text
C:\Users\phans\Downloads\SEMICON2\KLA_Semiconductor_Image_Restoration_Submission(1)\test\NoisyLR
```

run:

```powershell
python evaluate.py "C:\Users\phans\Downloads\SEMICON2\KLA_Semiconductor_Image_Restoration_Submission(1)\test\NoisyLR" "restored_outputs"
```

The script automatically loads:

```text
weights/best.pt
```

No source-code modification is required.

The two required positional arguments are:

1. **Test images directory** — directory containing the input `.npy` images.
2. **Output directory** — directory where the restored `.npy` files will be written.

## Output Format

The restored outputs are saved as `.npy` files in the output directory.

For example:

```text
restored_outputs/
├── 000000.npy
├── 000001.npy
├── 000002.npy
├── ...
└── 000399.npy
```

The output filenames correspond to the input filenames.

The predictions are stored as `float32` NumPy arrays and are safely clipped to the range `[0, 1]`.

## Benchmarking / Inference Behavior

The evaluation script:

1. Loads the trained checkpoint.
2. Constructs the restoration model.
3. Loads all `.npy` files from the supplied test directory.
4. Runs inference using CUDA when a compatible GPU is available.
5. Falls back to CPU when CUDA is unavailable.
6. Saves one restored `.npy` file for each input image.
7. Reports the number of processed images and average inference time.

The script uses PyTorch inference mode and synchronizes CUDA before and after GPU inference timing.

## Tested Inference

The evaluation pipeline was tested on a test set containing **400 images**.

Example successful execution:

```text
Loading checkpoint: weights/best.pt
Model configuration: {'base_channels': 64, 'num_blocks': 10}
Validation PSNR: 28.599383467151277
Validation SSIM: 0.7746947352344742
Test images: 400
Processed 50/400
Processed 100/400
Processed 150/400
Processed 200/400
Processed 250/400
Processed 300/400
Processed 350/400
Processed 400/400
```

The inference run completed successfully using CUDA and generated **400 restored output files**.

The exact inference time depends on the hardware and runtime environment.

## Training

The training implementation is provided in:

```text
train.py
```

Supporting training components are provided in:

```text
dataset.py
losses.py
metrics.py
model.py
```

The training script contains the complete model training pipeline, including dataset loading, model construction, optimization, validation, metric calculation, and checkpoint saving.

To reproduce training, first prepare the required training/validation dataset in the format expected by `dataset.py`, install the dependencies from `requirements.txt`, and run:

```bash
python train.py
```

The trained model checkpoint can then be used by `evaluate.py`.

## Evaluation Script

The standalone evaluation script is:

```text
evaluate.py
```

It is a regular Python script and does not require a Jupyter notebook.

Its interface is:

```bash
python evaluate.py <test_images_directory> <output_directory>
```

The script loads the trained model from:

```text
weights/best.pt
```

and writes the restored images to the output directory supplied by the user.

The script is designed to run without manual source-code edits.

## Trained Model Weights

The final trained model weights are included in the repository:

```text
weights/best.pt
```

The checkpoint is directly loadable by the provided `evaluate.py` script.

GitHub may display a message indicating that `.pt` files cannot be rendered in the browser. This is expected because the file is a binary PyTorch checkpoint. The file itself is included in the repository and is downloaded when the repository is cloned.

## Restored Test Outputs

The repository contains the actual restored outputs produced by the final trained model:

```text
restored_test_outputs/
```

The directory contains:

```text
400 .npy output files
```

corresponding to the 400 test images processed by the evaluation script.

## Reproducibility

All essential components required to reproduce inference are included in the repository:

* Model architecture
* Dataset utilities
* Loss functions
* Evaluation metrics
* Training script
* Standalone evaluation script
* Trained model weights
* Python package requirements
* Restored test outputs

The evaluation entry point is:

```bash
python evaluate.py <test_images_directory> <output_directory>
```

No manual modification of `evaluate.py` is required.

## Validation Results

The final checkpoint reports the following validation metrics:

| Metric          |     Value |
| --------------- | --------: |
| Validation PSNR | 28.599383 |
| Validation SSIM |  0.774695 |

These values are stored in the trained checkpoint and are printed by the evaluation script when the checkpoint is loaded.

GitHub Repository:

https://github.com/hansikaaaaaa/KLA-Semiconductor-Image-Restoration
