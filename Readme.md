# DIP-UP – Deep Image Prior for MRI Phase Unwrapping

[deepMRI collection](https://github.com/sunhongfu/deepMRI)

DIP-UP provides a DIP-based enhancement mechanism for pre-trained 3D CNN models in MRI phase unwrapping. It refines existing network outputs using two unsupervised constraints: Laplacian loss and total variation (TV) loss — no additional labeled data needed.

**Pipeline:** (1) Train a base phase unwrapping network → (2) Apply DIP enhancement at test time.

> 16–24 GB GPU memory recommended. Tested on Windows 10.

---

## Overview

### Pipeline

![DIP-UP Pipeline](https://github.com/user-attachments/assets/5bb02af5-2f37-4f3d-9182-f31520c544aa)

---

## Requirements

- Python 3.7+, PyTorch
- NVIDIA GPU (16–24 GB VRAM recommended)

---

## Checkpoints and Test Data

- **Pre-trained models** (PHU-NET3D and PhaseNet3D):
  [Dropbox](https://www.dropbox.com/scl/fo/r4qv54fsdznxxmqgdcwn0/AMCMT9M-hvgmZ276-hcgaV8?rlkey=di0g1drz4whpz308rn84cxnx9&st=yw6d5qfz&dl=0)

- **Test data** (simulation: 10 ms TE, σ=0.1; in vivo: 5.8 ms TE):
  [Dropbox](https://www.dropbox.com/scl/fo/iz98whja62v5ih6idg60g/AEyEcemOzOd6yY1A0n2sldM?rlkey=zol5wkvxn4onu6xi5mf55w3en&st=54svy08l&dl=0)

---

## Usage

### Step 1 – Prepare data

Two network variants are provided:

| Network | Data loader | Input channels |
|---------|-------------|----------------|
| PHU-NET3D | `TrainingDataLoad_ResidueLoss_2Chan.py` | `image_file`, `lap_file`, `Label_file` |
| PhaseNet3D | `TrainingDataLoad_ResidueLoss_1Chan.py` | `image_file`, `Label_file` |

### Step 2 – Train base network

Edit the following variables in the training script:

| Variable | Description |
|----------|-------------|
| `DATA_DIRECTORY` | Root directory of training dataset |
| `DATA_LIST_PATH` | Index file (e.g. `test_IDs_28800.txt`) |
| `ModelFolder` | Path to save trained model checkpoints |
| `ModelName` | Name for saved model files |

### Step 3 – DIP enhancement (test time)

Edit and run the inference script:

| Variable | Description |
|----------|-------------|
| `SourceDir` | Directory of inference data |
| `ReconType` | `Simulation` or `InVivo` |
| `SaveDir_NIFTI` | Output directory for `.nii` results |

**Learning rate options:**
- Variable: decays every 10 epochs during DIP
- Constant: uses default fixed value

---

[⬆ top](#dip-up--deep-image-prior-for-mri-phase-unwrapping) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)
