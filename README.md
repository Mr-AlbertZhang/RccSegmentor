# RccSegmentor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Task: Medical Segmentation](https://img.shields.io/badge/Task-Medical%20Segmentation-blue.svg)]()

> **Official implementation of the paper:** > **"RccSegmentor: Advancing Renal Cell Carcinoma Segmentation via Feature Aggregation and Exploration"** > *(Under Review / Accepted)*

---

## 🚀 Introduction

**RccSegmentor** is a novel deep learning framework tailored for the accurate and efficient segmentation of **Clear Cell Renal Cell Carcinoma (ccRCC)** from abdominal CT scans. It is designed to address critical clinical challenges, including small lesion volumes, diverse tumor locations, and significant shape variations.

### 🌟 Key Features
* **Backbone**: Built upon **Pyramid Vision Transformer v2 (PVTv2)** to capture robust global context.
* **Feature Exploration Module (FEM)**: Incorporates *Light-Channel/Spatial Attention* and a *Multi-scale Global Perception Unit (MGPU)* to suppress background noise and precisely localize small lesions.
* **Multi-scale Feature Aggregation Decoder (MFAD)**: Utilizes *Large Kernel Gated Cascade Modules (LK-GCM)* and *Group Gated Attention (GGA)* to enforce semantic alignment and capture intricate tumor morphology.
* **Efficiency**: Achieves **SOTA accuracy** (0.8506 Average Dice) with extremely low computational cost (**4.46 GFLOPs**, **136 FPS**).

---

## 🏗️ Architecture

![RccSegmentor Architecture](figure2.png)
The framework follows an encoder-decoder structure:
1.  **Encoder**: PVTv2 extracts multi-scale features ($X_1, X_2, X_3, X_4$).
2.  **Bottleneck**: FEM refines shallow features ($X_1$) to filter noise.
3.  **Decoder**: MFAD aggregates multi-scale features ($X_2, X_3, X_4$) with the refined shallow features.

---

## 🛠️ Usage

We provide the core model architecture implementation (`model.py`) to facilitate understanding of the network structure.

### 1. Requirements
* Python 3.8+
* PyTorch 1.10+
* CUDA 11.0+ (Recommended)

### 2. Inference Demo
You can initialize the model and perform a forward pass with a random tensor to verify the architecture.

```python
import torch
# Ensure 'model.py' is in your working directory
from model import RccSegmentor

# Define input: (Batch_size, Channels, Height, Width)
# Example: 1 image, 3 channels, 512x512 resolution
input_tensor = torch.randn(1, 3, 512, 512).cuda()

# Initialize model
# Note: Adjust parameters if your class __init__ requires specific args
model = RccSegmentor().cuda()

# Forward pass
output = model(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
# Expected Output: torch.Size([1, 1, 512, 512])
```
---

## ⚖️ Data & Code Availability

### Private Clinical Datasets
The multi-center clinical datasets (N Center, W Center, H Center) used in this study involve sensitive patient privacy and are subject to **strict institutional ethical governance**. Therefore, the raw imaging data cannot be hosted in this public repository.

### Code Access
The full training/testing scripts and pre-trained weights are tightly integrated with the internal data structure. To balance reproducibility with compliance:
* **Core Architecture**: Provided in this repository (`model.py`).
* **Full Access**: Researchers interested in reproducing the results may request access to the source code and/or data (subject to ethical approval) by contacting the corresponding author.

---

## 📝 Citation

If you find **RccSegmentor** useful for your research or clinical applications, please consider citing our paper:

```bibtex
@article{RccSegmentor2025,
  title={RccSegmentor: Advancing Renal Cell Carcinoma Segmentation via Feature Aggregation and Exploration},
  author={Zhang, Dong and Chen, Sihao and Wang, Lei and Shan, Shuai and Zhang, Yu-Dong},
  journal={Neurocomputing (Under Review)},
  year={2025}
}
```

(Citation details will be updated upon publication)

---

## 🙏 Acknowledgement

We thank **Jiangsu Province Hospital** for providing the clinical data and annotation support. We also acknowledge the **TCGA-KIRC** and **KiTS23** open-source communities for their valuable datasets.
