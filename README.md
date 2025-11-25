# RccSegmentor
Official implementation of RccSegmentor:Advancing Renal Cell Carcinoma Segmentation via Feature Aggregation and Exploration
This repository contains the official PyTorch implementation of the paper:

**"RccSegmentor: Advancing Renal Cell Carcinoma Segmentation via Feature Aggregation and Exploration"**

## 🚀 Introduction

RccSegmentor is a novel deep learning framework designed for accurate renal tumor segmentation. It integrates a **Pyramid Vision Transformer v2 (PVTv2)** backbone with two key modules:
- **Feature Exploration Module (FEM)**: Suppresses background noise and enhances small lesion localization.
- **Multi-scale Feature Aggregation Decoder (MFAD)**: Aligns features to capture complex tumor morphology.

## 🛠️ Usage

We provide the core model architecture code (`model.py`) for reference.

### Requirements
- Python 3.8+
- PyTorch 1.10+

### Example
```python
import torch
# Assuming the file is renamed to model.py
from model import RccSegmentor

# Input tensor: (Batch_size, Channels, Height, Width)
input_tensor = torch.randn(1, 3, 512, 512)
model = RccSegmentor()
output = model(input_tensor)

print("Output shape:", output.shape)
