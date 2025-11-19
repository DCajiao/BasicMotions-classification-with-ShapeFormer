# BasicMotions Classification with ShapeFormer

A data analytics project focused on the implementation and deployment of **ShapeFormer**, a state-of-the-art transformer-based architecture for multivariate time series classification.

## Base Paper

**Title:** ShapeFormer: Shapelet Transformer for Multivariate Time Series Classification

**Authors:** Xuan-May Le, Ling Luo, Uwe Aickelin, Minh-Tuan Tran

**Publication:** KDD 2024 (30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining)

**Original Repository:** [xuanmay2701/shapeformer](https://github.com/xuanmay2701/shapeformer)

---

## Overview

ShapeFormer is a novel deep learning architecture that combines **class-specific** and **generic** transformer modules to achieve state-of-the-art performance on multivariate time series classification (MTSC) tasks. This implementation extends the original work with a focus on practical deployment and production-ready inference for basic motion classification.

### Key Achievement

ShapeFormer achieved the **highest accuracy ranking** across all 30 UEA multivariate time series datasets, with an average rank of 2.5 and top-1 performance on 15 datasets.

---

## Model Architecture & Main Innovations

### Core Architecture

ShapeFormer addresses a critical limitation of existing transformer-based MTSC methods: **existing methods focus exclusively on generic features while ignoring class-specific features**. This leads to poor performance on:

1. **Imbalanced datasets** – where generic features favor majority classes
2. **Similar overall patterns** – where fine-grained class distinctions matter

The architecture consists of two complementary transformer modules:

#### 1. **Class-Specific Transformer Module**

Captures discriminative class-specific features through:

- **Offline Shapelet Discovery (OSD)**: Extracts high-quality shapelets from training data using Perceptually Important Points (PIPs), reducing computational complexity from 45M to ~5,900 candidates per dataset
- **Shapelet Filter**: Finds best-fit subsequences for each shapelet and computes difference features between shapelets and their matching subsequences
- **Dynamic Shapelet Optimization**: Shapelets are treated as learnable parameters and optimized during training
- **Position Embedding**: Captures spatial information using start index, end index, and variable positions
- **Transformer Encoder**: Multi-head attention mechanism that assigns higher attention scores to features within the same class

#### 2. **Generic Transformer Module**

Extracts generic features across all classes through:

- **CNN Feature Extraction**: Two convolutional blocks with Conv1D layers (kernel size = 8) to capture temporal patterns and inter-variable correlations
- **Transformer Encoder**: Standard multi-head attention mechanism with learnable position embeddings
- **Average Pooling**: Aggregates features into a single class token

### Main Innovations

| Innovation | Impact |
|-----------|--------|
| **Dual-module architecture** | Leverages both class-specific and generic features for superior discrimination |
| **Offline Shapelet Discovery for MTS** | Efficiently extracts discriminative subsequences; 7,600x reduction in shapelet candidates |
| **Shapelet Filter with difference features** | Captures relative changes between shapelets and input series, not just absolute distances |
| **Dynamic shapelet optimization** | Shapelets adapt during training rather than remaining fixed |
| **Selective class token design** | Uses only the highest information gain shapelet token instead of averaging all tokens |
| **Position-aware shapelet embeddings** | Encodes temporal and variable location information for better discrimination |

---

## Theoretical Architecture Summary

### Shapelet Discovery Phase (Preprocessing)

```
Input: Training dataset D with M time series, T time steps, D variables
Output: Shapelet pool S with N shapelets per class

1. Shapelet Extraction:
   - Identify Perceptually Important Points (PIPs) based on reconstruction distance
   - Extract shapelet candidates from windows around consecutive PIPs
   - For each PIP: maximum 3 candidates generated → ~5,900 candidates total

2. Shapelet Selection:
   - Compute Perceptual Subsequence Distance (PSD) to all training instances
   - Calculate optimal information gain for each candidate
   - Select top N shapelets ranked by information gain for each class
```

### Inference Phase

```
For each input time series X:

1. Class-Specific Path:
   ├─ Shapelet Filter: Find best-fit subsequences for each shapelet
   ├─ Difference Feature: Compute ΔF = P(s_m) - P(x_matched)
   ├─ Position Embedding: Encode shapelet location information
   ├─ Transformer Encoder: Multi-head attention → Z_spe
   └─ Class Token: Use first token (Z_spe_1) from highest information gain shapelet

2. Generic Path:
   ├─ CNN Block 1: Conv1D for temporal pattern extraction
   ├─ CNN Block 2: Conv1D for inter-variable correlation
   ├─ Transformer Encoder: Multi-head attention → Z_gen
   └─ Class Token: Average pooling → Z_gen

3. Classification:
   ├─ Concatenate: [Z_spe, Z_gen]
   ├─ Linear Classification Head: softmax(Linear([Z_spe, Z_gen]))
   └─ Output: Class prediction
```

### Key Hyperparameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| Number of PIPs (n_pips) | 0.2T | Controls shapelet discovery granularity |
| Window size | Dataset-dependent | Search radius for best-fit subsequences |
| Number of shapelets (N) | Dataset-dependent | Shapelets per class for class-specific module |
| Embedding size (d_spe) | 128 | Class-specific feature dimension |
| Embedding size (d_gen) | 32 | Generic feature dimension |
| Attention heads | 16 | Multi-head attention heads |
| Dropout ratio | 0.4 | Regularization |
| Conv kernel size | 8 | CNN filter kernel size |

---

## Installation & Setup

The project requires Python 3.12 and uses the uv package manager for fast, reproducible dependency installation. The system can be deployed either locally for development or via Docker for production and isolated environments.

### Prerequisites

| Component	| Version	| Purpose |
|-----------|---------|---------|
| Python	| 3.12.x	| Runtime environment |
| uv |	Latest |	Fast Python package manager |
| Docker |	20.10+ | Container runtime (optional) |
| Git |	Any	| Repository cloning |

### Local Development Setup

1. Clone Repository

```bash
git clone https://github.com/DCajiao/BasicMotions-classification-with-ShapeFormer.git
cd BasicMotions-classification-with-ShapeFormer
```

2. Create Virtual Environment
The virtual environment is excluded from version control via gitignore
```bash
uv venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

3. Install Dependencies
The project defines 22 core dependencies in pyproject.toml
```bash
uv pip install -e .
```



### Docker Installation

```bash
# Build Docker image
docker build -t shapeformer:latest .

# Run container
docker run -p 8501:8501 shapeformer:latest
```

---

## Usage Guide

### Quick Start with Streamlit

The repository includes an interactive Streamlit application for real-time inference and visualization:

```bash
# Run Streamlit app
streamlit run app.py

# Access at: http://localhost:8501
```

### Training

For training with the BasicMotions dataset, run the following command:

```bash
python main.py --dataset_pos=1 --num_shapelet=10 --window_size=80 --pre_shapelet_discovery=0 --processes=12 --epochs=100 --lr=0.001 --weight_decay=5e-4 --dropout=0.2 --shape_embed_dim=128 --pos_embed_dim=128 --emb_size=64 --local_embed_dim=48 --local_pos_dim=48 --dim_ff=256 --num_heads=4 --local_num_heads=4 --num_pip=0.1

```

### Inference on Single Instance

For command-line inference, use:

```bash
python infer_shapeformer.py --ts_path "Dataset/UEA/Multivariate_ts/BasicMotions/BasicMotions_TEST.ts" --checkpoint "Results/Dataset/UEA/2025-11-16_12-02/checkpoints/BasicMotionsmodel_last.pth" --shapelet_pkl "store/BasicMotions_80.pkl" --config "Results/Dataset/UEA/2025-11-16_12-02/configuration.json" --device cpu
```

---

## Model Weights & Inference

### Loading Pre-trained Weights

```python
import torch
from shapeformer import ShapeFormer

# Initialize model with same architecture as saved weights
model = ShapeFormer(
    num_classes=4,
    num_variables=6,
    num_shapelets_per_class=10,
    embedding_size_specific=128,
    embedding_size_generic=32
)

# Load pre-trained weights
checkpoint = torch.load('shapeformer_basicmotions.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Optional: Load optimizer state for continued training
optimizer = torch.optim.RAdam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

### Shapelet Loading

Shapelets are pre-computed during the training phase and stored separately:

```python
import pickle
import numpy as np

# Load shapelet pool
with open('shapelets_pool.pkl', 'rb') as f:
    shapelets = pickle.load(f)

# Shapelet structure:
# {
#     'shapelets': [M x L x D] array of shapelet values,
#     'start_indices': [M] array of start positions,
#     'end_indices': [M] array of end positions,
#     'variables': [M] array of variable indices,
#     'class_labels': [M] array of class assignments
# }

print(f"Number of shapelets: {len(shapelets['shapelets'])}")
print(f"Shapelet length: {shapelets['shapelets'][0].shape[0]}")
```
---

## Experimental Results

ShapeFormer achieves state-of-the-art performance on UEA MTSC benchmark:

| Metric | Value |
|--------|-------|
| Average Rank | 2.5 |
| Top-1 Datasets | 15/30 (50%) |
| Statistical Significance | p-value < 0.05 |
| Performance vs. WHEN | +0.617 rank improvement |
| Performance vs. SVP-T | +2.783 rank improvement |

### Dataset-Specific Performance

- **BasicMotions**: 100.0% accuracy
- **AtrialFibrillation**: 66.0% accuracy (challenging imbalanced dataset)
- **Japanese Vowels**: 99.7% accuracy
- **NATOPS**: 98.9% accuracy

---

## 🔧 Configuration

### Hyperparameter Tuning

Modify `config.yaml` for custom settings:

```yaml
# Shapelet Discovery
shapelet_discovery:
  num_pips: 0.2  # Fraction of time series length
  window_size: 100  # Search radius for best-fit subsequences
  num_shapelets_per_class: 10

# Model Architecture
model:
  num_classes: 4
  num_variables: 6
  embedding_size_specific: 128
  embedding_size_generic: 32
  attention_heads: 16
  dropout: 0.4

# Training
training:
  batch_size: 16
  epochs: 200
  learning_rate: 0.01
  weight_decay: 5e-4
  optimizer: radam
```

---


### Local Deployment

```bash
# Run Streamlit in production mode
streamlit run app.py
```

---

## 📖 Documentation

For detailed information, refer to:

- **Original Paper**: [arXiv:2405.14608](https://arxiv.org/abs/2405.14608)
- **Original Repository**: [xuanmay2701/shapeformer](https://github.com/xuanmay2701/shapeformer)
- **UEA Archive**: [timeseriesclassification.com](http://timeseriesclassification.com/)

---

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{le2024shapeformer,
  title={ShapeFormer: Shapelet Transformer for Multivariate Time Series Classification},
  author={Le, Xuan-May and Luo, Ling and Aickelin, Uwe and Tran, Minh-Tuan},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```

---

**Last Updated:** November 2025  
**Status:** Active Development  
**Version:** 1.0.0