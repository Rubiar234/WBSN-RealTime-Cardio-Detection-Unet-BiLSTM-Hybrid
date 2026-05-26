# FT-UNet + TW-MT-BiLSTM: Real-Time Cardiopulmonary Detection in WBSN

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Fine-Tuned U-Net and Time-Weighted Bi-Directional LSTM for Real-Time Cardiopulmonary Detection in WBSN Data**  
> *Rubia R., Dr. Sasikala S. — Velammal College of Engineering and Technology, Madurai*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start (Google Colab)](#quick-start-google-colab)
- [Preprocessing](#preprocessing)
- [Model Configuration](#model-configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Streaming Inference Configuration](#streaming-inference-configuration)
- [Clinical Label Generation](#clinical-label-generation)
- [Reproducibility Notes](#reproducibility-notes)

---

## Overview

This repository provides the full implementation for the paper **"Fine-Tuned U-Net and Time-Weighted Bi-Directional LSTM for Real-Time Cardiopulmonary Detection in WBSN Data"**. The proposed hybrid deep learning framework — **F-UTrans-BPNet** — combines:

- A **Fine-Tuned U-Net** encoder-decoder for spatial feature extraction from raw physiological waveforms (PPG, ECG, ABP).
- A **Time-Weighted Multi-Task Bi-Directional LSTM (TW-MT-BiLSTM)** for temporal sequence modeling with attention-based weighting.
- A **Multi-Task Learning** head that simultaneously predicts Blood Pressure (Systolic BP), SpO₂, and Respiratory Rate.

The model is validated on real `.mat` waveform data sampled from ICU physiological recordings and achieves strong regression and classification performance across all three vital sign targets.

---

## Architecture

```
Raw Signal Input (PPG / ECG)
        │
        ▼
 ┌──────────────────────┐
 │  Fine-Tuned U-Net    │  ← InterAxialBlock + IncBlock encoder/decoder
 │  (Spatial Features)  │    5-level encoder (32→64→128→256→512 channels)
 └──────────┬───────────┘    Skip connections + transposed convolutions
            │  Feature maps [batch, 1, seq_len]
            ▼
 ┌──────────────────────────────────┐
 │  TW-MT-BiLSTM                   │
 │  ├─ Bidirectional LSTM           │  ← hidden_dim=64 per direction
 │  ├─ Temporal Attention (Softmax) │  ← attn = Linear(128, 1)
 │  └─ Context Vector               │
 └──────┬──────────┬──────────┬────┘
        │          │          │
        ▼          ▼          ▼
    SBP Head   SpO₂ Head   RR Head
  (Linear→1) (Linear→1) (Linear→1)
```

**Key Innovations:**
- **IncBlock** — Inception-style multi-scale 1D convolution at each encoder/decoder level.
- **InterAxialBlock** — Cross-axial attention bridging 1D and 2D convolutional representations.
- **Temporal Attention** — Soft weighting over LSTM hidden states, emphasising clinically critical time steps.
- **Multi-Task Heads** — Shared latent representation reduces variance and captures physiological correlations between BP, SpO₂, and RR.

---

## Repository Structure

```
FT-UNet-TW-BiLSTM-Cardiopulmonary-WBSN-Detection/
│
├── main_2.ipynb              # Main notebook (data loading → training → evaluation)
├── EXECUTION_GUIDE.txt       # Step-by-step execution instructions
├── README.md                 # This file
│
├── Samples/                  # Dataset folder (download separately — see Dataset section)
│   ├── part_1.mat            # Physiological waveform data (PPG, ECG, ABP)
│   └── 
│
└── requirements.txt          # Python dependencies
```

> **Note:** The `Samples/` folder is **not** included in this repository due to size constraints. See the [Dataset](#dataset) section for download instructions.

---

## Dataset

This study uses a `.mat` waveform dataset containing multi-channel physiological signals (PPG, ECG, Blood Pressure) recorded at **125 Hz** from ICU patients.

### Dataset Format

Each `.mat` file (`part_1.mat`, etc.) contains a variable `p` of shape `(1, N)` where each entry is a matrix with rows:

| Row Index | Signal         |
|-----------|----------------|
| `0`       | PPG            |
| `1`       | Blood Pressure |
| `2`       | ECG            |

### How to Obtain the Data

The dataset is derived from the **MIMIC-III Waveform Database** available via PhysioNet:

1. Register at [https://physionet.org](https://physionet.org) (free, requires CITI training)
2. Accept the data use agreement for MIMIC-III
3. Download waveform records and convert to `.mat` format using the WFDB toolbox:
   ```bash
   pip install wfdb
   python -c "import wfdb; wfdb.dl_database('mimic3wdb', './data')"
   ```
4. Place processed `.mat` files in the `Samples/` folder.

The `Samples/` folder provided in the repository (for authorized users) contains pre-extracted, anonymized segments ready for direct use with the notebook.

### Signal Segmentation

- **Window size:** 125 samples per segment (1 second at 125 Hz)
- **Number of records loaded per run:** 1,000 subjects
- **Labels extracted per window:**
  - `SBP` = max(BP segment) — Systolic Blood Pressure
  - `DBP` = min(BP segment) — Diastolic Blood Pressure

---

## System Requirements

| Component         | Requirement                          |
|-------------------|--------------------------------------|
| Python            | 3.10+                                |
| Runtime           | Google Colab (recommended) or Jupyter|
| GPU               | Recommended (CUDA-enabled)           |
| TensorFlow        | 2.x                                  |
| PyTorch           | 2.x                                  |
| RAM               | ≥ 8 GB                               |
| Storage           | ≥ 2 GB (for dataset + checkpoints)   |

---

## Installation

### Option 1 — Google Colab (Recommended)

No local installation needed. See [Quick Start (Google Colab)](#quick-start-google-colab).

### Option 2 — Local Environment

```bash
# Clone the repository
git clone https://github.com/Rubiar234/FT-UNet-TW-BiLSTM-Cardiopulmonary-WBSN-Detection.git
cd FT-UNet-TW-BiLSTM-Cardiopulmonary-WBSN-Detection

# Install dependencies
pip install tensorflow numpy pandas scipy scikit-learn matplotlib imbalanced-learn shap lime torch
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Requirements

```
tensorflow>=2.10
torch>=2.0
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
imbalanced-learn>=0.11
shap>=0.42
lime>=0.2
```

---

## Quick Start (Google Colab)

Follow these steps to reproduce results in Google Colab:

### Step 1 — Upload Files to Google Drive

Upload the following to your Google Drive root:
- `main_2.ipynb`
- `Samples/` folder (containing `part_1.mat`)

### Step 2 — Open Notebook in Colab

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open Notebook → Google Drive**
3. Select `main_2.ipynb`

### Step 3 — Mount Google Drive

Run **Cell 1** (first cell):
```python
from google.colab import drive
drive.mount('/content/drive')
```
Authorize Google Drive access when prompted.

### Step 4 — Update Dataset Paths

In **Cell 2**, locate and update the following lines:

**Line 44** — Set the Samples folder path:
```python
# Before
os.listdir('Samples/')

# After
os.listdir('/content/drive/MyDrive/Samples')
```

**Lines 50 and 54** — Set the `.mat` file paths:
```python
# Before
sample_file = scipy.io.loadmat('Samples/part_1.mat')
test_sample  = scipy.io.loadmat('Samples/part_1.mat')['p']

# After
sample_file = scipy.io.loadmat('/content/drive/MyDrive/Samples/part_1.mat')
test_sample  = scipy.io.loadmat('/content/drive/MyDrive/Samples/part_1.mat')['p']
```

### Step 5 — Run All Cells

Run all cells sequentially (**Runtime → Run all**). The notebook will:

1. Load and segment the physiological signals (PPG, ECG, ABP)
2. Extract SBP and DBP labels per window
3. Define and instantiate the U-Net + TW-MT-BiLSTM architecture
4. Train a baseline Linear Regression with 5-fold cross-validation
5. Train and evaluate the proposed F-UTrans-BPNet model
6. Generate all performance tables and comparison plots

---

## Preprocessing

All preprocessing is performed within `main_2.ipynb`. The pipeline:

| Step | Operation | Details |
|------|-----------|---------|
| 1 | **Signal loading** | `.mat` file read via `scipy.io.loadmat`; variable `p` extracted |
| 2 | **Segmentation** | Fixed windows of 125 samples (1 s at 125 Hz) |
| 3 | **Label extraction** | SBP = max(BP window), DBP = min(BP window) |
| 4 | **Reshaping** | PPG, ECG, BP arrays reshaped to column vectors `(-1, 1)` |
| 5 | **Train/test split** | 70% train / 30% test via `train_test_split` |
| 6 | **Cross-validation** | 5-fold KFold on training set |
| 7 | **Normalization** | Min-max applied per channel on training fold only |

**Random seed:** `np.random.seed(42)` is set at the top of the notebook for reproducibility.

---

## Model Configuration

### F-UTrans-BPNet Hyperparameters

| Parameter | Value |
|-----------|-------|
| U-Net Encoder Channels | 1 → 32 → 64 → 128 → 256 → 512 |
| BiLSTM Hidden Units | 64 per direction (128 total) |
| BiLSTM Layers | 2 |
| Dropout Rate | 0.3 |
| Attention Mechanism | Softmax over Linear(128 → 1) |
| Input Signal Length | 125 samples |
| Output Heads | SBP, SpO₂, Respiratory Rate |
| Loss Function | Multi-Task MSE (sum of per-head MSE) |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 100 |
| Activation (hidden) | LeakyReLU (0.2) |
| Activation (output) | Linear (regression) |

---

## Training

The model training is executed directly within the notebook. To run training:

```python
# Instantiate model
model = F_UTrans_BPNet(lstm_input_dim=1, lstm_hidden_dim=64)

# Sample input: batch of 32, single-channel signal of length 125
sample_input = torch.rand(32, 1, 125)
sbp_out, spo2_out, rr_out = model(sample_input)
```

Baseline comparison (Linear Regression with 5-fold CV) is also included and runs automatically.

---

## Evaluation

The notebook computes all metrics reported in the paper:

**Regression metrics** (BP, SpO₂, RR):
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

**Classification metrics** (SpO₂ and RR using clinical thresholds):
- Accuracy, Precision, Recall, F1-Score

All metrics are printed as a summary table and bar charts are generated for visual comparison.

---

## Results

### Table 1 — TW-MT-BiLSTM Performance

| Vital Sign | MAE | RMSE | R² Score | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|
| Blood Pressure (BP) | 4.52 mmHg | 6.01 | 0.92 | 60% | 55% | 58% | 56% |
| SpO₂ | 1.87 % | 2.41 | 0.95 | 96% | 94% | 97% | 95% |
| Respiratory Rate | 1.65 br/min | 2.03 | 0.91 | 93% | 90% | 94% | 92% |

### Table 2 — Baseline Model Comparison

| Model | MAE (BP) | MAE (SpO₂) | MAE (RR) | RMSE (BP) | RMSE (SpO₂) | RMSE (RR) | Acc (SpO₂) | F1 (RR) |
|---|---|---|---|---|---|---|---|---|
| U-Net + LSTM | 5.24 | 2.96 | 1.92 | 7.89 | 3.75 | 2.31 | 85% | 80% |
| U-Net + BiLSTM | 4.95 | 2.65 | 1.83 | 6.84 | 3.42 | 2.05 | 89% | 85% |
| U-Net + TW-LSTM | 4.50 | 2.30 | 1.65 | 6.10 | 3.15 | 1.94 | 92% | 87% |
| **TW-MT-BiLSTM (Proposed)** | **4.12** | **1.87** | **1.45** | **5.61** | **2.61** | **1.76** | **96%** | **92%** |

### Table 3 — Multi-Task vs. Single-Task Ablation

| Task | Model Type | MAE | RMSE | Accuracy | F1-Score |
|---|---|---|---|---|---|
| Blood Pressure | Single-Task | 4.85 | 6.92 | — | — |
| Blood Pressure | **Multi-Task** | **4.12** | **5.61** | — | — |
| SpO₂ | Single-Task | 2.15 | 3.01 | 0.90 | — |
| SpO₂ | **Multi-Task** | **1.87** | **2.63** | **0.96** | — |
| Respiratory Rate | Single-Task | 1.72 | 2.60 | — | 0.86 |
| Respiratory Rate | **Multi-Task** | **1.45** | **2.18** | — | **0.92** |

---

## Streaming Inference Configuration

The following configuration was used for real-time streaming inference evaluation in Wireless Body Sensor Network (WBSN) conditions:

| Parameter | Value |
|---|---|
| Sampling Frequency | 100 Hz |
| Window Length | 5 s |
| Window Overlap | 50% |
| Future Context Buffer | 1 s |
| Batch Size | 32 |
| Learning Rate | 0.001 |

> **Note:** Window overlap of 50% means each new inference window advances by 2.5 s, with a 1 s future context buffer retained for bidirectional LSTM context completeness.

---

## Clinical Label Generation

Continuous physiological measurements were converted into binary classification labels using clinically validated threshold ranges. These thresholds were applied after regression prediction to compute Accuracy, Precision, Recall, and F1-Score.

| Physiological Parameter | Normal Range | Abnormal Condition |
|---|---|---|
| SpO₂ | ≥ 95% | < 95% |
| Respiratory Rate | 12–20 breaths/min | < 12 or > 20 breaths/min |
| Blood Pressure | Clinical reference range | Outside range |

**Python implementation (from notebook):**

```python
# SpO₂ label generation
spo2_labels      = (true_spo2 >= 95).astype(int)   # 1 = Normal, 0 = Abnormal
pred_spo2_labels = (pred_spo2  >= 95).astype(int)

# Respiratory Rate label generation
rr_labels      = (true_rr > 16).astype(int)
pred_rr_labels = (pred_rr > 16).astype(int)
```

---

## Reproducibility Notes

| Item | Detail |
|---|---|
| **Random seed** | `np.random.seed(42)` — set at top of notebook |
| **Train/test split** | 70% / 30%, fixed by `train_test_split` default state |
| **Cross-validation** | 5-fold KFold, `shuffle=False` |
| **Hardware** | Results obtained on Google Colab with GPU runtime (T4) |
| **TensorFlow version** | 2.x — minor numerical differences possible across versions |
| **Dataset files** | Use exactly the `.mat` files in the `Samples/` folder provided |
| **Path sensitivity** | All dataset paths must be updated before running (see [Quick Start](#quick-start-google-colab)) |

> ⚠️ Results may vary slightly depending on hardware, GPU driver, and TensorFlow/PyTorch version due to floating-point non-determinism in CUDA operations. Core trends and rankings will remain consistent.

---

*This repository is maintained for academic reproducibility purposes.*
