# Signal-3L 4.0 – Environment Setup

This document describes how to set up the Python environment required to run **Signal-3L 4.0**.

## 1. Create Conda Environment

First, create a new Conda environment named `signal3lv4` with **Python 3.8**:

```bash
conda create -n signal3lv4 python=3.8
conda activate signal3lv4
```

## 2. Install Pytorch

Install PyTorch (with CUDA 12.1 support) and related packages via Conda:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

## 3. Install Python Dependencies

Make sure you are in the project root directory (where `requirements.txt` is located), then run:

```bash
pip install -r requirements.txt
```
