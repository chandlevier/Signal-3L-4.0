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

## 3. Install ESM2 and Foldseek

Signal-3L 4.0 uses ESM2 as a pretrained protein language model to generate sequence embeddings and Foldseek to obtain structure-based representations.
```bash
pip install fair-esm==2.0.0
conda install -c conda-forge -c bioconda foldseek
```

## 4. Install Necessary Dependencies

Install from local `packages/` directory
```bash
cd packages
pip install --no-index --find-links . *.whl
cd ..
```

Then, make sure the following core libraries required by Signal-3L 4.0 are installed.
```bash
pip install \
  atom3d==0.2.6 \
  biopython==1.83 \
  Brotli==1.0.9 \
  egnn-pytorch==0.2.7 \
  fair-esm==2.0.0 \
  huggingface-hub==0.26.1 \
  openpyxl==3.1.5 \
  pandas==2.0.3 \
  peft==0.13.2 \
  pytorch-lightning \
  scikit-learn==1.3.2 \
  seaborn==0.13.2 \
  torch-geometric==2.6.1 \
  torchdrug==0.2.1 \
  transformers==4.28.0
```

> Note: fair-esm==2.0.0 is listed here again for completeness; if you already installed it in Step 3, pip will just skip it.

## 5. Install Other Dependencies

Make sure you are in the project root directory (where `requirements.txt` is located), then run:

```bash
pip install -r requirements.txt
```
