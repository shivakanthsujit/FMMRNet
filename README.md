# Factorized Multiscale Multiresolution Residual Network

> Shivakanth Sujit & Deivalakshmi S, "Factorized Multiscale Multiresolution Residual Network."

## Overview

This repository provides a PyTorch implementation of FMMRNet.

## Installation

### Clone the repository

```bash
git clone https://github.com/shivakanthsujit/FMMRNet.git
cd FMMRNet
```

### Install dependencies

```bash
conda create -n env python=3.6
conda activate env
pip install -r requirements.txt
```

### Download the datasets

Download the dataset from <https://www.kaggle.com/shivakanthsujit/jrdr-deraining-dataset> and place the `JRDR` folder in `data`.

## Training

```bash
python train.py
python eval.py
```

Use Tensorboard to monitor the training.

`tensorboard --logdir lightning_logs`
