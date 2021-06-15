# Factorized Multiscale Multiresolution Residual Network

> Shivakanth Sujit & Deivalakshmi S, "Factorized Multiscale Multiresolution Residual Network."

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

Setup the Kaggle-API for downloading from the command line

```bash
pip install kaggle --upgrade
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets -h # Tests if the command works
```

## Training

```bash
python train.py
python eval.py
```

Use Tensorboard to monitor the training.

`tensorboard --logdir logs`
