# Evaluating the Adversarial Robustness of I-JEPA

This repository contains the code and experimental setup for the project **"Evaluating the Adversarial Robustness of Image-Joint Embedding Predictive Architectures"**.

This work investigates whether Self-Supervised Learning (SSL) methods—specifically the Image Joint-Embedding Predictive Architecture (I-JEPA)—provide greater adversarial robustness compared to standard supervised pretraining in a linear probe classification setting.

**Authors:** Aditya Agarwala & Purva Parmar (Indian Institute of Science)  
**Course:** DS 307: Ethics in AI during August 2025 semester at IISc Bengaluru

## Project Overview

This project compares two ViT-H/14 backbones pretrained on ImageNet-21k:

1.  **Supervised ViT:** `google/vit-huge-patch14-224-in21k`
2.  **I-JEPA:** `facebook/ijepa_vith14_22k`

We evaluate these models using a **Linear Probing** setup (frozen backbone, trained head) on **CIFAR-100** and **ImageNet-100** against gradient-based attacks (FGSM, PGD) and ensemble attacks (AutoAttack).

## Repository Structure

```text
.
├── attacks/                 # Implementations of FGSM and PGD attacks
├── robust_eval.py           # Main script for running adversarial evaluations
├── train_vit_linear.py      # Script for training linear probes on frozen backbones
├── train_eval.py            # Training and evaluation loops
├── robust_utils.py          # Utilities for normalization, feature extraction, and projections
├── data.py                  # Dataloaders for CIFAR-100 and ImageNet-100
├── plotting.py              # Training loss plotting tools
├── robust_plots.py          # Adversarial robustness curve plotting tools
├── requirements.txt         # Python dependencies
└── *.txt                    # Reference commands for training and attacks
```

The plotting utilities have not been updated sufficiently at the moment.

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Models

The code automatically downloads the following models from Hugging Face:

  * **Supervised:** `google/vit-huge-patch14-224-in21k`
  * **I-JEPA:** `facebook/ijepa_vith14_22k`

## Usage

### 1\. Training the Linear Probe

Before evaluating robustness, we must train a linear classification head on top of the frozen backbone. The script `train_vit_linear.py` handles this.

**Arguments:**

  * `--model_type`: Choose `vit` (Supervised) or `ijepa` (Self-Supervised).
  * `--use_bn_head`: (ViT only) Applies Batch Norm before the head (recommended for Google ViT)
  * `--last4`: (I-JEPA only) Concatenates the last 4 avg pooled layer patches. Not using this flag results in the `last1` variant where just the last layer patches are average pooled.

**Example: Training Google ViT on CIFAR-100**

```bash
python train_vit_linear.py \
  --dataset cifar100 \
  --model_id google/vit-huge-patch14-224-in21k \
  --model_type vit \
  --use_bn_head \
  --batch_size 256 --epochs 25 \
  --precision amp --seed 0
```

**Example: Training I-JEPA (Last-4) on ImageNet-100**

```bash
python train_vit_linear.py \
  --dataset imagenet100 \
  --model_id facebook/ijepa_vith14_22k \
  --model_type ijepa \
  --last4 \
  --batch_size 256 --epochs 25 \
  --precision amp --seed 0
```

*See `model_train_commands.txt` for more configurations.*

### 2\. Adversarial Evaluation

Once the linear probe is trained (checkpoints saved in `runs/`), use `robust_eval.py` to run attacks.

**Supported Attacks:** `fgsm`, `pgd_inf`, `pgd_l2`, `autoattack`.

**Example: Running PGD-Linf on CIFAR-100**

```bash
python robust_eval.py \
  --dataset cifar100 \
  --model_type vit \
  --model_id google/vit-huge-patch14-224-in21k \
  --ckpt runs/cifar100/google_vit_bn_on/best-val-top1.pt \
  --attack pgd_inf \
  --eps 1/255 \
  --steps 10,20 \
  --restarts 3 \
  --out_dir robust_results/google_cifar100_pgd
```

**Example: Running AutoAttack on I-JEPA**

```bash
python robust_eval.py \
  --dataset imagenet100 \
  --model_type ijepa \
  --last4 \
  --model_id facebook/ijepa_vith14_22k \
  --ckpt runs/imagenet100/ijepa_last4/best-val-top1.pt \
  --attack autoattack \
  --eps 1/255 \
  --aa_norm linf \
  --aa_individual \
  --out_dir robust_results/ijepa_imagenet100_autoattack
```

*See `model_attack_commands.txt` for more configurations.*

### 3\. Multi-GPU Scheduling (Optional)

If you have access to multiple GPUs, you can use `training_scheduler.py` to queue and distribute training commands automatically across available devices.

## Report

More details about the project can be found in the Report file.