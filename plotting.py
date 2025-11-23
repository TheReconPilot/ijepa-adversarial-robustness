#!/usr/bin/env python3
# common/plotting.py — plot train/val loss curves

import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_curves(history_csv_path: str, out_png_path: str):
    df = pd.read_csv(history_csv_path)
    # Expect columns: epoch, train_loss, val_loss
    if not {"epoch","train_loss","val_loss"}.issubset(df.columns):
        raise ValueError("metrics.csv must contain columns: epoch, train_loss, val_loss")
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_path)
