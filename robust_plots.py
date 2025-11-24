#!/usr/bin/env python3
# plots/robust_plots.py — simple matplotlib plots for robust curves

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_fgsm_curve(csv_path: str, out_png: str):
    df = pd.read_csv(csv_path)
    df = df[df["attack"] == "fgsm"].copy()
    if df.empty:
        print("No FGSM rows found in csv.")
        return
    df = df.sort_values("eps")
    plt.figure()
    plt.plot(df["eps"], df["robust_top1"], marker="o", label="FGSM ℓ∞")
    plt.xlabel("ε (ℓ∞)"); plt.ylabel("Robust Top-1 (%)")
    plt.title("FGSM Robustness Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close()

def plot_pgd_grid(csv_path: str, out_png: str, eps: float = None):
    df = pd.read_csv(csv_path)
    df = df[df["attack"].str.startswith("pgd_")].copy()
    if eps is not None:
        df = df[abs(df["eps"] - eps) < 1e-8]
    if df.empty:
        print("No PGD rows found (after eps filter)." if eps is not None else "No PGD rows found.")
        return
    # assume fixed eps; group by steps
    df = df.sort_values("steps")
    plt.figure()
    for name, sub in df.groupby("attack"):
        plt.plot(sub["steps"], sub["robust_top1"], marker="o", label=name)
    plt.xlabel("PGD steps"); plt.ylabel("Robust Top-1 (%)")
    title = f"PGD Robustness vs Steps (ε={eps:g})" if eps is not None else "PGD Robustness vs Steps"
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close()

def plot_autoattack_curve(csv_path, out_png):
    df = pd.read_csv(csv_path)
    df = df[df["attack"] == "autoattack"].copy()
    if df.empty:
        print("No AutoAttack rows found.")
        return
    df = df.sort_values("eps")
    plt.figure()
    plt.plot(df["eps"], df["robust_top1"], marker="o", label="AutoAttack")
    plt.xlabel("ε (ℓ∞)")
    plt.ylabel("Robust Top-1 (%)")
    plt.title("AutoAttack Robustness Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close()


if __name__ == "__main__":
    plot_fgsm_curve('runs/google_vith_imagenet100/robust/robust_metrics.csv','runs/google_vith_imagenet100/robust/fgsm_curve.png')
    plot_pgd_grid('runs/google_vith_imagenet100/robust/robust_metrics.csv','runs/google_vith_imagenet100/robust/pgd_steps_curve.png',eps=8/255)