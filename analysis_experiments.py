#!/usr/bin/env python3
"""analysis_experiments.py

Runs four follow-up analyses (E1, E2, E3, E5 from REPRESENTATION_PROBE_REPORT.md
Appendix C) end to end:

  E1 — Cross-model adversarial transferability matrix (FGSM at one eps).
  E2 — CKA between model representations on a fixed image set.
  E3 — Patch leave-one-out importance maps (16x16 grid).
  E5 — Per-model logit-margin distributions (clean and under FGSM).

Each model is loaded once. Adversarial inputs and clean features are cached in
RAM/disk and then re-used for the cross-evals. MAE is loaded with mask_ratio=0
(per the corrected eval pipeline).

Outputs land in ``analysis_experiments/``:
  transferability/transfer_matrix.csv          + transfer_matrix.png
  cka/cka_matrix.csv                            + cka_matrix.png
  patch_loo/{model}_avg_importance.npy          + patch_loo_grid.png + patch_loo_summary.csv
  margins/{model}_margins.csv                   + margins_cdf.png + margins_summary.csv
  summary.json                                  (consolidated results)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from occlusion_eval import (
    DEFAULT_PROBES,
    ProbeSpec,
    build_dataset,
    load_probe_meta,
    resolve_device,
    resolve_probe_specs,
)
from representation_probe import _maybe_disable_mae_masking
from robust_eval import FullClassifier, build_eval_transform, load_encoder_and_head
from robust_utils import seed_all, set_extract_config, clamp01, extract_embed


# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------

def fgsm_attack(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                eps: float) -> torch.Tensor:
    """Single-step FGSM. x is in [0,1]."""
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y, reduction="sum")
    grad, = torch.autograd.grad(loss, x_adv)
    return clamp01(x_adv.detach() + eps * grad.sign())


def patch_dropout_batch(x: torch.Tensor, patch_size: int = 14,
                        fill: float = 0.5) -> torch.Tensor:
    """Given a single image x [3,H,W], return a [Pn, 3, H, W] batch where each
    item has one of the Pn = (H/p)*(W/p) non-overlapping patches replaced by
    ``fill`` (gray). The first item is patch index (0,0) -> top-left.
    """
    C, H, W = x.shape
    nh, nw = H // patch_size, W // patch_size
    n = nh * nw
    out = x.detach().clone().unsqueeze(0).expand(n, C, H, W).contiguous()
    for idx in range(n):
        r = idx // nw
        c = idx % nw
        out[idx, :, r * patch_size:(r + 1) * patch_size,
                  c * patch_size:(c + 1) * patch_size] = fill
    return out


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) between two feature matrices [N,D1]
    and [N,D2]. Centered, biased estimator."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    KX = X @ X.T
    KY = Y @ Y.T
    hsic_xy = (KX * KY).sum()
    hsic_xx = (KX * KX).sum()
    hsic_yy = (KY * KY).sum()
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12))


def margin_per_sample(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """z_y - max_{c != y} z_c per sample."""
    B, C = logits.shape
    z_y = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.unsqueeze(1), float("-inf"))
    z_other = masked.max(dim=1).values
    return (z_y - z_other).detach()


# ----------------------------------------------------------------------
#  Core pass: one model at a time
# ----------------------------------------------------------------------

@dataclass
class Pass1Out:
    model_name: str
    clean_features: np.ndarray  # [N, D]
    clean_logits: np.ndarray    # [N, C]
    margins_clean: np.ndarray   # [N]
    margins_fgsm: np.ndarray    # [N]
    own_clean_acc: float
    own_fgsm_acc: float
    x_adv_path: str             # saved tensor of adversarial inputs
    patch_loo_avg: np.ndarray   # [nh, nw] mean log-prob drop on true class
    patch_loo_per_image: np.ndarray  # [n_loo_imgs, nh, nw]
    patch_grid_size: Tuple[int, int]


def pass1_one_model(spec: ProbeSpec, X_pp: torch.Tensor, Y: torch.Tensor,
                    device: torch.device, eps: float, batch_size: int,
                    loo_indices: Sequence[int], loo_batch: int,
                    out_dir: Path) -> Pass1Out:
    meta = load_probe_meta(spec.ckpt_path)
    model_id = meta["model_id"]
    encoder_with_norm, head, processor = load_encoder_and_head(
        model_id=model_id,
        ckpt_path=str(spec.ckpt_path),
        device=device,
        model_type=spec.model_type,
        last4=spec.last4,
        n_classes=100,
        mae_mask_ratio=0.0 if model_id == "facebook/vit-mae-huge" else None,
    )
    set_extract_config(model_type=spec.model_type, last4=spec.last4)
    # extra defensive override (covers paths where load_encoder_and_head doesn't
    # plumb the flag through, e.g. if someone changed the loader earlier).
    _maybe_disable_mae_masking(encoder_with_norm,
                               0.0 if model_id == "facebook/vit-mae-huge" else None)
    model = FullClassifier(encoder_with_norm, head).to(device).eval()

    N = X_pp.size(0)
    feat_chunks: List[torch.Tensor] = []
    logit_chunks: List[torch.Tensor] = []
    margin_clean_list: List[torch.Tensor] = []
    margin_fgsm_list: List[torch.Tensor] = []
    correct_clean = 0
    correct_fgsm = 0
    x_adv_chunks: List[torch.Tensor] = []

    desc = f"{spec.name} pass1"
    for s in tqdm(range(0, N, batch_size), desc=desc):
        x = X_pp[s:s + batch_size].to(device).float()
        y = Y[s:s + batch_size].to(device)

        # clean forward (with grad off, then features w/o grad)
        with torch.no_grad():
            enc = encoder_with_norm(x)
            f = extract_embed(enc).float()
            logits = head(f)
            feat_chunks.append(f.cpu())
            logit_chunks.append(logits.detach().cpu())
            margin_clean_list.append(margin_per_sample(logits, y).cpu())
            correct_clean += (logits.argmax(1) == y).sum().item()

        # FGSM (needs grad)
        x_adv = fgsm_attack(model, x, y, eps=eps)

        with torch.no_grad():
            enc_a = encoder_with_norm(x_adv)
            f_a = extract_embed(enc_a).float()
            logits_a = head(f_a)
            margin_fgsm_list.append(margin_per_sample(logits_a, y).cpu())
            correct_fgsm += (logits_a.argmax(1) == y).sum().item()

        x_adv_chunks.append(x_adv.cpu())

    feats = torch.cat(feat_chunks, dim=0).numpy()
    logits = torch.cat(logit_chunks, dim=0).numpy()
    margins_clean = torch.cat(margin_clean_list, dim=0).numpy()
    margins_fgsm = torch.cat(margin_fgsm_list, dim=0).numpy()
    x_adv_all = torch.cat(x_adv_chunks, dim=0)

    # Save x_adv to disk (4 sets of [N,3,224,224] easily exceeds GPU VRAM but
    # not RAM; disk is the cleanest hand-off between pass1/pass2 across model
    # loads).
    x_adv_dir = out_dir / "transferability" / "adv_inputs"
    x_adv_dir.mkdir(parents=True, exist_ok=True)
    x_adv_path = x_adv_dir / f"{spec.name}_xadv.pt"
    torch.save(x_adv_all, x_adv_path)

    # ---------- patch LOO importance ----------
    # Use a small subset of images for per-image LOO.
    patch_size = 14
    H, W = X_pp.shape[-2:]
    nh, nw = H // patch_size, W // patch_size

    per_image_drops: List[np.ndarray] = []
    desc_loo = f"{spec.name} patch-LOO"
    for img_idx in tqdm(loo_indices, desc=desc_loo):
        x = X_pp[img_idx].to(device).float()
        y_true = int(Y[img_idx].item())
        with torch.no_grad():
            base_logits = model(x.unsqueeze(0))
            base_logp = F.log_softmax(base_logits, dim=1)[0, y_true].item()

            patched = patch_dropout_batch(x, patch_size=patch_size, fill=0.5).to(device)
            drops = np.zeros(nh * nw, dtype=np.float32)
            for s in range(0, patched.size(0), loo_batch):
                chunk = patched[s:s + loo_batch]
                lp = F.log_softmax(model(chunk), dim=1)[:, y_true]
                drops[s:s + chunk.size(0)] = (base_logp - lp.detach().cpu().numpy())
        per_image_drops.append(drops.reshape(nh, nw))

    per_image_drops_arr = np.stack(per_image_drops, axis=0)  # [n, nh, nw]
    patch_loo_avg = per_image_drops_arr.mean(axis=0)

    # cleanup
    del model, encoder_with_norm, head, processor, x_adv_all
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return Pass1Out(
        model_name=spec.name,
        clean_features=feats,
        clean_logits=logits,
        margins_clean=margins_clean,
        margins_fgsm=margins_fgsm,
        own_clean_acc=100.0 * correct_clean / N,
        own_fgsm_acc=100.0 * correct_fgsm / N,
        x_adv_path=str(x_adv_path),
        patch_loo_avg=patch_loo_avg,
        patch_loo_per_image=per_image_drops_arr,
        patch_grid_size=(nh, nw),
    )


def pass2_cross_eval(spec: ProbeSpec, all_passes: Dict[str, Pass1Out],
                     Y: torch.Tensor, device: torch.device,
                     batch_size: int) -> Dict[str, float]:
    """Returns a row of the transfer matrix for target=spec: maps source
    name -> robust top-1 (%)."""
    meta = load_probe_meta(spec.ckpt_path)
    model_id = meta["model_id"]
    encoder_with_norm, head, processor = load_encoder_and_head(
        model_id=model_id,
        ckpt_path=str(spec.ckpt_path),
        device=device,
        model_type=spec.model_type,
        last4=spec.last4,
        n_classes=100,
        mae_mask_ratio=0.0 if model_id == "facebook/vit-mae-huge" else None,
    )
    set_extract_config(model_type=spec.model_type, last4=spec.last4)
    _maybe_disable_mae_masking(encoder_with_norm,
                               0.0 if model_id == "facebook/vit-mae-huge" else None)
    model = FullClassifier(encoder_with_norm, head).to(device).eval()

    row: Dict[str, float] = {}
    for src_name, src in all_passes.items():
        x_adv_all = torch.load(src.x_adv_path, map_location="cpu", weights_only=False)
        N = x_adv_all.size(0)
        correct = 0
        with torch.no_grad():
            for s in tqdm(range(0, N, batch_size),
                          desc=f"target={spec.name} src={src_name}",
                          leave=False):
                x = x_adv_all[s:s + batch_size].to(device).float()
                y = Y[s:s + batch_size].to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
        row[src_name] = 100.0 * correct / N

    del model, encoder_with_norm, head, processor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ijepa", "mae", "vit", "dino"])
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--num_images", type=int, default=1000,
                    help="Total images for E1 (transferability), E2 (CKA), and E5 (margins).")
    ap.add_argument("--num_loo_images", type=int, default=24,
                    help="How many images get a per-patch LOO importance map (E3).")
    ap.add_argument("--patch_loo_batch", type=int, default=32)
    ap.add_argument("--fgsm_eps", type=float, default=4.0 / 255.0)
    ap.add_argument("--batch_size_pass1", type=int, default=8,
                    help="Forward + backward batch (FGSM step). Conservative because of dinov2-giant.")
    ap.add_argument("--batch_size_pass2", type=int, default=16,
                    help="Forward-only batch for the transfer matrix.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out_dir", type=str, default="analysis_experiments")
    args = ap.parse_args()

    seed_all(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    (out_dir / "transferability").mkdir(parents=True, exist_ok=True)
    (out_dir / "cka").mkdir(parents=True, exist_ok=True)
    (out_dir / "patch_loo").mkdir(parents=True, exist_ok=True)
    (out_dir / "margins").mkdir(parents=True, exist_ok=True)

    # ---------- 1. Build a stratified image batch ----------
    dset = build_dataset(args.split)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(dset))[:min(args.num_images * 2, len(dset))]
    by_class: Dict[int, List[int]] = {}
    for i in indices:
        lbl = int(dset[int(i)]["label"])
        by_class.setdefault(lbl, []).append(int(i))
    chosen: List[int] = []
    while len(chosen) < args.num_images:
        progressed = False
        for k in sorted(by_class):
            if by_class[k]:
                chosen.append(by_class[k].pop())
                progressed = True
                if len(chosen) >= args.num_images:
                    break
        if not progressed:
            break

    pil_items = [dset[int(i)] for i in chosen]
    images_pil = [it["image"].convert("RGB") for it in pil_items]
    labels = torch.tensor([int(it["label"]) for it in pil_items], dtype=torch.long)

    # All backbones in this study accept 224x224 [0,1] inputs (per-model
    # normalisation happens *inside* ModelWithNorm). DINOv2 processors default
    # to shortest_edge=256, so we force target=224 here to make sure the
    # preprocessed tensor is shared correctly by every model.
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    tf = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    X_pp = torch.stack([tf(im) for im in images_pil], dim=0)  # [N,3,224,224] in [0,1]

    print(f"[info] using {X_pp.size(0)} images across "
          f"{len(set(labels.tolist()))} classes; "
          f"FGSM eps={args.fgsm_eps:g}; LOO on {args.num_loo_images} images")

    loo_indices = list(range(args.num_loo_images))  # first N images get LOO

    # ---------- 2. Pass 1 per model ----------
    probe_specs = resolve_probe_specs(args.models)
    pass1: Dict[str, Pass1Out] = {}
    for spec in probe_specs:
        if not spec.ckpt_path.exists():
            raise FileNotFoundError(spec.ckpt_path)
        print(f"\n[pass1] {spec.name}")
        out = pass1_one_model(spec, X_pp, labels, device,
                              eps=args.fgsm_eps,
                              batch_size=args.batch_size_pass1,
                              loo_indices=loo_indices,
                              loo_batch=args.patch_loo_batch,
                              out_dir=out_dir)
        pass1[spec.name] = out
        print(f"  clean_top1={out.own_clean_acc:.2f}  fgsm_top1={out.own_fgsm_acc:.2f}")

    # ---------- 3. CKA matrix ----------
    names = [s.name for s in probe_specs]
    cka_mat = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            cka_mat[i, j] = linear_cka(pass1[ni].clean_features,
                                        pass1[nj].clean_features)
    cka_df = pd.DataFrame(cka_mat, index=names, columns=names)
    cka_df.to_csv(out_dir / "cka" / "cka_matrix.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cka_mat, vmin=0, vmax=1, cmap="viridis")
    for (i, j), v in np.ndenumerate(cka_mat):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v < 0.5 else "black", fontsize=10)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_title("Linear CKA between clean features\n(higher = more similar representation)")
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "cka" / "cka_matrix.png", dpi=160)
    plt.close(fig)
    print("\n[cka]")
    print(cka_df.round(3).to_string())

    # ---------- 4. Margins distributions ----------
    margin_rows = []
    margin_per_model = {}
    for n in names:
        m_clean = pass1[n].margins_clean
        m_fgsm = pass1[n].margins_fgsm
        margin_per_model[n] = {"clean": m_clean, "fgsm": m_fgsm}
        margin_rows.append({
            "model": n,
            "mean_clean_margin": float(np.mean(m_clean)),
            "median_clean_margin": float(np.median(m_clean)),
            "p10_clean_margin": float(np.percentile(m_clean, 10)),
            "p90_clean_margin": float(np.percentile(m_clean, 90)),
            "mean_fgsm_margin": float(np.mean(m_fgsm)),
            "median_fgsm_margin": float(np.median(m_fgsm)),
            "p10_fgsm_margin": float(np.percentile(m_fgsm, 10)),
            "p90_fgsm_margin": float(np.percentile(m_fgsm, 90)),
            "frac_clean_margin_lt_1": float(np.mean(m_clean < 1.0)),
            "frac_fgsm_margin_lt_0": float(np.mean(m_fgsm < 0)),
        })
        pd.DataFrame({"clean_margin": m_clean,
                      "fgsm_margin": m_fgsm}).to_csv(
            out_dir / "margins" / f"{n}_margins.csv", index=False)
    pd.DataFrame(margin_rows).to_csv(out_dir / "margins" / "margins_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for n in names:
        m = pass1[n].margins_clean
        axes[0].plot(np.sort(m), np.linspace(0, 1, len(m)), label=n)
        m2 = pass1[n].margins_fgsm
        axes[1].plot(np.sort(m2), np.linspace(0, 1, len(m2)), label=n)
    for ax, title in zip(axes, ["Clean logit margin (z_y - max_{c!=y} z_c)",
                                 "FGSM logit margin (eps={:.4f})".format(args.fgsm_eps)]):
        ax.set_title(title); ax.set_xlabel("margin"); ax.set_ylabel("CDF")
        ax.axvline(0, color="k", linewidth=0.7)
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "margins" / "margins_cdf.png", dpi=160)
    plt.close(fig)

    # ---------- 5. Patch LOO grids ----------
    # Save per-model average maps and a combined visualisation.
    patch_summary = []
    for n in names:
        avg = pass1[n].patch_loo_avg
        np.save(out_dir / "patch_loo" / f"{n}_avg_importance.npy", avg)
        np.save(out_dir / "patch_loo" / f"{n}_per_image.npy",
                pass1[n].patch_loo_per_image)
        # summary stats: total mass in central 0.5x0.5 box; entropy of map
        H_, W_ = avg.shape
        ch, cw = int(H_ * 0.25), int(W_ * 0.25)
        center_mass = float(np.abs(avg[ch:H_ - ch, cw:W_ - cw]).sum() /
                            max(np.abs(avg).sum(), 1e-12))
        flat = avg.flatten().clip(min=0)
        s = flat.sum()
        if s > 0:
            p = flat / s
            entropy = float(-(p * np.log(p + 1e-12)).sum())
        else:
            entropy = float("nan")
        # how many patches cause a >0.5 nat drop on avg
        n_critical = int((avg > 0.5).sum())
        patch_summary.append({"model": n,
                              "center_mass_loo": center_mass,
                              "entropy_loo": entropy,
                              "n_patches_drop_gt_0p5_nat": n_critical,
                              "max_drop": float(avg.max()),
                              "mean_drop": float(avg.mean())})
    pd.DataFrame(patch_summary).to_csv(out_dir / "patch_loo" / "patch_loo_summary.csv",
                                       index=False)

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    vmax = max(np.abs(pass1[n].patch_loo_avg).max() for n in names)
    for ax, n in zip(axes, names):
        im = ax.imshow(pass1[n].patch_loo_avg, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{n}\nmean log-prob drop")
        ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle(
        f"Patch LOO importance (16x16 grid, gray fill, mean over "
        f"{args.num_loo_images} images)", fontsize=11
    )
    fig.savefig(out_dir / "patch_loo" / "patch_loo_grid.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # ---------- 6. Pass 2: cross-model transferability ----------
    transfer_mat = pd.DataFrame(np.zeros((len(names), len(names))),
                                 index=names, columns=names)
    transfer_mat.index.name = "target"
    transfer_mat.columns.name = "source"
    for spec in probe_specs:
        print(f"\n[pass2] {spec.name}")
        row = pass2_cross_eval(spec, pass1, labels, device,
                                batch_size=args.batch_size_pass2)
        for src, acc in row.items():
            transfer_mat.loc[spec.name, src] = acc

    transfer_mat.to_csv(out_dir / "transferability" / "transfer_matrix.csv")
    print("\n[transferability]  rows=target, cols=source, cells=robust top-1 (%)")
    print(transfer_mat.round(2).to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    arr = transfer_mat.to_numpy()
    im = ax.imshow(arr, vmin=0, vmax=100, cmap="viridis")
    for (i, j), v in np.ndenumerate(arr):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                color="white" if v < 50 else "black", fontsize=10)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("source (where the FGSM was generated)")
    ax.set_ylabel("target (model evaluated)")
    ax.set_title(f"FGSM transfer matrix (eps={args.fgsm_eps:g}) — robust top-1 (%)\n"
                  "diagonal = own-model accuracy under own attack")
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "transferability" / "transfer_matrix.png", dpi=160)
    plt.close(fig)

    # ---------- 7. Summary JSON ----------
    summary = {
        "config": {
            "models": args.models,
            "split": args.split,
            "num_images": args.num_images,
            "num_loo_images": args.num_loo_images,
            "fgsm_eps": args.fgsm_eps,
            "seed": args.seed,
        },
        "own_accuracies": {n: {"clean": pass1[n].own_clean_acc,
                                "fgsm_own": pass1[n].own_fgsm_acc}
                            for n in names},
        "cka_matrix": cka_df.to_dict(orient="index"),
        "transfer_matrix": transfer_mat.to_dict(orient="index"),
        "margin_summary": margin_rows,
        "patch_loo_summary": patch_summary,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # cleanup big adv files? keep them, they're useful for follow-ups.
    print(f"\n[done] all artifacts under {out_dir}/")


if __name__ == "__main__":
    main()
