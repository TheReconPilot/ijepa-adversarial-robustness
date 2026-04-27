#!/usr/bin/env python3
"""
representation_probe.py

Visualization / hypothesis-testing script. Probes *why* different SSL ViT
backbones have different adversarial and occlusion robustness on ImageNet-100.

For each model in {ijepa, mae, vit, dino} it runs, on a fixed test subset:
  1. Clean accuracy.
  2. Input-gradient saliency (|dL/dx|) — saved as grids and summarized by spatial
     entropy and center-mass.
  3. Radial FFT spectra of (a) the saliency map and (b) the FGSM perturbation at
     a fixed epsilon.
  4. Feature-space sensitivity ("Lipschitz proxy"): how far the CLS / pooled
     feature moves under a random-noise step vs an FGSM step of the same L_inf.
  5. Adversarial feature drift: cosine angle between the feature delta under
     FGSM and the direction toward the closest wrong-class centroid (clean
     centroids computed on the same subset). High alignment => "semantic drift".

All numeric results go to a CSV, plus per-model PNG figures + one summary
markdown.

This is deliberately a single self-contained script. It reuses the encoder/head
loader from robust_eval.py for consistency.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from occlusion_eval import (
    DEFAULT_PROBES,
    ProbeSpec,
    build_dataset,
    load_probe_meta,
    pil_collate,
    preprocess_batch,
    resolve_device,
    resolve_probe_specs,
)
from robust_eval import FullClassifier, build_eval_transform, load_encoder_and_head
from robust_utils import seed_all, set_extract_config, clamp01, extract_embed


# ---------- helpers ----------

def spatial_entropy(saliency_2d: torch.Tensor, eps: float = 1e-12) -> float:
    """Shannon entropy of a non-negative saliency map, normalized to sum=1.

    Lower entropy = more concentrated evidence region.
    Higher entropy = more spread-out evidence region.
    """
    flat = saliency_2d.reshape(-1).clamp_min(0)
    s = flat.sum()
    if s <= 0:
        return float("nan")
    p = flat / s
    return float(-(p * (p + eps).log()).sum().item())


def center_mass_fraction(saliency_2d: torch.Tensor, frac: float = 0.5) -> float:
    """Fraction of total saliency mass that falls inside a centered box of
    side = frac * H.

    Higher value = saliency is more object/center focused.
    Lower value = saliency uses periphery/background too.
    """
    H, W = saliency_2d.shape[-2:]
    ch, cw = int(H * (1 - frac) / 2), int(W * (1 - frac) / 2)
    sub = saliency_2d[..., ch:H - ch, cw:W - cw]
    total = saliency_2d.abs().sum().item()
    if total <= 0:
        return float("nan")
    return float(sub.abs().sum().item() / total)


def radial_spectrum(x_2d: torch.Tensor, n_bins: int = 32) -> np.ndarray:
    """Radially averaged log-power spectrum of a 2D map (normalized frequency
    from 0 to 0.5). Returns 1D numpy array of length n_bins.
    """
    x = x_2d.detach().float().cpu()
    H, W = x.shape[-2:]
    fx = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
    power = (fx.abs() ** 2)
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32) - H / 2,
        torch.arange(W, dtype=torch.float32) - W / 2,
        indexing="ij",
    )
    r = torch.sqrt(yy ** 2 + xx ** 2)
    r_max = float(min(H, W) / 2)
    r = (r / max(r_max, 1.0)).clamp(0.0, 1.0)
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    out = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        m = (r >= bins[i]) & (r < bins[i + 1] + (1e-6 if i == n_bins - 1 else 0.0))
        if m.any():
            out[i] = float(power[..., m].mean().item())
    return out


def saliency_from_gradient(grad: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, H, W] via abs().sum over channels, per-image max-normed."""
    s = grad.detach().abs().sum(dim=1)
    m = s.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    return s / m


def l2_per_sample(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).norm(p=2, dim=1)


def linf_per_sample(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).abs().max(dim=1).values


def overlay_saliency(img_01: torch.Tensor, sal_01: torch.Tensor) -> np.ndarray:
    img = img_01.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    sal = sal_01.detach().float().clamp(0, 1).cpu().numpy()
    cmap = plt.get_cmap("inferno")
    heat = cmap(sal)[..., :3]
    return (0.5 * img + 0.5 * heat).clip(0, 1)


# ---------- core probe ----------

@dataclass
class ProbeRun:
    name: str
    model_id: str
    model_type: str
    last4: bool
    # scalar summaries
    clean_top1: float
    fgsm_top1: float
    rnd_top1: float
    mean_saliency_entropy: float
    mean_saliency_center_mass: float
    mean_lipschitz_fgsm: float
    mean_lipschitz_rnd: float
    median_lipschitz_fgsm: float
    median_lipschitz_rnd: float
    mean_cos_drift_wrong: float  # mean cos(Δf, direction to nearest wrong centroid). >0 = "semantic drift"
    mean_cos_drift_true: float   # mean cos(Δf, direction to own centroid). <0 = leaving true class
    frac_drift_toward_wrong: float  # fraction of samples with cos(Δf, wrong_dir) > 0
    mean_feature_ratio: float        # ||Δf_fgsm|| / ||Δf_rnd||


def _reduce_to_1d(feat) -> torch.Tensor:
    if torch.is_tensor(feat):
        return feat
    if hasattr(feat, "last_hidden_state"):
        return feat.last_hidden_state.mean(dim=1)
    raise ValueError("Unexpected encoder output")


def _maybe_disable_mae_masking(encoder_wrapped, force_mask_ratio: float | None) -> str:
    """If encoder is ViTMAEModel (or contains one) and force_mask_ratio is set,
    override the mask ratio. Returns a short status string for logging."""
    if force_mask_ratio is None:
        return "no override"
    # Unwrap ModelWithNorm
    backbone = getattr(encoder_wrapped, "backbone", encoder_wrapped)
    cfg = getattr(backbone, "config", None)
    touched = []
    if cfg is not None and hasattr(cfg, "mask_ratio"):
        cfg.mask_ratio = float(force_mask_ratio)
        touched.append("config.mask_ratio")
    emb = getattr(backbone, "embeddings", None) or getattr(getattr(backbone, "vit", None) or backbone, "embeddings", None)
    if emb is not None and hasattr(emb, "config") and hasattr(emb.config, "mask_ratio"):
        emb.config.mask_ratio = float(force_mask_ratio)
        touched.append("embeddings.config.mask_ratio")
    return f"forced mask_ratio={force_mask_ratio} ({', '.join(touched) or 'no-op'})"


def run_model(
    spec: ProbeSpec,
    images_pil: List[Image.Image],
    labels: torch.Tensor,
    device: torch.device,
    out_dir: Path,
    fgsm_eps: float,
    grad_batch: int,
    sample_vis: int,
    seed: int,
    force_mae_mask_ratio: float | None = None,
) -> Tuple[ProbeRun, np.ndarray, np.ndarray, Dict]:
    """Run every probe for one model. Returns (summary, grad_radial_spectrum,
    pert_radial_spectrum, per_image_dict)."""
    meta = load_probe_meta(spec.ckpt_path)
    model_id = meta["model_id"]

    encoder_with_norm, head, processor = load_encoder_and_head(
        model_id=model_id,
        ckpt_path=str(spec.ckpt_path),
        device=device,
        model_type=spec.model_type,
        last4=spec.last4,
        n_classes=100,
    )
    set_extract_config(model_type=spec.model_type, last4=spec.last4)
    status = _maybe_disable_mae_masking(encoder_with_norm, force_mae_mask_ratio)
    if force_mae_mask_ratio is not None:
        print(f"[mae-mask-override] {spec.name}: {status}")
    model = FullClassifier(encoder_with_norm, head).to(device).eval()

    tf, _, _ = build_eval_transform(processor)
    X = torch.stack([tf(im) for im in images_pil], dim=0).to(device).float()
    Y = labels.to(device)

    N = X.size(0)
    B = grad_batch

    # containers
    all_sal_entropy: List[float] = []
    all_sal_center: List[float] = []
    grad_spectrum_sum = None
    pert_spectrum_sum = None
    feat_fgsm_norm: List[float] = []
    feat_rnd_norm: List[float] = []
    pix_fgsm_norm: List[float] = []
    pix_rnd_norm: List[float] = []
    clean_correct = 0
    fgsm_correct = 0
    rnd_correct = 0

    clean_feats_chunks: List[torch.Tensor] = []
    fgsm_feats_chunks: List[torch.Tensor] = []

    vis_samples = {"img": [], "sal": [], "adv": [], "delta": [], "label": [], "pred": []}

    # Generate per-sample random noise once for reproducibility
    gen = torch.Generator(device=device).manual_seed(seed)
    rnd_noise = (torch.rand(X.shape, device=device, generator=gen) * 2 - 1) * fgsm_eps
    X_rnd = clamp01(X + rnd_noise)

    for start in tqdm(range(0, N, B), desc=f"{spec.name}"):
        x = X[start:start + B].detach().clone().requires_grad_(True)
        y = Y[start:start + B]

        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        grad, = torch.autograd.grad(loss, x)

        # clean predictions
        clean_correct += (logits.argmax(1) == y).sum().item()

        # Saliency per image
        sal = saliency_from_gradient(grad)  # [B,H,W], in [0,1]
        for i in range(sal.size(0)):
            all_sal_entropy.append(spatial_entropy(sal[i]))
            all_sal_center.append(center_mass_fraction(sal[i], frac=0.5))
            sp = radial_spectrum(sal[i].detach(), n_bins=32)
            grad_spectrum_sum = sp if grad_spectrum_sum is None else grad_spectrum_sum + sp

        # FGSM step
        delta = fgsm_eps * grad.detach().sign()
        x_adv = clamp01(x.detach() + delta).detach()

        # radial spectrum of perturbation (signed, mean across channels)
        with torch.no_grad():
            pert = (x_adv - x.detach()).mean(dim=1)  # [B,H,W]
            for i in range(pert.size(0)):
                sp = radial_spectrum(pert[i], n_bins=32)
                pert_spectrum_sum = sp if pert_spectrum_sum is None else pert_spectrum_sum + sp

        # Evaluate adversarial + collect feats
        with torch.no_grad():
            enc_clean = encoder_with_norm(x.detach())
            f_clean = extract_embed(enc_clean)

            enc_adv = encoder_with_norm(x_adv)
            f_adv = extract_embed(enc_adv)

            logits_adv = head(f_adv)
            fgsm_correct += (logits_adv.argmax(1) == y).sum().item()

            # random noise counterpart
            x_rnd = X_rnd[start:start + B]
            enc_rnd = encoder_with_norm(x_rnd)
            f_rnd = extract_embed(enc_rnd)
            logits_rnd = head(f_rnd)
            rnd_correct += (logits_rnd.argmax(1) == y).sum().item()

            # norms
            df_fgsm = (f_adv - f_clean).float()
            df_rnd = (f_rnd - f_clean).float()
            dx_fgsm = (x_adv - x.detach()).float()
            dx_rnd = (x_rnd - x.detach()).float()
            feat_fgsm_norm.extend(df_fgsm.flatten(1).norm(p=2, dim=1).cpu().tolist())
            feat_rnd_norm.extend(df_rnd.flatten(1).norm(p=2, dim=1).cpu().tolist())
            pix_fgsm_norm.extend(dx_fgsm.flatten(1).norm(p=2, dim=1).cpu().tolist())
            pix_rnd_norm.extend(dx_rnd.flatten(1).norm(p=2, dim=1).cpu().tolist())

            clean_feats_chunks.append(f_clean.float().cpu())
            fgsm_feats_chunks.append(f_adv.float().cpu())

        # collect a few samples for visualization
        if len(vis_samples["img"]) < sample_vis:
            take = min(sample_vis - len(vis_samples["img"]), x.size(0))
            vis_samples["img"].append(x.detach()[:take].cpu())
            vis_samples["sal"].append(sal[:take].cpu())
            vis_samples["adv"].append(x_adv[:take].cpu())
            vis_samples["delta"].append((x_adv - x.detach())[:take].cpu())
            vis_samples["label"].append(y[:take].cpu())
            vis_samples["pred"].append(logits_adv.argmax(1)[:take].cpu())

    clean_feats = torch.cat(clean_feats_chunks, dim=0)  # [N, D]
    fgsm_feats = torch.cat(fgsm_feats_chunks, dim=0)   # [N, D]

    # Compute class centroids on clean features only using the images that were
    # classified correctly at clean time (avoid noisy centroids). Fall back to
    # all if too few survive.
    with torch.no_grad():
        y_cpu = Y.cpu()
        # recompute clean preds cheaply via argmax on head output
        head_cpu = head.to("cpu")
        clean_logits = head_cpu(clean_feats)
        clean_pred = clean_logits.argmax(1)
        head.to(device)
        mask_correct = (clean_pred == y_cpu)
        if mask_correct.sum().item() < 20:
            mask_correct = torch.ones_like(y_cpu, dtype=torch.bool)
        # per-class sums and counts so we can do leave-one-out centroids cheaply
        class_sum: Dict[int, torch.Tensor] = {}
        class_cnt: Dict[int, int] = {}
        for i in range(clean_feats.size(0)):
            if not mask_correct[i]:
                continue
            yi = int(y_cpu[i].item())
            class_sum[yi] = clean_feats[i] + class_sum.get(yi, 0)
            class_cnt[yi] = class_cnt.get(yi, 0) + 1
        centroid_ids = [c for c, n in class_cnt.items() if n >= 1]
        centroid_stack = torch.stack([class_sum[c] / class_cnt[c] for c in centroid_ids], dim=0)
        cn = centroid_stack / centroid_stack.norm(dim=1, keepdim=True).clamp_min(1e-12)

        cos_drift_wrong: List[float] = []
        cos_drift_true: List[float] = []
        for i in range(clean_feats.size(0)):
            yi = int(y_cpu[i].item())
            f = clean_feats[i]
            fa = fgsm_feats[i]
            delta = fa - f
            if delta.norm().item() <= 1e-8:
                continue
            delta_n = delta / delta.norm().clamp_min(1e-12)

            # leave-one-out own-class centroid
            if yi in class_cnt and class_cnt[yi] > 1 and mask_correct[i]:
                loo = (class_sum[yi] - clean_feats[i]) / (class_cnt[yi] - 1)
                true_dir = loo - f
                if true_dir.norm() > 1e-8:
                    cos_drift_true.append(
                        float((delta_n @ (true_dir / true_dir.norm().clamp_min(1e-12))).item())
                    )
            elif yi in class_cnt and class_cnt[yi] > 0 and not mask_correct[i]:
                # i was wrong cleanly, so its class centroid doesn't include it already
                true_dir = centroid_stack[centroid_ids.index(yi)] - f
                if true_dir.norm() > 1e-8:
                    cos_drift_true.append(
                        float((delta_n @ (true_dir / true_dir.norm().clamp_min(1e-12))).item())
                    )

            # nearest wrong centroid direction (by cosine from clean feature)
            fn = f / f.norm().clamp_min(1e-12)
            sims = cn @ fn
            wrong_ids = [cid for cid in centroid_ids if cid != yi]
            if wrong_ids:
                idxs = [centroid_ids.index(cid) for cid in wrong_ids]
                sims_w = sims[idxs]
                best_local = int(sims_w.argmax().item())
                best_cid = wrong_ids[best_local]
                wrong_dir = centroid_stack[centroid_ids.index(best_cid)] - f
                if wrong_dir.norm() > 1e-8:
                    cos_drift_wrong.append(
                        float((delta_n @ (wrong_dir / wrong_dir.norm().clamp_min(1e-12))).item())
                    )

    # averaged radial spectra
    grad_spectrum = (grad_spectrum_sum / N).astype(np.float32)
    pert_spectrum = (pert_spectrum_sum / N).astype(np.float32)

    lip_fgsm = np.array(feat_fgsm_norm) / np.maximum(np.array(pix_fgsm_norm), 1e-12)
    lip_rnd = np.array(feat_rnd_norm) / np.maximum(np.array(pix_rnd_norm), 1e-12)
    feat_ratio = np.array(feat_fgsm_norm) / np.maximum(np.array(feat_rnd_norm), 1e-12)

    summary = ProbeRun(
        name=spec.name,
        model_id=model_id,
        model_type=spec.model_type,
        last4=spec.last4,
        clean_top1=100.0 * clean_correct / N,
        fgsm_top1=100.0 * fgsm_correct / N,
        rnd_top1=100.0 * rnd_correct / N,
        mean_saliency_entropy=float(np.nanmean(all_sal_entropy)),
        mean_saliency_center_mass=float(np.nanmean(all_sal_center)),
        mean_lipschitz_fgsm=float(np.nanmean(lip_fgsm)),
        mean_lipschitz_rnd=float(np.nanmean(lip_rnd)),
        median_lipschitz_fgsm=float(np.nanmedian(lip_fgsm)),
        median_lipschitz_rnd=float(np.nanmedian(lip_rnd)),
        mean_cos_drift_wrong=float(np.nanmean(cos_drift_wrong)) if cos_drift_wrong else float("nan"),
        mean_cos_drift_true=float(np.nanmean(cos_drift_true)) if cos_drift_true else float("nan"),
        frac_drift_toward_wrong=float(np.mean([c > 0 for c in cos_drift_wrong])) if cos_drift_wrong else float("nan"),
        mean_feature_ratio=float(np.nanmean(feat_ratio)),
    )

    # Save visualization grids: clean | saliency | adversarial | perturbation
    vis_img = torch.cat(vis_samples["img"], dim=0)[:sample_vis]
    vis_sal = torch.cat(vis_samples["sal"], dim=0)[:sample_vis]
    vis_adv = torch.cat(vis_samples["adv"], dim=0)[:sample_vis]
    vis_delta = torch.cat(vis_samples["delta"], dim=0)[:sample_vis]
    K = vis_img.size(0)
    fig, axes = plt.subplots(K, 4, figsize=(14, 3 * K))
    if K == 1:
        axes = axes.reshape(1, -1)
    for i in range(K):
        axes[i, 0].imshow(vis_img[i].clamp(0, 1).permute(1, 2, 0).numpy())
        axes[i, 0].set_title("clean" if i == 0 else "")
        axes[i, 1].imshow(overlay_saliency(vis_img[i], vis_sal[i]))
        axes[i, 1].set_title("saliency overlay" if i == 0 else "")
        axes[i, 2].imshow(vis_adv[i].clamp(0, 1).permute(1, 2, 0).numpy())
        axes[i, 2].set_title(f"adv (eps={fgsm_eps:.4f})" if i == 0 else "")
        d = vis_delta[i].mean(0).numpy()
        # symmetric colormap scaling for the signed perturbation
        m = float(np.abs(d).max() or 1e-6)
        axes[i, 3].imshow(d, cmap="bwr", vmin=-m, vmax=m)
        axes[i, 3].set_title("pert (mean over C)" if i == 0 else "")
        for a in axes[i]:
            a.axis("off")
    fig.suptitle(f"{spec.name}  ({model_id})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"{spec.name}_grid.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    per_image = {
        "saliency_entropy": all_sal_entropy,
        "saliency_center_mass": all_sal_center,
        "lipschitz_fgsm": lip_fgsm.tolist(),
        "lipschitz_rnd": lip_rnd.tolist(),
        "feature_ratio_fgsm_over_rnd": feat_ratio.tolist(),
        "cos_drift_to_wrong_centroid": cos_drift_wrong,
        "cos_drift_to_true_centroid": cos_drift_true,
    }

    # cleanup
    del model, encoder_with_norm, head, processor, X, X_rnd
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary, grad_spectrum, pert_spectrum, per_image


# ---------- driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ijepa", "mae", "vit", "dino"])
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--num_images", type=int, default=256)
    ap.add_argument("--grad_batch", type=int, default=4,
                    help="Batch size for forward+backward through the encoder.")
    ap.add_argument("--sample_vis", type=int, default=6,
                    help="How many images to save in the saliency grid.")
    ap.add_argument("--fgsm_eps", type=float, default=4.0 / 255.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out_dir", type=str, default="representation_probe")
    ap.add_argument("--force_mae_mask_ratio", type=float, default=None,
                    help="If set (e.g. 0.0), override MAE's random-masking at eval. Use to test whether MAE's adversarial robustness is a masking artifact.")
    args = ap.parse_args()

    seed_all(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # build subset of PIL images (stratified over ~first N classes for diversity)
    dset = build_dataset(args.split)
    # shuffle deterministically then take first num_images, but try to ensure
    # we cover a mix of classes: sort by label cyclically.
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(dset))[:min(args.num_images * 2, len(dset))]
    by_class: Dict[int, List[int]] = {}
    for i in indices:
        lbl = int(dset[int(i)]["label"])
        by_class.setdefault(lbl, []).append(int(i))
    chosen: List[int] = []
    class_keys = sorted(by_class.keys())
    ptr = 0
    while len(chosen) < args.num_images:
        progressed = False
        for k in class_keys:
            if by_class[k]:
                chosen.append(by_class[k].pop())
                progressed = True
                if len(chosen) >= args.num_images:
                    break
        if not progressed:
            break
        ptr += 1

    items = [dset[int(i)] for i in chosen]
    images_pil = [it["image"].convert("RGB") for it in items]
    labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
    print(f"[info] using {len(images_pil)} images across {len(set(labels.tolist()))} classes")

    probe_specs = resolve_probe_specs(args.models)
    runs: List[ProbeRun] = []
    spectra_grad: Dict[str, np.ndarray] = {}
    spectra_pert: Dict[str, np.ndarray] = {}
    all_per_image: Dict[str, Dict] = {}

    for spec in probe_specs:
        if not spec.ckpt_path.exists():
            raise FileNotFoundError(spec.ckpt_path)
        print(f"\n[run] {spec.name} <- {spec.ckpt_path}")
        summary, gs, ps, pi = run_model(
            spec=spec,
            images_pil=images_pil,
            labels=labels,
            device=device,
            out_dir=out_dir,
            fgsm_eps=args.fgsm_eps,
            grad_batch=args.grad_batch,
            sample_vis=args.sample_vis,
            seed=args.seed,
            force_mae_mask_ratio=args.force_mae_mask_ratio,
        )
        runs.append(summary)
        spectra_grad[spec.name] = gs
        spectra_pert[spec.name] = ps
        all_per_image[spec.name] = pi

    # ----- write CSV summary -----
    df = pd.DataFrame([asdict(r) for r in runs])
    df.to_csv(out_dir / "summary.csv", index=False)
    print("\n" + df.to_string(index=False))

    # ----- spectra figures -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    freqs = np.linspace(0.0, 0.5, spectra_grad[list(spectra_grad)[0]].shape[0])
    for name, spec_ in spectra_grad.items():
        axes[0].plot(freqs, np.log10(spec_ + 1e-12), label=name)
    axes[0].set_title("Radial log-power spectrum of |dL/dx| saliency")
    axes[0].set_xlabel("normalized radial frequency")
    axes[0].set_ylabel("log10 power")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for name, spec_ in spectra_pert.items():
        axes[1].plot(freqs, np.log10(spec_ + 1e-12), label=name)
    axes[1].set_title(f"Radial log-power of FGSM perturbation (eps={args.fgsm_eps:.4f})")
    axes[1].set_xlabel("normalized radial frequency")
    axes[1].set_ylabel("log10 power")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "radial_spectra.png", dpi=140)
    plt.close(fig)

    # ----- scalar comparison bar charts -----
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    names = [r.name for r in runs]
    xs = np.arange(len(names))
    axes[0, 0].bar(xs, [r.clean_top1 for r in runs], color="tab:blue")
    axes[0, 0].set_title("clean top-1 (%)")
    axes[0, 1].bar(xs, [r.fgsm_top1 for r in runs], color="tab:red")
    axes[0, 1].set_title(f"FGSM top-1 (%) eps={args.fgsm_eps:.4f}")
    axes[0, 2].bar(xs, [r.mean_saliency_center_mass for r in runs], color="tab:green")
    axes[0, 2].set_title("saliency center-mass (frac=0.5)")
    axes[1, 0].bar(xs, [r.mean_saliency_entropy for r in runs], color="tab:orange")
    axes[1, 0].set_title("saliency spatial entropy (lower=concentrated)")
    axes[1, 1].bar(xs, [r.mean_lipschitz_fgsm for r in runs], color="tab:purple")
    axes[1, 1].set_title("mean ||Δf||/||Δx||  (FGSM)")
    axes[1, 2].bar(xs, [r.mean_feature_ratio for r in runs], color="tab:brown")
    axes[1, 2].set_title("mean ||Δf_fgsm|| / ||Δf_rnd||")
    for ax in axes.ravel():
        ax.set_xticks(xs)
        ax.set_xticklabels(names)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "scalar_comparison.png", dpi=140)
    plt.close(fig)

    # ----- save JSON blob with per-image details -----
    with open(out_dir / "per_image.json", "w") as f:
        json.dump({
            "config": {
                "models": args.models,
                "split": args.split,
                "num_images": args.num_images,
                "fgsm_eps": args.fgsm_eps,
                "seed": args.seed,
            },
            "summary": [asdict(r) for r in runs],
            "radial_spectra_grad": {k: v.tolist() for k, v in spectra_grad.items()},
            "radial_spectra_pert": {k: v.tolist() for k, v in spectra_pert.items()},
            "per_image": all_per_image,
        }, f, indent=2)

    print(f"\n[done] results written under {out_dir}/")


if __name__ == "__main__":
    main()
