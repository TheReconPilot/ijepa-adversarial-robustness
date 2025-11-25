#!/usr/bin/env python3
# robust_eval.py — unified evaluation: FGSM, PGD, AutoAttack
# Unified model wrapper (FullClassifier) + FP32 evaluation for exact comparability.

from __future__ import annotations
import argparse, os, sys, json, time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
import pandas as pd
import torchvision.utils as vutils

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

# AutoAttack import (optional runtime dependency)
try:
    from autoattack import AutoAttack
except Exception:
    AutoAttack = None

from transformers import AutoImageProcessor, AutoModel

# local imports (robust_utils provides Normalize, ModelWithNorm, extract_embed, etc.)
sys.path.append(os.path.dirname(__file__))
from robust_utils import (
    seed_all, Normalize, ModelWithNorm, extract_embed, clamp01, set_extract_config,
    parse_float_list, parse_int_list, Timer, save_json, project_linf, project_l2
)

_BICUBIC = InterpolationMode.BICUBIC

# ------------------------
# Data pipeline / loaders
# ------------------------

def build_eval_transform(processor, target: Optional[int] = None):
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std  = getattr(processor, "image_std",  [0.229, 0.224, 0.225])
    size = getattr(processor, "size", None) or getattr(processor, "crop_size", None)
    if isinstance(size, dict):
        target = size.get("shortest_edge", size.get("height", 224))
    elif isinstance(size, int):
        target = size
    else:
        target = target or 224
    tf = transforms.Compose([
        transforms.Resize(256, interpolation=_BICUBIC, antialias=True),
        transforms.CenterCrop(target),
        transforms.ToTensor(),  # -> [0,1]
    ])
    return tf, mean, std


def build_loader(dataset: str, split: str, processor, batch_size: int, workers: int = 8, seed: int = 0) -> DataLoader:
    seed_all(seed)
    tf, _, _ = build_eval_transform(processor)

    if dataset.lower() == "imagenet100":
        if load_dataset is None:
            raise ImportError("Install `datasets` to evaluate ImageNet-100")
        hf_split = "validation" if split == "val" else split
        dset = load_dataset("ilee0022/ImageNet100", split=hf_split)
        def apply_transform(batch):
            batch["pixel_values"] = [tf(img.convert("RGB")) for img in batch["image"]]
            return batch
        dset.set_transform(apply_transform)
        def collate_fn(batch):
            pixel_values = torch.stack([b["pixel_values"] for b in batch])
            labels = torch.tensor([b["label"] for b in batch])
            return pixel_values, labels
        return DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=workers,
                          pin_memory=True, persistent_workers=(workers>0), collate_fn=collate_fn)

    elif dataset.lower() == "cifar100":
        if split == "val":
            full_train_for_val = datasets.CIFAR100(root="./data", train=True, download=True, transform=tf)
            idx = list(range(len(full_train_for_val)))
            val_idx = idx[-5000:]
            val_set = Subset(full_train_for_val, val_idx)
            return DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers,
                              pin_memory=True, persistent_workers=(workers>0))
        else:
            test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=tf)
            return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers,
                              pin_memory=True, persistent_workers=(workers>0))
    else:
        raise ValueError("Unknown dataset: " + dataset)


# ------------------------
# Model loading
# ------------------------

def load_encoder_and_head(model_id: str, ckpt_path: str, device: torch.device,
                          model_type: str, last4: bool, n_classes: int = 100):
    """
    Same contract as original repo: returns (wrapped_model, head, processor)
    where wrapped_model is ModelWithNorm(encoder, Normalize(...))
    """
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    tf, mean, std = build_eval_transform(processor)
    norm = Normalize(mean, std).to(device)

    encoder = AutoModel.from_pretrained(model_id, output_hidden_states=last4).to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    set_extract_config(model_type=model_type, last4=last4)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        enc_out = ModelWithNorm(encoder, norm)(dummy)
        feats = extract_embed(enc_out)
        feat_dim = feats.shape[-1]

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("head", ckpt)

    has_bn = any(k.startswith("bn.") for k in state.keys())

    class LinearHead(nn.Module):
        def __init__(self, in_dim, num_classes, use_bn=False):
            super().__init__()
            self.bn = nn.BatchNorm1d(in_dim) if use_bn else None
            self.fc = nn.Linear(in_dim, num_classes)
        def forward(self, x):
            if self.bn is not None:
                x = self.bn(x)
            return self.fc(x)

    head = LinearHead(feat_dim, n_classes, use_bn=has_bn).to(device)
    head.load_state_dict(state, strict=True)
    head.eval()

    wrapped_model = ModelWithNorm(encoder, norm)
    return wrapped_model, head, processor


# ------------------------
# Unified classifier wrapper
# ------------------------

class FullClassifier(nn.Module):
    """
    Wraps encoder_with_norm (ModelWithNorm) + head -> logits.
    This is the single model attacked by all methods (FGSM/PGD/AA).
    """
    def __init__(self, encoder_with_norm: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder_with_norm = encoder_with_norm
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x assumed in [0,1], float32
        enc_out = self.encoder_with_norm(x)
        feats = extract_embed(enc_out)
        logits = self.head(feats)
        return logits


# ------------------------
# Unified FP32 clean evaluator
# ------------------------

@torch.no_grad()
def eval_clean_fullmodel(full_model: nn.Module, loader, device: torch.device) -> float:
    """
    Evaluate clean top-1 in strict FP32 mode (no AMP / bf16).
    Ensures consistent baseline across FGSM/PGD/AA.
    """
    full_model.eval()
    n, correct = 0, 0
    for imgs, targets in tqdm(loader, desc="Clean Eval (FP32)", leave=False):
        imgs = imgs.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True)
        logits = full_model(imgs)
        pred = logits.argmax(1)
        correct += (pred == targets).sum().item()
        n += targets.numel()
    return 100.0 * correct / max(n, 1)


# ------------------------
# FP32 FGSM (one-step)
# ------------------------

def run_fgsm_fp32(full_model: nn.Module, loader, device: torch.device, eps_list: List[float], out_dir: str, save_k: int = 0):
    rows = []
    full_model.eval()
    for eps in eps_list:
        n, correct = 0, 0
        pbar = tqdm(loader, desc=f"FGSM eps={eps:g}", leave=False)
        t0 = time.time()
        for i, (x, y) in enumerate(pbar):
            x = x.to(device).float()
            y = y.to(device)
            x_adv = x.detach().clone().requires_grad_(True)

            logits = full_model(x_adv)
            loss = F.cross_entropy(logits, y)
            full_model.zero_grad(set_to_none=True)
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            loss.backward()

            grad_sign = x_adv.grad.detach().sign()
            with torch.no_grad():
                x_adv = clamp01(x_adv + eps * grad_sign)

            with torch.no_grad():
                logits_adv = full_model(x_adv)
                pred = logits_adv.argmax(1)
                correct += (pred == y).sum().item()
                n += y.numel()

            # Optionally save first K triplets (global across batches)
            if save_k > 0:
                # Save only the first save_k images overall (use a small global index)
                global_idx = i * loader.batch_size
                # We'll handle per-run saving later (outside loop) to keep ordering simple

        wall = time.time() - t0
        rows.append({
            "attack": "fgsm", "eps": eps, "steps": 1, "restarts": 1,
            "robust_top1": 100.0 * correct / max(n,1),
            "imgs_per_sec": n / max(wall, 1e-6), "wall_clock_sec": wall,
        })
    return rows


# ------------------------
# FP32 PGD (multi-step) - returns rows similar to original run_pgd
# ------------------------

def run_pgd_fp32(full_model: nn.Module, loader, device: torch.device,
                 eps: float, steps_list: List[int], restarts: int,
                 alpha: Optional[float], out_dir: str, save_k: int = 0,
                 norm: str = "linf", seed: int = 0):
    rows = []
    for steps in steps_list:
        n, correct = 0, 0
        pbar = tqdm(loader, desc=f"PGD-{norm} eps={eps:g} steps={steps}", leave=False)
        t0 = time.time()
        g = torch.Generator(device=device).manual_seed(seed)
        # We'll gather adversarial examples per-batch and compute accuracy
        for i, (x, y) in enumerate(pbar):
            x = x.to(device).float()
            y = y.to(device)
            # choose alpha
            if alpha is None:
                alpha_local = (2.0 * eps / steps) if norm == "linf" else (2.5 * eps / steps)
            else:
                alpha_local = alpha

            best_x = None
            best_loss = torch.full((x.size(0),), -float("inf"), device=device)

            for r in range(max(1, restarts)):
                # random init
                if norm == "linf":
                    delta = torch.rand(x.shape, device=device, generator=g) * (2*eps) - eps
                    x_adv = clamp01(x + delta).detach()
                    x_adv = project_linf(x_adv, x, eps)
                else:
                    delta = torch.randn(x.shape, device=device, generator=g)
                    flat = delta.view(delta.size(0), -1)
                    norms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
                    rscale = torch.rand((delta.size(0),1), device=device, generator=g)
                    scaled = (flat / norms) * (eps * rscale)
                    delta = scaled.view_as(delta)
                    x_adv = clamp01(x + delta).detach()
                    x_adv = project_l2(x_adv, x, eps)

                for _step in range(steps):
                    x_adv.requires_grad_(True)
                    logits = full_model(x_adv)
                    loss_per = F.cross_entropy(logits, y, reduction="none")
                    grad = torch.autograd.grad(loss_per.sum(), x_adv, only_inputs=True)[0]
                    gstep = grad.detach()

                    with torch.no_grad():
                        if norm == "linf":
                            x_adv = x_adv + alpha_local * gstep.sign()
                            x_adv = project_linf(x_adv, x, eps)
                        else:
                            flatg = gstep.view(gstep.size(0), -1)
                            gnorm = torch.linalg.vector_norm(flatg, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
                            gunit = (flatg / gnorm).view_as(gstep)
                            x_adv = x_adv + alpha_local * gunit
                            x_adv = project_l2(x_adv, x, eps)
                        x_adv = clamp01(x_adv)
                    x_adv = x_adv.detach()

                # evaluate final loss and select worst restart
                with torch.no_grad():
                    logits_f = full_model(x_adv)
                    final_loss = F.cross_entropy(logits_f, y, reduction="none")
                    if best_x is None:
                        best_x = x_adv.clone()
                        best_loss = final_loss
                    else:
                        pick = final_loss > best_loss
                        best_x = torch.where(pick.view(-1,1,1,1), x_adv, best_x)
                        best_loss = torch.where(pick, final_loss, best_loss)

            # now best_x contains the chosen adversarial examples for this batch
            with torch.no_grad():
                logits_adv = full_model(best_x)
                pred = logits_adv.argmax(1)
                correct += (pred == y).sum().item()
                n += y.numel()
            # (we do not save K here per-batch; saving will be handled after materializing when needed)

        wall = time.time() - t0
        rows.append({
            "attack": f"pgd_{norm}", "eps": eps, "steps": steps, "restarts": restarts,
            "robust_top1": 100.0 * correct / max(n,1),
            "imgs_per_sec": n / max(wall, 1e-6), "wall_clock_sec": wall,
        })
    return rows


# ------------------------
# Helper: collect full split into tensors
# ------------------------

def load_whole_split(loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for x, y in tqdm(loader, desc="Collecting split into memory"):
        xs.append(x)
        ys.append(y)
    x_all = torch.cat(xs, dim=0).to(device).float()
    y_all = torch.cat(ys, dim=0).to(device)
    return x_all, y_all


# ------------------------
# Main CLI
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar100","imagenet100"], required=True)
    ap.add_argument("--model_type", choices=["ijepa","vit"], required=True)
    ap.add_argument("--last4", action="store_true")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--attack", choices=["fgsm","pgd_inf","pgd_l2","autoattack"], required=True)
    ap.add_argument("--eps", required=True, help="float or list: e.g. 8/255 or 1/255,2/255,4/255,8/255")
    ap.add_argument("--steps", default="10,20,50")
    ap.add_argument("--alpha", default="auto", help="'auto' or float")
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", choices=["val","test"], default="val")
    ap.add_argument("--precision", choices=["fp32","amp","bf16"], default="fp32",
                   help="Precision option is kept for CLI compatibility; this unified script uses FP32 for attacks & eval.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_k", type=int, default=0, help="Save first K triplets (x, x_adv, delta)")
    # AutoAttack specific
    ap.add_argument("--aa_norm", choices=["linf","l2"], default="linf", help="Norm for AutoAttack")
    ap.add_argument("--aa_version", default="standard", help="AutoAttack version string")
    ap.add_argument("--aa_attacks", default="apgd-ce,apgd-dlr,fab,square",
                    help="Comma-separated list of AutoAttack sub-attacks to run")
    ap.add_argument("--aa_individual", action="store_true",
                help="Run each AutoAttack sub-attack separately and log per-attack results")
    args = ap.parse_args()

    # We force deterministic-ish behavior
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load encoder + head
    model, head, processor = load_encoder_and_head(
        args.model_id, args.ckpt, device,
        model_type=args.model_type, last4=args.last4,
        n_classes=100,
    )

    # Build loader
    loader = build_loader(args.dataset, args.split, processor,
                          batch_size=args.batch_size, workers=args.workers, seed=args.seed)

    # Build unified full model (FP32)
    set_extract_config(model_type=args.model_type, last4=args.last4)
    full_model = FullClassifier(model, head).to(device)
    full_model.eval()

    # Unified clean eval in FP32
    clean_top1 = eval_clean_fullmodel(full_model, loader, device)
    print(f"[INFO] Clean Top-1 (FP32, full model): {clean_top1:.2f}%")

    eps_list = parse_float_list(args.eps)
    steps_list = parse_int_list(args.steps) if args.attack.startswith("pgd") else [1]
    alpha_val = None if args.alpha == "auto" else float(args.alpha)

    rows: List[Dict] = []
    meta = dict(
        dataset=args.dataset,
        model_type=args.model_type,
        last4=bool(args.last4),
        model_id=args.model_id,
        ckpt=args.ckpt,
        split=args.split,
        precision="fp32",
        seed=args.seed,
        batch_size=args.batch_size,
    )

    if args.attack == "fgsm":
        rows = run_fgsm_fp32(full_model, loader, device, eps_list, args.out_dir, save_k=args.save_k)

    elif args.attack == "pgd_inf":
        assert len(eps_list) == 1, "For PGD-ℓ∞, pass exactly one --eps; use separate runs for others."
        rows = run_pgd_fp32(full_model, loader, device, eps_list[0], steps_list, args.restarts,
                            alpha_val, args.out_dir, save_k=args.save_k, norm="linf", seed=args.seed)

    elif args.attack == "pgd_l2":
        assert len(eps_list) == 1, "For PGD-ℓ2, pass exactly one --eps; use separate runs for others."
        rows = run_pgd_fp32(full_model, loader, device, eps_list[0], steps_list, args.restarts,
                            alpha_val, args.out_dir, save_k=args.save_k, norm="l2", seed=args.seed)

    else:  # autoattack
        if AutoAttack is None:
            raise ImportError("AutoAttack package not installed. Install with: pip install autoattack")

        # Materialize dataset into memory
        x_all, y_all = load_whole_split(loader, device)
        n_samples = x_all.size(0)
        print(f"[INFO] Loaded {n_samples} images for AutoAttack.")

        attacks_to_run = [s.strip() for s in args.aa_attacks.split(",") if s.strip()]

        use_individual = args.aa_individual

        for eps in eps_list:
            print("-------------------------------------------------------")
            print(f"[AutoAttack] norm={args.aa_norm}, eps={eps:g}, version={args.aa_version}")

            norm_str = "Linf" if args.aa_norm == "linf" else "L2"

            adversary = AutoAttack(
                full_model,
                norm=norm_str,
                eps=eps,
                version=args.aa_version,
                device=device,
            )

            adversary.attacks_to_run = attacks_to_run

            # -------------------------------------------------
            # CASE 1: standard AA (single worst-case accuracy)
            # -------------------------------------------------
            if not use_individual:
                with Timer() as t:
                    x_adv = adversary.run_standard_evaluation(x_all, y_all, bs=args.batch_size)

                full_model.eval()
                with torch.no_grad():
                    preds = []
                    for i in range(0, n_samples, args.batch_size):
                        logits = full_model(x_adv[i : i + args.batch_size])
                        preds.append(logits.argmax(1))
                    preds = torch.cat(preds, dim=0)

                robust_top1 = (preds == y_all).float().mean().item() * 100.0
                imgs_per_sec = n_samples / max(t.elapsed, 1e-6)

                print(f"[AutoAttack] eps={eps:g} robust Top-1: {robust_top1:.2f}%")

                row = dict(
                    eps=eps,
                    attack="autoattack",
                    subattack="overall",
                    robust_top1=robust_top1,
                    steps=None,
                    restarts=None,
                    imgs_per_sec=imgs_per_sec,
                    wall_clock_sec=t.elapsed,
                )
                rows.append(row)

            # -------------------------------------------------
            # CASE 2: individual sub-attack results
            # -------------------------------------------------
            else:
                with Timer() as t:
                    dict_adv = adversary.run_standard_evaluation_individual(
                        x_all, y_all, bs=args.batch_size
                    )

                # Evaluate each attack separately
                for attack_name, x_adv in dict_adv.items():
                    full_model.eval()
                    with torch.no_grad():
                        preds = []
                        for i in range(0, n_samples, args.batch_size):
                            logits = full_model(x_adv[i : i + args.batch_size])
                            preds.append(logits.argmax(1))
                        preds = torch.cat(preds, dim=0)

                    robust_top1 = (preds == y_all).float().mean().item() * 100.0

                    print(
                        f"[AutoAttack:{attack_name}] eps={eps:g} robust Top-1: {robust_top1:.2f}%"
                    )

                    row = dict(
                        eps=eps,
                        attack="autoattack",
                        subattack=attack_name,
                        robust_top1=robust_top1,
                        steps=None,
                        restarts=None,
                        imgs_per_sec=n_samples / max(t.elapsed, 1e-6),
                        wall_clock_sec=t.elapsed,
                    )
                    rows.append(row)


            # Save first K triplets (orig, adv, delta) if requested
            if args.save_k > 0:
                save_dir = os.path.join(args.out_dir, f"autoattack_eps={eps:g}")
                os.makedirs(save_dir, exist_ok=True)
                K = min(args.save_k, n_samples)
                for i in range(K):
                    orig = x_all[i].detach().cpu()
                    adv  = x_adv[i].detach().cpu()
                    delta = adv - orig
                    # Save: orig, adv, and scaled delta for visualization
                    vutils.save_image(orig, os.path.join(save_dir, f"img_{i:03d}_orig.png"))
                    vutils.save_image(adv, os.path.join(save_dir, f"img_{i:03d}_adv.png"))
                    # scale delta to make it visible (center at 0.5)
                    vutils.save_image(delta * 10.0 + 0.5, os.path.join(save_dir, f"img_{i:03d}_delta.png"))

    # append meta and write CSV/JSON
    for r in rows:
        r.update(meta)
        r["clean_top1_for_reference"] = clean_top1

    csv_path = os.path.join(args.out_dir, "robust_metrics.csv")
    df = pd.DataFrame(rows)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)

    summary = {"clean_top1": clean_top1, "n_rows": len(rows), "attack": args.attack, "runs": rows}
    save_json(os.path.join(args.out_dir, f"robust_summary_{args.attack}.json"), summary)

    print(f"Done. Clean Top-1: {clean_top1:.2f}%. Wrote: {csv_path}")


if __name__ == "__main__":
    main()
