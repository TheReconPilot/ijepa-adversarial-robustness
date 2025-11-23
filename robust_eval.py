from __future__ import annotations
import argparse, os, sys, json
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
import pandas as pd

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

from transformers import AutoImageProcessor, AutoModel

# local imports
sys.path.append(os.path.dirname(__file__))
from robust_utils import (
    seed_all, Normalize, ModelWithNorm, extract_embed, clamp01, set_extract_config,
    parse_float_list, parse_int_list, top1_accuracy, Timer, save_json
)
from attacks.fgsm import fgsm_linf_step
from attacks.pgd import pgd_attack

_BICUBIC = InterpolationMode.BICUBIC

# ---------- Data pipeline (deterministic eval) ----------

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
    """Unified eval loaders for cifar100 & imagenet100."""
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
            # Use train set with eval transforms as "val" (mirrors training pipeline)
            full_train_for_val = datasets.CIFAR100(root="./data", train=True, download=True, transform=tf)
            # simple fixed split: use last 5k as val for determinism
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


# ---------- Model loading (VI T / I-JEPA, last4 aware) ----------

def load_encoder_and_head(model_id: str, ckpt_path: str, device: torch.device,
                          model_type: str, last4: bool, n_classes: int = 100):

    # processor + transforms
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    tf, mean, std = build_eval_transform(processor)
    norm = Normalize(mean, std).to(device)

    # encoder (frozen)
    encoder = AutoModel.from_pretrained(model_id, output_hidden_states=last4).to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    # configure global extractor
    set_extract_config(model_type=model_type, last4=last4)

    # determine feature dimension
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        enc_out = ModelWithNorm(encoder, norm)(dummy)
        feats = extract_embed(enc_out)
        feat_dim = feats.shape[-1]

    # -----------------------------------------------------------
    # LOAD CHECKPOINT
    # -----------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("head", ckpt)

    # Detect BN head (same logic as training)
    has_bn = any(k.startswith("bn.") for k in state.keys())

    # Build head to match training architecture
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

    # remap keys ("fc.weight", "bn.weight", etc. remain valid)
    head.load_state_dict(state, strict=True)
    head.eval()

    wrapped_model = ModelWithNorm(encoder, norm)
    return wrapped_model, head, processor



# ---------- Evaluation ----------

@torch.no_grad()
def eval_clean(model, head, loader, device, precision="amp") -> float:
    use_bf16 = (precision == "bf16")
    use_amp = (precision == "amp")
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    n, correct = 0, 0
    for imgs, targets in tqdm(loader, desc="Clean Eval", leave=False):
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(use_amp or use_bf16), dtype=autocast_dtype):
            enc_out = model(imgs)
            feats = extract_embed(enc_out)
            logits = head(feats)
        pred = logits.argmax(1)
        correct += (pred == targets).sum().item()
        n += targets.numel()
    return 100.0 * correct / max(n,1)


def run_fgsm(model, head, loader, device, eps_list, precision, out_dir, save_k=0):
    rows = []
    for eps in eps_list:
        n, correct = 0, 0
        pbar = tqdm(loader, desc=f"FGSM eps={eps:g}", leave=False)
        with Timer() as t:
            for i, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)
                x_adv = fgsm_linf_step(model, head, x, y, eps=eps, precision=precision)
                with torch.no_grad():
                    enc_out = model(x_adv)
                    feats = extract_embed(enc_out)
                    logits = head(feats)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                n += y.numel()
        rows.append({
            "attack": "fgsm", "eps": eps, "steps": 1, "restarts": 1,
            "robust_top1": 100.0 * correct / max(n,1),
            "imgs_per_sec": n / max(t.elapsed, 1e-6), "wall_clock_sec": t.elapsed,
        })
    return rows


def run_pgd(model, head, loader, device, eps, steps_list, restarts, alpha, precision, out_dir, save_k=0, norm="linf", seed=0):
    rows = []
    g = torch.Generator(device=device).manual_seed(seed)
    for steps in steps_list:
        n, correct = 0, 0
        pbar = tqdm(loader, desc=f"PGD-{norm} eps={eps:g} steps={steps}", leave=False)
        with Timer() as t:
            for i, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)
                x_adv = pgd_attack(model, head, x, y, eps=eps, steps=steps, alpha=alpha,
                                   restarts=restarts, norm=norm, precision=precision, rng=g)
                with torch.no_grad():
                    enc_out = model(x_adv)
                    feats = extract_embed(enc_out)
                    logits = head(feats)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                n += y.numel()
        rows.append({
            "attack": f"pgd_{norm}", "eps": eps, "steps": steps, "restarts": restarts,
            "robust_top1": 100.0 * correct / max(n,1),
            "imgs_per_sec": n / max(t.elapsed, 1e-6), "wall_clock_sec": t.elapsed,
        })
    return rows


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar100","imagenet100"], required=True)
    ap.add_argument("--model_type", choices=["ijepa","vit"], required=True)
    ap.add_argument("--last4", action="store_true")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--attack", choices=["fgsm","pgd_inf","pgd_l2"], required=True)
    ap.add_argument("--eps", required=True, help="float or list: e.g. 8/255 or 1/255,2/255,4/255,8/255")
    ap.add_argument("--steps", default="10,20,50")
    ap.add_argument("--alpha", default="auto", help="'auto' or float")
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--split", choices=["val","test"], default="val")
    ap.add_argument("--precision", choices=["fp32","amp","bf16"], default="amp")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_k", type=int, default=0, help="Save first K triplets (x, x_adv, delta)")
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder/head and configure extractor
    model, head, processor = load_encoder_and_head(
        args.model_id, args.ckpt, device,
        model_type=args.model_type, last4=args.last4,
        n_classes=100,
    )

    # Build loader for requested dataset
    loader = build_loader(args.dataset, args.split, processor, batch_size=args.batch_size, workers=args.workers, seed=args.seed)

    # Eval
    clean_top1 = eval_clean(model, head, loader, device, precision=args.precision)

    eps_list = parse_float_list(args.eps)
    steps_list = parse_int_list(args.steps) if args.attack.startswith("pgd") else [1]
    alpha = None if args.alpha == "auto" else float(args.alpha)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    meta = dict(
        dataset=args.dataset, model_type=args.model_type, last4=bool(args.last4),
        model_id=args.model_id, ckpt=args.ckpt, split=args.split,
        precision=args.precision, seed=args.seed, batch_size=args.batch_size,
    )

    if args.attack == "fgsm":
        rows = run_fgsm(model, head, loader, device, eps_list, args.precision, args.out_dir, save_k=args.save_k)
    elif args.attack == "pgd_inf":
        assert len(eps_list) == 1, "For PGD-ℓ∞, pass exactly one --eps; use separate runs for others."
        rows = run_pgd(model, head, loader, device, eps_list[0], steps_list, args.restarts,
                       alpha, args.precision, args.out_dir, save_k=args.save_k, norm="linf", seed=args.seed)
    else:  # pgd_l2
        assert len(eps_list) == 1, "For PGD-ℓ2, pass exactly one --eps; use separate runs for others."
        rows = run_pgd(model, head, loader, device, eps_list[0], steps_list, args.restarts,
                       alpha, args.precision, args.out_dir, save_k=args.save_k, norm="l2", seed=args.seed)

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
    save_json(os.path.join(args.out_dir, "robust_summary.json"), summary)

    print(f"Done. Clean Top-1: {clean_top1:.2f}%. Wrote: {csv_path}")


if __name__ == "__main__":
    main()
