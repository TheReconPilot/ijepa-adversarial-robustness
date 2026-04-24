#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
from robust_utils import seed_all, set_extract_config


def parse_fraction_list(text: str) -> List[float]:
    values: List[float] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "/" in tok:
            num, den = tok.split("/", 1)
            values.append(float(num) / float(den))
        else:
            values.append(float(tok))
    if not values:
        raise ValueError("Expected a non-empty comma-separated list of radii.")
    if any(v <= 0 for v in values):
        raise ValueError("All radii must be positive.")
    return values


def circle_mask(h: int, w: int, radius_px: float, device: torch.device) -> torch.Tensor:
    ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    cy = h / 2.0
    cx = w / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return dist2 <= (radius_px ** 2)


def apply_occlusion(x: torch.Tensor, mask: torch.Tensor, fill_value: float) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    return x * (1.0 - mask) + fill_value * mask


def evaluate_model(
    spec: ProbeSpec,
    split: str,
    dataloader: DataLoader,
    radius_fracs: Sequence[float],
    device: torch.device,
    allow_cpu_fallback: bool,
    fill_value: float,
    max_examples: int | None,
    use_amp: bool,
) -> Tuple[List[Dict], List[Dict]]:
    meta = load_probe_meta(spec.ckpt_path)
    model_id = meta["model_id"]

    actual_device = device
    try:
        encoder_with_norm, head, processor = load_encoder_and_head(
            model_id=model_id,
            ckpt_path=str(spec.ckpt_path),
            device=actual_device,
            model_type=spec.model_type,
            last4=spec.last4,
            n_classes=100,
        )
    except torch.OutOfMemoryError:
        if device.type != "cuda" or not allow_cpu_fallback:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        actual_device = torch.device("cpu")
        print(f"[warn] CUDA OOM while loading {spec.name}; retrying on CPU.")
        encoder_with_norm, head, processor = load_encoder_and_head(
            model_id=model_id,
            ckpt_path=str(spec.ckpt_path),
            device=actual_device,
            model_type=spec.model_type,
            last4=spec.last4,
            n_classes=100,
        )

    set_extract_config(model_type=spec.model_type, last4=spec.last4)
    full_model = FullClassifier(encoder_with_norm, head).to(actual_device)
    full_model.eval()
    tf, _, _ = build_eval_transform(processor)

    stats_by_radius = {
        frac: {"correct": 0, "total": 0}
        for frac in radius_fracs
    }

    examples_seen = 0
    mask_cache: Dict[Tuple[int, int, float], torch.Tensor] = {}
    progress = tqdm(dataloader, desc=f"{spec.name}:{split}", leave=False)

    with torch.no_grad():
        for images, labels in progress:
            if max_examples is not None and examples_seen >= max_examples:
                break

            if max_examples is not None:
                keep = min(len(images), max_examples - examples_seen)
                images = images[:keep]
                labels = labels[:keep]
                if keep == 0:
                    break

            x = preprocess_batch(images, tf).to(actual_device, non_blocking=True).float()
            y = labels.to(actual_device, non_blocking=True)
            examples_seen += y.numel()

            h, w = x.shape[-2], x.shape[-1]
            smaller_dim = min(h, w)

            for frac in radius_fracs:
                radius_px = frac * smaller_dim
                cache_key = (h, w, radius_px)
                if cache_key not in mask_cache:
                    mask_cache[cache_key] = circle_mask(h, w, radius_px=radius_px, device=actual_device)

                x_occ = apply_occlusion(x, mask_cache[cache_key], fill_value)
                with torch.autocast(
                    device_type=actual_device.type,
                    enabled=(use_amp and actual_device.type == "cuda"),
                    dtype=torch.float16,
                ):
                    preds = full_model(x_occ).argmax(dim=1)

                stats_by_radius[frac]["correct"] += int(preds.eq(y).sum().item())
                stats_by_radius[frac]["total"] += int(y.numel())

            progress.set_postfix({"examples": examples_seen})

    rows_by_radius: List[Dict] = []
    rows_detail: List[Dict] = []
    for frac in radius_fracs:
        stats = stats_by_radius[frac]
        total = stats["total"]
        row = {
            "model": spec.name,
            "model_id": model_id,
            "ckpt": str(spec.ckpt_path),
            "split": split,
            "device": str(actual_device),
            "radius_frac_of_smaller_dim": frac,
            "radius_label": f"{frac:.4f}",
            "correct": stats["correct"],
            "total": total,
            "top1": 100.0 * stats["correct"] / max(total, 1),
            "fill_value": fill_value,
        }
        rows_by_radius.append(row)
        rows_detail.append(row.copy())

    return rows_by_radius, rows_detail


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate ImageNet-100 probes under centered circular occlusion."
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["ijepa", "mae", "vit", "dino"],
        help=f"Subset of models to run. Options: {sorted(DEFAULT_PROBES)}",
    )
    ap.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="ImageNet-100 split to evaluate. Test is smaller than validation in this dataset copy.",
    )
    ap.add_argument(
        "--radius_fracs",
        default="1/8,2/8,3/8,4/8,5/8,6/8,7/8,8/8",
        help="Circle radii as fractions of the smaller input dimension.",
    )
    ap.add_argument("--fill_value", type=float, default=0.5, help="Pixel value used to cover the circle.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_examples", type=int, default=None, help="Optional cap for a smoke test or quick pass.")
    ap.add_argument(
        "--allow_cpu_fallback",
        action="store_true",
        help="If CUDA runs out of memory while loading a model, retry that model on CPU.",
    )
    ap.add_argument("--amp", action="store_true", help="Use CUDA autocast during model inference.")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    seed_all(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    radius_fracs = parse_fraction_list(args.radius_fracs)
    dset = build_dataset(args.split)
    dataloader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=pil_collate,
    )

    probe_specs = resolve_probe_specs(args.models)
    rows_by_radius: List[Dict] = []
    rows_detail: List[Dict] = []

    for spec in probe_specs:
        if not spec.ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {spec.name}: {spec.ckpt_path}")

        model_rows, detail_rows = evaluate_model(
            spec=spec,
            split=args.split,
            dataloader=dataloader,
            radius_fracs=radius_fracs,
            device=device,
            allow_cpu_fallback=args.allow_cpu_fallback,
            fill_value=args.fill_value,
            max_examples=args.max_examples,
            use_amp=args.amp,
        )
        rows_by_radius.extend(model_rows)
        rows_detail.extend(detail_rows)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(rows_by_radius).to_csv(Path(args.out_dir) / "circle_occlusion_by_radius.csv", index=False)
    pd.DataFrame(rows_detail).to_csv(Path(args.out_dir) / "circle_occlusion_details.csv", index=False)

    summary = {
        "models": list(args.models),
        "split": args.split,
        "radius_fracs": radius_fracs,
        "fill_value": args.fill_value,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "requested_device": args.device,
        "allow_cpu_fallback": args.allow_cpu_fallback,
        "amp": args.amp,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "note": (
            "Occlusion is applied after each model's standard ImageNet preprocessing. "
            "Each circle is centered in the model input, and the radius is measured as a "
            "fraction of the smaller input dimension."
        ),
    }
    with open(Path(args.out_dir) / "circle_occlusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pivot = pd.DataFrame(rows_by_radius).pivot(
        index="radius_frac_of_smaller_dim", columns="model", values="top1"
    )
    print(pivot)


if __name__ == "__main__":
    main()
