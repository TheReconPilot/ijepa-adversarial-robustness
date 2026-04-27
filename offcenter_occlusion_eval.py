#!/usr/bin/env python3
"""offcenter_occlusion_eval.py

E4 from REPRESENTATION_PROBE_REPORT.md Appendix C: stress-test the
"I-JEPA uses the periphery" hypothesis (Finding 2) by sweeping the *center*
of a circular occluder across positions on the image plane and measuring
robust top-1 accuracy for each model at each position.

Hypothesis. If a model's evidence is centered (DINO, ViT, MAE), placing the
occluder at the corner should hurt LESS than placing it at the center, since
the occluder destroys mostly background. If a model's evidence is peripheral
(I-JEPA per Finding 2), the opposite should hold: corner occlusion should
hurt MORE than center occlusion.

We hold the radius fixed at fraction*min(H,W)/2 (default 0.25) and sweep the
*center* across:
    center_frac in {(0.5, 0.5), (0.25, 0.5), (0.5, 0.25), (0.25, 0.25),
                    (0.75, 0.5), (0.5, 0.75), (0.75, 0.75), (0.0, 0.0)}
where (0.5, 0.5) is the image center and (0.0, 0.0) is the top-left corner
(the radius is clamped to the visible region implicitly because we mask only
inside the image plane).

Output (under --out_dir):
  offcenter_by_position.csv  -- one row per (model, position)
  offcenter_summary.json
  offcenter_grid.png         -- one heatmap per model (one cell per position)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from circle_occlusion_eval import circle_mask, apply_occlusion
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
from representation_probe import _maybe_disable_mae_masking
from robust_eval import FullClassifier, build_eval_transform, load_encoder_and_head
from robust_utils import seed_all, set_extract_config


def _circle_mask_centered(h: int, w: int, cy_frac: float, cx_frac: float,
                          radius_px: float, device: torch.device) -> torch.Tensor:
    ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    cy = h * cy_frac
    cx = w * cx_frac
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius_px ** 2)


def evaluate_model(spec: ProbeSpec, dataloader: DataLoader,
                   positions: Sequence[Tuple[float, float]],
                   radius_frac: float, fill_value: float,
                   device: torch.device, max_examples: int | None) -> List[Dict]:
    meta = load_probe_meta(spec.ckpt_path)
    model_id = meta["model_id"]
    encoder_with_norm, head, processor = load_encoder_and_head(
        model_id=model_id, ckpt_path=str(spec.ckpt_path), device=device,
        model_type=spec.model_type, last4=spec.last4, n_classes=100,
        mae_mask_ratio=0.0 if model_id == "facebook/vit-mae-huge" else None,
    )
    set_extract_config(model_type=spec.model_type, last4=spec.last4)
    _maybe_disable_mae_masking(encoder_with_norm,
                               0.0 if model_id == "facebook/vit-mae-huge" else None)
    model = FullClassifier(encoder_with_norm, head).to(device).eval()
    tf, _, _ = build_eval_transform(processor)

    rows = []
    stats = {pos: {"correct": 0, "total": 0} for pos in positions}
    examples_seen = 0
    mask_cache: Dict[Tuple[int, int, float, float, float], torch.Tensor] = {}

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"{spec.name}", leave=False):
            if max_examples is not None and examples_seen >= max_examples:
                break
            if max_examples is not None:
                keep = min(len(images), max_examples - examples_seen)
                images = images[:keep]
                labels = labels[:keep]
                if keep == 0:
                    break
            x = preprocess_batch(images, tf).to(device, non_blocking=True).float()
            y = labels.to(device, non_blocking=True)
            examples_seen += y.numel()
            h, w = x.shape[-2:]
            radius_px = radius_frac * min(h, w)
            for pos in positions:
                key = (h, w, pos[0], pos[1], radius_px)
                if key not in mask_cache:
                    mask_cache[key] = _circle_mask_centered(h, w, pos[0], pos[1],
                                                            radius_px, device)
                x_occ = apply_occlusion(x, mask_cache[key], fill_value)
                preds = model(x_occ).argmax(dim=1)
                stats[pos]["correct"] += int(preds.eq(y).sum().item())
                stats[pos]["total"] += int(y.numel())

    for pos in positions:
        s = stats[pos]
        rows.append({
            "model": spec.name,
            "model_id": model_id,
            "ckpt": str(spec.ckpt_path),
            "cy_frac": pos[0],
            "cx_frac": pos[1],
            "radius_frac": radius_frac,
            "fill_value": fill_value,
            "correct": s["correct"],
            "total": s["total"],
            "top1": 100.0 * s["correct"] / max(s["total"], 1),
        })

    del model, encoder_with_norm, head, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def parse_positions(text: str) -> List[Tuple[float, float]]:
    """Parse a comma-separated list of cy:cx pairs, e.g.
    "0.5:0.5,0.25:0.25,0.0:0.0". Each value is a fraction in [0,1]."""
    out: List[Tuple[float, float]] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        cy_s, cx_s = tok.split(":")
        out.append((float(cy_s), float(cx_s)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Off-center circular occlusion sweep.")
    ap.add_argument("--models", nargs="+", default=["ijepa", "mae", "vit", "dino"])
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--max_examples", type=int, default=1000,
                    help="Cap evaluation to first N images of the split.")
    ap.add_argument("--radius_frac", type=float, default=0.25,
                    help="Circle radius as a fraction of the smaller image dim. "
                         "0.25 covers ~20%% of the area centered.")
    ap.add_argument("--positions", type=str,
                    default="0.5:0.5,0.25:0.5,0.5:0.25,0.75:0.5,0.5:0.75,"
                            "0.25:0.25,0.25:0.75,0.75:0.25,0.75:0.75,"
                            "0.0:0.0,0.0:1.0,1.0:0.0,1.0:1.0",
                    help="Comma-separated cy:cx fractions for the circle center.")
    ap.add_argument("--fill_value", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    seed_all(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positions = parse_positions(args.positions)
    print(f"[info] sweeping {len(positions)} positions at radius_frac={args.radius_frac}")

    dset = build_dataset(args.split)
    dataloader = DataLoader(dset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0,
                            collate_fn=pil_collate)

    probe_specs = resolve_probe_specs(args.models)
    all_rows: List[Dict] = []
    for spec in probe_specs:
        if not spec.ckpt_path.exists():
            raise FileNotFoundError(spec.ckpt_path)
        rows = evaluate_model(
            spec=spec, dataloader=dataloader, positions=positions,
            radius_frac=args.radius_frac, fill_value=args.fill_value,
            device=device, max_examples=args.max_examples,
        )
        all_rows.extend(rows)
        print(pd.DataFrame(rows).to_string(index=False))

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "offcenter_by_position.csv", index=False)

    # ---- summary plot: one heatmap per model ----
    fig, axes = plt.subplots(1, len(args.models), figsize=(4 * len(args.models), 4))
    if len(args.models) == 1:
        axes = [axes]

    for ax, mname in zip(axes, args.models):
        sub = df[df["model"] == mname]
        # Build a 2D grid keyed by unique cy and cx values
        cys = sorted(sub["cy_frac"].unique())
        cxs = sorted(sub["cx_frac"].unique())
        grid = np.full((len(cys), len(cxs)), np.nan, dtype=np.float64)
        for _, r in sub.iterrows():
            i = cys.index(r["cy_frac"])
            j = cxs.index(r["cx_frac"])
            grid[i, j] = r["top1"]
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=100, origin="upper")
        ax.set_title(f"{mname}", fontsize=12)
        ax.set_xticks(range(len(cxs)))
        ax.set_xticklabels([f"{c:.2f}" for c in cxs], rotation=45, fontsize=8)
        ax.set_yticks(range(len(cys)))
        ax.set_yticklabels([f"{c:.2f}" for c in cys], fontsize=8)
        ax.set_xlabel("cx_frac")
        ax.set_ylabel("cy_frac")
        for i in range(len(cys)):
            for j in range(len(cxs)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0f}",
                             ha="center", va="center",
                             color="white" if grid[i, j] < 50 else "black",
                             fontsize=8)
    fig.suptitle(
        f"Off-center circular occlusion (r={args.radius_frac:.2f}*minDim)\n"
        "robust top-1 (%) at each circle-center position",
        fontsize=11,
    )
    plt.colorbar(im, ax=axes, fraction=0.025)
    fig.savefig(out_dir / "offcenter_grid.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "config": {
            "models": args.models,
            "split": args.split,
            "max_examples": args.max_examples,
            "radius_frac": args.radius_frac,
            "fill_value": args.fill_value,
            "positions": positions,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "rows": all_rows,
    }
    with open(out_dir / "offcenter_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] wrote {out_dir}/offcenter_by_position.csv "
          f"and {out_dir}/offcenter_grid.png")


if __name__ == "__main__":
    main()
