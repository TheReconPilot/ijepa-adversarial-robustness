#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

from robust_eval import FullClassifier, build_eval_transform, load_encoder_and_head
from robust_utils import seed_all, set_extract_config


_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    ckpt_path: Path
    model_type: str
    last4: bool


DEFAULT_PROBES: Dict[str, ProbeSpec] = {
    "ijepa": ProbeSpec(
        name="ijepa",
        ckpt_path=_ROOT / "runs" / "imagenet100" / "ijepa_last4" / "best-val-top1.pt",
        model_type="ijepa",
        last4=True,
    ),
    "mae": ProbeSpec(
        name="mae",
        ckpt_path=_ROOT / "runs" / "imagenet100" / "mae_bn_on" / "best-val-top1.pt",
        model_type="vit",
        last4=False,
    ),
    "vit": ProbeSpec(
        name="vit",
        ckpt_path=_ROOT / "runs" / "imagenet100" / "google_vit_bn_on" / "best-val-top1.pt",
        model_type="vit",
        last4=False,
    ),
    "dino": ProbeSpec(
        name="dino",
        ckpt_path=_ROOT / "runs" / "imagenet100" / "dino_bn_on" / "best-val-top1.pt",
        model_type="vit",
        last4=False,
    ),
    "vit_large": ProbeSpec(
        name="vit_large",
        ckpt_path=_ROOT / "runs" / "imagenet100" / "vit_large_bn_on" / "best-val-top1.pt",
        model_type="vit",
        last4=False,
    ),
    "dino_large": ProbeSpec(
        name="dino_large",
        ckpt_path=_ROOT / "runs" / "imagenet100" / "dino_large_bn_on" / "best-val-top1.pt",
        model_type="vit",
        last4=False,
    ),
    "moco": ProbeSpec(
        name="moco",
        ckpt_path=_ROOT / "runs" / "imagenet100" / "moco_v3_large_bn_on" / "best-val-top1.pt",
        model_type="moco",
        last4=False,
    ),
}


REGION_NAMES = [
    "north_right",
    "east_upper",
    "east_lower",
    "south_right",
    "south_left",
    "west_lower",
    "west_upper",
    "north_left",
]


def parse_int_list(text: str) -> List[int]:
    values = [int(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list of integers.")
    return values


def parse_region_order(text: str) -> List[int]:
    values = parse_int_list(text)
    if sorted(values) != list(range(8)):
        raise ValueError("progressive_order must contain each region id 0..7 exactly once.")
    return values


def resolve_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg in {"cpu", "cuda"}:
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but no GPU is available.")
        return torch.device(device_arg)
    raise ValueError("device must be one of: auto, cpu, cuda")


def resolve_probe_specs(names: Sequence[str]) -> List[ProbeSpec]:
    probes: List[ProbeSpec] = []
    for name in names:
        key = name.lower()
        if key not in DEFAULT_PROBES:
            raise ValueError(f"Unknown model '{name}'. Available: {sorted(DEFAULT_PROBES)}")
        probes.append(DEFAULT_PROBES[key])
    return probes


def load_probe_meta(ckpt_path: Path) -> Dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    if "model_id" not in meta:
        raise ValueError(f"Checkpoint metadata missing model_id: {ckpt_path}")
    return meta


def pil_collate(batch) -> Tuple[List[Image.Image], torch.Tensor]:
    images = [item["image"].convert("RGB") for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return images, labels


def build_dataset(split: str):
    if load_dataset is None:
        raise ImportError("Install `datasets` to run occlusion evaluation.")
    return load_dataset("ilee0022/ImageNet100", split=split)


def build_combos(
    occlusion_counts: Sequence[int],
    max_combinations_per_k: int,
    seed: int,
) -> Dict[int, List[Tuple[int, ...]]]:
    combos_by_k: Dict[int, List[Tuple[int, ...]]] = {}
    for k in occlusion_counts:
        if not 1 <= k <= 7:
            raise ValueError("Occlusion counts must be between 1 and 7 inclusive.")

        combos = list(itertools.combinations(range(8), k))
        if max_combinations_per_k > 0 and len(combos) > max_combinations_per_k:
            rng = random.Random(seed + 1009 * k)
            combos = sorted(rng.sample(combos, max_combinations_per_k))
        combos_by_k[k] = combos
    return combos_by_k


def build_progressive_combos(
    occlusion_counts: Sequence[int],
    region_order: Sequence[int],
) -> Dict[int, List[Tuple[int, ...]]]:
    combos_by_k: Dict[int, List[Tuple[int, ...]]] = {}
    for k in occlusion_counts:
        if not 1 <= k <= 7:
            raise ValueError("Occlusion counts must be between 1 and 7 inclusive.")
        combos_by_k[k] = [tuple(region_order[:k])]
    return combos_by_k


def region_masks(h: int, w: int, device: torch.device) -> torch.Tensor:
    ys = ((torch.arange(h, device=device, dtype=torch.float32) + 0.5) - (h / 2.0)) / (h / 2.0)
    xs = ((torch.arange(w, device=device, dtype=torch.float32) + 0.5) - (w / 2.0)) / (w / 2.0)
    ny, nx = torch.meshgrid(ys, xs, indexing="ij")

    abs_x = nx.abs()
    abs_y = ny.abs()
    regions = torch.full((h, w), -1, device=device, dtype=torch.long)

    tr = (nx >= 0) & (ny < 0)
    br = (nx >= 0) & (ny >= 0)
    bl = (nx < 0) & (ny >= 0)
    tl = (nx < 0) & (ny < 0)

    regions[tr & (abs_y >= abs_x)] = 0
    regions[tr & (abs_y < abs_x)] = 1
    regions[br & (abs_x > abs_y)] = 2
    regions[br & (abs_x <= abs_y)] = 3
    regions[bl & (abs_y >= abs_x)] = 4
    regions[bl & (abs_y < abs_x)] = 5
    regions[tl & (abs_x > abs_y)] = 6
    regions[tl & (abs_x <= abs_y)] = 7

    if (regions < 0).any():
        raise RuntimeError("Failed to assign some pixels to an occlusion region.")

    return torch.stack([(regions == rid) for rid in range(8)], dim=0)


def combo_mask(mask_bank: torch.Tensor, combo: Sequence[int]) -> torch.Tensor:
    return mask_bank[list(combo)].any(dim=0, keepdim=False)


def apply_occlusion(x: torch.Tensor, mask: torch.Tensor, fill_value: float) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    return x * (1.0 - mask) + fill_value * mask


def chunked(seq: Sequence[Tuple[int, ...]], chunk_size: int) -> Iterable[Sequence[Tuple[int, ...]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    for start in range(0, len(seq), chunk_size):
        yield seq[start:start + chunk_size]


def preprocess_batch(images: Sequence[Image.Image], tf) -> torch.Tensor:
    return torch.stack([tf(img) for img in images], dim=0)


def evaluate_model(
    spec: ProbeSpec,
    split: str,
    dataloader: DataLoader,
    combos_by_k: Dict[int, List[Tuple[int, ...]]],
    device: torch.device,
    allow_cpu_fallback: bool,
    fill_value: float,
    combo_chunk_size: int,
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

    count_stats = {
        k: {"correct": 0, "total": 0, "num_combos": len(combos)}
        for k, combos in combos_by_k.items()
    }
    combo_stats = {
        (k, combo): {"correct": 0, "total": 0}
        for k, combos in combos_by_k.items()
        for combo in combos
    }

    examples_seen = 0
    mask_cache: Dict[Tuple[int, int], torch.Tensor] = {}
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

            hw = (x.shape[-2], x.shape[-1])
            if hw not in mask_cache:
                mask_cache[hw] = region_masks(hw[0], hw[1], device=actual_device)
            mask_bank = mask_cache[hw]

            for k, combos in combos_by_k.items():
                for combo_group in chunked(combos, combo_chunk_size):
                    x_group = torch.cat(
                        [apply_occlusion(x, combo_mask(mask_bank, combo), fill_value) for combo in combo_group],
                        dim=0,
                    )

                    with torch.autocast(
                        device_type=actual_device.type,
                        enabled=(use_amp and actual_device.type == "cuda"),
                        dtype=torch.float16,
                    ):
                        preds = full_model(x_group).argmax(dim=1).view(len(combo_group), y.size(0))
                    correct_per_combo = preds.eq(y.unsqueeze(0)).sum(dim=1).tolist()

                    for combo, n_correct in zip(combo_group, correct_per_combo):
                        combo_stats[(k, combo)]["correct"] += int(n_correct)
                        combo_stats[(k, combo)]["total"] += int(y.size(0))
                        count_stats[k]["correct"] += int(n_correct)
                        count_stats[k]["total"] += int(y.size(0))

            progress.set_postfix({"examples": examples_seen})

    by_count_rows: List[Dict] = []
    for k in sorted(count_stats):
        stats = count_stats[k]
        total = stats["total"]
        by_count_rows.append({
            "model": spec.name,
            "model_id": model_id,
            "ckpt": str(spec.ckpt_path),
            "split": split,
            "device": str(actual_device),
            "occluded_regions": k,
            "num_region_combinations": stats["num_combos"],
            "correct": stats["correct"],
            "total": total,
            "top1": 100.0 * stats["correct"] / max(total, 1),
            "fill_value": fill_value,
        })

    by_combo_rows: List[Dict] = []
    for k in sorted(combos_by_k):
        for combo in combos_by_k[k]:
            stats = combo_stats[(k, combo)]
            total = stats["total"]
            by_combo_rows.append({
                "model": spec.name,
                "model_id": model_id,
                "ckpt": str(spec.ckpt_path),
                "split": split,
                "device": str(actual_device),
                "occluded_regions": k,
                "combo": ",".join(map(str, combo)),
                "combo_region_names": ",".join(REGION_NAMES[idx] for idx in combo),
                "correct": stats["correct"],
                "total": total,
                "top1": 100.0 * stats["correct"] / max(total, 1),
                "fill_value": fill_value,
            })

    return by_count_rows, by_combo_rows


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate ImageNet-100 probes under 8-sector geometric occlusion."
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
        "--occlusion_counts",
        default="1,2,3,4,5,6,7",
        help="Comma-separated counts of regions to occlude. Must be within 1..7.",
    )
    ap.add_argument(
        "--mode",
        choices=["progressive", "combinations"],
        default="progressive",
        help="`progressive` occludes one additional sector at a time along a fixed order; `combinations` averages over many sector combinations.",
    )
    ap.add_argument(
        "--progressive_order",
        default="0,1,2,3,4,5,6,7",
        help="Order of sector ids used in progressive mode.",
    )
    ap.add_argument(
        "--max_combinations_per_k",
        type=int,
        default=0,
        help="If > 0, sample at most this many region combinations for each occlusion count. Used only in combinations mode.",
    )
    ap.add_argument("--fill_value", type=float, default=0.5, help="Pixel value used to cover occluded regions.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--combo_chunk_size", type=int, default=4, help="How many region combinations to score in one forward pass chunk.")
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

    occlusion_counts = sorted(set(parse_int_list(args.occlusion_counts)))
    progressive_order = parse_region_order(args.progressive_order)
    if args.mode == "progressive":
        combos_by_k = build_progressive_combos(
            occlusion_counts=occlusion_counts,
            region_order=progressive_order,
        )
    else:
        combos_by_k = build_combos(
            occlusion_counts=occlusion_counts,
            max_combinations_per_k=args.max_combinations_per_k,
            seed=args.seed,
        )

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
    by_count_rows: List[Dict] = []
    by_combo_rows: List[Dict] = []

    for spec in probe_specs:
        if not spec.ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {spec.name}: {spec.ckpt_path}")

        count_rows, combo_rows = evaluate_model(
            spec=spec,
            split=args.split,
            dataloader=dataloader,
            combos_by_k=combos_by_k,
            device=device,
            allow_cpu_fallback=args.allow_cpu_fallback,
            fill_value=args.fill_value,
            combo_chunk_size=args.combo_chunk_size,
            max_examples=args.max_examples,
            use_amp=args.amp,
        )
        by_count_rows.extend(count_rows)
        by_combo_rows.extend(combo_rows)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(by_count_rows).to_csv(Path(args.out_dir) / "occlusion_by_count.csv", index=False)
    pd.DataFrame(by_combo_rows).to_csv(Path(args.out_dir) / "occlusion_by_combo.csv", index=False)

    summary = {
        "models": list(args.models),
        "split": args.split,
        "occlusion_counts": occlusion_counts,
        "mode": args.mode,
        "progressive_order": progressive_order,
        "max_combinations_per_k": args.max_combinations_per_k,
        "fill_value": args.fill_value,
        "batch_size": args.batch_size,
        "combo_chunk_size": args.combo_chunk_size,
        "workers": args.workers,
        "requested_device": args.device,
        "allow_cpu_fallback": args.allow_cpu_fallback,
        "amp": args.amp,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "total_combinations_scored": int(sum(len(v) for v in combos_by_k.values())),
        "region_names": REGION_NAMES,
        "note": (
            "Occlusion is applied after each model's standard ImageNet preprocessing, "
            "using 8 regions induced by the center horizontal, center vertical, and two diagonals."
        ),
    }
    with open(Path(args.out_dir) / "occlusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(pd.DataFrame(by_count_rows).pivot(index="occluded_regions", columns="model", values="top1"))


if __name__ == "__main__":
    main()
