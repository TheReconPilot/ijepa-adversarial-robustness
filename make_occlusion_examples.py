#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

from occlusion_eval import region_masks
from circle_occlusion_eval import circle_mask


_ROOT = Path(__file__).resolve().parent
_BICUBIC = InterpolationMode.BICUBIC


def parse_float_list(text: str) -> List[float]:
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
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def parse_int_list(text: str) -> List[int]:
    values = [int(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def build_vis_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize(256, interpolation=_BICUBIC, antialias=True),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


def apply_occlusion(x: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.5) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype).unsqueeze(0)
    return x * (1.0 - mask) + fill_value * mask


def draw_title(img: Image.Image, title: str, band_h: int = 26) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + band_h), color=(245, 245, 245))
    canvas.paste(img, (0, band_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(20, 20, 20))
    return canvas


def make_grid(images: Sequence[Image.Image], cols: int, bg=(255, 255, 255), pad: int = 10) -> Image.Image:
    if not images:
        raise ValueError("Need at least one image to create a grid.")
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), color=bg)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        grid.paste(img, (x, y))
    return grid


def load_sample(split: str, index: int) -> Image.Image:
    if load_dataset is None:
        raise ImportError("Install `datasets` to generate example images.")
    dset = load_dataset("ilee0022/ImageNet100", split=split)
    row = dset[index]
    return row["image"].convert("RGB")


def progressive_sector_images(
    x: torch.Tensor,
    counts: Sequence[int],
    order: Sequence[int],
    fill_value: float,
) -> List[Image.Image]:
    masks = region_masks(x.shape[-2], x.shape[-1], device=torch.device("cpu"))
    images: List[Image.Image] = [draw_title(tensor_to_pil(x), "Original")]
    for count in counts:
        combo = order[:count]
        mask = masks[list(combo)].any(dim=0)
        img = tensor_to_pil(apply_occlusion(x, mask, fill_value=fill_value))
        images.append(draw_title(img, f"Sector k={count}"))
    return images


def circle_images(
    x: torch.Tensor,
    radius_fracs: Iterable[float],
    fill_value: float,
) -> List[Image.Image]:
    h, w = x.shape[-2], x.shape[-1]
    smaller_dim = min(h, w)
    images: List[Image.Image] = [draw_title(tensor_to_pil(x), "Original")]
    for frac in radius_fracs:
        mask = circle_mask(h, w, radius_px=frac * smaller_dim, device=torch.device("cpu"))
        img = tensor_to_pil(apply_occlusion(x, mask, fill_value=fill_value))
        images.append(draw_title(img, f"Circle r={frac:g}"))
    return images


def main():
    ap = argparse.ArgumentParser(description="Generate example images for the occlusion experiments.")
    ap.add_argument("--split", choices=["train", "validation", "test"], default="test")
    ap.add_argument("--index", type=int, default=0, help="Dataset example index.")
    ap.add_argument("--size", type=int, default=224, help="Visualization crop size.")
    ap.add_argument("--fill_value", type=float, default=0.5)
    ap.add_argument("--sector_counts", default="1,2,3,4,5,6,7")
    ap.add_argument("--sector_order", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--circle_radii", default="1/8,2/8,3/8,4/8,5/8,6/8,7/8,8/8")
    ap.add_argument("--out_dir", default=str(_ROOT / "occlusion_visualizations"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_sample(args.split, args.index)
    tf = build_vis_transform(size=args.size)
    x = tf(img)

    original_path = out_dir / "occlusion_example_original.png"
    tensor_to_pil(x).save(original_path)

    sector_grid = make_grid(
        progressive_sector_images(
            x=x,
            counts=parse_int_list(args.sector_counts),
            order=parse_int_list(args.sector_order),
            fill_value=args.fill_value,
        ),
        cols=4,
    )
    sector_path = out_dir / "sector_occlusion_examples.png"
    sector_grid.save(sector_path)

    circle_grid = make_grid(
        circle_images(
            x=x,
            radius_fracs=parse_float_list(args.circle_radii),
            fill_value=args.fill_value,
        ),
        cols=3,
    )
    circle_path = out_dir / "circle_occlusion_examples.png"
    circle_grid.save(circle_path)

    print(f"Saved original image to: {original_path}")
    print(f"Saved sector examples to: {sector_path}")
    print(f"Saved circle examples to: {circle_path}")


if __name__ == "__main__":
    main()
