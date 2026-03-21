from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor, AutoModel
from transformers.utils import logging as hf_logging

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

from robust_utils import Normalize, ModelWithNorm, extract_embed, set_extract_config

_BICUBIC = InterpolationMode.BICUBIC
_ROOT = Path(__file__).resolve().parent
_DEFAULT_BLACK_CAT = _ROOT / "domain_expansion.png"

hf_logging.set_verbosity_error()


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    model_type: str
    last4: bool
    ckpt_path: Path


DEFAULT_PROBES: List[ProbeSpec] = [
    ProbeSpec(
        name="ijepa_last1",
        model_type="ijepa",
        last4=False,
        ckpt_path=_ROOT / "runs" / "imagenet100" / "ijepa_last1" / "best-val-top1.pt",
    ),
    ProbeSpec(
        name="ijepa_last4",
        model_type="ijepa",
        last4=True,
        ckpt_path=_ROOT / "runs" / "imagenet100" / "ijepa_last4" / "best-val-top1.pt",
    ),
    ProbeSpec(
        name="mae_bn_on",
        model_type="vit",
        last4=False,
        ckpt_path=_ROOT / "runs" / "imagenet100" / "mae_bn_on" / "best-val-top1.pt",
    ),
]


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, use_bn: bool = False):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim) if use_bn else None
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bn is not None:
            x = self.bn(x)
        return self.fc(x)


def build_eval_transform(processor, target: Optional[int] = None):
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
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
        transforms.ToTensor(),
    ])
    return tf, mean, std


def find_default_black_cat() -> Path:
    if _DEFAULT_BLACK_CAT.exists():
        return _DEFAULT_BLACK_CAT

    patterns = ("*black-cat*.jpg", "*black*cat*.jpg", "*cat*.jpg", "*cat*.png")
    for pattern in patterns:
        matches = sorted(_ROOT.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError("Could not locate a black-cat image in the project folder.")


def load_label_texts() -> Dict[int, str]:
    if load_dataset is None:
        return {}

    try:
        dset = load_dataset("ilee0022/ImageNet100", split="validation")
    except Exception:
        return {}

    label_texts: Dict[int, str] = {}
    for row in dset:
        label = row.get("label")
        text = row.get("text")
        if isinstance(label, int) and isinstance(text, str) and label not in label_texts:
            label_texts[label] = text
            if len(label_texts) >= 100:
                break
    return label_texts


def load_probe(spec: ProbeSpec, device: torch.device):
    ckpt = torch.load(spec.ckpt_path, map_location=device)
    meta = ckpt.get("meta", {})
    model_id = meta.get("model_id")
    if not model_id:
        raise ValueError(f"Checkpoint {spec.ckpt_path} is missing model_id metadata.")

    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    _, mean, std = build_eval_transform(processor)
    norm = Normalize(mean, std).to(device)

    encoder = AutoModel.from_pretrained(model_id, output_hidden_states=spec.last4).to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    set_extract_config(model_type=spec.model_type, last4=spec.last4)
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        feats = extract_embed(ModelWithNorm(encoder, norm)(dummy))
        feat_dim = feats.shape[-1]

    state = ckpt.get("head", ckpt)
    num_classes = state["fc.weight"].shape[0]
    has_bn = any(k.startswith("bn.") for k in state.keys())
    head = LinearHead(feat_dim, num_classes, use_bn=has_bn).to(device)
    head.load_state_dict(state, strict=True)
    head.eval()

    return {
        "spec": spec,
        "meta": meta,
        "processor": processor,
        "encoder_with_norm": ModelWithNorm(encoder, norm),
        "head": head,
    }


def preprocess_image(image_path: Path, processor) -> torch.Tensor:
    tf, _, _ = build_eval_transform(processor)
    with Image.open(image_path) as img:
        tensor = tf(img.convert("RGB"))
    return tensor.unsqueeze(0)


def classify_image(
    image_tensor: torch.Tensor,
    encoder_with_norm: nn.Module,
    head: nn.Module,
    model_type: str,
    last4: bool,
    topk: int,
):
    set_extract_config(model_type=model_type, last4=last4)
    with torch.inference_mode():
        enc_out = encoder_with_norm(image_tensor)
        feats = extract_embed(enc_out)
        logits = head(feats)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(k=min(topk, probs.shape[-1]), dim=-1)

    return top_indices[0].tolist(), top_probs[0].tolist()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Quick playground for MAE and I-JEPA ImageNet-100 probes.")
    ap.add_argument("--image", type=Path, default=None, help="Image to classify.")
    ap.add_argument(
        "--models",
        nargs="*",
        default=[spec.name for spec in DEFAULT_PROBES],
        help="Subset of probes to run. Options: ijepa_last1 ijepa_last4 mae_bn_on",
    )
    ap.add_argument("--topk", type=int, default=5, help="How many classes to print per model.")
    ap.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda or cpu.")
    return ap.parse_args()


def main():
    args = parse_args()

    available = {spec.name: spec for spec in DEFAULT_PROBES}
    unknown = [name for name in args.models if name not in available]
    if unknown:
        raise ValueError(f"Unknown probe(s): {unknown}. Available: {sorted(available)}")

    image_path = (args.image if args.image is not None else find_default_black_cat()).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    label_texts = load_label_texts()

    print(f"Image: {image_path}")
    print(f"Device: {device}")
    print()

    for name in args.models:
        spec = available[name]
        if not spec.ckpt_path.exists():
            print(f"[skip] {spec.name}: missing checkpoint at {spec.ckpt_path}")
            print()
            continue

        bundle = load_probe(spec, device=device)
        image_tensor = preprocess_image(image_path, bundle["processor"]).to(device)
        pred_ids, pred_probs = classify_image(
            image_tensor=image_tensor,
            encoder_with_norm=bundle["encoder_with_norm"],
            head=bundle["head"],
            model_type=spec.model_type,
            last4=spec.last4,
            topk=args.topk,
        )

        meta = bundle["meta"]
        print(f"[{spec.name}]")
        print(f"backbone: {meta.get('model_id', 'unknown')}")
        print(f"checkpoint: {spec.ckpt_path}")
        print(f"val_top1: {meta.get('val_top1', 'unknown')}")
        for rank, (class_id, prob) in enumerate(zip(pred_ids, pred_probs), start=1):
            label = label_texts.get(class_id, f"class_{class_id}")
            print(f"{rank}. class_id={class_id:>2}  prob={prob:.4f}  label={label}")
        print()

        del bundle, image_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
