from __future__ import annotations
import json, math, os, random, time
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

import torch
import torch.nn as nn

# ---- Parsing helpers ----

def _parse_one_float(tok: str) -> float:
    tok = tok.strip()
    if "/" in tok:
        num, den = tok.split("/", 1)
        return float(num) / float(den)
    return float(tok)


def parse_float_list(s: str) -> List[float]:
    return [_parse_one_float(t) for t in s.split(",") if t.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(t.strip()) for t in s.split(",") if t.strip()]

# ---- RNG / determinism ----

def seed_all(seed: int = 0):
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ---- Normalization wrapper (mean/std in [0,1] space) ----

class Normalize(nn.Module):
    def __init__(self, mean: Iterable[float], std: Iterable[float]):
        super().__init__()
        mean = torch.tensor(list(mean)).view(1, -1, 1, 1)
        std = torch.tensor(list(std)).view(1, -1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ModelWithNorm(nn.Module):
    """Wrap an encoder so inputs x are expected in [0,1] and normalized inside."""
    def __init__(self, backbone: nn.Module, norm: Normalize):
        super().__init__()
        self.backbone = backbone
        self.norm = norm

    def forward(self, x: torch.Tensor):
        x_n = self.norm(x)
        try:
            out = self.backbone(pixel_values=x_n)
        except TypeError:
            out = self.backbone(x_n)
        return out

# ---- Feature extraction from HF outputs (configurable) ----

# Global extraction config set by robust_eval main()
_EXTRACT_CFG = {"model_type": "vit", "last4": False}


def set_extract_config(model_type: str = "vit", last4: bool = False):
    model_type = model_type.lower()
    if model_type not in {"vit", "ijepa"}:
        raise ValueError("model_type must be 'vit' or 'ijepa'")
    _EXTRACT_CFG["model_type"] = model_type
    _EXTRACT_CFG["last4"] = bool(last4)


def extract_embed(outputs) -> torch.Tensor:
    """
    Config-aware extractor matching training pipeline:
      - ViT: CLS (or last-4 CLS concat if last4=True)
      - I-JEPA: avg-pooled patches (or last-4 avg-pooled concat if last4=True)
    """
    mt = _EXTRACT_CFG["model_type"]
    last4 = _EXTRACT_CFG["last4"]

    if not hasattr(outputs, "last_hidden_state"):
        raise ValueError("Encoder outputs missing last_hidden_state")

    x = outputs.last_hidden_state  # [B, tokens, D]

    if last4:
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise ValueError("output_hidden_states=True required for last4 extraction")
        hs = outputs.hidden_states[-4:]
        if mt == "vit":
            cls4 = [h[:, 0] for h in hs]
            return torch.cat(cls4, dim=-1)
        else:  # ijepa
            pooled = [h.mean(dim=1) for h in hs]
            return torch.cat(pooled, dim=-1)

    # Single-layer
    if mt == "vit":
        return x[:, 0]
    else:  # ijepa
        return x.mean(dim=1)

# ---- Projections & clipping (pixel space [0,1]) ----

def clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def project_linf(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(x_adv, x - eps, x + eps)


def project_l2(x_adv: torch.Tensor, x: torch.Tensor, eps: float, eps_atol: float = 1e-12) -> torch.Tensor:
    diff = (x_adv - x).flatten(1)
    nrm = torch.norm(diff, p=2, dim=1, keepdim=True).clamp(min=eps_atol)
    scale = torch.clamp(eps / nrm, max=1.0)
    diff = (diff * scale).view_as(x_adv)
    return x + diff

# ---- Metrics ----

@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100.0

# ---- Timing ----

@dataclass
class Timer:
    start_t: float = None
    def __enter__(self): self.start_t = time.time(); return self
    def __exit__(self, exc_type, exc, tb): self.elapsed = time.time() - self.start_t

# ---- I/O helpers ----

def save_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
