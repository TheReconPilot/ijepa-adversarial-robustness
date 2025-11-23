# Supports ViT (CLS) and I-JEPA (avg-pool) + optional last4 extraction

from __future__ import annotations
import json, os, time
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from tqdm.auto import tqdm
import pandas as pd


@dataclass
class StepMeters:
    loss_sum: float = 0.0
    count: int = 0
    def update(self, loss: float, n: int):
        self.loss_sum += float(loss) * n
        self.count += n
    @property
    def avg(self) -> float:
        return self.loss_sum / max(self.count, 1)


def extract_embed(outputs, model_type: str = "vit", last4: bool = False):
    """
    Unified embedding extractor supporting:
    - ViT (supervised): CLS token or last-4 CLS concat
    - I-JEPA (self-supervised): avg patch features or last-4 avg-pooled concat
    """
    if not hasattr(outputs, "last_hidden_state"):
        raise ValueError("Encoder outputs missing last_hidden_state.")

    x = outputs.last_hidden_state  # [B, tokens, D]

    # If using last-4 hidden states
    if last4:
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise ValueError("output_hidden_states=True required for last4.")
        hs = outputs.hidden_states[-4:]  # last 4 layers

        if model_type == "vit":
            raise ValueError("--last4 only supported for ijepa model type")
            # cls_layers = [h[:, 0] for h in hs]
            # return torch.cat(cls_layers, dim=-1)

        elif model_type == "ijepa":
            pooled = [h.mean(dim=1) for h in hs]
            return torch.cat(pooled, dim=-1)

        else:
            raise ValueError("Unknown model_type: " + str(model_type))

    # Single-layer feature extraction
    if model_type == "vit":
        return x[:, 0]  # CLS

    elif model_type == "ijepa":
        return x.mean(dim=1)  # avg pooled patches

    else:
        raise ValueError("Unknown model_type: " + str(model_type))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
        correct = (pred == targets).sum().item()
        return 100.0 * correct / targets.numel()


def save_checkpoint(path: str, head: nn.Module, meta: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"head": head.state_dict(), "meta": meta}, path)


def train_one_epoch(
    encoder: nn.Module,
    head: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    scheduler,
    device: torch.device,
    epoch_idx: int,
    precision: str = "amp",
    max_steps: Optional[int] = None,
    model_type: str = "vit",
    last4: bool = False,
) -> Dict:

    encoder.eval()  # frozen
    head.train()

    step_m = StepMeters()
    start = time.time()
    total_images = 0
    last_t = start

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train E{epoch_idx}", leave=False)
    steps_done = 0

    for it, (imgs, targets) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        use_bf16 = (precision == "bf16")
        use_amp = (precision == "amp")
        autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

        with torch.cuda.amp.autocast(enabled=(use_amp or use_bf16), dtype=autocast_dtype):
            with torch.no_grad():
                enc_out = encoder(pixel_values=imgs) if "forward" in dir(encoder) else encoder(imgs)

            feats = extract_embed(enc_out, model_type=model_type, last4=last4)
            logits = head(feats)
            loss = cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs = imgs.size(0)
        step_m.update(loss.item(), bs)
        total_images += bs

        now = time.time()
        iter_time = now - last_t
        last_t = now
        img_per_s = bs / max(iter_time, 1e-6)
        lr = max(g["lr"] for g in optimizer.param_groups)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}", "img/s": f"{img_per_s:.1f}"})

        steps_done += 1
        if max_steps is not None and steps_done >= max_steps:
            break

    wall = time.time() - start
    return {
        "train_loss": step_m.avg,
        "images_per_sec": total_images / max(wall, 1e-6),
        "wall_time_sec": wall,
        "lr_max": max(g["lr"] for g in optimizer.param_groups),
    }


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    dataloader,
    device: torch.device,
    split: str = "val",
    precision: str = "amp",
    model_type: str = "vit",
    last4: bool = False,
) -> Dict:

    encoder.eval()
    head.eval()

    step_m = StepMeters()
    correct = 0
    n_total = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Eval:{split}", leave=False)

    use_bf16 = (precision == "bf16")
    use_amp = (precision == "amp")
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    for it, (imgs, targets) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(use_amp or use_bf16), dtype=autocast_dtype):
            enc_out = encoder(pixel_values=imgs) if "forward" in dir(encoder) else encoder(imgs)
            feats = extract_embed(enc_out, model_type=model_type, last4=last4)
            logits = head(feats)
            loss = cross_entropy(logits, targets)

        step_m.update(loss.item(), imgs.size(0))
        pred = torch.argmax(logits, dim=1)
        correct += (pred == targets).sum().item()
        n_total += targets.numel()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    top1 = 100.0 * correct / max(n_total, 1)
    key_loss = "val_loss" if split == "val" else "test_loss"
    key_top1 = "val_top1" if split == "val" else "test_top1"

    return {key_loss: step_m.avg, key_top1: top1}


def append_metrics(run_dir: str, row: Dict):
    os.makedirs(run_dir, exist_ok=True)
    jpath = os.path.join(run_dir, "metrics.jsonl")
    with open(jpath, "a") as f:
        f.write(json.dumps(row) + "\n")

    cpath = os.path.join(run_dir, "metrics.csv")
    df = pd.DataFrame([row])
    if not os.path.exists(cpath):
        df.to_csv(cpath, index=False)
    else:
        df.to_csv(cpath, mode="a", header=False, index=False)
