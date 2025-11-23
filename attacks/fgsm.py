#!/usr/bin/env python3
# attacks/fgsm.py — FGSM ℓ∞ in pixel space

from __future__ import annotations
import torch
import torch.nn.functional as F

# reuse helpers from robust_utils
from robust_utils import extract_embed, clamp01

def fgsm_linf_step(
    model,                   # ModelWithNorm (wraps HF encoder + Normalize)
    head,                    # nn.Linear
    x: torch.Tensor,         # [B,3,H,W] in [0,1]
    y: torch.Tensor,         # [B]
    eps: float,              # epsilon in pixel space
    precision: str = "amp",  # affects forward only
) -> torch.Tensor:
    """
    One-step untargeted FGSM in ℓ∞.
    Perturbations are applied in pixel space before normalization.
    """
    use_bf16 = (precision == "bf16")
    use_amp = (precision == "amp")
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    x_adv = x.detach().clone().requires_grad_(True)

    # Forward (can use AMP/bf16); grads w.r.t. x_adv are still computed in fp32 internally.
    with torch.cuda.amp.autocast(enabled=(use_amp or use_bf16), dtype=autocast_dtype):
        enc_out = model(x_adv)            # HF output object
        feats   = extract_embed(enc_out)  # -> Tensor [B,D]
        logits  = head(feats)
        loss    = F.cross_entropy(logits, y)

    # Backprop to input
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    if x_adv.grad is not None:
        x_adv.grad.zero_()
    loss.backward()

    # FGSM update in pixel space
    grad_sign = x_adv.grad.detach().sign()
    x_adv = x_adv + eps * grad_sign
    x_adv = clamp01(x_adv)  # stay in [0,1]
    return x_adv.detach()
