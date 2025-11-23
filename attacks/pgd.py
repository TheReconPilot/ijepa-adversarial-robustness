#!/usr/bin/env python3
# attacks/pgd.py — PGD ℓ∞ (and optional ℓ2) in pixel space

from __future__ import annotations
import torch
import torch.nn.functional as F

from robust_utils import extract_embed, clamp01

def _project_linf(x_adv, x_orig, eps):
    return torch.clamp(x_adv, min=x_orig - eps, max=x_orig + eps)

def _project_l2(x_adv, x_orig, eps, eps_atol=1e-12):
    # project x_adv to the ℓ2 ball centered at x_orig with radius eps
    diff = x_adv - x_orig
    flat = diff.view(diff.size(0), -1)
    norms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True).clamp_min(eps_atol)
    scale = (eps / norms).clamp(max=1.0)
    proj = (flat * scale).view_as(diff)
    return x_orig + proj

# ---- RNG helpers (backward-compatible across torch versions) ----
def _rand_uniform_like(x, low, high, rng=None):
    r = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=rng) if rng is not None \
        else torch.rand(x.shape, device=x.device, dtype=x.dtype)
    return low + (high - low) * r

def _randn_like(x, rng=None):
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=rng) if rng is not None \
        else torch.randn(x.shape, device=x.device, dtype=x.dtype)

def pgd_attack(
    model,
    head,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    alpha: float | None = None,   # if None, auto: 2*eps/steps for ℓ∞; ~2.5*eps/steps for ℓ2
    restarts: int = 1,
    norm: str = "linf",           # "linf" or "l2"
    precision: str = "amp",       # forward only (updates are fp32)
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Untargeted PGD in pixel space. Returns adversarial examples that maximize
    the per-sample loss across restarts.
    """
    use_bf16 = (precision == "bf16")
    use_amp  = (precision == "amp")
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if norm not in ("linf", "l2"):
        raise ValueError("norm must be 'linf' or 'l2'")

    if alpha is None:
        alpha = (2.0 * eps / steps) if norm == "linf" else (2.5 * eps / steps)

    device = x.device
    best_x = None
    best_loss = torch.full((x.size(0),), -float("inf"), device=device)

    for _ in range(max(1, restarts)):
        # ---------------- Random start inside the epsilon-ball ----------------
        if norm == "linf":
            delta = _rand_uniform_like(x, -eps, eps, rng)
            x_adv = clamp01(x + delta).detach()
            x_adv = _project_linf(x_adv, x, eps)
        else:  # l2
            delta = _randn_like(x, rng)
            flat  = delta.view(delta.size(0), -1)
            norms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
            # sample radius in [0, eps]
            r = torch.rand((delta.size(0), 1), device=device, generator=rng) if rng is not None \
                else torch.rand((delta.size(0), 1), device=device)
            scaled = (flat / norms) * (eps * r)
            delta  = scaled.view_as(delta)
            x_adv  = clamp01(x + delta).detach()
            x_adv  = _project_l2(x_adv, x, eps)

        # --------------------------- PGD iterations ---------------------------
        for _step in range(steps):
            x_adv.requires_grad_(True)
            # Forward may use autocast; keep update math in fp32
            with torch.cuda.amp.autocast(enabled=(use_amp or use_bf16), dtype=autocast_dtype):
                enc_out = model(x_adv)           # HF output object
                feats   = extract_embed(enc_out) # -> [B, D]
                logits  = head(feats)
                loss    = F.cross_entropy(logits, y, reduction="none")  # per-sample

            grad = torch.autograd.grad(loss.sum(), x_adv, only_inputs=True)[0]
            g = grad.detach().float()  # ensure fp32 for the update

            with torch.no_grad():
                if norm == "linf":
                    x_adv = x_adv + alpha * g.sign()
                    x_adv = _project_linf(x_adv, x, eps)
                else:
                    g_flat = g.view(g.size(0), -1)
                    g_norm = torch.linalg.vector_norm(g_flat, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
                    g_unit = (g_flat / g_norm).view_as(g)
                    x_adv  = x_adv + alpha * g_unit
                    x_adv  = _project_l2(x_adv, x, eps)

                x_adv = clamp01(x_adv)

            x_adv = x_adv.detach()  # break graph each step

        # ----------------------- Pick worst restart per sample ----------------
        with torch.no_grad():
            enc_out = model(x_adv)
            feats   = extract_embed(enc_out)
            logits  = head(feats)
            final_loss = F.cross_entropy(logits, y, reduction="none")

            if best_x is None:
                best_x   = x_adv.clone()
                best_loss = final_loss
            else:
                pick = final_loss > best_loss
                best_x    = torch.where(pick.view(-1, 1, 1, 1), x_adv, best_x)
                best_loss = torch.where(pick, final_loss, best_loss)

    return best_x.detach()
