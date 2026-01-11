from __future__ import annotations
import os, json, time, argparse, random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, get_cosine_schedule_with_warmup

from data import get_dataloaders as _get_dls, get_num_classes
from train_eval import train_one_epoch, evaluate, append_metrics, extract_embed, save_checkpoint, count_params

# Ignore certificate verification so any dataset download proceeds on server
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, use_bn: bool = False):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim) if use_bn else None
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        return self.fc(x)


def main():
    ap = argparse.ArgumentParser()
    # Dataset & model
    ap.add_argument("--dataset", type=str, choices=["cifar100", "imagenet100"], required=True,
                    help="Choose dataset.")
    ap.add_argument("--model_id", type=str, required=True, help="HF model id (e.g., google/vit-huge-patch14-224-in21k)")
    ap.add_argument("--model_nickname", type=str, default="google_vit", help="Nickname for the model to use in output directory")
    ap.add_argument("--model_type", type=str, choices=["vit", "ijepa"], default="vit",
                    help="'vit' uses CLS; 'ijepa' uses avg-pooled patches")
    ap.add_argument("--last4", action="store_true", help="Concatenate last-4 layers as features")
    ap.add_argument("--use_bn_head", action="store_true", help="Add BatchNorm before linear head (ViT eval recipe)")

    # Optimization & runtime
    ap.add_argument("--num_classes", type=int, default=-1, help="Override class count; default inferred by dataset")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=-1.0)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", type=str, choices=["amp", "bf16"], default="amp")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Only used for CIFAR-100 stratified split")
    ap.add_argument("--out_dir", type=str, default="runs/")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--checkpoint", type=str, default="best-val-top1.pt")

    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------
    # Build clean run directory name
    # -----------------------------
    # Backbone prefix
    if args.model_type == "vit":
        backbone_prefix = args.model_nickname
    else:
        backbone_prefix = "ijepa"

    # BN flag (ViT only)
    if args.model_type == "vit":
        bn_suffix = "bn_on" if args.use_bn_head else "bn_off"
    else:
        bn_suffix = None

    # last4 flag (I-JEPA only)
    if args.model_type == "ijepa":
        last_suffix = "last4" if args.last4 else "last1"
    else:
        last_suffix = None

    # Construct run name
    parts = [backbone_prefix]
    if bn_suffix:
        parts.append(bn_suffix)
    if last_suffix:
        parts.append(last_suffix)

    # parts.append(f"seed{args.seed}")

    run_name = "_".join(parts)

    # Final run directory:  out_dir/dataset/run_name/
    run_dir = os.path.join(args.out_dir, args.dataset, run_name)

    os.makedirs(run_dir, exist_ok=True)

    # Image processor + encoder
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    encoder = AutoModel.from_pretrained(args.model_id, output_hidden_states=args.last4)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval().to(device)

    # Determine feature dimension with a dummy forward
    with torch.no_grad():
        dummy = torch.zeros(2, 3, 224, 224).to(device)
        out = encoder(pixel_values=dummy)
        feat = extract_embed(out, model_type=args.model_type, last4=args.last4)
        embed_dim = feat.shape[-1]

    # Class count
    num_classes = args.num_classes if args.num_classes > 0 else get_num_classes(args.dataset)

    # Head
    use_bn = args.use_bn_head if args.model_type == "vit" else False
    head = LinearHead(embed_dim, num_classes, use_bn=use_bn).to(device)

    # Dataloaders
    train_loader, val_loader, test_loader = _get_dls(
        args.dataset,
        processor=processor,
        batch_size=args.batch_size,
        workers=args.workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Optimizer & schedule
    base_lr = 1e-3
    lr = base_lr * (args.batch_size / 256.0) if args.lr < 0 else args.lr
    opt = torch.optim.AdamW(head.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    total_steps = args.max_steps
    if total_steps is None:
        if args.epochs is None:
            raise ValueError("Provide --epochs or --max_steps")
        total_steps = args.epochs * len(train_loader)

    warmup_steps = max(args.warmup_steps, int(0.05 * total_steps))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "amp"))

    # Eval-only mode
    if args.eval_only:
        ckpt_path = os.path.join(run_dir, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        head.load_state_dict(ckpt["head"])
        head.eval()

        val_m = evaluate(encoder, head, val_loader, device, split="val", precision=args.precision,
                         model_type=args.model_type, last4=args.last4)
        test_m = evaluate(encoder, head, test_loader, device, split="test", precision=args.precision,
                          model_type=args.model_type, last4=args.last4)
        summary = {
            "backbone": args.model_id,
            "dataset": args.dataset,
            "head_params": count_params(head),
            "total_updates": 0,
            "val_top1": val_m.get("val_top1"),
            "test_top1": test_m.get("test_top1"),
        }
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return

    # Training
    best_val = -1.0
    steps_done = 0
    wall_start = time.time()
    epoch_idx = 0

    while steps_done < total_steps:
        remaining = total_steps - steps_done
        metrics = train_one_epoch(
            encoder, head, train_loader, opt, scaler, sched, device,
            epoch_idx, precision=args.precision, max_steps=remaining,
            model_type=args.model_type, last4=args.last4,
        )
        steps_done += len(train_loader) if remaining >= len(train_loader) else remaining

        val_m = evaluate(encoder, head, val_loader, device, split="val", precision=args.precision,
                         model_type=args.model_type, last4=args.last4)

        row = {
            "epoch": epoch_idx,
            "train_loss": metrics["train_loss"],
            "val_loss": val_m["val_loss"],
            "val_top1": val_m["val_top1"],
            "lr_max": metrics["lr_max"],
            "images_per_sec": metrics["images_per_sec"],
            "gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0,
            "wall_time_sec": metrics["wall_time_sec"],
        }
        append_metrics(run_dir, row)

        save_checkpoint(os.path.join(run_dir, "last.pt"), head,
                        {"embed_dim": embed_dim, "model_id": args.model_id, "epoch": epoch_idx,
                         "dataset": args.dataset})

        if val_m["val_top1"] > best_val:
            best_val = val_m["val_top1"]
            save_checkpoint(os.path.join(run_dir, "best-val-top1.pt"), head,
                            {"embed_dim": embed_dim, "model_id": args.model_id, "epoch": epoch_idx,
                             "dataset": args.dataset, "val_top1": best_val})

        epoch_idx += 1

    # Final eval
    ckpt = torch.load(os.path.join(run_dir, "best-val-top1.pt"), map_location="cpu")
    head.load_state_dict(ckpt["head"])
    head.eval()

    val_final = evaluate(encoder, head, val_loader, device, split="val", precision=args.precision,
                         model_type=args.model_type, last4=args.last4)
    test_final = evaluate(encoder, head, test_loader, device, split="test", precision=args.precision,
                          model_type=args.model_type, last4=args.last4)

    mins = (time.time() - wall_start) / 60.0
    summary = {
        "params": {
            "backbone": args.model_id,
            "dataset": args.dataset,
            "head_params": count_params(head),
            "optimizer": "AdamW",
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "use_bn_head": bool(use_bn),
            "last4": bool(args.last4),
            "model_type": args.model_type,
        },
        "FLOPs_per_img": None,
        "total_updates": total_steps,
        "wall_clock_min": mins,
        "val_top1": val_final["val_top1"],
        "test_top1": test_final["test_top1"],
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
