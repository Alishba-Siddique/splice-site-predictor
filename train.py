"""
Train SpliceSiteResNet on the HS3D dataset.

Industry-grade training pipeline:
  • Stratified train/val/test splits (guaranteed class balance in every split)
  • WeightedRandomSampler (balanced mini-batches)
  • Reverse-complement data augmentation
  • Focal loss (class-imbalance handling)
  • AdamW + linear warmup + cosine-annealing LR schedule
  • Automatic Mixed Precision (AMP) — free 2× speedup on GPU
  • Gradient clipping
  • NaN-safe early stopping on validation ROC-AUC
  • Full evaluation: accuracy / precision / recall / F1 / ROC-AUC / PR-AUC

Usage
-----
  python train.py --site_type donor               # real HS3D (downloads ~5 MB)
  python train.py --site_type acceptor
  python train.py --site_type donor --synthetic   # offline fallback
  python train.py --site_type donor --false_parts 1 --epochs 40
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

from model import SpliceSiteResNet
from data import download_hs3d, generate_synthetic_data, make_dataloaders
from utils import compute_metrics, plot_roc, plot_pr, plot_confusion_matrix


# ── Loss ─────────────────────────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,          # lower alpha: sampler already balances classes
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """
    Focal loss with optional label smoothing.

    alpha=0.25 complements WeightedRandomSampler: the sampler handles macro
    imbalance (equal class frequency per batch), focal loss handles micro
    imbalance (hard vs. easy examples within the batch).
    Label smoothing prevents over-confident predictions and improves calibration.
    """
    targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
    bce  = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probs = torch.sigmoid(logits.detach())
    pt    = torch.where(targets >= 0.5, probs, 1 - probs)
    at    = torch.where(
        targets >= 0.5,
        torch.full_like(probs, alpha),
        torch.full_like(probs, 1 - alpha),
    )
    return (at * (1 - pt) ** gamma * bce).mean()


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, scheduler, scaler, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        with autocast(enabled=scaler.is_enabled()):
            loss = focal_loss(model(X).squeeze(-1), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_list, labels_list = [], []
    for X, y in loader:
        with autocast(enabled=device == "cuda"):
            logits_list.append(model(X.to(device)).squeeze(-1).cpu().float())
        labels_list.append(y)
    logits = torch.cat(logits_list).numpy()
    labels = torch.cat(labels_list).numpy().astype(int)
    scores = 1.0 / (1.0 + np.exp(-logits))
    preds  = (scores >= 0.5).astype(int)

    # Guard: ROC-AUC is undefined when only one class is present in the split.
    # This should never happen with stratified splitting, but we guard anyway.
    if len(np.unique(labels)) < 2:
        warnings.warn("Only one class in split — ROC-AUC set to 0.0", stacklevel=2)
        metrics = compute_metrics(labels, preds, scores)
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"]  = 0.0
    else:
        metrics = compute_metrics(labels, preds, scores)

    return metrics, scores, labels


# ── LR schedule: linear warmup + cosine decay ────────────────────────────────

def build_scheduler(optimiser, total_steps: int, warmup_frac: float = 0.05):
    """
    Linear warmup for `warmup_frac` of total steps, then cosine annealing.
    OneCycleLR with pct_start controls the warmup fraction.
    """
    return OneCycleLR(
        optimiser,
        max_lr=optimiser.param_groups[0]["lr"],
        total_steps=total_steps,
        pct_start=warmup_frac,
        anneal_strategy="cos",
        div_factor=10.0,        # start at max_lr / 10
        final_div_factor=1e4,   # end at max_lr / 10000
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SpliceSiteResNet")
    parser.add_argument("--site_type",  choices=["donor", "acceptor"], default="donor")
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--patience",   type=int,   default=8,
                        help="Early-stopping patience on val ROC-AUC")
    parser.add_argument("--data_dir",    default="data/HS3D")
    parser.add_argument("--output_dir",  default="outputs")
    parser.add_argument("--false_parts", type=int, default=None,
                        help="Number of false-site zip files to use (None = all).")
    parser.add_argument("--synthetic",   action="store_true",
                        help="Use synthetic data instead of HS3D")
    parser.add_argument("--no_amp",      action="store_true",
                        help="Disable automatic mixed precision (AMP)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and not args.no_amp
    print(f"\nSpliceSiteResNet | {args.site_type} | device={device} | AMP={use_amp}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = f"splice_model_{args.site_type}.pt"

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.synthetic:
        sequences, labels = generate_synthetic_data(
            args.site_type, n_positive=2000, n_negative=10_000, seed=args.seed
        )
    else:
        try:
            sequences, labels = download_hs3d(
                args.site_type, dest_dir=args.data_dir, false_parts=args.false_parts
            )
        except Exception as exc:
            print(f"\nHS3D download failed: {exc}")
            print("Falling back to synthetic data.\n")
            sequences, labels = generate_synthetic_data(
                args.site_type, n_positive=2000, n_negative=10_000, seed=args.seed
            )

    train_dl, val_dl, test_dl = make_dataloaders(
        sequences, labels,
        batch_size=args.batch_size,
        seed=args.seed,
        augment_train=True,
        weighted_sampling=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SpliceSiteResNet().to(device)
    print(f"Parameters: {model.n_params:,}")

    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                      betas=(0.9, 0.999))
    total_steps = args.epochs * len(train_dl)
    scheduler   = build_scheduler(optimiser, total_steps, warmup_frac=0.05)
    scaler      = GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc   = 0.0
    patience_count = 0
    history        = {"train_loss": [], "val_auc": [], "val_f1": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dl, optimiser, scheduler, scaler, device)
        val_metrics, _, _ = evaluate(model, val_dl, device)
        val_auc = val_metrics["roc_auc"]
        val_f1  = val_metrics["f1"]
        lr_now  = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        history["lr"].append(lr_now)

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"loss={train_loss:.4f}  val_auc={val_auc:.4f}  "
            f"val_f1={val_f1:.4f}  lr={lr_now:.2e}"
        )

        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            patience_count = 0
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Best checkpoint  (val_auc={best_val_auc:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping — no improvement for {args.patience} epochs.")
                break

    # ── Final test evaluation ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_metrics, test_scores, test_labels = evaluate(model, test_dl, device)

    print("\n" + "=" * 52)
    print(f"  Test results  ({args.site_type}  |  {model.n_params:,} params)")
    print("=" * 52)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        print(f"  {k:<12}: {test_metrics[k]:.4f}")
    print("=" * 52)

    # ── Evaluation plots ───────────────────────────────────────────────────────
    o = args.output_dir

    fig_roc, _ = plot_roc(test_labels, test_scores,
                          title=f"ROC — {args.site_type}")
    fig_roc.savefig(f"{o}/roc_{args.site_type}.png", dpi=150, bbox_inches="tight")
    plt.close(fig_roc)

    fig_pr, _ = plot_pr(test_labels, test_scores,
                        title=f"PR — {args.site_type}")
    fig_pr.savefig(f"{o}/pr_{args.site_type}.png", dpi=150, bbox_inches="tight")
    plt.close(fig_pr)

    fig_cm = plot_confusion_matrix(test_metrics["confusion_matrix"])
    fig_cm.savefig(f"{o}/cm_{args.site_type}.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cm)

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["train_loss"], color="steelblue")
    axes[0].set(title="Training Loss (Focal)", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(history["val_auc"], color="darkorange", label="ROC-AUC")
    axes[1].plot(history["val_f1"],  color="seagreen",   label="F1", ls="--")
    axes[1].set(title="Validation Metrics", xlabel="Epoch", ylim=[0, 1])
    axes[1].legend()
    axes[2].plot(history["lr"], color="purple")
    axes[2].set(title="Learning Rate", xlabel="Step", ylabel="LR")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.suptitle(f"{args.site_type.capitalize()} model — SpliceSiteResNet", fontsize=13)
    plt.tight_layout()
    fig.savefig(f"{o}/training_curves_{args.site_type}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots → {o}/")
    print(f"Model → {model_path}")
    print("\nTrain the other model:")
    other = "acceptor" if args.site_type == "donor" else "donor"
    print(f"  python train.py --site_type {other}")
    print("Then launch the app:  python app.py")


if __name__ == "__main__":
    main()
