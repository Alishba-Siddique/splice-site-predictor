"""Preprocessing, inference helpers, and evaluation/plotting utilities."""
from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc,
)

# ── One-hot encoding ──────────────────────────────────────────────────────────

_NT_IDX = {"A": 0, "T": 1, "C": 2, "G": 3}


def one_hot_encode(seq: str) -> np.ndarray:
    """
    Return (4, L) float32 array.  N and unknown bases → all-zero column.
    """
    seq = seq.upper()
    enc = np.zeros((4, len(seq)), dtype=np.float32)
    for i, nt in enumerate(seq):
        idx = _NT_IDX.get(nt)
        if idx is not None:
            enc[idx, i] = 1.0
    return enc


def encode_batch(seqs: list[str]) -> torch.Tensor:
    """Stack one-hot arrays into a (N, 4, L) tensor."""
    return torch.from_numpy(np.stack([one_hot_encode(s) for s in seqs]))


# ── Sliding-window inference ──────────────────────────────────────────────────

def find_dinucleotide_positions(seq: str, dinucleotide: str) -> list[int]:
    seq = seq.upper()
    di = dinucleotide.upper()
    return [i for i in range(len(seq) - 1) if seq[i : i + 2] == di]


def extract_window(seq: str, center: int, half: int = 70) -> str | None:
    """Extract [center-half, center+half) from seq; return None if out of bounds."""
    start, end = center - half, center + half
    if start < 0 or end > len(seq):
        return None
    return seq[start:end]


def predict_on_sequence(
    model,
    sequence: str,
    site_type: str = "donor",
    half_window: int = 70,
    batch_size: int = 256,
    device: str = "cpu",
) -> tuple[list[int], list[float]]:
    """
    Evaluate every GT (donor) or AG (acceptor) position in *sequence*.

    Returns (positions, probabilities), both sorted by position.
    Positions that lack sufficient flanking sequence are skipped.
    """
    dinucleotide = "GT" if site_type == "donor" else "AG"
    candidates = find_dinucleotide_positions(sequence, dinucleotide)

    windows, valid_pos = [], []
    for pos in candidates:
        w = extract_window(sequence, pos, half_window)
        if w is not None:
            windows.append(w)
            valid_pos.append(pos)

    if not windows:
        return [], []

    model.eval()
    all_probs: list[float] = []
    for i in range(0, len(windows), batch_size):
        batch = encode_batch(windows[i : i + batch_size]).to(device)
        with torch.no_grad():
            logits = model(batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
        all_probs.extend(probs)

    return valid_pos, all_probs


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_prediction_track(
    sequence: str,
    positions: list[int],
    probs: list[float],
    site_type: str = "donor",
    threshold: float = 0.5,
) -> plt.Figure:
    """Two-panel figure: probability scatter + dinucleotide position track."""
    dinucleotide = "GT" if site_type == "donor" else "AG"
    colour = "#2196F3" if site_type == "donor" else "#FF5722"
    label = f"{site_type.capitalize()} splice probability"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Panel 1: probability scatter
    if positions:
        ax1.scatter(positions, probs, s=18, alpha=0.7, color=colour, zorder=3, label=label)
        ax1.vlines(positions, 0, probs, alpha=0.12, color=colour, linewidth=0.9)
    ax1.axhline(threshold, color="crimson", linestyle="--", lw=1.5,
                label=f"Threshold = {threshold:.2f}")
    predicted = [p for p, s in zip(positions, probs) if s >= threshold]
    if predicted:
        ax1.vlines(predicted, 0, [probs[positions.index(p)] for p in predicted],
                   color="orange", alpha=0.5, lw=2, label=f"Predicted ({len(predicted)})")
    ax1.set_ylabel("Splice probability", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_title(
        f"{site_type.capitalize()} splice-site predictions  |  sequence length: {len(sequence):,} bp",
        fontsize=12, fontweight="bold",
    )
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: dinucleotide positions
    all_di = find_dinucleotide_positions(sequence, dinucleotide)
    ax2.vlines(all_di, 0, 0.5, color="grey", alpha=0.35, lw=0.9,
               label=f"All {dinucleotide} ({len(all_di)})")
    if predicted:
        ax2.vlines(predicted, 0, 1.0, color=colour, alpha=0.85, lw=1.5,
                   label=f"Predicted ({len(predicted)})")
    ax2.set_xlabel("Position in sequence (bp)", fontsize=11)
    ax2.set_ylabel(f"{dinucleotide} sites", fontsize=11)
    ax2.set_ylim(0, 1.3)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ── Metrics & evaluation plots ────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   auc(fpr, tpr),
        "pr_auc":    auc(rec, prec),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def plot_roc(y_true, y_score, title: str = "ROC Curve") -> tuple[plt.Figure, float]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=title, xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, roc_auc


def plot_pr(y_true, y_score, title: str = "Precision-Recall Curve") -> tuple[plt.Figure, float]:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, color="darkorange", label=f"AUC = {pr_auc:.4f}")
    baseline = y_true.mean()
    ax.axhline(baseline, color="grey", linestyle="--", lw=1, label=f"Baseline = {baseline:.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, pr_auc


def plot_confusion_matrix(cm: np.ndarray, labels=("Non-splice", "Splice")) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], xticklabels=labels,
        yticks=[0, 1], yticklabels=labels,
        xlabel="Predicted", ylabel="True",
        title="Confusion Matrix",
    )
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.tight_layout()
    return fig
