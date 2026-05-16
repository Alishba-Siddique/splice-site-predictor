"""
HS3D dataset download, parsing, and DataLoader construction.

Downloads from the Altervista mirror of the HS3D dataset:
  https://salvatorerampone.altervista.org/wp-content/HS3D/

Each component is a separate ZIP file containing a .seq file in HS3D format:
  ACCESSION  (n,n,n): <140-nt sequence>

If the download fails for any reason, a synthetic dataset is generated so
training can still run and produce a deployable (if weaker) model.
"""
from __future__ import annotations

import io
import random
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils import one_hot_encode

# ── Mirror config ─────────────────────────────────────────────────────────────

_BASE = "https://salvatorerampone.altervista.org/wp-content/HS3D/"

# True sites: one zip each. False sites: split across multiple zips.
_ZIPS = {
    "donor": {
        "true":  ["EI_true.zip"],
        "false": ["EI_false_1.zip", "EI_false_2.zip", "EI_false_3.zip"],
    },
    "acceptor": {
        "true":  ["IE_true.zip"],
        "false": ["IE_false_1.zip", "IE_false_2.zip", "IE_false_3.zip", "IE_false_4.zip"],
    },
}

# ── SSL context (mirror cert is self-signed) ──────────────────────────────────

def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ── Download + parse ──────────────────────────────────────────────────────────

def _fetch_zip(url: str) -> bytes:
    """Download a URL to memory, bypassing SSL verification."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=60) as r:
        return r.read()


def _parse_hs3d_seq(content: str) -> list[str]:
    """
    Parse HS3D .seq file content.

    Lines look like:
        AB000381  ( 1,2,2): CTCCTCTTTGCC...   (140 nt after '): ')
    Header/info lines (Date/Time, descriptions) are ignored.
    """
    seqs: list[str] = []
    for line in content.splitlines():
        if "): " not in line:
            continue
        seq = line.split("): ", 1)[1].strip().upper()
        if len(seq) == 140 and set(seq).issubset(set("ACGTN")):
            seqs.append(seq)
    return seqs


def _load_zip_sequences(zip_name: str) -> list[str]:
    """Download one HS3D zip and return all 140-nt sequences inside it."""
    url = _BASE + zip_name
    data = _fetch_zip(url)
    seqs: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            # Read every .seq or .seq.NNN file; skip .inf and .BAT
            lower = member.lower()
            if ".inf" in lower or ".bat" in lower:
                continue
            content = zf.read(member).decode("utf-8", errors="ignore")
            seqs.extend(_parse_hs3d_seq(content))
    return seqs


def download_hs3d(
    site_type: str,
    dest_dir: str | Path = "data/HS3D",
    false_parts: int | None = None,
) -> tuple[list[str], list[int]]:
    """
    Download and parse the HS3D dataset for one site type.

    Args:
        site_type:   'donor' or 'acceptor'.
        dest_dir:    Cache directory for downloaded zips.
        false_parts: How many false-site zip files to download (None = all).
                     Use 1 for a fast run; use None for the full dataset.

    Returns:
        (sequences, labels) — parallel lists, label 1 = true splice site.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    cfg        = _ZIPS[site_type]
    true_zips  = cfg["true"]
    false_zips = cfg["false"] if false_parts is None else cfg["false"][:false_parts]

    true_seqs: list[str] = []
    for zname in true_zips:
        cache = dest / zname
        if cache.exists():
            with open(cache, "rb") as f:
                raw = f.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for m in zf.namelist():
                    if ".inf" not in m.lower() and ".bat" not in m.lower():
                        true_seqs.extend(_parse_hs3d_seq(
                            zf.read(m).decode("utf-8", errors="ignore")
                        ))
        else:
            print(f"  Downloading {zname} …", end=" ", flush=True)
            raw = _fetch_zip(_BASE + zname)
            cache.write_bytes(raw)
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for m in zf.namelist():
                    if ".inf" not in m.lower() and ".bat" not in m.lower():
                        true_seqs.extend(_parse_hs3d_seq(
                            zf.read(m).decode("utf-8", errors="ignore")
                        ))
            print(f"{len(true_seqs):,} seqs so far")

    false_seqs: list[str] = []
    for zname in false_zips:
        cache = dest / zname
        before = len(false_seqs)
        if cache.exists():
            with open(cache, "rb") as f:
                raw = f.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for m in zf.namelist():
                    if ".inf" not in m.lower() and ".bat" not in m.lower():
                        false_seqs.extend(_parse_hs3d_seq(
                            zf.read(m).decode("utf-8", errors="ignore")
                        ))
        else:
            print(f"  Downloading {zname} …", end=" ", flush=True)
            raw = _fetch_zip(_BASE + zname)
            cache.write_bytes(raw)
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for m in zf.namelist():
                    if ".inf" not in m.lower() and ".bat" not in m.lower():
                        false_seqs.extend(_parse_hs3d_seq(
                            zf.read(m).decode("utf-8", errors="ignore")
                        ))
            print(f"+{len(false_seqs)-before:,}")

    sequences = true_seqs + false_seqs
    labels    = [1] * len(true_seqs) + [0] * len(false_seqs)
    print(
        f"HS3D {site_type}: {len(true_seqs):,} positives + "
        f"{len(false_seqs):,} negatives = {len(sequences):,} total"
    )
    return sequences, labels


# ── Synthetic fallback ────────────────────────────────────────────────────────

_BASES = "ACGT"


def _random_seq(length: int, gc: float = 0.5) -> str:
    at = 1.0 - gc
    weights = [at / 2, at / 2, gc / 2, gc / 2]  # A T C G
    return "".join(random.choices(_BASES, weights=weights, k=length))


def _donor_positive() -> str:
    exon   = _random_seq(70, gc=0.52)
    intron = "AAGT" + _random_seq(64, gc=0.38)
    return exon + "GT" + intron


def _donor_negative() -> str:
    return _random_seq(69, gc=0.45) + "GT" + _random_seq(69, gc=0.45)


def _acceptor_positive() -> str:
    tract  = _random_seq(60, gc=0.28)          # polypyrimidine-rich intron
    intron = (tract + "TTTTTTTTTT")[:70]
    exon   = _random_seq(68, gc=0.50)
    return intron + "AG" + exon


def _acceptor_negative() -> str:
    return _random_seq(69, gc=0.45) + "AG" + _random_seq(69, gc=0.45)


def generate_synthetic_data(
    site_type: str,
    n_positive: int = 2000,
    n_negative: int = 10_000,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Generate synthetic data when HS3D is unavailable (lower performance)."""
    random.seed(seed)
    pos_fn = _donor_positive    if site_type == "donor" else _acceptor_positive
    neg_fn = _donor_negative    if site_type == "donor" else _acceptor_negative
    positives = [pos_fn() for _ in range(n_positive)]
    negatives = [neg_fn() for _ in range(n_negative)]
    sequences = positives + negatives
    labels    = [1] * n_positive + [0] * n_negative
    print(
        f"Synthetic {site_type}: {n_positive:,} positives + "
        f"{n_negative:,} negatives (HS3D unavailable)"
    )
    return sequences, labels


# ── Reverse-complement augmentation ──────────────────────────────────────────

_RC = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(seq: str) -> str:
    return seq.upper().translate(_RC)[::-1]


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class SpliceSiteDataset(Dataset):
    def __init__(
        self,
        sequences: list[str],
        labels: list[int],
        augment: bool = False,
    ):
        self.augment = augment
        self.X = [one_hot_encode(s) for s in sequences]
        self.y = labels

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        # Random reverse-complement flip (only during training)
        if self.augment and random.random() < 0.5:
            x = x[::-1, ::-1].copy()   # RC is channel-flip + sequence-flip
        return (
            torch.from_numpy(x),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# ── Stratified split + weighted-sampler DataLoaders ──────────────────────────

def make_dataloaders(
    sequences: list[str],
    labels: list[int],
    split: tuple[float, float, float] = (0.70, 0.15, 0.15),
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    augment_train: bool = True,
    weighted_sampling: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Stratified train/val/test split with optional reverse-complement augmentation
    and WeightedRandomSampler for balanced mini-batches.

    Stratified splitting guarantees both classes appear in every split —
    critical for imbalanced datasets like HS3D where positives are <3 %.
    WeightedRandomSampler ensures each mini-batch is ~50 % positive,
    complementing the focal loss's per-sample down-weighting.
    """
    seqs   = np.array(sequences, dtype=object)
    labs   = np.array(labels,    dtype=int)

    # ── Stratified 70 / 15 / 15 split ──
    val_test_frac = split[1] + split[2]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        seqs, labs,
        test_size=val_test_frac,
        stratify=labs,
        random_state=seed,
    )
    val_frac = split[1] / val_test_frac
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=1.0 - val_frac,
        stratify=y_tmp,
        random_state=seed,
    )

    train_ds = SpliceSiteDataset(X_tr.tolist(), y_tr.tolist(), augment=augment_train)
    val_ds   = SpliceSiteDataset(X_va.tolist(), y_va.tolist(), augment=False)
    test_ds  = SpliceSiteDataset(X_te.tolist(), y_te.tolist(), augment=False)

    # ── Weighted sampler: over-sample positives to balance mini-batches ──
    if weighted_sampling:
        counts   = np.bincount(y_tr)
        weights  = 1.0 / counts[y_tr]        # inverse-frequency weights
        sampler  = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(y_tr),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    eval_kwargs = dict(
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Split (stratified): {len(X_tr):,} train "
        f"/ {len(X_va):,} val / {len(X_te):,} test  "
        f"| pos rate train={y_tr.mean():.3f}  val={y_va.mean():.3f}  test={y_te.mean():.3f}"
    )

    return (
        train_loader,
        DataLoader(val_ds,  **eval_kwargs),
        DataLoader(test_ds, **eval_kwargs),
    )
