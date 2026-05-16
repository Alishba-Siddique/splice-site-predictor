# SpliceSiteResNet — Technical Documentation

**Version**: 1.0  
**Python**: 3.10+  
**PyTorch**: 2.0+

---

## Table of Contents

1. [Architecture Specification](#1-architecture-specification)
2. [Data Pipeline](#2-data-pipeline)
3. [Training Pipeline](#3-training-pipeline)
4. [Inference API](#4-inference-api)
5. [Gradio Application](#5-gradio-application)
6. [Configuration Reference](#6-configuration-reference)
7. [Performance Characteristics](#7-performance-characteristics)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Architecture Specification

### `SpliceSiteResNet` (`model.py`)

The primary production model. A dilated pre-activation residual 1D CNN.

```
SpliceSiteResNet(dr=0.1, df=0.3)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `dr` | float | 0.1 | Dropout rate inside residual blocks |
| `df` | float | 0.3 | Dropout rate in the classification head |

**Forward signature**: `(B, 4, 140) → (B, 1)` — returns raw logits. Apply `torch.sigmoid` for probabilities.

#### Layer-by-layer dimensions

| Stage | Module | Input shape | Output shape | Parameters |
|---|---|---|---|---|
| Stem | Conv1d(4→64, k=1) + BN + GELU | (B, 4, 140) | (B, 64, 140) | 448 |
| Tower block 1 | ResBlock1D(64, dil=1) | (B, 64, 140) | (B, 64, 140) | 74,112 |
| Tower block 2 | ResBlock1D(64, dil=4) | (B, 64, 140) | (B, 64, 140) | 74,112 |
| Tower block 3 | ResBlock1D(64, dil=16) | (B, 64, 140) | (B, 64, 140) | 74,112 |
| Tower block 4 | ResBlock1D(64, dil=1) | (B, 64, 140) | (B, 64, 140) | 74,112 |
| MaxPool | MaxPool1d(2) | (B, 64, 140) | (B, 64, 70) | 0 |
| Down1 block 1 | ResBlock1D(64→128, dil=1) | (B, 64, 70) | (B, 128, 70) | 131,584 |
| Down1 block 2 | ResBlock1D(128, dil=2) | (B, 128, 70) | (B, 128, 70) | 230,912 |
| MaxPool | MaxPool1d(2) | (B, 128, 70) | (B, 128, 35) | 0 |
| Down2 block 1 | ResBlock1D(128→256, dil=1) | (B, 128, 35) | (B, 256, 35) | 264,192 |
| Down2 block 2 | ResBlock1D(256, dil=2) | (B, 256, 35) | (B, 256, 35) | 787,968 |
| GlobalAvgPool | AdaptiveAvgPool1d(1) | (B, 256, 35) | (B, 256) | 0 |
| Head | LayerNorm + Linear(256→128) + GELU + Dropout + Linear(128→1) | (B, 256) | (B, 1) | 33,409 |
| **Total** | | | | **~1,069,825** |

#### `ResBlock1D` internal structure

```
Input x  (B, C_in, L)
  │
  ├─ BN(C_in)
  ├─ GELU
  ├─ Conv1d(C_in → C_out, k, dilation=d, padding=(k-1)*d//2, bias=False)
  ├─ BN(C_out)
  ├─ GELU
  ├─ Dropout(dr)
  ├─ Conv1d(C_out → C_out, k=1, bias=False)
  │
  └─ Skip: Identity (C_in==C_out) or Conv1d(C_in→C_out, k=1, bias=False)
       ↕
  Output: block(x) + skip(x)  (B, C_out, L)
```

**Receptive field formula**: For a single dilated conv with kernel `k` and dilation `d`, the receptive field is `k + (k-1)*(d-1)`. Stacked through the pre-activation block (two convolutions), effective RF is additive.

### `SpliceSitePredictor` (`model.py`)

The legacy baseline model, retained for backward compatibility.

```
SpliceSitePredictor(seq_len=140, dropout=0.3)
```

Four-layer 1D CNN: Conv(4→32, k=9) → Pool → Conv(32→64, k=7) → Pool → Conv(64→128, k=5) → Pool → Conv(128→256, k=3) → GlobalAvgPool → FC(256→128) → FC(128→1). ~189 K parameters.

---

## 2. Data Pipeline

### `data.py` — Functions

#### `download_hs3d(site_type, dest_dir, false_parts)`

Downloads and caches HS3D component zip files from the Altervista mirror.

```python
download_hs3d(
    site_type:   str,              # 'donor' or 'acceptor'
    dest_dir:    str | Path,       # cache directory, default='data/HS3D'
    false_parts: int | None,       # number of false-site zips to use, None=all
) -> tuple[list[str], list[int]]   # (sequences, labels)
```

**Downloads per call**:

| File | Content | Size | Count |
|---|---|---|---|
| `EI_true.zip` | True donor sequences | ~138 KB | 2,796 |
| `EI_false_1.zip` | False donor (part 1 of 3) | ~1.6 MB | ~90,000 |
| `IE_true.zip` | True acceptor sequences | ~140 KB | 2,880 |
| `IE_false_1.zip` | False acceptor (part 1 of 4) | ~1.5 MB | ~82,000 |

Cached files are validated against ZIP magic bytes (`PK\x03\x04`) on every read. Corrupt or incomplete downloads are deleted and re-fetched automatically.

**SSL**: The Altervista mirror uses a self-signed certificate. The downloader bypasses hostname verification (`ctx.check_hostname = False`, `ctx.verify_mode = ssl.CERT_NONE`). This is acceptable for a fixed, known academic mirror.

#### `parse_hs3d_seq(content: str) -> list[str]`

Parses HS3D `.seq` file content. Line format:
```
AB000381  ( 1,2,2): CTCCTCTTTGCC...   ← 140 characters after '): '
```

Returns only lines with exactly 140 characters in `[ACGTN]`. Lines with `): ` absent (header, metadata) are silently skipped.

#### `generate_synthetic_data(site_type, n_positive, n_negative, seed)`

Generates a synthetic dataset when HS3D is unavailable.

- **Positives**: random sequence with correct GC composition, correct dinucleotide at position 70, and canonical intron-start AAGT (donor) or polypyrimidine-biased upstream (acceptor).
- **Negatives**: fully random sequence with the correct dinucleotide at position 70 but no splice-site context.

Synthetic negatives are easier to classify than real HS3D negatives (real negatives are drawn from genuine splice-site-adjacent genomic regions). Models trained on synthetic data will be overoptimistic during evaluation.

#### `reverse_complement(seq: str) -> str`

Returns the Watson-Crick reverse complement of a DNA string. Uses `str.translate` on the complement map `ACGTN→TGCAN` followed by string reversal. N maps to N.

#### `SpliceSiteDataset`

```python
SpliceSiteDataset(sequences, labels, augment=False)
```

A `torch.utils.data.Dataset` wrapping pre-encoded one-hot tensors. With `augment=True`, each `__getitem__` call flips the sequence to its reverse complement with probability 0.5. The RC operation on a `(4, L)` one-hot tensor is `x[::-1, ::-1].copy()` — simultaneous channel permutation (A↔T, C↔G) and position reversal.

#### `make_dataloaders(sequences, labels, ...)`

```python
make_dataloaders(
    sequences:         list[str],
    labels:            list[int],
    split:             tuple = (0.70, 0.15, 0.15),
    batch_size:        int = 128,
    num_workers:       int = 0,
    seed:              int = 42,
    augment_train:     bool = True,
    weighted_sampling: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]
```

Returns (train, val, test) DataLoaders.

**Stratification**: Uses `sklearn.model_selection.train_test_split` with `stratify=labels`. Both the 70/30 and 50/50 sub-splits are stratified. This guarantees that the positive rate in each split equals the dataset-level positive rate to within rounding.

**WeightedRandomSampler**: Training DataLoader uses `WeightedRandomSampler` with inverse-frequency weights. This makes each mini-batch approximately 50% positive, independent of the dataset-level class ratio.

**Validation and test DataLoaders** use a standard sequential sampler with `shuffle=False`. No augmentation is applied at evaluation time.

---

## 3. Training Pipeline

### `train.py` — CLI

```bash
python train.py [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--site_type` | `donor` | `donor` or `acceptor` |
| `--epochs` | `40` | Maximum training epochs |
| `--batch_size` | `256` | Mini-batch size |
| `--lr` | `3e-4` | Peak learning rate (after warmup) |
| `--patience` | `8` | Early-stopping patience on val ROC-AUC |
| `--data_dir` | `data/HS3D` | HS3D cache directory |
| `--output_dir` | `outputs` | Directory for evaluation plots |
| `--false_parts` | `None` (all) | Number of false-site zips (1 = fast run) |
| `--synthetic` | `False` | Skip HS3D download, use synthetic data |
| `--no_amp` | `False` | Disable Automatic Mixed Precision |
| `--seed` | `42` | Random seed |

#### Loss function: Focal loss

```python
def focal_loss(logits, targets, gamma=2.0, alpha=0.25, label_smoothing=0.05):
    targets = targets * (1 - ls) + 0.5 * ls          # label smoothing
    bce     = BCEWithLogitsLoss(logits, targets)
    pt      = sigmoid(logits) if y==1 else 1-sigmoid(logits)
    at      = alpha if y==1 else 1-alpha
    return mean(at * (1 - pt)^gamma * bce)
```

**α=0.25**: mild positive-class upweighting. Lower than the standard 0.75 because `WeightedRandomSampler` already balances class marginals at the batch level.

**γ=2**: standard value from Lin _et al._ (2017). Down-weights easy examples by a factor of `(1-p)^2`; a correctly classified example with `p=0.9` is weighted 100× less than one with `p=0.5`.

**label_smoothing=0.05**: replaces hard 0/1 labels with `0.025` and `0.975`. Prevents logit saturation and improves probability calibration.

#### Learning rate schedule

`OneCycleLR` with:
- `pct_start=0.05` — 5% of total steps as linear warmup
- `anneal_strategy='cos'` — cosine decay for remaining 95%
- `div_factor=10` — start LR = peak LR / 10
- `final_div_factor=1e4` — end LR = peak LR / 10,000

At `lr=3e-4`: warmup from `3e-5` to `3e-4`, decay to `3e-8`.

#### Training step

```python
# Forward
with autocast(enabled=use_amp):
    logits = model(X)
    loss   = focal_loss(logits.squeeze(-1), y)

# Backward (AMP-safe)
scaler.scale(loss).backward()
scaler.unscale_(optimiser)
clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimiser)
scaler.update()
scheduler.step()
```

#### Saved artefacts

After training, `train.py` saves:

| File | Description |
|---|---|
| `splice_model_{site_type}.pt` | Best checkpoint (highest val ROC-AUC) |
| `outputs/roc_{site_type}.png` | ROC curve |
| `outputs/pr_{site_type}.png` | Precision-recall curve |
| `outputs/cm_{site_type}.png` | Confusion matrix |
| `outputs/training_curves_{site_type}.png` | Loss, AUC, LR vs. epoch |

---

## 4. Inference API

### `utils.py` — Key functions

#### `one_hot_encode(seq: str) -> np.ndarray`

```python
one_hot_encode("ATCGN") -> np.ndarray of shape (4, 5), dtype=float32
```

Channel order: A=0, T=1, C=2, G=3. N and unknown bases → all-zero column.

#### `predict_on_sequence(model, sequence, site_type, half_window, batch_size, device)`

```python
positions, probs = predict_on_sequence(
    model       = trained_model,
    sequence    = "ATGCAG...GTAAGT...",   # arbitrary length DNA
    site_type   = "donor",                 # or "acceptor"
    half_window = 70,                      # bp on each side of candidate site
    batch_size  = 256,
    device      = "cpu",
)
```

**Algorithm**:
1. Find all occurrences of GT (donor) or AG (acceptor) in `sequence`.
2. For each candidate at position `i`, extract `sequence[i-70 : i+70]` (140 bp window).
3. Discard candidates within 70 bp of either end (insufficient flanking context).
4. One-hot encode and batch-score with the model.
5. Return `(positions, probabilities)`, both lists of the same length.

**Returns**: 0-indexed positions in the original sequence. To get 1-indexed (biological convention), add 1.

#### `extract_window(seq, center, half=70) -> str | None`

Extracts `seq[center-half : center+half]`. Returns `None` if the window would extend outside sequence bounds.

#### Plotting functions

```python
fig           = plot_prediction_track(sequence, positions, probs, site_type, threshold)
fig, roc_auc  = plot_roc(y_true, y_score, title)
fig, pr_auc   = plot_pr(y_true, y_score, title)
fig           = plot_confusion_matrix(cm, labels)
```

All plotting functions return a `matplotlib.Figure`. The backend is set to `Agg` (non-interactive) so plots can be generated in server environments.

### Minimal inference example

```python
import torch
from model import SpliceSiteResNet
from utils import predict_on_sequence

# Load
model = SpliceSiteResNet()
model.load_state_dict(torch.load("splice_model_donor.pt", map_location="cpu"))
model.eval()

# Score a sequence
sequence = open("my_gene.fa").read().replace("\n","").upper()[1:]  # strip FASTA header
positions, probs = predict_on_sequence(model, sequence, site_type="donor")

# Print high-confidence sites
for pos, prob in sorted(zip(positions, probs), key=lambda x: -x[1]):
    if prob > 0.7:
        print(f"pos={pos+1:6d}  P={prob:.4f}  context={sequence[pos:pos+6]}")
```

---

## 5. Gradio Application

### `app.py`

The Gradio Space loads both models at startup:

```python
donor_model,    donor_ok    = _load("splice_model_donor.pt")
acceptor_model, acceptor_ok = _load("splice_model_acceptor.pt")
```

If a model file is missing, an untrained randomly-initialised model is used with a warning banner. Predictions from an untrained model are meaningless — upload trained `.pt` files to the Space.

#### Prediction callback: `predict(sequence, site_type, threshold)`

| Input | Type | Validation |
|---|---|---|
| `sequence` | str | Stripped, uppercased; invalid chars rejected; min 142 bp |
| `site_type` | `"donor"` or `"acceptor"` | Radio button |
| `threshold` | float [0.1, 0.95] | Confidence threshold for "predicted" calls |

**Outputs**:
1. `matplotlib.Figure` — two-panel prediction track (probability scatter + dinucleotide positions)
2. `pd.DataFrame` — predicted sites above threshold, sorted by probability descending
3. `str` — status line (number of candidates scanned, number predicted)

#### Running locally

```bash
pip install -r requirements.txt
python app.py           # serves at http://127.0.0.1:7860
```

The app auto-detects GPU and uses CUDA if available. CPU inference on a 5 kb sequence takes approximately 200 ms.

---

## 6. Configuration Reference

### Model files

| File | Created by | Purpose |
|---|---|---|
| `splice_model_donor.pt` | `train.py` | Trained donor model weights |
| `splice_model_acceptor.pt` | `train.py` | Trained acceptor model weights |

Both files are `state_dict` dicts loadable with `torch.load(..., map_location="cpu")` + `model.load_state_dict(...)`.

### Data cache

Zip files in `data/HS3D/` are cached after first download. To reset the cache (e.g., after a corrupt download is detected), delete the relevant `.zip` files. They will be re-fetched on next run. `fetch_zip` validates ZIP magic bytes before using any cached file.

### Output plots

All evaluation plots are saved to `outputs/` (configurable with `--output_dir`). Files:

```
outputs/
├── roc_donor.png
├── roc_acceptor.png
├── pr_donor.png
├── pr_acceptor.png
├── cm_donor.png
├── cm_acceptor.png
├── training_curves_donor.png
└── training_curves_acceptor.png
```

Embed `cm_donor.png` and `roc_donor.png` in `README.md` by replacing the placeholder links in the Results section.

---

## 7. Performance Characteristics

### Training time (approximate)

| Hardware | Site type | Time per epoch | Total (40 epochs) |
|---|---|---|---|
| NVIDIA T4 (Colab) | Donor | ~25 s | ~17 min |
| NVIDIA T4 (Colab) | Acceptor | ~30 s | ~20 min |
| CPU only | Donor | ~180 s | ~2 h |

Times assume `false_parts=1` (~90K sequences total per site type). Using all false parts (`false_parts=None`) increases data size ~3× and training time proportionally.

### Memory

| Configuration | GPU VRAM | System RAM |
|---|---|---|
| Training, batch=256, T4 | ~1.2 GB | ~4 GB |
| Inference, batch=256, CPU | 0 | ~500 MB |

### Inference throughput

| Device | Throughput |
|---|---|
| NVIDIA T4 | ~50,000 windows/sec |
| CPU (8-core) | ~3,000 windows/sec |

For a 10 kb genomic sequence with ~500 GT positions, inference takes approximately 10 ms on GPU and 170 ms on CPU.

---

## 8. Troubleshooting

### `val_auc=nan` during training

**Cause**: The validation split contains only one class (all positives or all negatives), making ROC-AUC undefined.

**Solution**: Ensure `make_dataloaders` is called with stratified splitting (default). If you're using the notebook, restart the kernel and re-run the data loading cell — stale variables from a previous failed run may be in scope.

**Safety net**: `safe_auc()` in the training loop returns 0.0 instead of NaN and emits a warning. If you see this warning with stratified splitting, your dataset is likely all one class (check `sum(labels)`).

### `FileNotFoundError: splice_model_donor.pt`

The model file is only saved when at least one epoch improves validation ROC-AUC. If every epoch shows `val_auc=0.0` (single-class split), no checkpoint is ever saved. Fix the data loading issue first.

### Download fails: `gaierror: getaddrinfo failed`

DNS resolution is blocked in some corporate/university networks. Options:
1. Run on **Google Colab** where network access is unrestricted.
2. Use `--synthetic` flag: `python train.py --site_type donor --synthetic`. Performance will be ~5% lower than real HS3D.
3. Download the zip files manually from `https://salvatorerampone.altervista.org/wp-content/HS3D/` and place them in `data/HS3D/`.

### `BadZipFile: File is not a zip file`

A previously cached file is not a valid ZIP (corrupt partial download or redirect HTML). The updated `fetch_zip` detects this via magic byte check and re-downloads automatically. If it persists, manually delete `data/HS3D/*.zip` and retry.

### OOM (out of memory) on GPU

Reduce `--batch_size` (try 128 or 64). Alternatively, enable gradient checkpointing by modifying the forward pass in `model.py` to use `torch.utils.checkpoint.checkpoint` around each residual block.

### Gradio Space shows "warning: untrained model"

The model files `splice_model_donor.pt` and `splice_model_acceptor.pt` are not present in the Space repository. Upload them via `git lfs` or the Hugging Face web interface after training.

```bash
# In your Space repo:
git lfs track "*.pt"
git add splice_model_donor.pt splice_model_acceptor.pt
git commit -m "add trained model weights"
git push
```
