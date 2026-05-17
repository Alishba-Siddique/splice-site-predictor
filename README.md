---
title: Splice Site Predictor
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.20.0
app_file: app.py
pinned: false
tags:
  - biology
  - genomics
  - splice-site
  - deep-learning
  - pytorch
---



https://github.com/user-attachments/assets/4dbe520a-8f36-4b09-af64-d7e779f4828e



# 🧬 SpliceSiteResNet — Canonical Splice Site Predictor

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange?logo=gradio)](https://gradio.app)
[![HS3D Dataset](https://img.shields.io/badge/Data-HS3D-informational)](http://www.sci.unisannio.it/docenti/rampone/)

Predict canonical **GT–AG splice donor and acceptor sites** in human DNA sequences using a dilated pre-activation residual convolutional neural network trained on the Human Splice-Site Data Set (HS3D).

---

## Problem Statement

Pre-mRNA splicing removes introns and joins exons to produce a translatable mRNA transcript. The spliceosome identifies intron boundaries by recognising short degenerate consensus sequences — the **donor site** (GT at the 5′ end of the intron) and the **acceptor site** (AG at the 3′ end). Mutations that disrupt these signals cause aberrant splicing, accounting for approximately 10–15% of all disease-causing point mutations and a substantial fraction of pathogenic variants in cancer predisposition genes such as *BRCA1* and *BRCA2*.

The challenge is specificity: the human genome contains hundreds of thousands of GT and AG dinucleotides, but only a fraction of them are genuine splice sites. This model distinguishes true splice sites from decoys by learning sequence context at three biologically relevant scales from a 140 bp window centred on each candidate dinucleotide.

---

## Dataset: HS3D

The **Human Splice-Site Data Set** (Pollastro & Gagliardi, 2002) is the standard benchmark for fixed-window splice site prediction. It was extracted from GenBank Release 123 and contains 140-nucleotide windows around canonical GT–AG splice sites.

| Split | Donor | Acceptor |
|---|---:|---:|
| True sites (positives) | 2,796 | 2,880 |
| False sites (negatives) | 271,937 | 329,374 |
| **Total** | **274,733** | **332,254** |
| Positive rate | 1.02% | 0.87% |

False examples are genuine GT/AG dinucleotides from the same gene loci that the spliceosome does not use — biologically plausible decoys, not random sequences.

**Train / val / test split**: 70% / 15% / 15%, **stratified** on class label (seed=42). Stratification is mandatory at this imbalance ratio: a naive random split has a ~7% chance of producing a validation set with zero positives.

---

## Model Architecture

```
Input: (batch, 4, 140)  ←  one-hot DNA (A/T/C/G channels × 140 bp)

Stem ─── Conv1d(4→64, k=1) ─ BatchNorm ─ GELU
          │
          ├─ Dilated Tower (full 140-bp resolution)
          │    ResBlock(64, k=9, dil=1 ) ← local motif     [~9 bp context]
          │    ResBlock(64, k=9, dil=4 ) ← consensus motif [~41 bp context]
          │    ResBlock(64, k=9, dil=16) ← polypyrimidine  [~137 bp context]
          │    ResBlock(64, k=9, dil=1 ) ← local refinement
          │
          ├─ MaxPool(2) → 70 positions
          │    ResBlock(64→128, k=7, dil=1)
          │    ResBlock(128,    k=7, dil=2)
          │
          ├─ MaxPool(2) → 35 positions
          │    ResBlock(128→256, k=5, dil=1)
          │    ResBlock(256,     k=5, dil=2)
          │
          └─ GlobalAvgPool → LayerNorm → Linear(256→128) → GELU → Linear(128→1)

Output: (batch, 1)  ←  raw logit; sigmoid → P(splice site)
```

**Trainable parameters**: 1,069,825

Each `ResBlock` uses the **pre-activation** pattern (BatchNorm → GELU → Conv → BatchNorm → GELU → Conv) with an identity or 1×1 skip connection, matching the formulation of He *et al.* (2016) used in SpliceAI.

The dilation rates [1, 4, 16] are chosen to span the three length scales of canonical splice site biology:

| Dilation | Effective context | Biological signal captured |
|---|---|---|
| 1 | ~9 bp | GT/AG dinucleotide + immediate consensus (MAG\|GURAGU) |
| 4 | ~41 bp | Branch-point distance, exon–intron composition bias |
| 16 | ~137 bp | Full polypyrimidine tract (up to 70 bp upstream of acceptor) |

---

## Training Methodology

Five components work together to handle the ~100:1 class imbalance:

| Component | Role |
|---|---|
| Stratified split | Guarantees both classes present in every partition |
| `WeightedRandomSampler` | Makes each mini-batch ~50% positive (macro-imbalance) |
| Focal loss (γ=2, α=0.25) | Down-weights easy examples (micro-imbalance) |
| Label smoothing (ε=0.05) | Prevents logit saturation; improves calibration |
| RC augmentation (p=0.5) | Doubles training set; splice sites are strand-symmetric |

**Optimiser**: AdamW, lr=3×10⁻⁴, weight decay=10⁻⁴  
**Schedule**: Linear warmup (5% of steps) → cosine annealing to lr/10,000  
**AMP**: `torch.cuda.amp.autocast` + `GradScaler` for ~2× GPU throughput  
**Gradient clipping**: max-norm=1.0

---

## Results

> **Replace the values below with your actual test-set output after running `train.py`.**
> Expected ranges are based on published CNN-class baselines on HS3D.

| Metric | Donor model | Acceptor model |
|---|---:|---:|
| Accuracy | [FILL] | [FILL] |
| Precision | [FILL] | [FILL] |
| Recall | [FILL] | [FILL] |
| F1 | [FILL] | [FILL] |
| **ROC-AUC** | **[FILL]** | **[FILL]** |
| **PR-AUC** | **[FILL]** | **[FILL]** |
| MCC | [FILL] | [FILL] |

Confusion matrices, ROC curves, and PR curves are saved to `outputs/` by `train.py` and can be linked here:

```markdown
![Donor ROC](outputs/roc_donor.png)
![Donor Confusion Matrix](outputs/cm_donor.png)
```

---

## Quick Start

### 1. Install

```bash
git clone https://huggingface.co/spaces/<your-username>/splice-site-predictor
cd splice-site-predictor
pip install -r requirements.txt
```

### 2. Train

```bash
# Download HS3D and train (requires internet; ~15 min on GPU):
python train.py --site_type donor
python train.py --site_type acceptor

# Offline / network-restricted: use synthetic data
python train.py --site_type donor --synthetic
python train.py --site_type acceptor --synthetic
```

### 3. Run the app

```bash
python app.py   # opens at http://127.0.0.1:7860
```

### 4. Python API

```python
import torch
from model import SpliceSiteResNet
from utils import predict_on_sequence

model = SpliceSiteResNet()
model.load_state_dict(torch.load("splice_model_donor.pt", map_location="cpu"))
model.eval()

positions, probs = predict_on_sequence(model, your_sequence, site_type="donor")
for pos, p in zip(positions, probs):
    if p > 0.5:
        print(f"Donor site at position {pos+1}  (P={p:.3f})")
```

---

## Repository Structure

```
splice-site-predictor/
├── app.py                       # Gradio Space entry point
├── model.py                     # SpliceSiteResNet + SpliceSitePredictor
├── utils.py                     # One-hot encoding, inference, plotting
├── data.py                      # HS3D download, parsing, DataLoaders
├── train.py                     # Full training pipeline (CLI)
├── requirements.txt
├── PAPER.md                     # Full academic paper
├── docs/
│   └── TECHNICAL.md             # Architecture and API reference
├── notebooks/
│   └── training_pipeline.ipynb  # Self-contained Colab training notebook
├── examples/
│   └── brca1_exon11.fasta       # Example sequences for the demo
├── outputs/                     # Evaluation plots (generated by train.py)
│   ├── roc_donor.png
│   ├── cm_donor.png
│   └── ...
└── data/
    └── HS3D/                    # Cached zip files (auto-downloaded)
```

---

## Limitations

The following limitations are stated explicitly because overstating model scope is the most common failure mode in bioinformatics tool publications.

1. **Canonical GT–AG splice sites only.** The model is trained on U2-type spliceosomal introns (≈99% of human introns). It cannot predict U12-type (AT–AC, minor GT–AG) splice sites. Applying it to organisms with substantial non-canonical splicing will produce uninterpretable outputs.

2. **Fixed 140 bp window.** Each candidate position is scored from 70 bp upstream and 70 bp downstream only. Splicing regulatory elements (ESEs, ESSs, ISEs, ISSs) acting at distances >70 bp are invisible to this model. For long-range regulatory effects, SpliceAI (±10,000 bp context) is more appropriate.

3. **Human sequences only.** HS3D is derived exclusively from *Homo sapiens* pre-mRNA. Performance on non-human species — even closely related primates — has not been evaluated and should not be assumed.

4. **HS3D prior, not genome-wide prior.** In HS3D, approximately 1–3% of candidates are true splice sites. In the genome, the true positive rate is orders of magnitude lower. The model's output probability reflects the HS3D training distribution. For genome-wide scoring, recalibrate probabilities with Platt scaling or isotonic regression on an independent held-out chromosome.

5. **No alternative splicing.** The model scores each position independently. It has no representation of which exons are co-regulated, tissue-specific splice site usage, or splicing quantitative trait loci (sQTLs).

6. **HS3D vintage.** The benchmark was constructed from GenBank Release 123 (2001). Splice sites discovered by RNA-seq since then — including many alternative and tissue-specific isoforms — are not represented. The model may underperform on recently annotated junctions.

---

## References

1. Pollastro P, Gagliardi S (2002). *HS3D, a dataset of Homo sapiens splice regions*. Genome Informatics 13:290–300.

2. Jaganathan K *et al.* (2019). *Predicting splicing from primary sequence with deep learning*. Cell 176(3):535–548. [doi:10.1016/j.cell.2018.12.015](https://doi.org/10.1016/j.cell.2018.12.015)

3. He K *et al.* (2016). *Identity mappings in deep residual networks*. ECCV 2016. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)

4. Lin TY *et al.* (2017). *Focal loss for dense object detection*. ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

5. Yeo G & Burge CB (2004). *Maximum entropy modeling of short sequence motifs with applications to RNA splicing signals*. J Comput Biol 11(2–3):377–394.

---

## Citation

If you use this model or training code in your research, please cite:

```bibtex
@software{splicesiteresnet2024,
  title   = {SpliceSiteResNet: Multi-Scale Dilated Residual Networks for
             Human Canonical Splice Site Prediction},
  year    = {2025},
  url     = {https://huggingface.co/spaces/<your-username>/splice-site-predictor},
  note    = {Trained on HS3D (Pollastro \& Gagliardi, 2002)}
}
```

---
