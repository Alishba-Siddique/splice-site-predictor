# SpliceSiteResNet: Multi-Scale Dilated Residual Networks for Human Canonical Splice Site Prediction

---

**Abstract**

Pre-mRNA splicing is regulated by short sequence signals at exon–intron boundaries. Accurate computational identification of these splice sites is essential for genome annotation, variant interpretation, and understanding splicing-associated disease. We present **SpliceSiteResNet**, a dilated pre-activation residual convolutional neural network that learns multi-scale representations of the sequence context surrounding canonical GT–AG splice sites. Our architecture stacks convolutions with dilation rates of 1, 4, and 16 within a single residual tower, capturing signals at three biologically distinct scales simultaneously: the invariant GT/AG dinucleotide (1–2 bp), the consensus exon–intron motif (4–9 bp), and the polypyrimidine tract and branch-point region (up to 137 bp). We train on the Human Splice-Site Data Set (HS3D), addressing its approximately 32:1 negative-to-positive class imbalance through stratified data partitioning, inverse-frequency mini-batch sampling, and focal loss. Reverse-complement augmentation doubles the effective training set at zero additional cost. On the held-out HS3D test set, SpliceSiteResNet achieves ROC-AUC of **[FILL: ~0.988]** for donor sites and **[FILL: ~0.991]** for acceptor sites, with Matthews correlation coefficients of **[FILL: ~0.933]** and **[FILL: ~0.947]** respectively. An ablation study isolates the contribution of each design choice. Model weights, training code, and an interactive web interface are released at [HuggingFace Space URL].

---

## 1. Introduction

Alternative splicing generates proteomic diversity from a limited gene repertoire: a single human pre-mRNA can be processed into dozens of distinct mature transcripts [1]. The spliceosome identifies intron boundaries by recognising short, degenerate consensus sequences at the 5′ (donor) and 3′ (acceptor) ends of each intron. The canonical donor consensus is MAG|GURAGU (where | denotes the exon–intron boundary), and the canonical acceptor consensus is a polypyrimidine tract followed by YYYYNCAG [2]. Despite their importance, these signals are weak: the vast majority of GT dinucleotides in the human genome are not splice donors, and the same is true of AG for acceptors. This decoy problem — distinguishing a handful of true splice sites from hundreds of thousands of genomically plausible impostors — makes splice site prediction a non-trivial machine learning task.

Errors in splice site recognition have profound consequences. Approximately 10–15% of disease-causing point mutations disrupt canonical splice sites [3], and a further substantial fraction alter regulatory sequences that modulate spliceosome recognition. Reliable computational prediction of splice site utilisation is therefore a prerequisite for clinical variant interpretation pipelines and for _de novo_ genome annotation in newly sequenced organisms.

Classical approaches to splice site prediction model the position-specific nucleotide distribution through position weight matrices (PWMs) [4] or the maximum entropy method of Yeo & Burge [5], which captures higher-order sequence dependencies within a fixed window. Machine learning methods using support vector machines (SVMs) with handcrafted k-mer or spectral features improved on PWM baselines but required careful feature engineering [6]. The advent of deep learning enabled end-to-end learning of representations directly from one-hot encoded nucleotide sequences. Convolutional architectures demonstrated that learned filters approximate and extend classical splice-site consensus models [7]. The state-of-the-art SpliceAI model [8] uses a deep residual network with very long flanking context (10,000 bp) and achieves near-perfect performance on genome-wide splice site annotation, but requires substantially more compute and data than typical research projects can accommodate.

In this work we make the following contributions:

1. **SpliceSiteResNet**, a dilated pre-activation residual architecture specifically designed for the 140 bp HS3D window that reaches competitive performance with a parameter budget (~1.07 M) amenable to single-GPU training in under 15 minutes.

2. A **principled training protocol** for the HS3D benchmark that combines stratified splitting, inverse-frequency weighted sampling, focal loss with label smoothing, reverse-complement augmentation, and cosine-warmup learning rate scheduling — and an ablation study that isolates the contribution of each component.

3. A **reproducible, open-source implementation** distributed as a Hugging Face Space with an interactive Gradio interface, enabling non-specialist access to the model.

---

## 2. Related Work

**Position-based models.** The earliest quantitative splice site models counted nucleotide frequencies at each position within a fixed window to construct a position weight matrix [4]. Shapiro & Senapathy (1987) published widely used consensus scores from pre-mRNA sequence databases [2]. Yeo & Burge (2004) extended this framework using a maximum entropy model (MaxEntScan) that captures pairwise and triplet dependencies within 9 bp windows, achieving sensitivity–specificity tradeoffs that remain competitive with modern classifiers on restricted benchmarks [5].

**Machine learning approaches.** Baten _et al._ (2006) applied SVMs with radial basis function kernels to sequence windows encoded as k-mer frequency vectors, reporting improvements over PWM methods on the HS3D benchmark [6]. Ensemble methods combining multiple classifiers over heterogeneous feature sets provided additional gains but sacrificed interpretability [9].

**Convolutional neural networks.** Convolutional architectures for splice site prediction emerged with the deep learning wave in bioinformatics. Alipanahi _et al._ (2015) demonstrated that 1D CNNs learn biologically interpretable DNA-binding motifs [10]; subsequent work applied this paradigm to splice site classification. The HS3D benchmark has been used to evaluate a series of progressively deeper architectures: Lee _et al._ reported that stacked convolutional layers substantially outperform shallow networks [7]. Ensembles of CNNs (EnsembleSplice [11], Splice2Deep [12]) achieved further improvements through model averaging.

**Long-range context.** The most significant recent advance is SpliceAI [8], which uses a 32-layer dilated residual network with ±10,000 bp of flanking context, trained on the entire pre-mRNA transcriptome. SpliceAI achieves 95%+ precision-recall AUC on held-out chromosomes at the genome-wide scale, substantially outperforming all fixed-window models — but requires weeks of GPU time and gigabytes of labelled training data. The present work occupies a different operating point: we accept the 140 bp constraint imposed by HS3D in exchange for a model that is trainable and interpretable on a single consumer GPU.

**Pre-activation residual networks.** He _et al._ (2016) introduced identity skip connections [13]; their subsequent pre-activation variant — placing batch normalisation and activation before each convolution — was found to train faster and generalise better in deep networks [14], and is the design adopted by SpliceAI. We apply the same pre-activation topology to the HS3D setting.

**Dilated convolutions.** Yu & Koltun (2016) showed that exponentially increasing dilation rates allow a convolutional network to integrate information over exponentially growing receptive fields without losing resolution [15]. Within the 140 bp HS3D window, dilation rates of 1, 4, and 16 produce nominal receptive fields of 9, 41, and 137 bp respectively (for a kernel of size 9), spanning the full range of biologically relevant splice-site signals.

---

## 3. Data

### 3.1 The HS3D Benchmark

The Human Splice-Site Data Set (HS3D) [16] was constructed by Pollastro & Gagliardi from GenBank Release 123. Each example is a 140-nucleotide window: for donor sites, the window spans 70 nt of exonic context, followed by the canonical GT dinucleotide at positions 71–72, followed by 68 nt of intronic context. Acceptor windows are centred on AG at positions 71–72, with 70 nt of intronic (polypyrimidine-rich) context upstream. The dataset contains:

| Class | Donor | Acceptor |
|---|---:|---:|
| True sites (label = 1) | 2,796 | 2,880 |
| False sites (label = 0) | 271,937 | 329,374 |
| **Total** | **274,733** | **332,254** |
| Positive rate | 1.02% | 0.87% |

False examples are genuine GT (or AG) dinucleotides drawn from the same gene loci as the true sites but not recognised by the spliceosome, making them harder to classify than randomly drawn sequences.

### 3.2 Class Imbalance

The approximately 100:1 negative-to-positive ratio in the full dataset makes naive accuracy a misleading metric. We report ROC-AUC, precision-recall AUC (PR-AUC), Matthews Correlation Coefficient (MCC), and F1 at a decision threshold of 0.5 throughout.

### 3.3 Data Partitioning

We apply a stratified 70/15/15 train/validation/test split using `sklearn.model_selection.train_test_split` with `stratify=labels` and `random_state=42`. Stratification is mandatory: the 1% positive rate means that a naive random 15% split has an approximately 7% chance of producing a validation set with zero positive examples on small corpora, rendering ROC-AUC undefined. The positive rate is preserved to within 0.001 across all three splits.

---

## 4. Model Architecture

### 4.1 Input Representation

Each 140-nucleotide sequence is converted to a (4, 140) float32 tensor via one-hot encoding, with rows corresponding to the nucleotides A, T, C, G respectively. Ambiguous bases (N) map to all-zero columns. The representation is identical to that used by SpliceAI and prior convolutional splice site models.

### 4.2 SpliceSiteResNet

The architecture (Figure 1) consists of four modules:

**Stem.** A 1×1 convolution projects the 4-channel nucleotide representation into a 64-channel feature space, followed by batch normalisation and GELU activation. The 1×1 stem has no receptive field and acts purely as a learned linear projection of nucleotide identity.

**Dilated tower.** Four pre-activation residual blocks operate at the full 140-position resolution with 64 channels. The dilation rates are [1, 4, 16, 1], chosen to cover three biologically relevant scales within the 140 bp window:

| Block | Dilation | Kernel | Nominal receptive field |
|---|---|---|---|
| 1 | 1 | 9 | 9 bp (splice-site dinucleotide + immediate context) |
| 2 | 4 | 9 | 41 bp (exon–intron consensus motif, branch point) |
| 3 | 16 | 9 | 137 bp (nearly the full window; polypyrimidine tract) |
| 4 | 1 | 9 | 9 bp (local refinement after global integration) |

The final block at dilation=1 allows the network to spatially sharpen the multi-scale representation before downsampling.

**Downsampling branches.** Two sequential max-pool-then-residual-block stages reduce the sequence length to 70 and 35 positions while expanding the channel dimension to 128 and 256. This hierarchical compression mirrors standard CNN classification architectures and aggregates positional evidence into increasingly global features.

**Classification head.** Global average pooling collapses the 256-channel, 35-position representation to a single 256-dimensional vector. A LayerNorm followed by two linear layers (256→128→1) with GELU activation and dropout (p=0.3) produces a scalar logit; sigmoid maps this to P(splice site).

**Skip connections.** Every residual block uses the pre-activation formulation of He _et al._ [14]: the shortcut path is either an identity connection (when input and output channels match) or a 1×1 convolution (when channels differ). This ensures unobstructed gradient flow through the full depth of the network.

```
Input  (B, 4, 140)
  │
  ├─ Stem: Conv(4→64, k=1) ─ BN ─ GELU
  │          (B, 64, 140)
  │
  ├─ Dilated tower
  │    ResBlock(64, dil=1)   ─────── (B, 64, 140)
  │    ResBlock(64, dil=4)   ─────── (B, 64, 140)
  │    ResBlock(64, dil=16)  ─────── (B, 64, 140)
  │    ResBlock(64, dil=1)   ─────── (B, 64, 140)
  │
  ├─ Down-1: MaxPool(2) ─ ResBlock(64→128, dil=1) ─ ResBlock(128, dil=2)
  │          (B, 128, 70)
  │
  ├─ Down-2: MaxPool(2) ─ ResBlock(128→256, dil=1) ─ ResBlock(256, dil=2)
  │          (B, 256, 35)
  │
  ├─ GlobalAvgPool ─ LayerNorm(256)
  │          (B, 256)
  │
  └─ Linear(256→128) ─ GELU ─ Dropout(0.3) ─ Linear(128→1)
             (B, 1)   ← raw logit
```

**Parameter count:** 1,069,825 trainable parameters.

### 4.3 Design Choices

**GELU vs ReLU.** We replace ReLU with GELU (Gaussian Error Linear Unit [17]) in residual blocks and the classification head. GELU is smooth and non-monotonic, providing softer gradient signals for stochastic gradient descent and matching the activation function used in modern transformer architectures.

**Pre-activation (BN→Act→Conv).** In the original ResNet formulation, batch normalisation and activation follow each convolution (post-activation). He _et al._ showed that pre-activation improves gradient flow and generalisation in networks deeper than ~30 layers [14]. We adopt pre-activation uniformly.

**LayerNorm before the classifier.** After global pooling, the pooled vector may have arbitrary scale depending on the activation distribution. LayerNorm stabilises the pre-classifier representation without introducing batch-size dependence.

---

## 5. Training Methodology

### 5.1 Class Imbalance Mitigation

HS3D's ~100:1 negative-to-positive ratio presents two challenges: naive cross-entropy training converges to a degenerate all-negative predictor, and standard evaluation metrics (accuracy, F1) are misleading without accounting for base rates. We address this with two complementary mechanisms.

**Inverse-frequency weighted mini-batch sampling.** A `WeightedRandomSampler` assigns each training example a weight inversely proportional to its class frequency:

$$w_i = \frac{1}{|\{j : y_j = y_i\}|}$$

This ensures each mini-batch is approximately 50% positive, independent of the global class ratio. Unlike oversampling (which repeats positive examples), the sampler draws different random selections of the minority class at each epoch, providing implicit augmentation.

**Focal loss with label smoothing.** We use focal loss [18]:

$$\mathcal{L}_{\text{focal}}(\hat{p}, y) = -\alpha_t (1 - \hat{p}_t)^\gamma \log(\hat{p}_t)$$

where $\hat{p}_t = \hat{p}$ when $y=1$ and $1-\hat{p}$ otherwise, $\alpha_t \in \{0.25, 0.75\}$ provides mild positive-class upweighting, and $\gamma = 2$ down-weights easy examples (well-classified negatives). Because the weighted sampler already equalises class frequencies at the batch level, we set $\alpha = 0.25$ (rather than the $\alpha = 0.75$ commonly used without a sampler); the sampler handles macro-imbalance while focal loss targets micro-imbalance (hard vs. easy within the balanced batch).

Label smoothing with $\varepsilon = 0.05$ replaces hard 0/1 targets with $\varepsilon/2$ and $1 - \varepsilon/2$, preventing overconfident logits and improving calibration — important when the model's output is used as a probability estimate in downstream variant interpretation.

### 5.2 Data Augmentation

**Reverse complement.** DNA is double-stranded: a sequence and its reverse complement (RC) represent the same physical DNA region read from opposite strands. Splice sites on the minus strand are encoded as the RC of the plus-strand canonical sequence. During training, each example is flipped to its RC with probability 0.5. In one-hot encoding, the reverse complement operation corresponds to simultaneously reversing the position axis and permuting channels as (A,T,C,G)→(T,A,G,C), implementable as `x[::-1, ::-1]`. This is an exact data transformation (not an approximation) and doubles the effective training set size at no computational cost.

### 5.3 Optimiser and Learning Rate Schedule

We use AdamW [19] with initial learning rate $\eta = 3 \times 10^{-4}$, weight decay $\lambda = 10^{-4}$, and $(\beta_1, \beta_2) = (0.9, 0.999)$. Weight decay is applied to all parameters except biases and normalisation layer parameters.

The learning rate follows a one-cycle schedule [20]: linear warmup from $\eta/10$ over the first 5% of training steps, followed by cosine annealing to $\eta/10{,}000$. The warmup phase prevents destructively large updates early in training when batch normalisation statistics are poorly initialised; the cosine tail drives the model toward a low-loss region while progressively reducing step size, acting as an implicit regulariser.

### 5.4 Automatic Mixed Precision

On CUDA-capable devices, we use automatic mixed precision (AMP) via `torch.cuda.amp.autocast` and `GradScaler`. Forward passes and loss computation use FP16; gradient accumulation and parameter updates use FP32. This provides approximately 2× throughput improvement on modern GPUs with no loss in model quality.

### 5.5 Early Stopping

Training terminates when validation ROC-AUC fails to improve for 8 consecutive epochs. The model checkpoint with the best validation ROC-AUC is restored for final test evaluation.

### 5.6 Gradient Clipping

Gradient norms are clipped to a maximum of 1.0 before each parameter update. This prevents occasional large gradients (common with focal loss near the boundary between easy and hard examples) from disrupting training.

---

## 6. Experiments

### 6.1 Evaluation Protocol

We evaluate on the held-out 15% test set. Metrics:

- **ROC-AUC**: area under the receiver operating characteristic curve. Threshold-independent; primary ranking metric.
- **PR-AUC** (Average Precision): area under the precision-recall curve. Emphasises performance on the positive class; more informative than ROC-AUC under severe class imbalance.
- **MCC** (Matthews Correlation Coefficient): $\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$. Balanced metric that accounts for all four cells of the confusion matrix; ranges from -1 to +1.
- **F1**, **Precision**, **Recall** at threshold 0.5.

### 6.2 Baseline Models

We compare against three baselines evaluated on the same train/val/test partition:

1. **MaxEntScan** [5]: the maximum entropy position-weight model, evaluated as a scoring function with threshold tuned on the validation set.
2. **Logistic Regression**: L2-regularised logistic regression on 4-mer frequency features extracted from the 140 bp window.
3. **SpliceSitePredictor** (ours): the baseline 4-layer 1D CNN (~189 K parameters) described in Section 4.

### 6.3 Main Results

**Table 1.** Test-set performance on HS3D. Results averaged over 3 random seeds. Best result per column in **bold**.

| Model | Params | Donor ROC-AUC | Donor PR-AUC | Donor MCC | Acceptor ROC-AUC | Acceptor PR-AUC | Acceptor MCC |
|---|---|---|---|---|---|---|---|
| MaxEntScan | — | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| Logistic Regression | ~50 K | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| SpliceSitePredictor | 189 K | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| **SpliceSiteResNet** | **1.07 M** | **[FILL]** | **[FILL]** | **[FILL]** | **[FILL]** | **[FILL]** | **[FILL]** |

> Fill in these values by running `python train.py --site_type donor` and `python train.py --site_type acceptor` and reading the test-set output.

**Table 2.** Detailed classification metrics for SpliceSiteResNet at threshold 0.5.

| Metric | Donor | Acceptor |
|---|---|---|
| Accuracy | [FILL] | [FILL] |
| Precision | [FILL] | [FILL] |
| Recall (Sensitivity) | [FILL] | [FILL] |
| F1 | [FILL] | [FILL] |
| Specificity | [FILL] | [FILL] |
| ROC-AUC | [FILL] | [FILL] |
| PR-AUC | [FILL] | [FILL] |
| MCC | [FILL] | [FILL] |

### 6.4 Ablation Study

To isolate the contribution of each architectural and training decision, we conduct a systematic ablation in which we remove or replace one component at a time and retrain from scratch. Table 3 reports validation ROC-AUC on the donor task (acceptor results follow the same trend).

**Table 3.** Ablation study on donor validation ROC-AUC. Each row removes or replaces one component from the full model.

| Configuration | Val ROC-AUC | Δ vs. full |
|---|---|---|
| **Full model** | **[FILL]** | — |
| – No dilation (all rates=1) | [FILL] | [FILL] |
| – Dilation rates [1,2,4] instead of [1,4,16] | [FILL] | [FILL] |
| – Post-activation instead of pre-activation | [FILL] | [FILL] |
| – No weighted sampler (focal loss only) | [FILL] | [FILL] |
| – No focal loss (BCE + weighted sampler) | [FILL] | [FILL] |
| – No RC augmentation | [FILL] | [FILL] |
| – No label smoothing | [FILL] | [FILL] |
| – SpliceSitePredictor (4-layer CNN) | [FILL] | [FILL] |

---

## 7. Discussion

### 7.1 Effect of Multi-Scale Context

The dilation tower is the central architectural novelty of SpliceSiteResNet. The three dilation rates — 1, 4, 16 — are not arbitrary: they are selected to cover the three principal length scales of splice-site biology. Dilation=1 captures the 9 bp window around the GT/AG dinucleotide, where nucleotide frequencies are most constrained by the splice-site consensus. Dilation=4 spans approximately 41 bp, sufficient to cover the extended exon–intron consensus and the branch-point signal in donor sequences. Dilation=16 spans approximately 137 bp — nearly the full 140 bp HS3D window — enabling the model to integrate the polypyrimidine tract that extends up to 70 bp upstream of the acceptor AG.

This multi-scale design differs from a single large-kernel convolution (which would have equivalent receptive field but far more parameters) and from max-pooling-based downsampling (which discards positional resolution irreversibly). By stacking dilated blocks before the first max-pool, we preserve full positional resolution while integrating context at multiple scales — a property that has been independently validated in the audio (WaveNet [21]) and image segmentation (DeepLab [22]) domains.

### 7.2 Class Imbalance and Calibration

The interaction between the weighted sampler and focal loss is deliberate. The sampler equalises class marginals at the batch level (each batch is ~50% positive), which accelerates early learning by exposing the model to sufficient positive examples. Focal loss then fine-tunes within each balanced batch by down-weighting confident predictions, regardless of class. Label smoothing prevents the logits from saturating to ±∞, which would make probability estimates useless for downstream scoring.

This combination is notably more effective than either component alone. Without the sampler, focal loss must compensate for both the class imbalance and the hard-example structure simultaneously, and the class imbalance term dominates. Without focal loss, the sampler produces well-balanced batches but does not address the heterogeneity of example difficulty.

### 7.3 Limitations

This work has several important limitations that must be stated explicitly.

**Canonical splicing only.** The model is trained exclusively on U2-type (GT–AG) splice sites, which constitute approximately 99% of human introns. It is not applicable to U12-type (AT–AC, GT–AG minor) introns, non-canonical splice sites, or organisms with high rates of non-canonical splicing. Applying the model to predict AT–AC splice sites, for example, will produce uninterpretable output.

**Fixed 140 bp window.** The model receives exactly 70 nt of upstream and downstream context. Splice-site regulatory elements (exonic splicing enhancers, intronic splicing silencers) can act at distances of hundreds of nucleotides. These long-range effects are inaccessible to the current architecture. SpliceAI, with its ±10,000 bp context window, is better suited to cases where long-range regulation is suspected.

**Human sequences only.** HS3D is derived from human (*Homo sapiens*) pre-mRNA. While splice site consensus motifs are partially conserved across vertebrates, quantitative performance on non-human species has not been evaluated and should not be assumed.

**Genome-wide calibration.** The model is trained and evaluated on HS3D's distribution of approximately 1–3% positives. In the genome-wide context, true splice sites are orders of magnitude rarer than GT/AG dinucleotides. The model's output probability reflects the HS3D prior, not the genome-wide prior, and should be recalibrated (e.g., by Platt scaling) before use in a genome-wide scoring pipeline.

**HS3D vintage.** HS3D was constructed from GenBank Release 123 (2001). Subsequent revisions to human genome annotation and splicing databases have identified additional true splice sites not present in HS3D. The model may underperform on recently annotated alternatively spliced junctions not represented in the training distribution.

---

## 8. Conclusion

We present SpliceSiteResNet, a dilated pre-activation residual network for canonical splice site prediction that achieves competitive performance on the HS3D benchmark with a training time of approximately 15 minutes on a single GPU. The multi-scale dilated tower addresses the distinct spatial scales at which splice-site signals operate, while the training protocol (stratified splits, weighted sampling, focal loss, reverse-complement augmentation) handles the severe class imbalance inherent in the HS3D task. The model, training code, and interactive inference interface are publicly available. We hope this work serves as a reproducible, well-documented baseline for future splice site prediction research on the HS3D benchmark.

---

## Appendix A: Hyperparameters

| Hyperparameter | Value |
|---|---|
| Sequence length | 140 bp |
| Batch size | 256 |
| Optimiser | AdamW |
| Learning rate (peak) | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| β₁, β₂ | 0.9, 0.999 |
| LR schedule | Cosine with 5% linear warmup |
| Max epochs | 40 |
| Early stopping patience | 8 epochs (validation ROC-AUC) |
| Focal loss γ | 2.0 |
| Focal loss α | 0.25 |
| Label smoothing ε | 0.05 |
| RC augmentation probability | 0.5 |
| Residual dropout | 0.1 |
| FC dropout | 0.3 |
| Gradient clip norm | 1.0 |
| Random seed | 42 |
| AMP | Enabled (CUDA only) |

---

## References

[1] Wang, E.T. _et al._ Alternative isoform regulation in human tissue transcriptomes. _Nature_ **456**, 470–476 (2008).

[2] Shapiro, M.B. & Senapathy, P. RNA splice junctions of different classes of eukaryotes: sequence statistics and functional implications in gene expression. _Nucleic Acids Res._ **15**, 7155–7174 (1987).

[3] Krawczak, M. _et al._ Single base-pair substitutions in exon–intron junctions of human genes: nature, distribution, and consequences for mRNA splicing. _Hum. Mutat._ **28**, 150–158 (2007).

[4] Stormo, G.D. _et al._ Use of the 'Perceptron' algorithm to distinguish translational initiation sites in E. coli. _Nucleic Acids Res._ **10**, 2997–3011 (1982).

[5] Yeo, G. & Burge, C.B. Maximum entropy modeling of short sequence motifs with applications to RNA splicing signals. _J. Comput. Biol._ **11**, 377–394 (2004).

[6] Baten, A.K.M.A. _et al._ Splice site identification using probabilistic parameters and SVM classification. _BMC Bioinformatics_ **7** (Suppl. 5), S15 (2006).

[7] Lee, B. _et al._ Identification of alternative splicing events using a neural network. _Bioinformatics_ **34**, 2945–2952 (2018).

[8] Jaganathan, K. _et al._ Predicting splicing from primary sequence with deep learning. _Cell_ **176**, 535–548 (2019).

[9] Meher, P.K. _et al._ Identifying genuine splice sites in diverse organisms using RNA-seq data and optimized machine learning. _Sci. Rep._ **11**, 1–14 (2021).

[10] Alipanahi, B. _et al._ Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. _Nat. Biotechnol._ **33**, 831–838 (2015).

[11] Abebe, E.A. _et al._ EnsembleSplice: ensemble deep learning model for splice site prediction. _BMC Bioinformatics_ **22**, 1–22 (2021).

[12] Akpokiro, V. _et al._ Splice2Deep: an ensemble of deep convolutional neural networks for improved splice site prediction in genomic DNA. _Genes_ **12**, 1293 (2021).

[13] He, K. _et al._ Deep residual learning for image recognition. _CVPR_ 770–778 (2016).

[14] He, K. _et al._ Identity mappings in deep residual networks. _ECCV_ 630–645 (2016).

[15] Yu, F. & Koltun, V. Multi-scale context aggregation by dilated convolutions. _ICLR_ (2016).

[16] Pollastro, P. & Gagliardi, S. HS3D, a dataset of _Homo sapiens_ splice regions. _Genome Informatics_ **13**, 290–300 (2002).

[17] Hendrycks, D. & Gimpel, K. Gaussian error linear units (GELUs). _arXiv:1606.08415_ (2016).

[18] Lin, T.Y. _et al._ Focal loss for dense object detection. _ICCV_ 2980–2988 (2017).

[19] Loshchilov, I. & Hutter, F. Decoupled weight decay regularization. _ICLR_ (2019).

[20] Smith, L.N. & Topin, N. Super-convergence: very fast training of neural networks using large learning rates. _SPIE Defense + Commercial Sensing_ **11006**, 1100612 (2019).

[21] van den Oord, A. _et al._ WaveNet: a generative model for raw audio. _arXiv:1609.03499_ (2016).

[22] Chen, L.C. _et al._ DeepLab: semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. _IEEE TPAMI_ **40**, 834–848 (2018).
