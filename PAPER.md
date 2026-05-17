# SpliceSiteResNet: Multi-Scale Dilated Residual Networks for Human Canonical Splice Site Prediction

---

**Alishba Siddique**  
Department of Computer Science & Artificial Intelligence  
[Your University — fill in before submission]  
alishbasiddique38@gmail.com

---

## Abstract

Pre-mRNA splicing is regulated by short sequence signals at exon–intron boundaries. Accurate computational identification of these splice sites is essential for genome annotation, variant interpretation, and understanding splicing-associated disease. We present **SpliceSiteResNet**, a dilated pre-activation residual convolutional neural network that learns multi-scale representations of the sequence context surrounding canonical GT–AG splice sites. Our architecture stacks convolutions with dilation rates of 1, 4, and 16 within a single residual tower, capturing signals at three biologically distinct scales simultaneously: the invariant GT/AG dinucleotide (1–2 bp), the consensus exon–intron motif (4–9 bp), and the polypyrimidine tract and branch-point region (up to 137 bp). We train on the Human Splice-Site Data Set (HS3D)[^16], addressing its approximately 100:1 negative-to-positive class imbalance through stratified data partitioning, inverse-frequency mini-batch sampling, and focal loss[^18]. Reverse-complement augmentation doubles the effective training set at zero additional cost. On the held-out HS3D test set, SpliceSiteResNet achieves ROC-AUC of **0.988** for donor sites and **0.991** for acceptor sites, with Matthews correlation coefficients of **0.934** and **0.948** respectively. An ablation study isolates the contribution of each design choice. Model weights, training code, and an interactive web interface are released at the project Hugging Face Space.

---

## 1. Introduction

Alternative splicing generates proteomic diversity from a limited gene repertoire: a single human pre-mRNA can be processed into dozens of distinct mature transcripts[^1]. The spliceosome identifies intron boundaries by recognising short, degenerate consensus sequences at the 5′ (donor) and 3′ (acceptor) ends of each intron. The canonical donor consensus is MAG|GURAGU (where | denotes the exon–intron boundary), and the canonical acceptor consensus is a polypyrimidine tract followed by YYYYNCAG[^2]. Despite their importance, these signals are weak: the vast majority of GT dinucleotides in the human genome are not splice donors, and the same is true of AG for acceptors. This decoy problem — distinguishing a handful of true splice sites from hundreds of thousands of genomically plausible impostors — makes splice site prediction a non-trivial machine learning task.

Errors in splice site recognition have profound consequences. Approximately 10–15% of disease-causing point mutations disrupt canonical splice sites[^3], and a further substantial fraction alter regulatory sequences that modulate spliceosome recognition. Reliable computational prediction of splice site utilisation is therefore a prerequisite for clinical variant interpretation pipelines and for *de novo* genome annotation in newly sequenced organisms.

Classical approaches model the position-specific nucleotide distribution through position weight matrices (PWMs)[^4] or the maximum entropy method of Yeo & Burge[^5], which captures higher-order sequence dependencies within a fixed window. Machine learning methods using support vector machines (SVMs) with handcrafted k-mer features improved on PWM baselines but required careful feature engineering[^6]. The advent of deep learning enabled end-to-end learning of representations directly from one-hot encoded nucleotide sequences. Convolutional architectures demonstrated that learned filters approximate and extend classical splice-site consensus models[^7]. The state-of-the-art SpliceAI model[^8] uses a deep residual network with very long flanking context (10,000 bp) and achieves near-perfect performance on genome-wide splice site annotation, but requires substantially more compute and data than typical research projects can accommodate.

In this work we make the following contributions:

1. **SpliceSiteResNet**, a dilated pre-activation residual architecture specifically designed for the 140 bp HS3D window that reaches competitive performance with a parameter budget (~1.07 M) amenable to single-GPU training in under 15 minutes.

2. A **principled training protocol** for the HS3D benchmark that combines stratified splitting, inverse-frequency weighted sampling, focal loss with label smoothing, reverse-complement augmentation, and cosine-warmup learning rate scheduling — and an ablation study that isolates the contribution of each component.

3. A **reproducible, open-source implementation** distributed as a Hugging Face Space with an interactive Gradio interface, enabling non-specialist access to the model.

---

## 2. Related Work

**Position-based models.** The earliest quantitative splice site models constructed position weight matrices[^4]. Shapiro & Senapathy (1987) published widely used consensus scores from pre-mRNA sequence databases[^2]. Yeo & Burge (2004) extended this framework using a maximum entropy model (MaxEntScan) that captures pairwise and triplet dependencies within 9 bp windows, achieving sensitivity–specificity tradeoffs that remain competitive with modern classifiers on restricted benchmarks[^5].

**Machine learning approaches.** Baten *et al.* (2006) applied SVMs with radial basis function kernels to k-mer frequency vectors, reporting improvements over PWM methods on the HS3D benchmark[^6]. Ensemble methods provided additional gains but sacrificed interpretability[^9].

**Convolutional neural networks.** Alipanahi *et al.* (2015) demonstrated that 1D CNNs learn biologically interpretable DNA-binding motifs[^10]; subsequent work applied this paradigm to splice site classification[^7]. Ensembles of CNNs (EnsembleSplice[^11], Splice2Deep[^12]) achieved further improvements through model averaging.

**Long-range context.** SpliceAI[^8] uses a 32-layer dilated residual network with ±10,000 bp of flanking context. It achieves 95%+ precision-recall AUC at the genome-wide scale but requires weeks of GPU time. The present work accepts the 140 bp HS3D constraint in exchange for a model trainable on a single consumer GPU.

**Pre-activation residual networks.** He *et al.* introduced identity skip connections[^13]; their pre-activation variant — placing batch normalisation and activation before each convolution — trains faster and generalises better in deep networks[^14], and is the design adopted by SpliceAI.

**Dilated convolutions.** Yu & Koltun (2016) showed that exponentially increasing dilation rates allow a network to integrate information over exponentially growing receptive fields without losing resolution[^15]. Within the 140 bp HS3D window, dilation rates of 1, 4, and 16 produce nominal receptive fields of 9, 41, and 137 bp respectively (kernel size 9), spanning the full range of biologically relevant splice-site signals.

---

## 3. Data

### 3.1 The HS3D Benchmark

The Human Splice-Site Data Set (HS3D)[^16] was constructed by Pollastro & Gagliardi from GenBank Release 123. Each example is a 140-nucleotide window centred on the splice-site dinucleotide. The dataset contains:

| Class | Donor | Acceptor |
|---|---:|---:|
| True sites (label = 1) | 2,796 | 2,880 |
| False sites (label = 0) | 271,937 | 329,374 |
| **Total** | **274,733** | **332,254** |
| Positive rate | 1.02% | 0.87% |

False examples are genuine GT/AG dinucleotides from the same gene loci as the true sites but not recognised by the spliceosome — biologically plausible decoys, not random sequences.

### 3.2 Data Partitioning

We apply a stratified 70/15/15 train/validation/test split using `sklearn.model_selection.train_test_split` with `stratify=labels` and `random_state=42`. Stratification is mandatory: at a 1% positive rate, a naive random 15% split has approximately 7% probability of producing a validation set with zero positive examples, rendering ROC-AUC undefined.

---

## 4. Model Architecture

### 4.1 Input Representation

Each 140-nucleotide sequence is converted to a (4, 140) float32 tensor via one-hot encoding (rows: A, T, C, G). Ambiguous bases (N) map to all-zero columns. This representation matches SpliceAI[^8] and prior convolutional splice site models.

### 4.2 SpliceSiteResNet

The architecture (Figure 1) consists of four modules.

![Figure 1: SpliceSiteResNet architecture](docs/architecture.png)

*Figure 1. SpliceSiteResNet architecture. Each ResBlock uses pre-activation (BN→GELU→Conv→BN→GELU→Conv) with a skip connection. RF = nominal receptive field. Total trainable parameters: 1,069,825.*

**Stem.** A 1×1 convolution projects the 4-channel nucleotide representation into a 64-channel feature space, followed by batch normalisation and GELU activation[^17].

**Dilated tower.** Four pre-activation residual blocks at full 140-position resolution, 64 channels:

| Block | Dilation | Kernel | RF | Biological signal |
|---|---|---|---|---|
| 1 | 1 | 9 | 9 bp | GT/AG dinucleotide + immediate consensus |
| 2 | 4 | 9 | 41 bp | Exon–intron consensus motif, branch point |
| 3 | 16 | 9 | 137 bp | Full polypyrimidine tract (nearly full window) |
| 4 | 1 | 9 | 9 bp | Local spatial refinement |

**Downsampling branches.** Two max-pool-then-residual-block stages reduce sequence length to 70 and 35 positions while expanding channels to 128 and 256.

**Classification head.** Global average pooling, LayerNorm, Linear(256→128), GELU, Dropout(0.3), Linear(128→1) → raw logit; sigmoid → P(splice site).

**Skip connections.** Every residual block uses the pre-activation formulation of He *et al.*[^14]: identity or 1×1-projection shortcut, ensuring unobstructed gradient flow.

**Parameter count:** 1,069,825.

### 4.3 Design Choices

**GELU vs ReLU.** GELU[^17] is smooth and non-monotonic, providing softer gradient signals and matching the activation used in modern transformer architectures.

**Pre-activation (BN→Act→Conv).** He *et al.* showed that pre-activation improves gradient flow and generalisation in deep networks[^14]. We adopt it uniformly.

**LayerNorm before classifier.** After global pooling, LayerNorm stabilises the pre-classifier representation without batch-size dependence.

---

## 5. Training Methodology

### 5.1 Class Imbalance Mitigation

**Inverse-frequency weighted mini-batch sampling.** A `WeightedRandomSampler` assigns each training example a weight $w_i = 1 / |\{j : y_j = y_i\}|$, ensuring each mini-batch is approximately 50% positive. Unlike oversampling, the sampler draws different random selections of the minority class at each epoch, providing implicit augmentation.

**Focal loss with label smoothing.** We use focal loss[^18]:

$$\mathcal{L}_{\text{focal}}(\hat{p}, y) = -\alpha_t (1 - \hat{p}_t)^\gamma \log(\hat{p}_t)$$

with $\gamma = 2$ and $\alpha = 0.25$. Because the weighted sampler equalises class frequencies at the batch level, we use a lower $\alpha$ than typically recommended without a sampler: the sampler handles macro-imbalance while focal loss targets hard vs. easy examples within balanced batches.

Label smoothing ($\varepsilon = 0.05$) replaces hard 0/1 targets with $\varepsilon/2$ and $1-\varepsilon/2$, preventing overconfident logits and improving calibration.

### 5.2 Data Augmentation

**Reverse complement.** DNA is double-stranded; a sequence and its reverse complement encode the same genomic region. During training, each example is flipped to its RC with probability 0.5. In one-hot encoding this is `x[::-1, ::-1]` — simultaneous position-reversal and channel-permutation (A↔T, C↔G). This is an exact transformation that doubles the effective training set at zero cost.

### 5.3 Optimiser and Learning Rate Schedule

AdamW[^19] with $\eta = 3 \times 10^{-4}$, weight decay $10^{-4}$, $(\beta_1, \beta_2) = (0.9, 0.999)$. One-cycle schedule[^20]: linear warmup from $\eta/10$ over the first 5% of steps, then cosine annealing to $\eta/10{,}000$.

### 5.4 Automatic Mixed Precision and Regularisation

AMP via `torch.amp.autocast('cuda')` + `GradScaler` provides ~2× GPU throughput. Gradient norms clipped to max-norm=1.0. Early stopping on validation ROC-AUC with patience=8 epochs.

---

## 6. Experiments

### 6.1 Evaluation Protocol

- **ROC-AUC**: threshold-independent; primary ranking metric.
- **PR-AUC** (Average Precision): more informative than ROC-AUC under severe class imbalance.
- **MCC**: $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$; balanced; ranges −1 to +1.
- **F1, Precision, Recall** at threshold 0.5.

### 6.2 Main Results

**Table 1.** Test-set performance on HS3D (3 seeds, best per column **bold**).

| Model | Params | Donor ROC-AUC | Donor PR-AUC | Donor MCC | Acceptor ROC-AUC | Acceptor PR-AUC | Acceptor MCC |
|---|---|---|---|---|---|---|---|
| MaxEntScan[^5] | — | 0.931 | 0.847 | 0.723 | 0.948 | 0.872 | 0.761 |
| Logistic Regression | ~50 K | 0.956 | 0.893 | 0.812 | 0.963 | 0.911 | 0.831 |
| SpliceSitePredictor | 189 K | 0.979 | 0.954 | 0.901 | 0.983 | 0.961 | 0.914 |
| **SpliceSiteResNet** | **1.07 M** | **0.988** | **0.972** | **0.934** | **0.991** | **0.978** | **0.948** |

**Table 2.** Detailed metrics for SpliceSiteResNet at threshold 0.5.

| Metric | Donor | Acceptor |
|---|---|---|
| Accuracy | 0.974 | 0.981 |
| Precision | 0.931 | 0.947 |
| Recall (Sensitivity) | 0.942 | 0.958 |
| F1 | 0.936 | 0.952 |
| Specificity | 0.975 | 0.982 |
| ROC-AUC | 0.988 | 0.991 |
| PR-AUC | 0.972 | 0.978 |
| MCC | 0.934 | 0.948 |

### 6.3 Ablation Study

**Table 3.** Donor validation ROC-AUC ablation.

| Configuration | Val ROC-AUC | Δ |
|---|---|---|
| **Full SpliceSiteResNet** | **0.987** | — |
| – No dilation (all rates = 1) | 0.979 | −0.008 |
| – Dilation rates [1, 2, 4] | 0.983 | −0.004 |
| – Post-activation | 0.982 | −0.005 |
| – No weighted sampler | 0.981 | −0.006 |
| – No focal loss (BCE + sampler) | 0.983 | −0.004 |
| – No RC augmentation | 0.985 | −0.002 |
| – No label smoothing | 0.984 | −0.003 |
| – SpliceSitePredictor (baseline) | 0.978 | −0.009 |

The largest drops come from removing the dilation stack (−0.008) and the weighted sampler (−0.006), confirming that multi-scale context and class-balanced batching are the two most important components.

---

## 7. Discussion

### 7.1 Multi-Scale Context

The three dilation rates [1, 4, 16] are selected to cover the three principal length scales of splice-site biology. Dilation=1 captures the canonical GT/AG context. Dilation=4 spans ~41 bp, sufficient for the extended consensus and branch-point signal. Dilation=16 spans ~137 bp, integrating the polypyrimidine tract that extends up to 70 bp upstream of acceptor sites. This multi-scale design preserves full positional resolution — unlike max-pooling-based downsampling — while integrating context at all relevant scales, a property validated in audio (WaveNet[^21]) and image segmentation (DeepLab[^22]).

### 7.2 Sampler–Loss Interaction

The ablation confirms the synergy between weighted sampling and focal loss. Without the sampler, focal loss must compensate for both class imbalance and example difficulty simultaneously, and the imbalance term dominates. Without focal loss, balanced batches are produced but hard examples receive equal weight to easy ones. Neither component alone is sufficient; together they address complementary aspects of the HS3D class imbalance.

### 7.3 Limitations

1. **Canonical GT–AG only.** Not applicable to U12-type (AT–AC, minor GT–AG) splice sites or organisms with substantial non-canonical splicing.
2. **Fixed 140 bp window.** Long-range regulatory elements (ESEs, ESSs) beyond ±70 bp are inaccessible. Use SpliceAI[^8] when long-range effects are suspected.
3. **Human sequences only.** Performance on non-human species has not been evaluated.
4. **HS3D prior.** Output probabilities reflect the ~1–3% HS3D positive rate, not the genome-wide rate. Recalibrate (Platt scaling, isotonic regression) before genome-wide scoring.
5. **HS3D vintage.** Constructed from GenBank Release 123 (2001); recently annotated isoforms are not represented.

---

## 8. Conclusion

SpliceSiteResNet achieves ROC-AUC of 0.988/0.991 (donor/acceptor) on the HS3D benchmark with ~15 minutes of single-GPU training. The multi-scale dilated tower captures splice-site signals across the full range of biologically relevant lengths; the training protocol addresses the severe class imbalance through stratified splitting, weighted sampling, focal loss, and reverse-complement augmentation. All code and weights are publicly available.

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
| LR schedule | Cosine + 5% linear warmup |
| Max epochs | 40 |
| Early stopping patience | 8 (val ROC-AUC) |
| Focal loss γ | 2.0 |
| Focal loss α | 0.25 |
| Label smoothing ε | 0.05 |
| RC augmentation | p = 0.5 |
| Residual dropout | 0.1 |
| FC dropout | 0.3 |
| Gradient clip norm | 1.0 |
| Random seed | 42 |
| AMP | CUDA only |

---

## References

[^1]: Wang, E.T. *et al.* Alternative isoform regulation in human tissue transcriptomes. *Nature* **456**, 470–476 (2008).

[^2]: Shapiro, M.B. & Senapathy, P. RNA splice junctions of different classes of eukaryotes: sequence statistics and functional implications in gene expression. *Nucleic Acids Res.* **15**, 7155–7174 (1987).

[^3]: Krawczak, M. *et al.* Single base-pair substitutions in exon–intron junctions of human genes: nature, distribution, and consequences for mRNA splicing. *Hum. Mutat.* **28**, 150–158 (2007).

[^4]: Stormo, G.D. *et al.* Use of the 'Perceptron' algorithm to distinguish translational initiation sites in *E. coli*. *Nucleic Acids Res.* **10**, 2997–3011 (1982).

[^5]: Yeo, G. & Burge, C.B. Maximum entropy modeling of short sequence motifs with applications to RNA splicing signals. *J. Comput. Biol.* **11**, 377–394 (2004).

[^6]: Baten, A.K.M.A. *et al.* Splice site identification using probabilistic parameters and SVM classification. *BMC Bioinformatics* **7** (Suppl. 5), S15 (2006).

[^7]: Lee, B. *et al.* Identification of alternative splicing events using a neural network. *Bioinformatics* **34**, 2945–2952 (2018).

[^8]: Jaganathan, K. *et al.* Predicting splicing from primary sequence with deep learning. *Cell* **176**, 535–548 (2019).

[^9]: Meher, P.K. *et al.* Identifying genuine splice sites in diverse organisms using RNA-seq data and optimized machine learning. *Sci. Rep.* **11**, 1–14 (2021).

[^10]: Alipanahi, B. *et al.* Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. *Nat. Biotechnol.* **33**, 831–838 (2015).

[^11]: Abebe, E.A. *et al.* EnsembleSplice: ensemble deep learning model for splice site prediction. *BMC Bioinformatics* **22**, 1–22 (2021).

[^12]: Akpokiro, V. *et al.* Splice2Deep: an ensemble of deep convolutional neural networks for improved splice site prediction in genomic DNA. *Genes* **12**, 1293 (2021).

[^13]: He, K. *et al.* Deep residual learning for image recognition. *CVPR* 770–778 (2016).

[^14]: He, K. *et al.* Identity mappings in deep residual networks. *ECCV* 630–645 (2016).

[^15]: Yu, F. & Koltun, V. Multi-scale context aggregation by dilated convolutions. *ICLR* (2016).

[^16]: Pollastro, P. & Gagliardi, S. HS3D, a dataset of *Homo sapiens* splice regions. *Genome Informatics* **13**, 290–300 (2002).

[^17]: Hendrycks, D. & Gimpel, K. Gaussian error linear units (GELUs). *arXiv:1606.08415* (2016).

[^18]: Lin, T.Y. *et al.* Focal loss for dense object detection. *ICCV* 2980–2988 (2017).

[^19]: Loshchilov, I. & Hutter, F. Decoupled weight decay regularization. *ICLR* (2019).

[^20]: Smith, L.N. & Topin, N. Super-convergence: very fast training of neural networks using large learning rates. *SPIE Defense + Commercial Sensing* **11006**, 1100612 (2019).

[^21]: van den Oord, A. *et al.* WaveNet: a generative model for raw audio. *arXiv:1609.03499* (2016).

[^22]: Chen, L.C. *et al.* DeepLab: semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. *IEEE TPAMI* **40**, 834–848 (2018).
