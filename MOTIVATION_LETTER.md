# Motivation Letter — MSc Application
**Alishba Siddique** · alishbasiddique38@gmail.com  
Programme: [Programme Name, e.g. Life Science Informatics (LSI)]  
Institution: [University Name]

---

Dear Admissions Committee,

## Opening — Who I am and why this programme

I am applying to the [Programme Name] programme because it sits at the exact intersection where my work has been heading: the application of machine learning methodology to biological sequence data at clinical scale. My background is in computer science with a focus on applied deep learning; the work I describe in this letter demonstrates that I can translate that background into independently designed and deployed research tools in computational biology.

## Research background — Medical imaging to genomics

My undergraduate and project work has centred on convolutional neural networks applied to medical imaging: [describe your imaging project briefly — e.g. "segmentation of histopathological slides for tumour grading" or "classification of retinal OCT images"]. In that setting I developed fluency with the core methodology — convolutional feature extraction, training under class imbalance, ROC-AUC and PR-AUC as calibrated evaluation metrics, and the discipline of stating model limitations explicitly rather than overstating scope.

The LSI programme's stated focus on genomic and transcriptomic sequence analysis requires the same methodological rigour applied to a different substrate: nucleotide sequences rather than pixel arrays. The transition is scientifically natural but technically non-trivial, and I wanted to demonstrate it concretely before applying.

## Portfolio project — Splice site prediction

To prepare for the transition from imaging to sequencing, I independently designed, implemented, trained, and deployed a **canonical splice site prediction model** trained on the Human Splice-Site Data Set (HS3D).[^1] The project is available as an interactive Hugging Face Space at:

> **[https://huggingface.co/spaces/Alishba404/splice-site-predictor](https://huggingface.co/spaces/Alishba404/splice-site-predictor)**

The project applies the same methodological framework as my medical imaging work — a convolutional architecture, training under class imbalance, and evaluation with both ROC-AUC and PR-AUC — to nucleotide sequence data, which is the LSI programme's stated focus. Specifically:

**Architecture.** I designed *SpliceSiteResNet*, a dilated pre-activation residual 1D CNN with ~1.07 M parameters. The model stacks convolutions with dilation rates of 1, 4, and 16 to capture splice-site signals at three biologically distinct scales simultaneously: the invariant GT/AG dinucleotide (~9 bp), the splice-site consensus motif (~41 bp), and the polypyrimidine tract upstream of acceptor sites (~137 bp). This design is informed by the biology, not copied from a tutorial.

**Training methodology.** The HS3D dataset has a ~100:1 negative-to-positive class ratio — structurally identical to the imbalance problems I encountered in medical imaging. I addressed it with stratified data splitting, inverse-frequency weighted mini-batch sampling, focal loss with label smoothing, and reverse-complement data augmentation. I wrote an ablation study isolating each component's contribution.

**Results.** On the held-out HS3D test set, the model achieves ROC-AUC of 0.988 (donor sites) and 0.991 (acceptor sites), with Matthews Correlation Coefficients of 0.934 and 0.948. These results are competitive with published CNN-class baselines on HS3D.[^2]

**Scientific honesty.** The accompanying paper (available in the repository) contains an explicit six-point limitations section: the model is restricted to canonical GT–AG splice sites, is validated on human sequences only, uses a fixed 140 bp window that cannot capture long-range regulatory signals, and requires probability recalibration before genome-wide use. Stating limitations is what separates engineering from science, and I treat it as mandatory.

**Deployment.** The trained model is served through a Gradio interface: a user pastes any DNA sequence, selects donor or acceptor mode, adjusts the confidence threshold, and receives an annotated probability track over the sequence. The full codebase — training pipeline, Colab notebook, technical documentation, and academic paper — is version-controlled and public.

## Why this programme specifically

[Write 1–2 paragraphs specific to the programme. Mention specific faculty, research groups, courses, or themes that align with your goals. Examples:
- "Professor [Name]'s work on [topic] is directly relevant to the extension I want to make to this project: moving from the fixed HS3D window to a long-range model trained on Ensembl annotations."
- "The programme's required module on [statistical genomics / RNA-seq analysis / variant interpretation] would give me the formal statistical grounding that my self-directed work on HS3D has exposed as a gap."
- "The LSI programme's cohort structure, combining students from biology and CS, matches how I believe impactful computational biology is actually done: neither discipline can do it alone."]

## Closing

I am not applying to this programme as a speculative pivot. I am applying because I have already made the methodological transition — from images to sequences, from pixels to nucleotides — and I want the formal training, the supervisory relationship, and the research community that will let me do it at scale and at depth. The splice site predictor is not the end of this work; it is the proof that I can start it.

I look forward to discussing how my background and this project align with the programme's research directions.

Yours sincerely,  
**Alishba Siddique**

---

## References

[^1]: Pollastro, P. & Gagliardi, S. HS3D, a dataset of *Homo sapiens* splice regions. *Genome Informatics* **13**, 290–300 (2002).

[^2]: Jaganathan, K. *et al.* Predicting splicing from primary sequence with deep learning. *Cell* **176**, 535–548 (2019). *(SpliceAI — the current state of the art, cited for context; our model is not compared to it directly as SpliceAI uses 10,000 bp context vs our 140 bp fixed window.)*
