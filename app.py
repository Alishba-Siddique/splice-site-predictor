"""
Gradio Space: Splice Site Predictor
Predicts canonical GT (donor) and AG (acceptor) splice sites
in user-supplied DNA sequences using a trained 1D CNN.
"""
from __future__ import annotations

import io
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch

from model import SpliceSitePredictor
from utils import predict_on_sequence, plot_prediction_track

# ── Model loading ─────────────────────────────────────────────────────────────

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN    = 140
HALF       = 70


def _load(path: str) -> tuple[SpliceSitePredictor, bool]:
    """Load model from *path*; return (model, is_trained)."""
    model = SpliceSitePredictor(seq_len=SEQ_LEN).to(DEVICE)
    if Path(path).exists():
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        return model, True
    model.eval()
    return model, False


donor_model,    donor_ok    = _load("splice_model_donor.pt")
acceptor_model, acceptor_ok = _load("splice_model_acceptor.pt")

_UNTRAINED_MSG = ""
if not (donor_ok and acceptor_ok):
    missing = []
    if not donor_ok:    missing.append("`splice_model_donor.pt`")
    if not acceptor_ok: missing.append("`splice_model_acceptor.pt`")
    _UNTRAINED_MSG = (
        f"> **Warning:** {' and '.join(missing)} not found. "
        "Predictions are from a randomly-initialised model and will be meaningless. "
        "Run `python train.py --site_type donor` (and `acceptor`) to train the model, "
        "then restart this app."
    )


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(
    sequence: str,
    site_type: str,
    threshold: float,
) -> tuple[object, pd.DataFrame, str]:
    """Gradio callback: returns (plot, table, status_string)."""
    # ── Input validation ──
    seq = sequence.strip().upper().replace("\n", "").replace(" ", "").replace("\r", "")
    if not seq:
        return None, _empty_table(), "Please enter a DNA sequence."

    min_len = 2 * HALF + 2
    if len(seq) < min_len:
        return None, _empty_table(), f"Sequence must be at least {min_len} bp (got {len(seq)})."

    invalid = set(seq) - set("ATCGN")
    if invalid:
        return None, _empty_table(), f"Invalid characters detected: {sorted(invalid)}"

    model = donor_model if site_type == "donor" else acceptor_model
    dinucleotide = "GT" if site_type == "donor" else "AG"

    # ── Run model ──
    positions, probs = predict_on_sequence(
        model, seq,
        site_type=site_type,
        half_window=HALF,
        device=DEVICE,
    )

    # ── Build figure ──
    fig = plot_prediction_track(seq, positions, probs, site_type=site_type, threshold=threshold)

    # ── Build results table ──
    rows = []
    for pos, prob in zip(positions, probs):
        if prob >= threshold:
            rows.append({
                "Position (1-based)": pos + 1,
                "Dinucleotide":       seq[pos : pos + 2],
                "Probability":        f"{prob:.4f}",
                "Call":               "Splice site ✓",
            })
    rows.sort(key=lambda r: -float(r["Probability"]))
    table = pd.DataFrame(rows) if rows else _empty_table()

    status = (
        f"Scanned {len(positions)} {dinucleotide} position(s) in a {len(seq):,} bp sequence. "
        f"Predicted {len(rows)} splice site(s) above threshold {threshold:.2f}."
    )
    return fig, table, status


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Position (1-based)", "Dinucleotide", "Probability", "Call"]
    )


# ── Example sequences ─────────────────────────────────────────────────────────

# Each entry: [sequence, site_type, threshold]
EXAMPLES = [
    [
        # Strong canonical donor site (GT surrounded by exon/intron consensus)
        "CATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCAGGTAAGTATTTAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATA",
        "donor",
        0.5,
    ],
    [
        # Strong canonical acceptor site (polypyrimidine tract + AG)
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCA",
        "acceptor",
        0.5,
    ],
    [
        # Mini-gene: two donor sites, two acceptor sites
        (
            "ATGGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG"
            "GTAAGTATTTATTTAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATCAG"
            "CATCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG"
            "GTAAGTATTTATTTAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATAAATCAG"
            "CATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAGCATCAG"
        ),
        "donor",
        0.5,
    ],
]


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_CSS = """
.gradio-container { font-family: 'Inter', sans-serif; max-width: 1200px; margin: auto; }
#title { text-align: center; }
"""

with gr.Blocks(title="Splice Site Predictor", css=_CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# 🧬 Splice Site Predictor\n"
        "Predict canonical **GT–AG** splice donor and acceptor sites in DNA sequences "
        "using a 1D convolutional neural network trained on the "
        "[HS3D dataset](http://www.sci.unich.it/~ambesi/genomics/hs3d/).",
        elem_id="title",
    )

    if _UNTRAINED_MSG:
        gr.Markdown(_UNTRAINED_MSG)

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=1, min_width=340):
            seq_box = gr.Textbox(
                label="DNA Sequence (A / T / C / G / N)",
                placeholder="Paste at least 142 bp of genomic DNA here …",
                lines=9,
                max_lines=30,
            )
            with gr.Row():
                site_radio = gr.Radio(
                    choices=["donor", "acceptor"],
                    value="donor",
                    label="Site type",
                )
                threshold_slider = gr.Slider(
                    minimum=0.10, maximum=0.95, value=0.50, step=0.05,
                    label="Confidence threshold",
                )
            predict_btn = gr.Button("Predict Splice Sites", variant="primary", size="lg")
            status_box  = gr.Textbox(label="Status", interactive=False, lines=2)

        # ── Right column: outputs ─────────────────────────────────────────────
        with gr.Column(scale=2, min_width=500):
            output_plot  = gr.Plot(label="Prediction track")
            output_table = gr.Dataframe(
                label="Predicted splice sites (above threshold)",
                headers=["Position (1-based)", "Dinucleotide", "Probability", "Call"],
                row_count=(6, "dynamic"),
                interactive=False,
            )

    predict_btn.click(
        fn=predict,
        inputs=[seq_box, site_radio, threshold_slider],
        outputs=[output_plot, output_table, status_box],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[seq_box, site_radio, threshold_slider],
        label="Try an example",
    )

    gr.Markdown("""
---
### How it works
1. **Paste** a DNA sequence (minimum 142 bp; N allowed for ambiguous bases).
2. **Select** site type — *donor* scans every **GT**, *acceptor* scans every **AG**.
3. **Adjust** the threshold (default 0.50); lower it to find weak sites, raise it to reduce false positives.
4. **Click** Predict.

Each candidate dinucleotide is scored by extracting a 140 bp window (70 bp upstream + 70 bp downstream)
and passing it through the CNN. Positions lacking sufficient flanking sequence are skipped.

### Model
Four-layer 1D CNN (Conv → BN → ReLU × 4, then two fully-connected layers).
~189 K trainable parameters. Trained with focal loss to handle the ~5 : 1 class imbalance in HS3D.
See the [README](README.md) for the full architecture, training procedure, and evaluation metrics.

### Limitations
- Trained only on canonical **GT–AG** splice sites. Non-canonical (AT–AC, GC–AG) sites are not predicted.
- Fixed 140 bp window size matches the HS3D training distribution; very short flanking regions receive no prediction.
- Validated on human sequences only. Performance on non-human species is not characterised.
- Does not model long-range splicing regulatory elements (ESEs, ESSs, ISEs, ISSs).
    """)

if __name__ == "__main__":
    demo.launch()
