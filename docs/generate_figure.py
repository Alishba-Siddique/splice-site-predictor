"""Generate publication-quality architecture figure for SpliceSiteResNet."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(10, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis("off")

# ── Colour palette ────────────────────────────────────────────────────────────
C_INPUT  = "#E3F2FD"   # light blue  – input/output
C_STEM   = "#E8F5E9"   # light green – stem
C_DILATE = "#FFF3E0"   # light amber – dilated tower
C_POOL   = "#F3E5F5"   # light purple – pooling
C_DOWN   = "#FCE4EC"   # light pink  – downsampling
C_HEAD   = "#E0F7FA"   # light cyan  – classifier head
EDGE     = "#37474F"   # dark grey
TXT      = "#212121"   # near-black

def box(ax, x, y, w, h, label, sublabel="", color="#FFFFFF", fontsize=9, subsize=7.5):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.06",
                          facecolor=color, edgecolor=EDGE, linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w/2, cy + 0.12, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=TXT, zorder=4)
        ax.text(x + w/2, cy - 0.22, sublabel, ha="center", va="center",
                fontsize=subsize, color="#546E7A", zorder=4, style="italic")
    else:
        ax.text(x + w/2, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=TXT, zorder=4)

def arrow(ax, x, y1, y2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=EDGE,
                                lw=1.4, mutation_scale=12),
                zorder=2)

def bracket(ax, x1, x2, y, label, color="#78909C"):
    """Horizontal brace-style label for a group."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0, linestyle="dashed"))
    ax.text((x1+x2)/2, y + 0.05, label, ha="center", va="bottom",
            fontsize=7, color=color, style="italic")

BX, BW, BH = 1.5, 7.0, 0.6   # box left-x, width, height
CX = BX + BW/2                # centre-x for arrows
GAP = 0.18                    # gap between boxes
CUR = 19.0                    # current y (top-down)

def step(h=BH):
    global CUR
    top = CUR
    CUR -= h
    return top

# ── INPUT ─────────────────────────────────────────────────────────────────────
y = step(); box(ax, BX, y-BH, BW, BH, "Input", "(B, 4, 140)  one-hot DNA", C_INPUT)
arrow(ax, CX, y-BH, y-BH-GAP); CUR -= GAP

# ── STEM ──────────────────────────────────────────────────────────────────────
y = step(); box(ax, BX, y-BH, BW, BH,
                "Stem  Conv1d(4→64, k=1) + BN + GELU",
                "output: (B, 64, 140)", C_STEM)
arrow(ax, CX, y-BH, y-BH-GAP); CUR -= GAP

# ── DILATED TOWER ─────────────────────────────────────────────────────────────
tower_top = CUR
DBH = 0.52
dblocks = [
    ("ResBlock(64, k=9,  dil=1)",  "RF ≈  9 bp  |  splice-site dinucleotide context"),
    ("ResBlock(64, k=9,  dil=4)",  "RF ≈ 41 bp  |  consensus motif + branch point"),
    ("ResBlock(64, k=9,  dil=16)", "RF ≈ 137 bp |  polypyrimidine tract (full window)"),
    ("ResBlock(64, k=9,  dil=1)",  "RF ≈  9 bp  |  local spatial refinement"),
]
for i, (lbl, sub) in enumerate(dblocks):
    y = CUR
    box(ax, BX, y-DBH, BW, DBH, lbl, sub, C_DILATE, fontsize=8.5, subsize=7)
    CUR -= DBH
    if i < len(dblocks)-1:
        arrow(ax, CX, CUR, CUR-GAP*0.6); CUR -= GAP*0.6
tower_bot = CUR

# side brace label for tower
ax.annotate("", xy=(BX-0.25, tower_bot), xytext=(BX-0.25, tower_top),
            arrowprops=dict(arrowstyle="-", color="#FB8C00", lw=2))
ax.text(BX-0.55, (tower_top+tower_bot)/2, "Dilated\nTower",
        ha="center", va="center", fontsize=8, color="#FB8C00",
        fontweight="bold", rotation=90)

arrow(ax, CX, CUR, CUR-GAP); CUR -= GAP

# ── MAXPOOL 1 ─────────────────────────────────────────────────────────────────
y = CUR; PH = 0.42
box(ax, BX, y-PH, BW, PH, "MaxPool1d(2)  →  (B, 64, 70)", color=C_POOL, fontsize=8.5)
CUR -= PH; arrow(ax, CX, CUR, CUR-GAP); CUR -= GAP

# ── DOWN BRANCH 1 ─────────────────────────────────────────────────────────────
d1_top = CUR
for lbl, sub in [("ResBlock(64→128, k=7, dil=1)", "channel expansion + local features"),
                 ("ResBlock(128,     k=7, dil=2)", "RF ≈ 27 bp  |  broader context")]:
    y = CUR
    box(ax, BX, y-DBH, BW, DBH, lbl, sub, C_DOWN, fontsize=8.5, subsize=7)
    CUR -= DBH; arrow(ax, CX, CUR, CUR-GAP*0.6); CUR -= GAP*0.6
d1_bot = CUR

ax.annotate("", xy=(BX+BW+0.25, d1_bot), xytext=(BX+BW+0.25, d1_top),
            arrowprops=dict(arrowstyle="-", color="#E91E63", lw=2))
ax.text(BX+BW+0.55, (d1_top+d1_bot)/2, "Down-1\n(B,128,70)",
        ha="center", va="center", fontsize=7.5, color="#E91E63",
        fontweight="bold", rotation=90)

arrow(ax, CX, CUR, CUR-GAP); CUR -= GAP

# ── MAXPOOL 2 ─────────────────────────────────────────────────────────────────
box(ax, BX, CUR-PH, BW, PH, "MaxPool1d(2)  →  (B, 128, 35)", color=C_POOL, fontsize=8.5)
CUR -= PH; arrow(ax, CX, CUR, CUR-GAP); CUR -= GAP

# ── DOWN BRANCH 2 ─────────────────────────────────────────────────────────────
d2_top = CUR
for lbl, sub in [("ResBlock(128→256, k=5, dil=1)", "channel expansion to 256"),
                 ("ResBlock(256,      k=5, dil=2)", "RF ≈ 17 bp  |  final spatial integration")]:
    y = CUR
    box(ax, BX, y-DBH, BW, DBH, lbl, sub, C_DOWN, fontsize=8.5, subsize=7)
    CUR -= DBH; arrow(ax, CX, CUR, CUR-GAP*0.6); CUR -= GAP*0.6
d2_bot = CUR

ax.annotate("", xy=(BX+BW+0.25, d2_bot), xytext=(BX+BW+0.25, d2_top),
            arrowprops=dict(arrowstyle="-", color="#E91E63", lw=2))
ax.text(BX+BW+0.55, (d2_top+d2_bot)/2, "Down-2\n(B,256,35)",
        ha="center", va="center", fontsize=7.5, color="#E91E63",
        fontweight="bold", rotation=90)

arrow(ax, CX, CUR, CUR-GAP); CUR -= GAP

# ── GLOBAL AVG POOL ───────────────────────────────────────────────────────────
box(ax, BX, CUR-PH, BW, PH,
    "AdaptiveAvgPool1d(1)  +  LayerNorm(256)  →  (B, 256)",
    color=C_POOL, fontsize=8.5)
CUR -= PH; arrow(ax, CX, CUR, CUR-GAP); CUR -= GAP

# ── HEAD ──────────────────────────────────────────────────────────────────────
head_blocks = [
    ("Linear(256→128)  +  GELU  +  Dropout(0.3)", ""),
    ("Linear(128→1)", ""),
]
for lbl, sub in head_blocks:
    box(ax, BX, CUR-DBH, BW, DBH, lbl, sub, C_HEAD, fontsize=8.5)
    CUR -= DBH; arrow(ax, CX, CUR, CUR-GAP*0.6); CUR -= GAP*0.6

# ── OUTPUT ────────────────────────────────────────────────────────────────────
box(ax, BX, CUR-BH, BW, BH,
    "Output  (B, 1)  —  logit",
    "sigmoid(logit) → P(splice site)",
    C_INPUT)

# ── LEGEND ────────────────────────────────────────────────────────────────────
legend_items = [
    (C_INPUT,  "Input / Output"),
    (C_STEM,   "Stem projection"),
    (C_DILATE, "Dilated residual block"),
    (C_POOL,   "Pooling / normalisation"),
    (C_DOWN,   "Downsampling residual block"),
    (C_HEAD,   "Classification head"),
]
for i, (col, lbl) in enumerate(legend_items):
    lx, ly = 0.15, 1.15 - i*0.25
    rect = FancyBboxPatch((lx, ly), 0.35, 0.18,
                          boxstyle="round,pad=0.02",
                          facecolor=col, edgecolor=EDGE, linewidth=0.8, zorder=3)
    ax.add_patch(rect)
    ax.text(lx+0.45, ly+0.09, lbl, va="center", fontsize=7.5, color=TXT)

ax.text(5, -0.15,
        "Figure 1. SpliceSiteResNet architecture. "
        "RF = nominal receptive field (kernel size 9, formula: k + (k−1)×(d−1)).\n"
        "All ResBlocks use pre-activation (BN→GELU→Conv→BN→GELU→Conv) with skip connection.\n"
        "Total trainable parameters: 1,069,825.",
        ha="center", va="top", fontsize=7.5, color="#546E7A",
        wrap=True, style="italic")

ax.set_title("SpliceSiteResNet Architecture", fontsize=13, fontweight="bold",
             pad=8, color=TXT)

plt.tight_layout(rect=[0, 0.04, 1, 1])
out = "docs/architecture.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
