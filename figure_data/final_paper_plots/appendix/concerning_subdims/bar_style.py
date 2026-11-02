"""Shared styling for the baseline-vs-cr2bo4 horizontal bar figures."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PANEL_BG = "#f0f0f0"
COLORS = {"baseline": "tab:blue", "cr2bo4": "tab:orange"}
LABELS = {"baseline": "Baseline", "cr2bo4": "cr2bo4"}
TARGETS = [("sonnet-4.6", "Sonnet 4.6"), ("opus-4.8", "Opus 4.8"), ("gpt-5.5", "GPT-5.5")]
# Four-target variant (adds Gemini 3.5 Flash; used by the *_four_models figures).
TARGETS4 = TARGETS + [("gemini-3.5-flash", "Gemini 3.5 Flash")]


def mean_ci(x: np.ndarray) -> tuple[float, float]:
    """Mean and 95% CI half-width (1.96 * SEM)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else float("nan"), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(n))


def style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.grid(axis="x", color="white", linewidth=1.2)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def three_panel_bars(stats, dims, titles, out_base, xlabel="Judge score",
                     figsize=(11, 5.2), label_fs=11, dpi=200, sharex=False):
    """stats[(target, cell, dim)] = (mean, ci). One panel per target, rows = dims
    (top to bottom), two bars per row: baseline above cr2bo4.
    A dim starting with "## " is a group header row: no bars, bold label, and a
    thin separator line above it (except for the first row)."""
    fig, axes = plt.subplots(1, len(titles), figsize=figsize, sharey=True,
                             sharex=sharex)
    h = 0.36
    y = np.arange(len(dims))[::-1]
    is_hdr = [d.startswith("## ") for d in dims]
    for ax, (target, title) in zip(axes, titles):
        for k, cell in enumerate(["baseline", "cr2bo4"]):
            m = [0 if hd else stats[(target, cell, d)][0] for d, hd in zip(dims, is_hdr)]
            e = [0 if hd else stats[(target, cell, d)][1] for d, hd in zip(dims, is_hdr)]
            ax.barh(y + (h / 2 if k == 0 else -h / 2), m, height=h, xerr=e,
                    color=COLORS[cell], edgecolor="black", linewidth=0.6,
                    error_kw=dict(elinewidth=0.9, capsize=2.5, capthick=0.9))
        style(ax)
        ax.set_title(title, fontsize=14)
        ax.set_yticks(y)
        ax.set_yticklabels([d[3:] if hd else d for d, hd in zip(dims, is_hdr)], fontsize=label_fs)
        for lab, hd in zip(ax.get_yticklabels(), is_hdr):
            if hd:
                lab.set_fontweight("bold")
                lab.set_fontstyle("italic")
        for i, hd in enumerate(is_hdr):
            if hd and i > 0:
                ax.axhline(y[i] + 0.5, color="0.55", linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_xlim(left=0)
        ax.margins(y=0.04)
    fig.supxlabel(xlabel, fontsize=13, y=0.08)
    handles = [Patch(facecolor=COLORS[c], edgecolor="black") for c in COLORS]
    fig.legend(handles, [LABELS[c] for c in COLORS], loc="lower center",
               ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_base + ".{png,pdf}")
