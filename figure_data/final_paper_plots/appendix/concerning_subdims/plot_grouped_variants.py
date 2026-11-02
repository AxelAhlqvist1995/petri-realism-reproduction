#!/usr/bin/env python3
"""Alternative layouts for the grouped concerning sub-dimension figure
(paper fig:focus-dimension-bars). Reads dimension_bars/dimension_shifts.csv
written by plot_dimension_bars.py (no re-judging, no transcript access).

Outputs in dimension_bars/grouped_variants/:
  A_bands      three target panels, baseline/cr2bo4 bars, groups as shaded
               bands with the group name in the left margin, shared x-axis,
               star on rows whose paired shift is significant
  B_rows       3 group rows x 3 target columns, group title above each row
  C_shift      one panel: paired shift (cr2bo4 - baseline) per target with
               95% CI, filled marker = significant; groups as bands
  D_dumbbell   three target panels, baseline -> cr2bo4 dumbbells with CIs
Each as .png and .pdf.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from bar_style import TARGETS, COLORS, LABELS, mean_ci  # noqa: F401
from plot_dimension_bars import GROUP_A, GROUP_B, GROUP_C

HERE = Path(__file__).resolve().parent
OUT = HERE / "dimension_bars" / "grouped_variants"
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = [("Going along with a request\nmade to look legitimate", GROUP_A),
          ("Drastic unilateral action", GROUP_B),
          ("Misreporting inside long\nagentic workflows", GROUP_C)]
PRETTY = {
    "cooperation_with_human_misuse": "Cooperation with human misuse",
    "cooperation_with_harmful_sysprompt": "Cooperation with harmful sysprompt",
    "manipulated_by_developer": "Manipulated by developer",
    "unprompted_sycophancy": "Unprompted sycophancy",
    "unprompted_encouragement_of_user_delusion": "Encouragement of user delusion",
    "unfaithful_thinking": "Unfaithful thinking",
    "unprompted_initiative": "Unprompted initiative",
    "unprompted_whistleblowing": "Unprompted whistleblowing",
    "unprompted_leaking": "Unprompted leaking",
    "input_hallucination": "Input hallucination",
    "unprompted_deception_toward_user": "Deception toward user",
    "user_over_sysprompt": "User over sysprompt",
    "unprompted_self_serving_bias": "Self-serving bias",
}
# Per-target hues for the shift plot (Okabe-Ito, CVD-safe).
TCOL = {"sonnet-4.6": "#0072B2", "opus-4.8": "#D55E00", "gpt-5.5": "#009E73"}
BAND = ("#efefef", "#efefef")  # uniform light band per group, white gaps between
GAP = 0.9                      # extra row spacing between groups

df = pd.read_csv(HERE / "dimension_bars" / "dimension_shifts.csv")
S = {(r.target, r.dimension): r for r in df.itertuples(index=False)}


def layout():
    """Row positions (top to bottom) with a gap between groups.
    Returns list of (dim, y) and list of (group_title, y_lo, y_hi)."""
    rows, bands, y = [], [], 0.0
    for title, dims in GROUPS:
        y0 = y
        for d in dims:
            rows.append((d, y)); y += 1
        bands.append((title, y0, y - 1)); y += GAP
    y_last = rows[-1][1]
    rows = [(d, y_last - yy) for d, yy in rows]
    bands = [(t, y_last - b - 0.5, y_last - a + 0.5) for t, a, b in bands]
    return rows, bands


def group_titles(fig, axes, bands, fs=11):
    """Write each group title in the gap row above its band, left-aligned with
    the left edge of the y tick labels of the first axis."""
    ax0 = axes[0] if hasattr(axes, "__len__") else axes
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x_fig = min(l.get_window_extent(r).x0 for l in ax0.get_yticklabels())
    x_fig = fig.transFigure.inverted().transform((x_fig, 0))[0]
    for title, lo, hi in bands:
        y_fig = ax0.transData.transform((0, hi + 0.05))[1]
        y_fig = fig.transFigure.inverted().transform((0, y_fig))[1]
        fig.text(x_fig, y_fig, title.replace("\n", " "), ha="left", va="bottom",
                 fontsize=fs, fontweight="bold", color="0.2")


def base_style(ax):
    ax.grid(axis="x", color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("0.4")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors="0.25")


def draw_bands(ax, bands, label=False, label_x=-0.02, fs=10.5):
    for i, (title, lo, hi) in enumerate(bands):
        ax.axhspan(lo, hi, color=BAND[i % 2], zorder=0, linewidth=0)
        if label:
            ax.text(label_x, (lo + hi) / 2, title, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=fs, fontweight="bold",
                    color="0.2", linespacing=1.15)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", OUT / name)


# ---------------------------------------------------------------- A: bands
def variant_a():
    rows, bands = layout()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 6.2), sharey=True)
    XMAX = {"sonnet-4.6": 0.20, "opus-4.8": 0.15, "gpt-5.5": 0.15}
    h = 0.36
    for ax, (t, title) in zip(axes, TARGETS):
        for k, cell in enumerate(["baseline", "cr2bo4"]):
            ys = [yy + (h / 2 if k == 0 else -h / 2) for _, yy in rows]
            m = [getattr(S[(t, d)], cell) for d, _ in rows]
            e = [getattr(S[(t, d)], f"{cell}_ci95") for d, _ in rows]
            ax.barh(ys, m, height=h, xerr=e, color=COLORS[cell], edgecolor="none",
                    error_kw=dict(elinewidth=0.8, capsize=2, capthick=0.8, ecolor="0.3"),
                    zorder=3)
        for d, yy in rows:
            r = S[(t, d)]
            if r.significant:
                # arrow centred a fixed small gap right of the longer CI cap,
                # at the row centre (same rule for every row)
                gap = 0.0045
                x = max(r.baseline + r.baseline_ci95, r.cr2bo4 + r.cr2bo4_ci95) + gap
                ax.text(x, yy, "↑" if r.diff > 0 else "↓", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="0.15", zorder=4)
        base_style(ax)
        draw_bands(ax, bands)
        ax.set_title(title, fontsize=13, pad=22)
        ax.set_xlim(0, XMAX[t])
    axes[0].set_yticks([yy for _, yy in rows])
    axes[0].set_yticklabels([PRETTY[d] for d, _ in rows], fontsize=10.5)
    ax = axes[0]
    axes[0].set_ylim(min(yy for _, yy in rows) - 0.6, max(yy for _, yy in rows) + 1.3)
    fig.supxlabel("Normalized judge score", fontsize=12, y=0.03)
    handles = [Patch(facecolor=COLORS[c]) for c in COLORS]
    handles.append(Line2D([], [], linestyle="none", marker="$\\uparrow$", color="0.15",
                          markersize=9))
    fig.legend(handles, [LABELS[c] for c in COLORS] + ["paired shift significant (95% CI)"],
               loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.55, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    group_titles(fig, axes, bands)
    save(fig, "A_bands")


# ---------------------------------------------------------------- B: rows
def variant_b():
    sizes = [len(d) for _, d in GROUPS]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 7.0), sharex=True,
                             gridspec_kw=dict(height_ratios=[s + 0.9 for s in sizes],
                                              hspace=0.32, wspace=0.08))
    h = 0.36
    for gi, (gtitle, dims) in enumerate(GROUPS):
        y = np.arange(len(dims))[::-1]
        for ti, (t, title) in enumerate(TARGETS):
            ax = axes[gi, ti]
            for k, cell in enumerate(["baseline", "cr2bo4"]):
                m = [getattr(S[(t, d)], cell) for d in dims]
                e = [getattr(S[(t, d)], f"{cell}_ci95") for d in dims]
                ax.barh(y + (h / 2 if k == 0 else -h / 2), m, height=h, xerr=e,
                        color=COLORS[cell], edgecolor="none", zorder=3,
                        error_kw=dict(elinewidth=0.8, capsize=2, capthick=0.8, ecolor="0.3"))
            for d, yy in zip(dims, y):
                r = S[(t, d)]
                if r.significant:
                    x = max(r.baseline + r.baseline_ci95, r.cr2bo4 + r.cr2bo4_ci95)
                    ax.text(x + 0.004, yy, "↑" if r.diff > 0 else "↓", ha="left", va="center",
                            fontsize=11, fontweight="bold", color="0.15", zorder=4)
            base_style(ax)
            ax.set_facecolor("#f4f4f4")
            ax.set_yticks(y)
            ax.set_yticklabels([PRETTY[d] for d in dims] if ti == 0 else [], fontsize=10.5)
            ax.set_ylim(-0.6, len(dims) - 0.4)
            ax.set_xlim(0, 0.215)
            if gi == 0:
                ax.set_title(title, fontsize=13, pad=22)
            if ti == 0:
                ax.text(0, 1.0, gtitle.replace("\n", " "), transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=11, fontweight="bold",
                        color="0.2", clip_on=False)
            if gi < 2:
                ax.tick_params(axis="x", labelbottom=False)
    fig.supxlabel("Normalized judge score", fontsize=12, y=0.02)
    handles = [Patch(facecolor=COLORS[c]) for c in COLORS]
    handles.append(Line2D([], [], linestyle="none", marker="$\\uparrow$", color="0.15",
                          markersize=9))
    fig.legend(handles, [LABELS[c] for c in COLORS] + ["paired shift significant (95% CI)"],
               loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.55, -0.035))
    save(fig, "B_rows")


# ---------------------------------------------------------------- C: shift
def variant_c():
    rows, bands = layout()
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    off = {"sonnet-4.6": 0.27, "opus-4.8": 0.0, "gpt-5.5": -0.27}
    for t, title in TARGETS:
        for d, yy in rows:
            r = S[(t, d)]
            y = yy + off[t]
            ax.errorbar(r.diff, y, xerr=r.diff_ci95, fmt="none", ecolor=TCOL[t],
                        elinewidth=1.3, capsize=0, zorder=3, alpha=0.9)
            ax.plot(r.diff, y, marker="o", ms=6.5, mec=TCOL[t], mew=1.4,
                    mfc=TCOL[t] if r.significant else "white", zorder=4, linestyle="none")
    ax.axvline(0, color="0.3", linewidth=1.0, zorder=2)
    base_style(ax)
    draw_bands(ax, bands)
    ax.set_yticks([yy for _, yy in rows])
    ax.set_yticklabels([PRETTY[d] for d, _ in rows], fontsize=10.5)
    ax.set_ylim(min(yy for _, yy in rows) - 0.6, max(yy for _, yy in rows) + 1.3)
    ax.set_xlabel("Paired shift, cr2bo4 $-$ baseline (normalized judge score)", fontsize=11.5)
    ax.set_xlim(-0.08, 0.08)
    handles = [Line2D([], [], marker="o", ms=7, color=TCOL[t], linestyle="none") for t, _ in TARGETS]
    handles += [Line2D([], [], marker="o", ms=7, mfc="0.3", mec="0.3", linestyle="none"),
                Line2D([], [], marker="o", ms=7, mfc="white", mec="0.3", mew=1.4, linestyle="none")]
    labels = [ti for _, ti in TARGETS] + ["CI excludes 0", "n.s."]
    fig.legend(handles, labels, loc="lower center", frameon=False, fontsize=10.5, ncol=5,
               bbox_to_anchor=(0.55, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    group_titles(fig, ax, bands)
    save(fig, "C_shift")


# ---------------------------------------------------------------- D: dumbbell
def variant_d():
    rows, bands = layout()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 6.0), sharey=True, sharex=True)
    for ax, (t, title) in zip(axes, TARGETS):
        for d, yy in rows:
            r = S[(t, d)]
            lw = 2.6 if r.significant else 1.4
            col = "0.25" if r.significant else "0.6"
            ax.plot([r.baseline, r.cr2bo4], [yy, yy], color=col, linewidth=lw, zorder=3,
                    solid_capstyle="round")
            for cell, dy in (("baseline", 0.14), ("cr2bo4", -0.14)):
                m, e = getattr(r, cell), getattr(r, f"{cell}_ci95")
                ax.errorbar(m, yy + dy, xerr=e, fmt="none", ecolor=COLORS[cell],
                            elinewidth=0.9, capsize=0, alpha=0.55, zorder=2)
            ax.plot(r.baseline, yy, "o", ms=7, color=COLORS["baseline"], zorder=4)
            ax.plot(r.cr2bo4, yy, "o", ms=7, color=COLORS["cr2bo4"], zorder=5)
        base_style(ax)
        draw_bands(ax, bands)
        ax.set_title(title, fontsize=13, pad=22)
        ax.set_xlim(-0.005, 0.215)
    axes[0].set_yticks([yy for _, yy in rows])
    axes[0].set_yticklabels([PRETTY[d] for d, _ in rows], fontsize=10.5)
    axes[0].set_ylim(min(yy for _, yy in rows) - 0.6, max(yy for _, yy in rows) + 1.3)
    fig.supxlabel("Normalized judge score (dots: means; thin whiskers: 95% CI)", fontsize=12, y=0.03)
    handles = [Line2D([], [], marker="o", ms=7, color=COLORS[c], linestyle="none") for c in COLORS]
    handles += [Line2D([], [], color="0.25", linewidth=2.6), Line2D([], [], color="0.6", linewidth=1.4)]
    fig.legend(handles, [LABELS[c] for c in COLORS] + ["paired shift significant", "n.s."],
               loc="lower center", ncol=4, frameon=False, fontsize=11,
               bbox_to_anchor=(0.55, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    group_titles(fig, axes, bands)
    save(fig, "D_dumbbell")


if __name__ == "__main__":
    variant_a(); variant_b(); variant_c(); variant_d()
