#!/usr/bin/env python3
"""Appendix: sensitivity of the multibranch held-out realism win rate (paper style).

Title-less. Shows the SAME information as the heldout_family bar figure, plus how
the choice of which comparisons you hold out affects the estimate: the held-out
realism win rate is a DISTRIBUTION over the full independent grid of balanced 5/5
splits (C(10,5)^2 = 63,504 splits, cc and wildchat free to differ), while the
other strategies — optimize RWR, first branch, baseline, cr2bo4 — are scored on
the full 10/10 and so are single values (vertical lines). Same seed set, labels
and colours as heldout_family.png (via cv_heldout_hist).

Output: multibranch_heldout/cv_sensitivity_full_grid.{png,csv}
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_CWR = _HERE.parent / "cost_winrate_reasoning"
if str(_CWR) not in sys.path:
    sys.path.insert(0, str(_CWR))

from plot_multibranch_measure_optimization import cv_heldout_hist  # noqa: E402

OUT = _HERE / "multibranch_heldout"
DPI = 200
HELD_OUT_LABEL = "Multibranch (held-out RWR)"
HELD_OUT_COLOR = "#7570b3"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    dist, lines = cv_heldout_hist()
    ho_mean = st.mean(dist)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f0f0f0")

    # held-out RWR = the distribution (same purple as its bar in heldout_family);
    # its black mean line needs no label (the purple mass is self-evident).
    ax.hist(dist, bins=40, color=HELD_OUT_COLOR, edgecolor="black", zorder=3)
    ax.axvline(ho_mean, color="black", lw=2, zorder=6)

    # the other strategies: single values on the full 10/10 (dashed vertical lines),
    # labelled inline (rotated, mid-height) instead of via an obstructive legend.
    INLINE = {
        "Baseline": "Baseline",
        "cr2bo4": "cr2bo4",
        "Multibranch (optimize RWR)": "Optimize RWR",
        "Multibranch (first branch)": "First branch",
    }
    ymax = ax.get_ylim()[1]
    for lbl, val, col in lines:
        if lbl == HELD_OUT_LABEL:
            continue
        ax.axvline(val, color=col, lw=2, ls="--", zorder=5)
        ax.text(
            val, ymax * 0.5, INLINE.get(lbl, lbl), rotation=90, va="center",
            ha="center", color=col, fontsize=11, fontweight="bold", zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        )

    ax.set_xlabel("Held-out realism win rate (per 5/5 split out of 10/10)", fontsize=12)
    ax.set_ylabel("# of splits", fontsize=12)
    ax.grid(axis="y", color="white", lw=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    png = OUT / "cv_sensitivity_full_grid.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with open(OUT / "cv_sensitivity_full_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        w.writerow(["n_splits", len(dist)])
        w.writerow(["heldout_mean", ho_mean])
        w.writerow(["heldout_std", st.pstdev(dist)])
        w.writerow(["heldout_min", min(dist)])
        w.writerow(["heldout_max", max(dist)])
        for lbl, val, _col in lines:
            w.writerow([lbl, val])

    print(f"Saved {png}")
    print(f"full grid: n_splits={len(dist)} held-out mean={ho_mean:.4f} std={st.pstdev(dist):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
