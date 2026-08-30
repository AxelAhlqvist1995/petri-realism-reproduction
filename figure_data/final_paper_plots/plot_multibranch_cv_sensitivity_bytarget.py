#!/usr/bin/env python3
"""Sensitivity histogram (cv_sensitivity_full_grid) side by side per target.

Same figure as plot_multibranch_cv_sensitivity.py, twice: left panel target
sonnet-4.6 (cell baseline_multibranch_320), right panel target opus-4.8
(cell baseline_multibranch_320). Each panel is self-contained (its own seed set,
x-range and inline strategy lines) because the two targets live on very
different win-rate scales.

Implementation: cv_heldout_hist() in plot_multibranch_measure_optimization is
sonnet-hardcoded via the module-level MULTIBRANCH_CELL / MULTIBRANCH_TARGET;
this script rebinds those (and clears the grid cache) per target — the same
data the per-target bar figures use.

Output: multibranch_heldout/cv_sensitivity_full_grid_bytarget.{png,csv}
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

import plot_multibranch_measure_optimization as _mo  # noqa: E402
from plot_multibranch_cv_sensitivity import HELD_OUT_COLOR, HELD_OUT_LABEL  # noqa: E402

OUT = _HERE / "multibranch_heldout"
DPI = 200

_PAIRWISE_ROOT = _mo.MULTIBRANCH_CELL.parents[2]
TARGETS = [  # (target, display, multibranch cell) — the 320-turn runs
    ("sonnet-4.6", "Sonnet 4.6",
     _PAIRWISE_ROOT / "target_sonnet-4.6" / "auditor_sonnet-4.6"
     / "baseline_multibranch_320"),
    ("opus-4.8", "Opus 4.8",
     _PAIRWISE_ROOT / "target_opus-4.8" / "auditor_sonnet-4.6"
     / "baseline_multibranch_320"),
]

INLINE = {
    "Baseline": "Baseline",
    "cr2bo4": "cr2bo4",
    "Multibranch (optimize RWR)": "Optimize RWR",
    "Multibranch (first branch)": "First branch",
}


def hist_for_target(target: str, cell: Path):
    _mo.MULTIBRANCH_TARGET = target
    _mo.MULTIBRANCH_CELL = cell
    _mo._CVGRID_CACHE.clear()  # branch seeds + grid results are cell-specific
    return _mo.cv_heldout_hist()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    results = [(t, disp, *hist_for_target(t, cell)) for t, disp, cell in TARGETS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("white")
    for ax, (target, display, dist, lines) in zip(axes, results):
        ax.set_facecolor("#f0f0f0")
        ho_mean = st.mean(dist)
        ax.hist(dist, bins=40, color=HELD_OUT_COLOR, edgecolor="black",
                zorder=3)
        ax.axvline(ho_mean, color="black", lw=2, zorder=6)
        ymax = ax.get_ylim()[1]
        for lbl, val, col in lines:
            if lbl == HELD_OUT_LABEL:
                continue
            ax.axvline(val, color=col, lw=2, ls="--", zorder=5)
            ax.text(
                val, ymax * 0.5, INLINE.get(lbl, lbl), rotation=90,
                va="center", ha="center", color=col, fontsize=11,
                fontweight="bold", zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                          pad=1.5),
            )
        ax.set_title("Held-out realism win rate (per 5/5 split out of 10/10)",
                     fontsize=13)
        ax.set_xlabel(display, fontsize=14)
        ax.set_ylabel("# of splits", fontsize=12)
        ax.grid(axis="y", color="white", lw=0.8)
        ax.set_axisbelow(True)

    fig.tight_layout()
    png = OUT / "cv_sensitivity_full_grid_bytarget.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {png}")

    with open(OUT / "cv_sensitivity_full_grid_bytarget.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "quantity", "value"])
        for target, _display, dist, lines in results:
            w.writerow([target, "n_splits", len(dist)])
            w.writerow([target, "heldout_mean", st.mean(dist)])
            w.writerow([target, "heldout_std", st.pstdev(dist)])
            w.writerow([target, "heldout_min", min(dist)])
            w.writerow([target, "heldout_max", max(dist)])
            for lbl, val, _col in lines:
                w.writerow([target, lbl, val])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
