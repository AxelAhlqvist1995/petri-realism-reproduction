#!/usr/bin/env python3
"""Appendix (Compute matched baseline): whole-transcript measures.

Unlike the branch-selection figures, this judges the ENTIRE multibranch audit as
one transcript (opus-5 whole-transcript judge) and compares it against the
single-branch Baseline and $cr2bo4$ and the multibranch first branch (b00). No
realism win rate (it is per single-branch and undefined for the whole multibranch
audit); concerning is the large leftmost panel, with eval awareness and seed
adherence stacked on the right.

Per measure a seed is kept only where all four methods have a value (identical
seed-instruction distribution within each panel). Title-less (paper styling);
Ns in the CSV sidecar.

Output: multibranch_heldout/wholetranscript_measures.{png,csv}
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_CWR = _HERE.parent / "cost_winrate_reasoning"
if str(_CWR) not in sys.path:
    sys.path.insert(0, str(_CWR))

from plot_method_comparison_bars import Z_95, mean_and_sem  # noqa: E402
from plot_method_comparison_bars_multibranch import (  # noqa: E402
    MULTIBRANCH_CELL,
    MULTIBRANCH_TARGET,
)
from plot_method_comparison_bars_multibranch_pd_selected import (  # noqa: E402
    load_per_seed_opus5,
)

OUT = _HERE / "multibranch_heldout"
DPI = 200

# bar order + colours (baseline/cr2bo4/first branch match the other section
# figures; whole-transcript multibranch gets its own colour)
BARS = [
    ("Baseline", "baseline", "#1f77b4"),
    ("cr2bo4", "cr2bo4", "#ff7f0e"),
    ("Multibranch (first branch)", "_first", "#7f7f7f"),
    ("Multibranch (whole transcript)", "multibranch", "#8c564b"),
]
# concerning is the big leftmost panel; the other two stack on the right
MEASURES = [("concerning", "Concerning"), ("eval_awareness", "Eval awareness"),
            ("s_seed_adherence", "Seed adherence")]


def _first_branch_measures():
    """b00 per-branch concerning/eval_awareness/s_seed_adherence (normalised)."""
    out = {mk: {} for mk, _ in MEASURES}
    with open(MULTIBRANCH_CELL / "branch_judging" / "summary.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["branch_index"] != "0":
                continue
            s = int(r["seed"])
            for mk, _ in MEASURES:
                v = r.get(mk, "")
                if v not in ("", "None"):
                    out[mk][s] = (float(v) - 1.0) / 9.0
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    T = MULTIBRANCH_TARGET
    mkeys = [mk for mk, _ in MEASURES]
    conds = [("baseline", "Baseline", "#1f77b4"), ("cr2bo4", "cr2bo4", "#ff7f0e"),
             ("multibranch", "Multibranch", "#8c564b")]
    data, _present = load_per_seed_opus5(conds, mkeys)
    fb = _first_branch_measures()

    def series(cond_key, mk):
        return fb[mk] if cond_key == "_first" else data.get((T, cond_key, mk), {})

    # per-measure fairness intersection + stats
    stats = {}  # mk -> list of (label, mean, ci, color); plus n
    ns = {}
    for mk, _disp in MEASURES:
        seeds = set.intersection(*[set(series(ck, mk)) for _l, ck, _c in BARS])
        seeds = sorted(seeds)
        ns[mk] = len(seeds)
        row = []
        for label, ck, color in BARS:
            vals = [series(ck, mk)[s] for s in seeds]
            m, sem = mean_and_sem(vals)
            row.append((label, m, sem * Z_95, color))
        stats[mk] = row

    # ---- layout: concerning big on the left, EA + seed adherence stacked right ----
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], wspace=0.28, hspace=0.25)
    ax_big = fig.add_subplot(gs[:, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_br = fig.add_subplot(gs[1, 1])
    panels = [("concerning", ax_big, 15), ("eval_awareness", ax_tr, 13),
              ("s_seed_adherence", ax_br, 13)]
    disp = dict(MEASURES)

    for mk, ax, fs in panels:
        ax.set_facecolor("#f0f0f0")
        top = 0.0
        for i, (label, m, ci, color) in enumerate(stats[mk]):
            ax.bar(i, m, width=0.72, color=color, edgecolor="black", linewidth=0.8,
                   yerr=ci, capsize=3,
                   error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "black"},
                   zorder=3)
            top = max(top, m + ci)
        ax.set_xticks(range(len(BARS)))
        ax.set_xticklabels([])
        ax.set_ylabel(disp[mk], fontsize=fs)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(top * 1.15, 0.05))

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="black", label=l) for l, _k, c in BARS]
    fig.legend(handles, [l for l, _k, _c in BARS], loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=12, ncol=4)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    png = OUT / "wholetranscript_measures.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with open(OUT / "wholetranscript_measures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measure", "bar", "mean", "ci_half", "n"])
        for mk, _d in MEASURES:
            for label, m, ci, _c in stats[mk]:
                w.writerow([mk, label, m, ci, ns[mk]])

    print(f"Saved {png}")
    for mk, _d in MEASURES:
        print(f"  {mk} (n={ns[mk]}): " + ", ".join(f"{l}={m:.3f}" for l, m, _ci, _c in stats[mk]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
