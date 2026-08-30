#!/usr/bin/env python3
"""APPENDIX FIGURE — logprob vs pairwise depth scaling on Sonnet 4.6.

Two panels sharing the cost-relative-to-own-baseline x-axis: realism win
rate (left, with rung labels + legend) and seed adherence (right, singular
judge score from 0_transcripts.csv, normalized (raw-1)/9) for the two
critique-refinement selector versions on target sonnet-4.6 — the PAIRWISE
protocol (auditor Sonnet 4.6, outputs/reasoning, costs priced per the shared
loaders' COST_MODE — "corrected" correct-implementation estimates by
default, matching the main-text scaling figures) and the LOGPROB protocol
(auditor Opus 4.7, outputs/ablation_protocol_no_stopping, analytical cost
model). Every dot of
both curves is averaged over the SAME seed set (the intersection of the
seeds scored under every rung of both ladders), and each curve's costs are
normalised by ITS OWN baseline, so the two protocols are compared per
compute multiple rather than per dollar. Colours match the cr4
selector-comparison figure (pairwise green, logprob orange); each
protocol's depth-x-breadth ladder (cr2 -> cr2bo2 -> ...) is drawn in a
darker tone of the same colour, as in the scaling figures, labelled but
kept out of the legend.

Output: logprob_vs_pairwise_sonnet/{plot.png, summary.csv}
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CWR = HERE.parents[2] / "cost_winrate_reasoning"
for p in (CWR, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import logprob_cells as lp  # noqa: E402
from plot_cost_winrate_reasoning_sonnet import (  # noqa: E402
    REPO_ROOT,
    Z_95,
    cost_by_seed,
    mean_and_sem,
    winrate_by_seed,
)
from plot_cost_grid_reasoning import (  # noqa: E402
    baseline_cost_by_seed,
    score_by_seed,
)

TARGET = "sonnet-4.6"
OUT_DIR = HERE / "logprob_vs_pairwise_sonnet"

PAIRWISE_BASE = REPO_ROOT / "outputs" / "reasoning"


def pairwise_cell(cond: str) -> Path:
    if cond == "baseline":
        return PAIRWISE_BASE / "baseline" / f"target_{TARGET}" / f"auditor_{TARGET}"
    return (
        PAIRWISE_BASE / "pairwise" / f"target_{TARGET}" / f"auditor_{TARGET}" / cond
    )


def pairwise_cost(cell: Path) -> dict[int, float]:
    # The baseline cell has no cost_logs; its measured cost comes from the
    # cached .eval streaming pricer used by the scaling figures.
    if cell.name.startswith("auditor_"):
        return baseline_cost_by_seed(cell)
    return cost_by_seed(cell)


def _darken(color: str, factor: float = 0.58) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    return tuple(c * factor for c in to_rgb(color))


def logprob_cell(cond: str) -> Path:
    return lp.cell(TARGET, cond)


# (label, colour, depth rungs, mix rungs, cell fn, cost fn); the mix
# ladder branches off cr2 and is drawn in the darkened tone, legend-less.
PROTOCOLS = [
    ("Pairwise", "tab:green",
     ["baseline", "cr1", "cr2", "cr4", "cr8", "cr16"],
     ["cr2", "cr2bo2", "cr2bo4"],
     pairwise_cell, pairwise_cost),
    ("Logprob", "tab:orange",
     ["baseline", "cr1", "cr2", "cr4", "cr8", "cr11", "cr22"],
     ["cr2", "cr2bo2", "cr2bo4", "cr2bo8"],
     logprob_cell, lp.analytical_cost_by_seed),
]

YLABEL_FONTSIZE = 24
TICK_FONTSIZE = 17
ANNOTATION_FONTSIZE = 16
LEGEND_FONTSIZE = 21
FIG_SIZE = (17, 6.6)
DPI = 200

# Rung labels drawn to the right of their dot (the two curves criss-cross,
# so top-left labels would collide with the other curve).
LABEL_RIGHT = {"cr22", "cr16", "cr11", "cr2bo8"}
# The logprob mix curve threads between the two depth curves; its labels go
# under the dots. (protocol label, rung) -> below-dot placement.
LABEL_BELOW = {("Logprob", "cr2bo2"), ("Logprob", "cr2bo4"),
               ("Pairwise", "cr2"), ("Pairwise", "cr2bo2"),
               ("Pairwise", "cr2bo4")}


def main() -> int:
    # Per-measure values: realism win rate and the singular seed-adherence
    # judge score (0_transcripts.csv `s_seed_adherence`, normalized (raw-1)/9
    # like the other scaling figures).
    wr: dict[tuple[str, str], dict[int, float]] = {}
    sa: dict[tuple[str, str], dict[int, float]] = {}
    for label, _c, rungs, mix, cell_fn, _cost in PROTOCOLS:
        for cond in {*rungs, *mix}:
            wr[(label, cond)] = winrate_by_seed(cell_fn(cond))
            sa[(label, cond)] = score_by_seed(
                cell_fn(cond), "s_seed_adherence", normalize=True
            )

    def common_seeds(values: dict[tuple[str, str], dict[int, float]]) -> list[int]:
        common: set[int] | None = None
        for d in values.values():
            common = set(d) if common is None else common & set(d)
        return sorted(common or set())

    wr_seeds = common_seeds(wr)
    sa_seeds = common_seeds(sa)
    print(f"[seeds] winrate common: n={len(wr_seeds)}  "
          f"seed-adherence common: n={len(sa_seeds)}")

    # Costs (x positions) are shared across panels: mean over the winrate
    # seed set, normalised per ladder by its own baseline.
    cost_norms: dict[tuple[str, str], float] = {}
    for label, _c, rungs, mix, cell_fn, cost_fn in PROTOCOLS:
        base_cost = None
        for cond in dict.fromkeys(rungs + mix):
            costs = cost_fn(cell_fn(cond))
            cvals = [costs[s] for s in wr_seeds if s in costs]
            mean_cost = sum(cvals) / len(cvals)
            if cond == "baseline":
                base_cost = mean_cost
            cost_norms[(label, cond)] = mean_cost / base_cost

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    rows = []
    panels = [
        (axes[0], wr, wr_seeds, "Realism win rate", True),
        (axes[1], sa, sa_seeds, "Seed adherence", False),
    ]
    for ax, values, seeds, ylabel, annotate in panels:
        ax.set_facecolor("#f0f0f0")
        base_ys = []
        for label, color, rungs, mix, cell_fn, _cost_fn in PROTOCOLS:
            stats: dict[str, tuple[float, float, float]] = {}
            for cond in dict.fromkeys(rungs + mix):
                m, sem = mean_and_sem([values[(label, cond)][s] for s in seeds])
                stats[cond] = (cost_norms[(label, cond)], m, sem * Z_95)
                rows.append([ylabel, label, cond, cost_norms[(label, cond)],
                             m, sem * Z_95, len(seeds)])
            base_ys.append(stats["baseline"][1])
            for series, series_color, legend in (
                (rungs, color, label),
                (mix, _darken(color), None),
            ):
                xs = [stats[c][0] for c in series]
                ys = [stats[c][1] for c in series]
                ax.plot(xs, ys, color=series_color, linewidth=2.2, alpha=0.55,
                        zorder=2, label=legend if annotate else None)
                for cond in series:
                    if legend is None and not cond.startswith("cr2bo"):
                        continue  # the mix anchor (cr2) is drawn by the depth pass
                    x, y, e = stats[cond][0], stats[cond][1], stats[cond][2]
                    is_base = cond == "baseline"
                    ax.errorbar(
                        x, y, yerr=e, fmt="o", color=series_color,
                        ecolor=series_color,
                        markersize=10 if is_base else 7,
                        markerfacecolor="white" if is_base else series_color,
                        markeredgecolor=series_color,
                        markeredgewidth=2.0 if is_base else 0,
                        capsize=3, elinewidth=1, zorder=4 if is_base else 3,
                    )
                    if not is_base and annotate:
                        va = "bottom"
                        if (label, cond) in LABEL_BELOW:
                            xytext, ha, va = (0, -12), "center", "top"
                        elif cond in LABEL_RIGHT:
                            xytext, ha = (8, 7), "left"
                        else:
                            xytext, ha = (-8, 7), "right"
                        ax.annotate(cond, (x, y), xytext=xytext,
                                    textcoords="offset points",
                                    fontsize=ANNOTATION_FONTSIZE,
                                    color=series_color,
                                    ha=ha, va=va, zorder=5)
        if annotate:
            # One shared "Baselines" callout above the pair of open dots at 1x.
            ax.annotate("Baselines", (1.0, max(base_ys)), xytext=(30, 34),
                        textcoords="offset points", fontsize=ANNOTATION_FONTSIZE,
                        color="black", ha="center", va="bottom", zorder=6)
            ax.legend(loc="upper left", fontsize=18, frameon=False,
                      handlelength=1.4)

        ax.set_xscale("log")
        # Match the sonnet sweep's tick set (32x dropped: too close to 45x max).
        xticks = [1, 2, 4, 8, 16, 45]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t}x" for t in xticks])
        ax.minorticks_off()
        ax.set_xlabel("Cost relative to own baseline", fontsize=YLABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(color="white", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "plot.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measure", "protocol", "folder", "cost_norm", "mean",
                    "ci95_half", "n"])
        w.writerows(rows)
    print(f"Saved {OUT_DIR / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
