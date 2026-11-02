#!/usr/bin/env python3
"""PAPER FIGURE — concerning-score distributions, baseline vs cr2bo4.

Recreates the paper's `concerning_score_distributions_combined.png` (the old
version was built on another machine and pooled both judges' scores without
distinguishing them). Data: raw 1-10 `concerning` rubric scores from BOTH
judges (claude-opus-5 and gpt-5.6-sol) on the same cells as paper Fig. 2
(`cr_comparison`): baseline and cr2bo4, targets Sonnet 4.6 / Opus 4.8 /
GPT-5.5, via the shared `judge_scores.json` cache (rebuild with
`python judge_scores.py --refresh` after any rejudge). The figure's point:
cr2bo4 shifts the whole distribution, not a few outliers.

Fairness: within each judge, a seed counts only if that judge scored it for
BOTH methods (baseline ∩ cr2bo4), so every baseline-vs-cr2bo4 comparison is
over an identical seed set. Each (method, judge) histogram is normalised to
100% over its own N.

Outputs in `concerning_distributions/`:
  plot.png           1x3 (one panel per target), colour = method
                     (baseline tab:blue, cr2bo4 tab:orange, as in Fig. 2),
                     hatching = judge (plain = Opus 5, /// = GPT-5.6), both
                     explained in the legend below the figure; no x-axis
                     label (the caption says these are raw concerning
                     scores).
  plot_perjudge.png  2x3 alternative: rows = judges, columns = targets,
                     plain baseline/cr2bo4 pairs per score.
  plot_avg.{png,pdf} 1x3, per-transcript AVERAGE of the two judges' scores
                     (half-point bins; seeds scored by both judges for both
                     methods). This is the version the paper uses
                     (`concerning_score_distributions_combined`).
  distributions.csv  per (target, judge, method, score) counts + percentages.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import judge_scores as js  # noqa: E402

OUT_DIR = HERE / "concerning_distributions"

TARGETS: list[tuple[str, str]] = [
    ("sonnet-4.6", "Sonnet 4.6"),
    ("opus-4.8", "Opus 4.8"),
    ("gpt-5.5", "GPT-5.5"),
]
# (folder, label, colour) — same method colours as Fig. 2 (plot_cr_comparison).
METHODS: list[tuple[str, str, str]] = [
    ("baseline", "Baseline", "tab:blue"),
    ("cr2bo4", "cr2bo4", "tab:orange"),
]
# (judge_scores key, display, hatch)
JUDGES: list[tuple[str, str, str]] = [
    ("opus5", "Opus 5 judge", ""),
    ("gpt56", "GPT-5.6 judge", "///"),
]

MEASURE = "concerning"
DPI = 200
PANEL_BG = "#f0f0f0"
TITLE_FS = 24
YLABEL_FS = 24
TICK_FS = 17
LEGEND_FS = 21


def build_dists() -> dict[tuple[str, str, str], tuple[Counter, int]]:
    """(target, judge, method) -> (Counter[int score], N)."""
    cache = js.load()
    dists: dict[tuple[str, str, str], tuple[Counter, int]] = {}
    for target, _tl in TARGETS:
        for judge, _jl, _h in JUDGES:
            per = {
                m: cache[target][m][MEASURE].get(judge, {})
                for m, _l, _c in METHODS
            }
            seeds = set.intersection(*(set(v) for v in per.values()))
            for m, _l, _c in METHODS:
                scores = [int(round(float(per[m][s]))) for s in seeds]
                dists[(target, judge, m)] = (Counter(scores), len(scores))
            print(f"[seeds] {target:<11} {judge:<6} N={len(seeds)}")
    return dists


def build_avg_dists() -> dict[tuple[str, str], tuple[Counter, int]]:
    """(target, method) -> (Counter[avg score in 0.5 steps], N); per
    transcript the two judges' raw scores are averaged."""
    cache = js.load()
    dists: dict[tuple[str, str], tuple[Counter, int]] = {}
    for target, _tl in TARGETS:
        per = {
            (m, j): cache[target][m][MEASURE].get(j, {})
            for m, _l, _c in METHODS for j, _jl, _h in JUDGES
        }
        seeds = set.intersection(*(set(v) for v in per.values()))
        for m, _l, _c in METHODS:
            vals = [
                round(2 * sum(float(per[(m, j)][s]) for j, _jl, _h in JUDGES)
                      / len(JUDGES)) / 2
                for s in seeds
            ]
            dists[(target, m)] = (Counter(vals), len(vals))
        print(f"[seeds/avg] {target:<11} N={len(seeds)}")
    return dists


def render_avg(dists, out_base):
    """1x3, colour = method, x = judge-averaged score (0.5 steps)."""
    scores = [x / 2 for x in range(2, 21)]
    width = 0.22
    ymax = min(100.0, 1.08 * max(
        pct(c, n, s) for c, n in dists.values() for s in scores))
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(16, 4.8), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, (target, tlabel) in zip(axes, TARGETS):
        for k, (m, _ml, color) in enumerate(METHODS):
            cnt, n = dists[(target, m)]
            xs = [s + (k - 0.5) * width for s in scores]
            ax.bar(xs, [pct(cnt, n, s) for s in scores], width=width,
                   color=color, edgecolor="black", linewidth=0.6)
        ax.set_facecolor(PANEL_BG)
        ax.grid(axis="y", color="white", linewidth=1.2)
        ax.set_axisbelow(True)
        ax.set_xticks(range(1, 11))
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(0, ymax)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(tlabel, fontsize=TITLE_FS)
    axes[0].set_ylabel("Samples (%)", fontsize=YLABEL_FS)
    handles = [Patch(facecolor=c, edgecolor="black") for _m, _l, c in METHODS]
    fig.legend(handles, [l for _m, l, _c in METHODS], loc="lower center",
               bbox_to_anchor=(0.5, -0.075), frameon=False,
               fontsize=LEGEND_FS, ncol=len(METHODS),
               handlelength=1.6, handleheight=1.3, columnspacing=1.6)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_base}.{{png,pdf}}")


def pct(counter: Counter, n: int, score) -> float:
    return 100.0 * counter.get(score, 0) / n if n else 0.0


def score_range(dists) -> list[int]:
    hi = max(max(c) for c, _n in dists.values())
    return list(range(1, hi + 1))


def _ymax(dists, scores) -> float:
    top = max(
        pct(cnt, n, s) for cnt, n in dists.values() for s in scores
    )
    return min(100.0, top * 1.08)


def style_axis(ax, scores, ymax):
    ax.set_facecolor(PANEL_BG)
    ax.grid(axis="y", color="white", linewidth=1.2)
    ax.set_axisbelow(True)
    ax.set_xticks(scores)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_xlim(scores[0] - 0.65, scores[-1] + 0.65)
    ax.set_ylim(0, ymax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def render_combined(dists, out_path):
    """1x3, colour = method, hatch = judge, 4 bars per score."""
    scores = score_range(dists)
    # (method folder, judge key, colour, hatch) in bar order per score.
    combos = [
        (m, j, color, hatch)
        for m, _ml, color in METHODS
        for j, _jl, hatch in JUDGES
    ]
    width = 0.19
    ymax = _ymax(dists, scores)

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(16, 4.8), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, (target, tlabel) in zip(axes, TARGETS):
        for k, (m, j, color, hatch) in enumerate(combos):
            cnt, n = dists[(target, j, m)]
            xs = [s + (k - (len(combos) - 1) / 2) * width for s in scores]
            ax.bar(xs, [pct(cnt, n, s) for s in scores], width=width,
                   color=color, hatch=hatch or None,
                   edgecolor="black", linewidth=0.6)
        style_axis(ax, scores, ymax)
        ax.set_title(tlabel, fontsize=TITLE_FS)
    axes[0].set_ylabel("Samples (%)", fontsize=YLABEL_FS)

    handles = [
        Patch(facecolor=color, edgecolor="black") for _m, _l, color in METHODS
    ] + [
        Patch(facecolor="white", edgecolor="black", hatch=hatch or None)
        for _j, _jl, hatch in JUDGES
    ]
    labels = [l for _m, l, _c in METHODS] + [jl for _j, jl, _h in JUDGES]
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.075), frameon=False,
               fontsize=LEGEND_FS, ncol=len(labels),
               handlelength=1.6, handleheight=1.3, columnspacing=1.6)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def render_perjudge(dists, out_path):
    """2x3 alternative: rows = judges, columns = targets."""
    scores = score_range(dists)
    width = 0.38
    ymax = _ymax(dists, scores)

    fig, axes = plt.subplots(
        len(JUDGES), len(TARGETS), figsize=(16, 8.6),
        sharey=True, sharex=True,
    )
    fig.patch.set_facecolor("white")
    for r, (judge, jlabel, _h) in enumerate(JUDGES):
        for c, (target, tlabel) in enumerate(TARGETS):
            ax = axes[r, c]
            for k, (m, _ml, color) in enumerate(METHODS):
                cnt, n = dists[(target, judge, m)]
                xs = [s + (k - (len(METHODS) - 1) / 2) * width
                      for s in scores]
                ax.bar(xs, [pct(cnt, n, s) for s in scores], width=width,
                       color=color, edgecolor="black", linewidth=0.6)
            style_axis(ax, scores, ymax)
            if r == 0:
                ax.set_title(tlabel, fontsize=TITLE_FS)
            if c == 0:
                ax.set_ylabel("Samples (%)", fontsize=YLABEL_FS)
    fig.tight_layout(rect=(0.0, 0.045, 0.965, 1.0))
    # rotated judge label on the right edge of each row (post-layout so the
    # axes positions are final)
    for r, (_j, jlabel, _h) in enumerate(JUDGES):
        pos = axes[r, -1].get_position()
        fig.text(0.985, (pos.y0 + pos.y1) / 2, jlabel, rotation=270,
                 va="center", ha="center", fontsize=TITLE_FS)

    handles = [
        Patch(facecolor=color, edgecolor="black") for _m, _l, color in METHODS
    ]
    labels = [l for _m, l, _c in METHODS]
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.035), frameon=False,
               fontsize=LEGEND_FS, ncol=len(labels),
               handlelength=1.6, handleheight=1.3, columnspacing=1.6)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def write_csv(dists, out_path):
    scores = score_range(dists)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "judge", "method", "n", "score", "count", "pct"])
        for target, _tl in TARGETS:
            for judge, _jl, _h in JUDGES:
                for m, _ml, _c in METHODS:
                    cnt, n = dists[(target, judge, m)]
                    for s in scores:
                        w.writerow([target, judge, m, n, s, cnt.get(s, 0),
                                    f"{pct(cnt, n, s):.3f}"])
    print(f"Saved {out_path}")


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dists = build_dists()
    render_combined(dists, OUT_DIR / "plot.png")
    render_perjudge(dists, OUT_DIR / "plot_perjudge.png")
    write_csv(dists, OUT_DIR / "distributions.csv")
    avg = build_avg_dists()
    render_avg(avg, str(OUT_DIR / "plot_avg"))
    with open(OUT_DIR / "distributions_avg.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "method", "avg_score", "count", "n", "pct"])
        for (t, m), (cnt, n) in sorted(avg.items()):
            for sc in sorted(cnt):
                w.writerow([t, m, sc, cnt[sc], n, round(pct(cnt, n, sc), 2)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
