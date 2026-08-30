#!/usr/bin/env python3
"""APPENDIX FIGURES — what makes seed adherence fall faster on some targets.

Two single-panel seed-adherence bar charts, sized for a HALF-WIDTH slot the
body text wraps around, drawn with the grouped-bar style of paper Fig. 15
(`cr4_selector_comparison`) via `plot_method_comparison_bars._render_panel`.

Both extend the existing channel ablations (paper Figs. 18/19) with a
TARGET-SWAP arm run on 2026-08-27. The channel ablations hold the target fixed
and move the critique-refine machinery; the target swap does the reverse, so
the pair separates "the audited model causes it" from "the machinery causes
it".

Bars are ordered as the argument reads: the two standard configurations
(target = selector = feedback), then one component at a time swapped to the
better-behaved model. Colour is the repo-wide per-target hue (Opus 4.8 orange,
GPT-5.5 green, Sonnet 4.6 blue, imported from
`plot_cost_grid_combined_targets.TARGET_COLOR` so these bars keep matching the
scaling figures); hatch marks which component was swapped, so the variants read
as one configuration with a single change rather than as unrelated conditions.

  gpt55_vs_opus48   Why GPT-5.5 loses seed adherence faster than Opus 4.8.
                    Target Opus 4.8 (0.74) vs target GPT-5.5 (0.49); swapping
                    the SELECTOR to Opus 4.8 (0.52) or the TARGET to Opus 4.8
                    (0.50) changes nothing, while swapping the FEEDBACK model
                    to Opus 4.8 recovers most of the gap (0.70). The collapse
                    therefore travels with the feedback model, not with the
                    audited model.

  opus48_vs_sonnet46  The same treatment for the Opus-4.8-vs-Sonnet-4.6
                    comparison. Included for completeness: on these seeds the
                    two targets do not actually differ (0.73 vs 0.74), so
                    there is no gap to explain and this figure is not expected
                    to appear in the paper.

The baselines are deliberately NOT plotted — without critique refinement the
two targets sit at 0.83 and 0.84, close enough that a sentence of body text
carries it better than two more bars.

All arms are cr8 with all three seed-adherence mitigations active, auditor
Sonnet 4.6, on the 10-seed channel-ablation set (the 8 GPT-5.5 focus seeds plus
4 and 138) — so "cr8" is left out of the legend labels too. Fairness comes from
`plot_method_comparison_bars.pool`: a seed counts only if EVERY arm in that
figure has it, so each figure compares one identical seed set (its N is printed
and written to the CSV). `opus48_vs_sonnet46` pools over N=9 rather than 10
because its target-swap arm lost seed 52, where the Sonnet 4.6 auditor refused
the seed instruction and produced zero target turns.

Seed adherence defaults to the mean of the Opus 5 and GPT-5.6 singular judges,
matching the paper's convention for singular measures; `--judge opus5|gpt56`
renders the single-judge versions.

Realism win rate is not plotted but is still computed into the CSV sidecar: it
is the measure that tracks the TARGET rather than the machinery (a Sonnet 4.6
target stays ~4x more realistic whichever machinery drives it), which is worth
keeping to hand even though it is not the point of these figures.

Outputs land next to this file: <name>.png + <name>_summary.csv, suffixed
`_opus5` / `_gpt56` for the single-judge variants.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def find_repo_root(explicit: Path | None) -> Path:
    """The checkout that actually holds outputs/.

    This file can live in a git worktree that has no outputs/ of its own, so
    walking up for the directory that does keeps the script runnable from
    either checkout without editing paths.
    """
    if explicit is not None:
        return explicit.expanduser().resolve()
    for cand in HERE.parents:
        if (cand / "outputs").exists() and (cand / "experiments").is_dir():
            return cand
    raise SystemExit(
        "Could not find a checkout containing outputs/ above this file; "
        "pass --repo-root explicitly."
    )


# Shared plotting helpers live in cost_winrate_reasoning/. Imported at module
# level (not inside __main__) because the per-target palette below is needed
# while the module body runs.
_CWR = find_repo_root(None) / "experiments" / "general_plotting" / "cost_winrate_reasoning"
if str(_CWR) not in sys.path:
    sys.path.insert(0, str(_CWR))

import plot_method_comparison_bars as mcb  # noqa: E402
from plot_cost_grid_combined_targets import TARGET_COLOR  # noqa: E402
from plot_cost_grid_reasoning import score_by_seed, winrate_by_seed  # noqa: E402


# ── Arms ─────────────────────────────────────────────────────────────────────
# Bars are a flat list per figure: (cell key, legend label, colour, hatch),
# ordered as the argument reads: the two standard configurations first
# (target = selector = feedback, so their legend entries need no qualifier),
# then one component at a time swapped to the better-behaved model, ENDING on
# the target swap — that is the arm that rules out the audited model, so it is
# the note to finish on. Every arm is cr8, so "cr8" is left out of the labels.
#
# ENCODING: colour = which TARGET the configuration belongs to, hatch = which
# component was swapped. Colours are the repo-wide per-target hues
# (`plot_cost_grid_combined_targets.TARGET_COLOR`: Opus 4.8 orange, GPT-5.5
# green, Sonnet 4.6 blue), imported rather than copied so these bars keep
# matching the scaling figures if that palette ever moves. The family under
# study keeps ONE solid colour and each single-component variant of it reuses
# that colour with a distinct hatch, so the later bars read at a glance as "the
# same configuration with one thing changed" rather than as unrelated
# conditions — the recovery then shows up purely as bar height.
C_OPUS48 = TARGET_COLOR["opus-4.8"]        # tab:orange
C_GPT55 = TARGET_COLOR["gpt-5.5"]          # tab:green
C_SONNET46 = TARGET_COLOR["sonnet-4.6"]    # tab:blue
# Hatches, in swap order. Kept coarse: at half-width a fine hatch turns to mud.
# The fourth is vertical, not "\\\\" — a back-diagonal is nearly indistinguishable
# from the forward-diagonal H_SEL once the figure is scaled to half width.
H_SEL, H_TGT, H_FB, H_BOTH = "//", "xx", "..", "||"
H_DRAFT = "++"

FIGURES: dict[str, dict] = {
    "gpt55_vs_opus48": {
        "arms": [
            ("cr8_o48", "Target Opus 4.8", C_OPUS48, None),
            ("cr8_g55", "Target GPT-5.5", C_GPT55, None),
            ("cr8_g55_sel_o48", "Selector → Opus 4.8", C_GPT55, H_SEL),
            ("cr8_g55_fb_o48", "Feedback → Opus 4.8", C_GPT55, H_FB),
            ("swap_b", "Target → Opus 4.8", C_GPT55, H_TGT),
        ],
    },
    # Does the FORM of the feedback matter? Puts the draft-suppression null
    # beside the feedback-model swap that does recover adherence, so the reader
    # can see that one intervention on the feedback channel works and the other
    # does not.
    "norewrite_drafts": {
        "arms": [
            ("cr8_g55", "Target GPT-5.5", C_GPT55, None),
            ("cr8_g55_norewrite", "Feedback: notes only", C_GPT55, H_DRAFT),
            ("cr8_g55_fb_o48", "Feedback → Opus 4.8", C_GPT55, H_FB),
        ],
    },
    "opus48_vs_sonnet46": {
        "arms": [
            ("cr8_s46", "Target Sonnet 4.6", C_SONNET46, None),
            ("cr8_o48", "Target Opus 4.8", C_OPUS48, None),
            ("cr8_o48_sel_s46", "Selector → Sonnet 4.6", C_OPUS48, H_SEL),
            ("cr8_o48_fb_s46", "Feedback → Sonnet 4.6", C_OPUS48, H_FB),
            ("cr8_o48_selfb_s46", "Selector + feedback → Sonnet 4.6", C_OPUS48, H_BOTH),
            ("swap_a", "Target → Sonnet 4.6", C_OPUS48, H_TGT),
        ],
    },
}

# Only seed adherence is plotted. Realism win rate is still computed and
# written to the CSV sidecar — it is the measure that tracks the TARGET rather
# than the machinery, which is worth keeping on hand even though it is not the
# point of these figures.
PLOT_MEASURE = ("s_seed_adherence", "Seed adherence")
CSV_MEASURES = ["winrate", "s_seed_adherence"]
ADH_COLUMN = {
    "avg": "s_seed_adherence__mean",
    "opus5": "s_seed_adherence__opus-5",
    "gpt56": "s_seed_adherence__gpt-5.6-sol",
}
JUDGE_SUFFIX = {"avg": "", "opus5": "_opus5", "gpt56": "_gpt56"}
PSEUDO_TARGET = "_"
# Sized for a wrapfigure the body text flows around, included at
# 0.45\linewidth (~2.48in). The full-width figures in this repo are ~14in wide
# at \linewidth (~5.5in), i.e. scaled to ~0.39; 6.4in at 0.45\linewidth is the
# same 0.39, so the house font sizes land at the same on-page size as everywhere
# else — do not shrink the fonts to compensate. Aspect is kept close to the
# original 7.0x5.4 so the wrap block stays short enough to sit beside one
# paragraph.
FIG_SIZE = (6.4, 4.9)
DPI = 200
LEGEND_FONTSIZE = 14
LEGEND_FONTSIZE_MIN = 10
# Fraction of the axes width the legend may occupy before the font steps down.
# The real constraint is simply not overflowing the panel; a stricter budget
# costs a font step for nothing (at 0.98 the five-arm figure dropped to 13.5
# despite fitting at 14).
LEGEND_WIDTH_BUDGET = 1.0
# Clearance left between the topmost drawn datum and the bottom of the in-axes
# legend, in points.
LEGEND_CLEARANCE_PT = 5.0
# Seed adherence is normalised to 0-1, so end the axis exactly at 1.0 whenever
# the legend fits below it.
PREFERRED_TOP = 1.0


def build_cells(repo: Path) -> dict[str, Path]:
    r = repo / "outputs" / "reasoning"
    abl = r / "pairwise" / "seed_adherence_ablations"
    swap = abl / "target_swap"
    p48 = r / "pairwise" / "target_opus-4.8"
    return {
        "base_g55": r / "baseline/target_gpt-5.5/auditor_sonnet-4.6",
        "base_o48": r / "baseline/target_opus-4.8/auditor_sonnet-4.6",
        "base_s46": r / "baseline/target_sonnet-4.6/auditor_sonnet-4.6",
        "cr8_g55": r / "pairwise/target_gpt-5.5/auditor_sonnet-4.6/cr8",
        "cr8_g55_sel_o48": abl
        / "all_mitigations/target_gpt-5.5_selection_opus-4.8/auditor_sonnet-4.6/cr8",
        "cr8_g55_norewrite": abl
        / "all_mitigations/target_gpt-5.5_feedback_no_rewrite/auditor_sonnet-4.6/cr8",
        "cr8_g55_fb_o48": abl
        / "all_mitigations/target_gpt-5.5_feedback_opus-4.8/auditor_sonnet-4.6/cr8",
        "cr8_o48": p48 / "auditor_sonnet-4.6/cr8",
        "cr8_o48_sel_s46": p48 / "auditor_sonnet-4.6_select_sonnet-4.6/cr8",
        "cr8_o48_fb_s46": p48 / "auditor_sonnet-4.6_feedback_sonnet-4.6/cr8",
        "cr8_o48_selfb_s46": p48 / "auditor_sonnet-4.6_select+feedback_sonnet-4.6/cr8",
        "cr8_s46": r / "pairwise/target_sonnet-4.6/auditor_sonnet-4.6/cr8",
        "swap_a": swap
        / "target_sonnet-4.6_select+feedback_opus-4.8/auditor_sonnet-4.6/cr8",
        "swap_b": swap
        / "target_opus-4.8_select+feedback_gpt-5.5/auditor_sonnet-4.6/cr8",
    }


def _place_legend(fig, ax, handles, labels, ncol):
    """Legend inside the axes, shrunk until it fits the axes width.

    LEGEND_FONTSIZE is the size we want; whether it fits depends on the longest
    label, which differs per figure ("Selector + feedback -> Sonnet 4.6" is far
    wider than "Target GPT-5.5"). At the requested size the six-arm figure
    overflows the panel and runs into the y-tick labels, so step down until it
    fits rather than hard-coding a different size per figure.
    """
    size = LEGEND_FONTSIZE
    while True:
        # Tight handle and column spacing: every px saved here is width the
        # font can spend instead.
        leg = ax.legend(handles, labels, loc="upper center", ncol=ncol,
                        fontsize=size, handlelength=1.3, labelspacing=0.3,
                        columnspacing=1.0, handletextpad=0.5,
                        borderpad=0.4, frameon=False)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        lw = leg.get_window_extent(renderer=r).width
        aw = ax.get_window_extent(renderer=r).width
        if lw <= aw * LEGEND_WIDTH_BUDGET or size <= LEGEND_FONTSIZE_MIN:
            print(f"[legend] fontsize {size:g}  width {lw:.0f}px "
                  f"of {aw:.0f}px axes ({lw / aw:.0%})")
            return leg
        leg.remove()
        size -= 0.5


def _fit_headroom_to_legend(fig, ax, leg, all_raw, arms) -> None:
    """Raise the y-axis top by exactly enough to clear the in-axes legend.

    A fixed multiplier either wastes a band of empty axes or lets the legend
    collide with the tallest bar, and the right value depends on the legend's
    row count, font size and label widths. So measure instead: the legend is
    anchored in AXES coordinates, so its pixel height is independent of the
    y-limits, while the bars and their value labels scale with them. That makes
    the required top solvable in one step rather than by iterating.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax_h = ax.get_window_extent(renderer=r).height
    leg_h = leg.get_window_extent(renderer=r).height
    txt_h = max(
        (t.get_window_extent(renderer=r).height for t in ax.texts), default=0.0
    )
    gap = LEGEND_CLEARANCE_PT * fig.dpi / 72.0

    # Topmost drawn datum: the highest error-bar cap across the arms.
    tops = [
        m + e
        for _c, _l, _co in arms
        for m, e, _n in [all_raw[(PSEUDO_TARGET, PLOT_MEASURE[0], _c)]]
        if m == m  # skip NaN
    ]
    if not tops:
        return
    content_top = max(tops)

    frac = (ax_h - leg_h - txt_h - gap) / ax_h
    if frac <= 0.05:  # legend taller than the axes; leave the default padding
        return
    lo, _hi = ax.get_ylim()
    required = lo + (content_top - lo) / frac
    # Prefer to stop exactly at the top of the measure's range: seed adherence
    # is normalised to 0-1, so an axis ending at 1.0 reads as the full scale
    # rather than an arbitrary number, and the last tick is also the frame.
    # Only exceed it when the legend genuinely does not fit underneath.
    ax.set_ylim(lo, PREFERRED_TOP if required <= PREFERRED_TOP else required)


def render(name: str, arms_full, all_raw, out_png: Path) -> None:
    mkey, display = PLOT_MEASURE
    arms = [(k, lab, col) for k, lab, col, _h in arms_full]
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    drawn: dict[str, object] = {}
    # No per-bar value annotations: the other bar figures in the paper do not
    # carry them, and the exact means are in the CSV sidecar. Dropping them also
    # lets the y-axis end lower, since _fit_headroom_to_legend no longer has to
    # reserve room for a line of text above the tallest error bar.
    mcb._render_panel(
        ax, mkey, display, [PSEUDO_TARGET], arms, all_raw, drawn,
        hide_xticklabels=True, annotate_values=False,
    )
    # Hatch the single-component variants. `drawn[label]` IS the bar's Rectangle
    # (one pseudo-target => one bar per arm) and doubles as the legend handle,
    # so setting the hatch here updates both the bar and the key. Applied after
    # _render_panel rather than by extending it, to leave that shared house
    # helper -- used by every other bar figure -- untouched.
    for _k, label, _col, hatch in arms_full:
        if hatch and label in drawn:
            drawn[label].set_hatch(hatch)

    handles = [drawn[a[1]] for a in arms_full if a[1] in drawn]
    labels = [a[1] for a in arms_full if a[1] in drawn]
    # Legend INSIDE the axes, top-centre, two columns. Buys back the strip a
    # below-axes legend costs, which matters at half width.
    ncol = 2
    leg = _place_legend(fig, ax, handles, labels, ncol)
    # Seed adherence is normalised to 0-1, so stop the ticks at 1.0 even though
    # the axis runs past it: the band above 1.0 exists only to hold the legend,
    # and labelled ticks up there would imply the measure can exceed 1.
    ax.set_yticks([t / 10 for t in range(0, 11, 2)])
    fig.tight_layout()
    _fit_headroom_to_legend(fig, ax, leg, all_raw, arms)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_png}")


def build_figure(name: str, spec: dict, cells: dict[str, Path], judge: str) -> None:
    arms_full = spec["arms"]
    # The shared mcb helpers unpack (cond, label, colour) triples; hatch is
    # this figure's own concern and is applied in render().
    arms = [(k, lab, col) for k, lab, col, _h in arms_full]
    keys = list(CSV_MEASURES)
    data: dict[tuple[str, str, str], dict[int, float]] = {}
    present: dict[str, list[str]] = {PSEUDO_TARGET: []}

    print(f"\n=== {name}  (seed adherence judge: {judge}) ===")
    for cond, label, _c in arms:
        cell = cells[cond]
        if not cell.is_dir():
            print(f"[skip] {cond:<18} missing cell: {cell}")
            continue
        present[PSEUDO_TARGET].append(cond)
        data[(PSEUDO_TARGET, cond, "winrate")] = winrate_by_seed(cell)
        data[(PSEUDO_TARGET, cond, "s_seed_adherence")] = score_by_seed(
            cell, ADH_COLUMN[judge], normalize=True
        )
        print(
            f"[load] {cond:<18} "
            + "  ".join(
                f"{k}={len(data[(PSEUDO_TARGET, cond, k)]):>3}" for k in keys
            )
        )

    all_raw = mcb.pool([PSEUDO_TARGET], arms, keys, data, present)
    for mkey in keys:
        ns = {c: all_raw[(PSEUDO_TARGET, mkey, c)][2] for c, _l, _co in arms}
        n_set = sorted({v for v in ns.values() if v})
        print(f"[pool] {mkey:<18} common-seed N: {n_set}  per-arm {ns}")

    suffix = JUDGE_SUFFIX[judge]
    render(name, arms_full, all_raw, HERE / f"{name}{suffix}.png")
    mcb.TARGETS = [PSEUDO_TARGET]
    mcb.CONDITIONS = arms
    mcb.write_csv(all_raw, HERE, f"{name}{suffix}_summary.csv", keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judge", choices=["avg", "opus5", "gpt56", "all"],
                    default="avg",
                    help="seed-adherence judge (default: avg of Opus 5 and "
                         "GPT-5.6, the paper convention for singular measures)")
    ap.add_argument("--figure", choices=[*FIGURES, "both"], default="both")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="checkout holding outputs/ (default: auto-detected)")
    args = ap.parse_args()

    repo = find_repo_root(args.repo_root)
    print(f"repo root: {repo}")
    cells = build_cells(repo)

    names = list(FIGURES) if args.figure == "both" else [args.figure]
    judges = ["avg", "opus5", "gpt56"] if args.judge == "all" else [args.judge]
    for judge in judges:
        for name in names:
            build_figure(name, FIGURES[name], cells, judge)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
