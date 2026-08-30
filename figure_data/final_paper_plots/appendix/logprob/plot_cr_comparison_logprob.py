#!/usr/bin/env python3
"""APPENDIX FIGURE — logprob-selector critique-refinement comparison.

The logprob twin of `../../plot_cr_comparison.py` (same layout, colours and
styling): Baseline vs Realism filter vs cr11 grouped bars for Sonnet 4.6 /
Opus 4.7 / Haiku 4.5, auditor opus-4.7, data from
`outputs/ablation_protocol_no_stopping`. Realism win rate spans the left-hand
side; eval awareness (singular, opus-5) top-right; concerning (full judge,
opus-5) bottom-right. Only one judge exists for these cells (opus-5 after the
2026-08-27 rejudge; gpt-5.4 preserved in history), so there are no judge
versions.

Everything lands in `cr_comparison_logprob/`:

  plot.png                       main figure
  plot_controlled_strict.png     + seed-adherence strict control (cr11 >=
                                 baseline per seed; Ns shown, 2x2 layout
                                 with seed adherence)
  summary*.csv                   matching CSVs

Fairness rule (via `plot_method_comparison_bars.pool`): per target and
measure, a seed counts only if all three conditions have it. Win rate is
judge-independent (`realism_win_rate/summary.json`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
CWR = HERE.parents[2] / "cost_winrate_reasoning"
for p in (CWR, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import logprob_cells as lp  # noqa: E402
import plot_method_comparison_bars as mcb  # noqa: E402
from plot_cost_grid_reasoning import score_by_seed, winrate_by_seed  # noqa: E402

OUT_DIR = HERE / "cr_comparison_logprob"

CONDITIONS: list[tuple[str, str, str]] = [
    ("baseline", "Baseline", "tab:blue"),
    ("realism_filter", "Realism filter", "tab:green"),
    ("cr11", "cr11", "tab:orange"),
]
EA_KEY = "s_eval_awareness"
JUDGE_MEASURE_KEYS = [EA_KEY, "concerning", "s_seed_adherence"]
MEASURE_KEYS = ["winrate"] + JUDGE_MEASURE_KEYS

BAR_WIDTH = 0.22
TARGET_GAP = 0.55
FIG_SIZE = (16, 6.1)
DPI = 200
YLABEL_FONTSIZE = 19

# Point the shared pooling/CSV machinery at the logprob cells.
mcb.TARGETS = list(lp.TARGETS)
mcb.TARGET_DISPLAY = dict(lp.TARGET_DISPLAY)
mcb.CONDITIONS = CONDITIONS


def load_measure(cell: Path, key: str):
    if key == "winrate":
        return winrate_by_seed(cell)
    return score_by_seed(cell, key, normalize=True)


def render_panel(ax, mkey, display, all_raw, drawn, hide_xticklabels=False,
                 target_n=None, winrate_ymax=None):
    """Same styling as plot_cr_comparison.render_panel; the win-rate axis is
    auto-scaled (haiku reaches ~0.8 here, vs ~0.3 in the main figure)."""
    n_cond = len(CONDITIONS)
    x = np.arange(len(mcb.TARGETS)) * (n_cond * BAR_WIDTH + TARGET_GAP)
    ax.set_facecolor("#f0f0f0")
    max_top = 0.0
    for c_idx, (cond, label, color) in enumerate(CONDITIONS):
        offset = (c_idx - (n_cond - 1) / 2) * BAR_WIDTH
        heights, errs = [], []
        any_real = False
        for target in mcb.TARGETS:
            m, e, _n = all_raw[(target, mkey, cond)]
            if np.isnan(m):
                heights.append(np.nan)
                errs.append(0.0)
            else:
                heights.append(m)
                errs.append(e)
                max_top = max(max_top, m + e)
                any_real = True
        bars = ax.bar(
            x + offset, heights, width=BAR_WIDTH, color=color,
            edgecolor="black", linewidth=0.8, yerr=errs, capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "black"},
            label=label if label not in drawn else None, zorder=3,
        )
        if any_real and label not in drawn:
            drawn[label] = bars[0]
    ax.set_xticks(x)
    if hide_xticklabels:
        ax.set_xticklabels([""] * len(mcb.TARGETS))
    else:
        labels = []
        for t in mcb.TARGETS:
            lab = mcb.TARGET_DISPLAY.get(t, t)
            if target_n is not None and t in target_n:
                lab += f"\nN={target_n[t]}"
            labels.append(lab)
        ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel(display, fontsize=YLABEL_FONTSIZE)
    ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if mkey == "winrate" and winrate_ymax is not None:
        ax.set_ylim(0.0, winrate_ymax)
    else:
        ax.set_ylim(0.0, max_top * 1.07 + 0.005)
    if mkey == EA_KEY and max_top <= 0.10:
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.set_ylim(0.0, 0.10)


def render_fig(all_raw, measures, out_path, target_n=None, title=""):
    grid2x2 = any(slot == "bot_left" for _m, _d, slot in measures)
    fig = plt.figure(figsize=(15, 9.5) if grid2x2 else FIG_SIZE)
    fig.patch.set_facecolor("white")
    if grid2x2:
        gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.20)
        axes = {
            "top_left": fig.add_subplot(gs[0, 0]),
            "top_right": fig.add_subplot(gs[0, 1]),
            "bot_left": fig.add_subplot(gs[1, 0]),
            "bot_right": fig.add_subplot(gs[1, 1]),
        }
        hidden = {"top_left", "top_right"}
    else:
        gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.25,
                              wspace=0.22)
        axes = {
            "left": fig.add_subplot(gs[:, 0]),
            "top_right": fig.add_subplot(gs[0, 1]),
            "bot_right": fig.add_subplot(gs[1, 1]),
        }
        hidden = {"top_right"}
    wr_tops = [
        m + e
        for (t, k, c), (m, e, _n) in all_raw.items()
        if k == "winrate" and m == m
    ]
    winrate_ymax = (max(wr_tops) * 1.12) if wr_tops else None
    drawn: dict[str, object] = {}
    for mkey, display, slot in measures:
        render_panel(
            axes[slot], mkey, display, all_raw, drawn,
            hide_xticklabels=(slot in hidden),
            target_n=target_n if slot not in hidden else None,
            winrate_ymax=winrate_ymax,
        )
    handles = [drawn[c[1]] for c in CONDITIONS if c[1] in drawn]
    labels = [c[1] for c in CONDITIONS if c[1] in drawn]
    bottom = 0.15 if target_n is not None else 0.11
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               frameon=False, fontsize=18, ncol=len(labels))
    if title:
        fig.suptitle(title, fontsize=15, y=0.99)
    fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0 if not title else 0.93))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def build_data():
    data: dict[tuple[str, str, str], dict[int, float]] = {}
    present: dict[str, list[str]] = {}
    for target in mcb.TARGETS:
        present[target] = []
        for cond, _l, _c in CONDITIONS:
            cell = lp.cell(target, cond)
            if not cell.is_dir():
                print(f"[skip] {target} {cond} (no cell)")
                continue
            present[target].append(cond)
            for mkey in MEASURE_KEYS:
                data[(target, cond, mkey)] = load_measure(cell, mkey)
        print(
            f"[load] {target:<12} "
            + "  ".join(
                f"{k}={min((len(data.get((target, c, k), {})) for c in present[target]), default=0):>3}"
                for k in MEASURE_KEYS
            )
        )
    return data, present


def controlled_seeds(data, max_drop: float = 0.0) -> dict[str, set[int]]:
    """Per target, seeds where cr11's seed adherence is at most `max_drop`
    below baseline's (both scored)."""
    out: dict[str, set[int]] = {}
    for t in mcb.TARGETS:
        base = data.get((t, "baseline", "s_seed_adherence"), {})
        cr = data.get((t, "cr11", "s_seed_adherence"), {})
        out[t] = {s for s in (set(base) & set(cr)) if cr[s] >= base[s] - max_drop}
    return out


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data, present = build_data()
    measures = [
        ("winrate", "Realism win rate", "left"),
        (EA_KEY, "Eval awareness", "top_right"),
        ("concerning", "Concerning", "bot_right"),
    ]

    all_raw = mcb.pool(mcb.TARGETS, CONDITIONS, MEASURE_KEYS, data, present)
    render_fig(all_raw, measures, OUT_DIR / "plot.png")
    mcb.write_csv(all_raw, OUT_DIR, "summary.csv", MEASURE_KEYS)

    ctrl = controlled_seeds(data, max_drop=0.0)
    target_n = {t: len(ctrl[t]) for t in mcb.TARGETS}
    all_raw_ctrl = mcb.pool(
        mcb.TARGETS, CONDITIONS, MEASURE_KEYS, data, present, allowed_seeds=ctrl
    )
    strict_measures = [
        ("winrate", "Realism win rate", "top_left"),
        (EA_KEY, "Eval awareness", "top_right"),
        ("s_seed_adherence", "Seed adherence", "bot_left"),
        ("concerning", "Concerning", "bot_right"),
    ]
    render_fig(
        all_raw_ctrl, strict_measures,
        OUT_DIR / "plot_controlled_strict.png",
        target_n=target_n,
        title="Controlled for seed adherence (cr11 \N{GREATER-THAN OR EQUAL TO} baseline per seed)",
    )
    # Paper version: no suptitle (the caption carries the control description).
    render_fig(
        all_raw_ctrl, strict_measures,
        OUT_DIR / "plot_controlled_strict_notitle.png",
        target_n=target_n,
    )
    mcb.write_csv(all_raw_ctrl, OUT_DIR, "summary_controlled_strict.csv",
                  MEASURE_KEYS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
