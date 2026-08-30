#!/usr/bin/env python3
"""APPENDIX FIGURE — cr4 selector comparison (logprob vs pairwise vs scoring).

Style twin of the channel-ablation figures (paper Fig. 19,
`cost_winrate_reasoning/plot_opus48_channel_ablation.py`): a single-target 2x2
grid rendered by `plot_method_comparison_bars._render_panel` with full-width
bar clusters, no x-tick labels, and value annotations on the seed-adherence
panel. Layout: realism win rate top-left, eval awareness top-right, concerning
bottom-left, seed adherence bottom-right. Compares HOW the critique-refine
preference signal is obtained at fixed depth cr4, target Sonnet 4.6, auditor
opus-4.7, data from `outputs/ablation_protocol_no_stopping`:

  Baseline          baseline/target_sonnet-4.6/auditor_opus-4.7
  cr4 (pairwise)    pairwise/cr4/target_sonnet-4.6/auditor_opus-4.7
  cr4 (logprob)     cr4/target_sonnet-4.6/auditor_opus-4.7
  cr4 (scoring)     score/cr4/target_sonnet-4.6/auditor_opus-4.7

Same four measures as plot_controlled_strict.png but with NO seed-adherence
control. Fairness comes from `plot_method_comparison_bars.pool`
instead: per measure, a seed counts only if ALL FOUR conditions have it, so
every within-panel comparison is over the identical seed set (the per-panel N).
Ns can differ between panels (singular refusal gaps differ by measure).

Judge measures are opus-5 (post the 2026-08-27 rejudge); the script warns
loudly if any cell's CSV still carries another judge. Win rate is
judge-independent (`realism_win_rate/summary.json`).

Everything lands in `cr4_selector_comparison/`: plot.png + summary.csv.
"""

from __future__ import annotations

import argparse
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
import plot_method_comparison_bars as mcb  # noqa: E402
from plot_cost_grid_reasoning import score_by_seed, winrate_by_seed  # noqa: E402

OUT_DIR = HERE / "cr4_selector_comparison"

TARGET = "sonnet-4.6"

# (condition path under ablation_protocol_no_stopping, label, colour)
CONDITIONS: list[tuple[str, str, str]] = [
    ("baseline", "Baseline", "tab:blue"),
    ("pairwise/cr4", "cr4 (pairwise)", "tab:green"),
    ("cr4", "cr4 (logprob)", "tab:orange"),
    ("score/cr4", "cr4 (scoring)", "tab:red"),
]
EA_KEY = "s_eval_awareness"
JUDGE_MEASURE_KEYS = [EA_KEY, "concerning", "s_seed_adherence"]
MEASURE_KEYS = ["winrate"] + JUDGE_MEASURE_KEYS
# (measure key, ylabel, grid position) — Fig.-19 layout, seed adherence
# bottom-right. The EA panel is rendered under the alias "eval_awareness" so
# _render_panel applies its eval-awareness axis rules.
MEASURES: list[tuple[str, str, tuple[int, int]]] = [
    ("winrate", "Realism win rate", (0, 0)),
    ("eval_awareness", "Eval awareness", (0, 1)),
    ("s_seed_adherence", "Seed adherence", (1, 0)),
    ("concerning", "Concerning", (1, 1)),
]

FIG_SIZE = (14, 8.4)  # matches the adherent cr11 figure (fig 17)
DPI = 200

# Point the shared pooling/CSV machinery at this figure's cells.
mcb.TARGETS = [TARGET]
mcb.TARGET_DISPLAY = {TARGET: lp.TARGET_DISPLAY[TARGET]}
mcb.CONDITIONS = CONDITIONS


def load_measure(cell: Path, key: str):
    if key == "winrate":
        return winrate_by_seed(cell)
    return score_by_seed(cell, key, normalize=True)


def check_judges() -> None:
    """The four cells are only comparable on judge measures if one judge
    scored them all; warn loudly if that's not (yet) the case."""
    by_cond: dict[str, set[str]] = {}
    for cond, _l, _c in CONDITIONS:
        path = lp.cell(TARGET, cond) / "0_transcripts.csv"
        if not path.is_file():
            continue
        with open(path, newline="") as f:
            by_cond[cond] = {
                (row.get("judge") or "?").strip() for row in csv.DictReader(f)
            }
    judges = set().union(*by_cond.values()) if by_cond else set()
    if judges != {"claude-opus-5"}:
        print("=" * 78)
        print("[WARNING] judge measures are NOT single-judge yet:")
        for cond, js in by_cond.items():
            print(f"  {cond:<14} judge column: {sorted(js)}")
        print("  -> rejudge the offending cell(s) with opus-5 and re-run.")
        print("=" * 78)


def render_fig(all_raw, out_path):
    """Fig.-19 styling: 2x2 grid via mcb._render_panel, hidden x-tick labels,
    value annotations on the seed-adherence panel, bottom legend."""
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    drawn: dict[str, object] = {}
    for mkey, display, (r, c) in MEASURES:
        mcb._render_panel(
            axes[r, c], mkey, display, [TARGET], CONDITIONS, all_raw, drawn,
            hide_xticklabels=True,
            annotate_values=(mkey == "s_seed_adherence"),
        )
    handles = [drawn[c[1]] for c in CONDITIONS if c[1] in drawn]
    labels = [c[1] for c in CONDITIONS if c[1] in drawn]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=15, ncol=len(labels))
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def build_data():
    data: dict[tuple[str, str, str], dict[int, float]] = {}
    present: dict[str, list[str]] = {TARGET: []}
    for cond, _l, _c in CONDITIONS:
        cell = lp.cell(TARGET, cond)
        if not cell.is_dir():
            print(f"[skip] {cond} (no cell at {cell})")
            continue
        present[TARGET].append(cond)
        for mkey in MEASURE_KEYS:
            data[(TARGET, cond, mkey)] = load_measure(cell, mkey)
        print(
            f"[load] {cond:<14} "
            + "  ".join(
                f"{k}={len(data.get((TARGET, cond, k), {})):>3}"
                for k in MEASURE_KEYS
            )
        )
    return data, present


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    check_judges()
    data, present = build_data()
    all_raw = mcb.pool([TARGET], CONDITIONS, MEASURE_KEYS, data, present)
    # Alias the singular EA scores under the key _render_panel special-cases
    # for its eval-awareness axis rules.
    for cond, _l, _c in CONDITIONS:
        all_raw[(TARGET, "eval_awareness", cond)] = all_raw[(TARGET, EA_KEY, cond)]
    for mkey in MEASURE_KEYS:
        ns = {c: all_raw[(TARGET, mkey, c)][2] for c, _l, _co in CONDITIONS}
        print(f"[pool] {mkey:<18} common-seed N per condition: {ns}")
    render_fig(all_raw, OUT_DIR / "plot.png")
    mcb.write_csv(all_raw, OUT_DIR, "summary.csv", MEASURE_KEYS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
