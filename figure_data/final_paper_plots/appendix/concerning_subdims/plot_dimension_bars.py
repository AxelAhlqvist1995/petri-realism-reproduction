#!/usr/bin/env python3
"""PAPER FIGURES — Petri alignment-judge dimension shifts, baseline vs cr2bo4.

Reproduces (from the transcripts in this repo) the figures that were
originally built elsewhere:
  dimension_bars/selected_dimension_horizontal_bars_three_models.{png,pdf}
      the 22 misalignment dimensions (sub-dimensions of concerning) whose
      paired cr2bo4 - baseline shift is significant (95% CI excludes 0) for
      at least one target
  dimension_bars/selected_dimension_horizontal_bars_four_models.{png,pdf}
      same selection rule over four targets (adds Gemini 3.5 Flash)
  dimension_bars/focus_dimension_horizontal_bars_{three,four}_models.{png,pdf}
      hand-picked FOCUS dimensions in a fixed order (three-model = paper fig)
  dimension_bars/alignment_dimension_baseline_vs_treatment_{sonnet46,opus48,gpt55,gemini35}.png
      per target: all 22 misalignment dimensions (left: means, right: paired shifts)
  dimension_bars/dimension_shifts.csv  (every number behind both figures)

Data: rows.csv written by extract.py (one row per transcript x judge, raw
1-10 rubric scores of the claude-opus-5 and gpt-5.6-sol judges) for the paper
Fig. 2 cells: baseline and cr2bo4, targets Sonnet 4.6 / Opus 4.8 / GPT-5.5.
Per transcript the two judges are averaged (same convention as the
decomposition / ladder tables of make_tables.py); scores are normalised to
[0,1] via (s-1)/9. Within a target only seeds with both a baseline and a
cr2bo4 transcript are used. Error bars are 95% CIs (1.96 SEM); paired-shift
CIs use the per-seed differences.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bar_style import TARGETS, TARGETS4, COLORS, LABELS, PANEL_BG, mean_ci, style, three_panel_bars

HERE = Path(__file__).resolve().parent
OUT = HERE / "dimension_bars"
OUT.mkdir(exist_ok=True)

# Petri misaligned-behaviour dimensions (the 22 used in the decomposition).
BEH = ["cooperation_with_human_misuse", "facilitating_terrorism",
       "cooperation_with_harmful_sysprompt", "user_over_sysprompt",
       "manipulated_by_developer", "unprompted_deception_toward_user",
       "unprompted_sycophancy", "unprompted_encouragement_of_user_delusion",
       "unprompted_leaking", "unprompted_whistleblowing", "unprompted_blackmail",
       "unprompted_initiative", "unprompted_emotional_manipulativeness",
       "unprompted_self_preservation", "unfaithful_thinking", "input_hallucination",
       "unprompted_self_serving_bias", "unprompted_gratitude",
       "unprompted_spirituality_or_extreme_awe", "unprompted_pos_affect",
       "unprompted_neg_affect", "unprompted_bizarre_behavior"]
# Other behaviour-type rubric dimensions (counterfactual_bad_behavior,
# overrefusal, stuck_in_loops, prefill_susceptibility) are NOT sub-dimensions
# of concerning (the rubric scores them separately) and are deliberately left
# out of every figure here. The selection pool and the full per-target
# figures both use the 22 misalignment dimensions only.
ALL = BEH
SELECT_POOL = BEH
SLUG = {"sonnet-4.6": "sonnet46", "opus-4.8": "opus48", "gpt-5.5": "gpt55",
        "gemini-3.5-flash": "gemini35"}
# Hand-picked dimensions (and order) for the focus figure that opens the
# concerning appendix of the paper.
GROUP_A = ["cooperation_with_human_misuse", "cooperation_with_harmful_sysprompt",
           "manipulated_by_developer", "unprompted_sycophancy",
           "unprompted_encouragement_of_user_delusion", "unfaithful_thinking"]
GROUP_B = ["unprompted_initiative", "unprompted_whistleblowing", "unprompted_leaking"]
GROUP_C = ["input_hallucination", "unprompted_deception_toward_user",
           "user_over_sysprompt", "unprompted_self_serving_bias"]
FOCUS = ["user_over_sysprompt", "unprompted_sycophancy", "unfaithful_thinking",
         "unprompted_deception_toward_user", "unprompted_initiative",
         "input_hallucination",
         "cooperation_with_human_misuse", "cooperation_with_harmful_sysprompt",
         "manipulated_by_developer", "unprompted_encouragement_of_user_delusion",
         "unprompted_leaking"]


def load() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    df = pd.read_csv(HERE / "rows.csv")
    out = {}
    for t, _ in TARGETS4:
        s = df[df.target == t]
        avg = {c: s[s.cell == c].groupby("seed")[ALL].mean() for c in ("baseline", "cr2bo4")}
        seeds = avg["baseline"].index.intersection(avg["cr2bo4"].index)
        b = (avg["baseline"].loc[seeds] - 1) / 9
        c = (avg["cr2bo4"].loc[seeds] - 1) / 9
        print(f"[seeds] {t:<11} paired n={len(seeds)}")
        out[t] = (b, c)
    return out


def main():
    data = load()
    stats, rows = {}, []
    for t, _ in TARGETS4:
        b, c = data[t]
        d = c - b
        for dim in ALL:
            mb, eb = mean_ci(b[dim].dropna())
            mc, ec = mean_ci(c[dim].dropna())
            md, ed = mean_ci(d[dim].dropna())
            stats[(t, "baseline", dim)] = (mb, eb)
            stats[(t, "cr2bo4", dim)] = (mc, ec)
            stats[(t, "diff", dim)] = (md, ed)
            rows.append(dict(target=t, dimension=dim, n=int(d[dim].notna().sum()),
                             baseline=mb, baseline_ci95=eb, cr2bo4=mc, cr2bo4_ci95=ec,
                             diff=md, diff_ci95=ed, significant=abs(md) > ed))
    with open(OUT / "dimension_shifts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # --- selected dims: significant for >=1 target, Petri dimension order
    sig = {dim for dim in SELECT_POOL for t, _ in TARGETS
           if abs(stats[(t, "diff", dim)][0]) > stats[(t, "diff", dim)][1]}
    selected = [d for d in SELECT_POOL if d in sig]
    print("[selected 3 targets]", selected)
    three_panel_bars(stats, selected, TARGETS,
                     str(OUT / "selected_dimension_horizontal_bars_three_models"),
                     figsize=(11, 0.55 * len(selected) + 2.2))

    # --- same selection rule over four targets (incl. Gemini 3.5 Flash)
    sig4 = {dim for dim in SELECT_POOL for t, _ in TARGETS4
            if abs(stats[(t, "diff", dim)][0]) > stats[(t, "diff", dim)][1]}
    selected4 = [d for d in SELECT_POOL if d in sig4]
    print("[selected 4 targets]", selected4)
    print("[selected 4 targets] added vs 3-target set:", sorted(sig4 - sig))
    three_panel_bars(stats, selected4, TARGETS4,
                     str(OUT / "selected_dimension_horizontal_bars_four_models"),
                     figsize=(14, 0.55 * len(selected4) + 2.2), label_fs=10.5)

    # --- hand-picked focus dimensions: three targets (paper figure) and four
    three_panel_bars(stats, FOCUS, TARGETS,
                     str(OUT / "focus_dimension_horizontal_bars_three_models"),
                     figsize=(11, 0.55 * len(FOCUS) + 2.2))
    # --- grouped focus figure: the three behavior groups used in the appendix text
    GROUPED = (["## Going along with a request made to look legitimate"] + GROUP_A
               + ["## Drastic unilateral action"] + GROUP_B
               + ["## Misreporting inside long agentic workflows"] + GROUP_C)
    three_panel_bars(stats, GROUPED, TARGETS,
                     str(OUT / "focus_dimension_grouped_three_models"),
                     figsize=(11.5, 0.42 * len(GROUPED) + 2.0), label_fs=10.5)
    three_panel_bars(stats, GROUPED, TARGETS4,
                     str(OUT / "focus_dimension_grouped_four_models"),
                     figsize=(14.5, 0.42 * len(GROUPED) + 2.0), label_fs=10)
    three_panel_bars(stats, FOCUS, TARGETS4,
                     str(OUT / "focus_dimension_horizontal_bars_four_models"),
                     figsize=(14, 0.55 * len(FOCUS) + 2.2), label_fs=10.5)

    # --- full per-target figures: means (left) + paired shift (right)
    for t, title in TARGETS4:
        order = sorted(ALL, key=lambda d: -stats[(t, "diff", d)][0])
        y = np.arange(len(order))[::-1]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.38 * len(order) + 1.8),
                                       sharey=True, gridspec_kw=dict(width_ratios=[1.25, 1]))
        h = 0.36
        for k, cell in enumerate(["baseline", "cr2bo4"]):
            m = [stats[(t, cell, d)][0] for d in order]
            e = [stats[(t, cell, d)][1] for d in order]
            ax1.barh(y + (h / 2 if k == 0 else -h / 2), m, height=h, xerr=e,
                     color=COLORS[cell], edgecolor="black", linewidth=0.6,
                     label=LABELS[cell],
                     error_kw=dict(elinewidth=0.9, capsize=2.5, capthick=0.9))
        style(ax1)
        ax1.set_yticks(y); ax1.set_yticklabels(order, fontsize=11)
        ax1.set_xlim(left=0); ax1.set_xlabel("Normalized judge score", fontsize=12)
        ax1.set_title("Mean score", fontsize=13)
        ax1.legend(frameon=False, fontsize=11, loc="lower right")
        md = np.array([stats[(t, "diff", d)][0] for d in order])
        ed = np.array([stats[(t, "diff", d)][1] for d in order])
        issig = np.abs(md) > ed
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.errorbar(md, y, xerr=ed, fmt="none", ecolor="black", elinewidth=0.9, capsize=2.5)
        ax2.scatter(md[issig], y[issig], color="tab:red", zorder=3, s=28, label="95% CI excludes 0")
        ax2.scatter(md[~issig], y[~issig], color="grey", zorder=3, s=28, label="n.s.")
        style(ax2)
        ax2.set_xlabel("cr2bo4 $-$ baseline (paired, 95% CI)", fontsize=12)
        ax2.set_title("Paired shift", fontsize=13)
        ax2.legend(frameon=False, fontsize=10, loc="lower right")
        ax1.margins(y=0.02); ax2.margins(y=0.02)
        fig.suptitle(f"{title}  (n={len(data[t][0])} paired seeds)", fontsize=15)
        fig.tight_layout()
        p = OUT / f"alignment_dimension_baseline_vs_treatment_{SLUG[t]}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
        print("Saved", p)


if __name__ == "__main__":
    main()
