#!/usr/bin/env python3
"""FINAL PAPER FIGURE — critique-refinement comparison (cr_comparison/).

Baseline vs Realism filter vs cr2bo4 grouped bars for Sonnet 4.6 / Opus 4.8 /
GPT-5.5 / Gemini 3.5 Flash (auditor sonnet-4.6). Layout: realism win rate
spans the full left-hand side; eval awareness (singular) top-right;
concerning (full judge) bottom-right. Seed adherence is NOT shown but still
defines the controlled-strict variant (seeds where cr2bo4's singular seed
adherence >= baseline's).

One folder per paper figure: everything lands in `cr_comparison/`.

  plot.png                        main figure   (judge avg)
  plot_x*.png                     Petri vs cr2bo4 only (no realism filter);
  plot_x_diff*.png                paired per-seed (cr2bo4 - Petri) differences
  plot_petri_labels*.png          main figure with Petri / + realism filter /
                                  + cr2bo4 legend labels
  plot_controlled_strict.png      + seed-adherence strict control (Ns shown)
  plot_opus5.png / plot_opus5_controlled_strict.png      opus-5-only judge
  plot_gpt56.png / plot_gpt56_controlled_strict.png      gpt-5.6-only judge
  summary*.csv                    matching CSVs (Ns always in the CSVs)

Judge handling: EA / concerning / (control-only) seed adherence come from
`judge_scores.py` views with the fairness rule — per target and measure a
seed counts only if the serving judge scored it for ALL three conditions;
`avg` uses the two-judge mean where both are complete and falls back to the
single complete judge otherwise. The EA axis names the judge; concerning is
labelled plainly. Ns appear in x-tick labels only on the controlled-strict
figures. Win rate is judge-independent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
CWR = HERE.parent / "cost_winrate_reasoning"
for p in (CWR, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import judge_scores as js  # noqa: E402
import plot_method_comparison_bars as mcb  # noqa: E402

OUT_DIR = HERE / "cr_comparison"
VERSIONS = ["avg", "opus5", "gpt56"]
VERSION_LABEL = {"avg": "judge avg", "opus5": "opus-5", "gpt56": "gpt-5.6"}

CONDITIONS: list[tuple[str, str, str]] = [
    ("baseline", "Baseline", "tab:blue"),
    ("realism_filter", "Realism filter", "tab:green"),
    ("cr2bo4", "cr2bo4", "tab:orange"),
]
EA_KEY = "s_eval_awareness"
JUDGE_MEASURE_KEYS = [EA_KEY, "concerning", "s_seed_adherence"]

STAR_TARGETS: set[str] = set()

BAR_WIDTH = 0.22
TARGET_GAP = 0.55
FIG_SIZE = (16, 6.1)
DPI = 200
YLABEL_FONTSIZE = 19


def render_panel(ax, mkey, display, all_raw, drawn, hide_xticklabels=False,
                 target_n=None, winrate_ymax=0.31):
    """Adapted from plot_method_comparison_bars._render_panel: no limited-
    reasoning stars, smaller y-axis titles, N labels only when target_n is
    given."""
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
    # In the N-labelled (adherence-controlled) figures every panel shows its
    # own labels: the per-measure fairness intersections can differ, so one
    # bottom-row N cannot speak for the top row.
    if hide_xticklabels and target_n is None:
        ax.set_xticklabels([""] * len(mcb.TARGETS))
    else:
        labels = []
        for t in mcb.TARGETS:
            lab = mcb.TARGET_DISPLAY.get(t, t)
            if t in STAR_TARGETS and mkey != "winrate":
                lab += "*"
            if target_n is not None:
                # THIS panel's N (equal across conditions by the fairness
                # rule); the control-set size is only a fallback.
                ns = [int(all_raw[(t, mkey, cond)][2])
                      for cond, _l, _c in CONDITIONS
                      if not np.isnan(all_raw[(t, mkey, cond)][0])]
                n_here = max(ns) if ns else target_n.get(t)
                if n_here is not None:
                    lab += f"\nN={n_here}"
            labels.append(lab)
        ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel(display, fontsize=YLABEL_FONTSIZE)
    ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    # Win rate gets a fixed short axis; the judge panels hug the tallest
    # error bar so the bars fill the panel.
    if mkey == "winrate":
        ax.set_ylim(0.0, winrate_ymax)
    else:
        ax.set_ylim(0.0, max_top * 1.07 + 0.005)
    if mkey == EA_KEY and max_top <= 0.10:
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.set_ylim(0.0, 0.10)


def render_fig(all_raw, measures, out_path, target_n=None, title=""):
    grid2x2 = any(slot == "bot_left" for _m, _d, slot in measures)
    # 2x2 controlled figures share the adherent cr11 figure's dimensions.
    fig = plt.figure(figsize=(14, 8.4) if grid2x2 else FIG_SIZE)
    fig.patch.set_facecolor("white")
    if grid2x2:
        gs = fig.add_gridspec(
            2, 2, hspace=0.32 if target_n is not None else 0.18, wspace=0.20
        )
        axes = {
            "top_left": fig.add_subplot(gs[0, 0]),
            "top_right": fig.add_subplot(gs[0, 1]),
            "bot_left": fig.add_subplot(gs[1, 0]),
            "bot_right": fig.add_subplot(gs[1, 1]),
        }
        hidden = {"top_left", "top_right"}
    else:
        gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.25, wspace=0.22)
        axes = {
            "left": fig.add_subplot(gs[:, 0]),
            "top_right": fig.add_subplot(gs[0, 1]),
            "bot_right": fig.add_subplot(gs[1, 1]),
        }
        hidden = {"top_right"}
    winrate_ymax = 0.325 if target_n is not None else 0.31
    drawn: dict[str, object] = {}
    for mkey, display, slot in measures:
        render_panel(
            axes[slot], mkey, display, all_raw, drawn,
            hide_xticklabels=(slot in hidden),
            target_n=target_n,
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


OVERVIEW_IMG = Path("/root/paper/petri_realism_paper/x_overview.png")


def render_x_overview_fig(all_raw, out_path, overview_img):
    """Method-overview diagram on the left; realism win rate (top) and eval
    awareness (bottom) bar panels stacked on the right."""
    img = plt.imread(str(overview_img))
    # Crop the diagram's internal white/transparent margins so it fills its
    # cell instead of floating in its own padding.
    rgb = img[..., :3]
    content = rgb.min(axis=-1) < 0.98
    if img.shape[-1] == 4:
        content &= img[..., 3] > 0.01
    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    if rows.size and cols.size:
        pad = 8
        img = img[max(rows[0] - pad, 0): rows[-1] + pad,
                  max(cols[0] - pad, 0): cols[-1] + pad]
    fig = plt.figure(figsize=(16, 5.0))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], hspace=0.25,
                          wspace=0.16)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img)
    ax_img.axis("off")
    # Hug the plots (bbox_inches='tight' trims the leading whitespace).
    ax_img.set_anchor("E")
    drawn: dict[str, object] = {}
    ax_wr = fig.add_subplot(gs[0, 1])
    ax_ea = fig.add_subplot(gs[1, 1])
    render_panel(
        ax_wr, "winrate", "Realism win rate", all_raw,
        drawn, hide_xticklabels=True,
    )
    render_panel(ax_ea, EA_KEY, "Eval awareness", all_raw, drawn)
    # Shorter right-hand column -> smaller fonts than the standalone figure.
    for ax in (ax_wr, ax_ea):
        ax.yaxis.label.set_size(16)
        ax.tick_params(axis="y", labelsize=13)
    ax_ea.tick_params(axis="x", labelsize=12)
    handles = [drawn[c[1]] for c in CONDITIONS if c[1] in drawn]
    labels = [c[1] for c in CONDITIONS if c[1] in drawn]
    # Legend inside the win-rate panel (the short middle bars leave room),
    # so the figure needs no bottom legend strip.
    ax_wr.legend(handles, labels, loc="upper center", ncol=len(labels),
                 frameon=False, fontsize=15, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def render_diff_fig(diffs, out_path):
    """Paired-difference view of the no-filter comparison: per target the mean
    per-seed difference (cr2bo4 - Petri) with its paired 95% CI; a CI clear of
    the zero line = confident cr2bo4 is higher (or lower). Same 3-slot layout
    as the main figure."""
    slots = [
        ("winrate", "\N{GREEK CAPITAL LETTER DELTA} Realism win rate", "left"),
        (EA_KEY, "\N{GREEK CAPITAL LETTER DELTA} Eval awareness", "top_right"),
        ("concerning", "\N{GREEK CAPITAL LETTER DELTA} Concerning", "bot_right"),
    ]
    fig = plt.figure(figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.25, wspace=0.22)
    axes = {
        "left": fig.add_subplot(gs[:, 0]),
        "top_right": fig.add_subplot(gs[0, 1]),
        "bot_right": fig.add_subplot(gs[1, 1]),
    }
    x = np.arange(len(mcb.TARGETS))
    for mkey, display, slot in slots:
        ax = axes[slot]
        ax.set_facecolor("#f0f0f0")
        means = [diffs[(t, mkey)][0] for t in mcb.TARGETS]
        errs = [diffs[(t, mkey)][1] for t in mcb.TARGETS]
        ax.bar(
            x, means, width=0.55, color="tab:orange", edgecolor="black",
            linewidth=0.8, yerr=errs, capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "black"},
            zorder=3,
        )
        ax.axhline(0.0, color="black", linewidth=1.4, zorder=4)
        ax.set_xticks(x)
        if slot == "top_right":
            ax.set_xticklabels([""] * len(mcb.TARGETS))
        else:
            ax.set_xticklabels(
                [mcb.TARGET_DISPLAY.get(t, t) for t in mcb.TARGETS], fontsize=14
            )
        ax.tick_params(axis="y", labelsize=16)
        ax.set_ylabel(display, fontsize=YLABEL_FONTSIZE)
        ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        lo = min(0.0, min(m - e for m, e in zip(means, errs)))
        hi = max(0.0, max(m + e for m, e in zip(means, errs)))
        pad = (hi - lo) * 0.08
        ax.set_ylim(lo - pad, hi + pad)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def build_data(version: str, scores: dict):
    data: dict[tuple[str, str, str], dict[int, float]] = {}
    present: dict[str, list[str]] = {}
    for target in mcb.TARGETS:
        present[target] = []
        for cond, _l, _c in CONDITIONS:
            cell = mcb.cell_path(target, cond)
            if cell is None or not cell.is_dir():
                print(f"[skip] {target} {cond} (no cell)")
                continue
            present[target].append(cond)
            data[(target, cond, "winrate")] = mcb.load_measure(cell, "winrate")
        td = scores.get(target, {})
        for mkey in JUDGE_MEASURE_KEYS:
            view, tag = js.version_view(td, present[target], mkey, version)
            if tag == "opus5-fallback" and version != "opus5":
                print(f"[warn] {target} {mkey}: opus-5 fallback in {version}")
            for cond in present[target]:
                data[(target, cond, mkey)] = view.get(cond, {})
        print(
            f"[load] {target:<16} "
            + "  ".join(
                f"{k}={min(len(data.get((target, c, k), {})) for c in present[target]):>3}"
                for k in ["winrate"] + JUDGE_MEASURE_KEYS
            )
        )
    return data, present


def run_version(version: str, scores: dict) -> None:
    global CONDITIONS
    data, present = build_data(version, scores)
    ea_label = "Eval awareness"
    # Paper figure: the 3-panel layout (big win-rate panel). The 4-panel
    # 2x2 companion adds seed adherence (bottom-left, concerning
    # bottom-right) and is the only layout produced per judge.
    measures4 = [
        ("winrate", "Realism win rate", "top_left"),
        (EA_KEY, ea_label, "top_right"),
        ("s_seed_adherence", "Seed adherence", "bot_left"),
        ("concerning", "Concerning", "bot_right"),
    ]
    measures3 = [
        ("winrate", "Realism win rate", "left"),
        (EA_KEY, ea_label, "top_right"),
        ("concerning", "Concerning", "bot_right"),
    ]
    measure_keys = ["winrate", EA_KEY, "concerning", "s_seed_adherence"]
    suffix = "" if version == "avg" else f"_{version}"

    all_raw = mcb.pool(mcb.TARGETS, CONDITIONS, measure_keys, data, present)
    if version == "avg":
        render_fig(all_raw, measures3, OUT_DIR / "plot.png")
    render_fig(all_raw, measures4, OUT_DIR / f"plot_4panel{suffix}.png")
    mcb.write_csv(all_raw, OUT_DIR, f"summary{suffix}.csv", measure_keys)
    if version != "avg":
        # Per-judge versions carry only the 4-panel figure; every other
        # variant exists as judge-avg only.
        return

    # Same figure with method-style legend labels (identical data):
    # Petri / + realism filter / + cr2bo4.
    orig_conditions = CONDITIONS
    alt_label = {
        "baseline": "Petri",
        "realism_filter": "+ realism filter",
        "cr2bo4": "+ cr2bo4",
    }
    CONDITIONS = [(f, alt_label[f], c) for f, _l, c in orig_conditions]
    render_fig(all_raw, measures3, OUT_DIR / f"plot_petri_labels{suffix}.png")
    CONDITIONS = orig_conditions

    # Variant without the realism filter: Baseline vs cr2bo4 only, same
    # layout. Data is rebuilt so the fairness rule (and the win-rate seed
    # pooling) runs over just the two compared methods — seeds are not
    # dropped because of realism-filter judge gaps.
    CONDITIONS = [
        ("baseline", "Petri", "tab:blue"),
        ("cr2bo4", "cr2bo4", "tab:orange"),
    ]
    mcb.CONDITIONS = CONDITIONS
    data2, present2 = build_data(version, scores)
    all_raw2 = mcb.pool(mcb.TARGETS, CONDITIONS, measure_keys, data2, present2)
    render_fig(all_raw2, measures3, OUT_DIR / f"plot_x{suffix}.png")
    mcb.write_csv(all_raw2, OUT_DIR, f"summary_x{suffix}.csv", measure_keys)

    # Overview-diagram version: x_overview.png (paper repo) on the left,
    # win rate + eval awareness stacked on the right.
    if OVERVIEW_IMG.is_file():
        render_x_overview_fig(
            all_raw2, OUT_DIR / f"plot_x_overview{suffix}.png", OVERVIEW_IMG
        )
    else:
        print(f"[warn] overview image missing: {OVERVIEW_IMG}")

    # Paired-difference companion: per target/measure the per-seed
    # (cr2bo4 - Petri) mean with a paired 95% CI.
    diffs = {}
    diff_rows = []
    for target in mcb.TARGETS:
        for mkey in measure_keys:
            db = data2.get((target, "baseline", mkey), {})
            dc = data2.get((target, "cr2bo4", mkey), {})
            common2 = sorted(set(db) & set(dc))
            vals = [dc[sd] - db[sd] for sd in common2]
            m = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            diffs[(target, mkey)] = (m, 1.96 * sem)
            diff_rows.append([target, mkey, m, 1.96 * sem, len(vals)])
    render_diff_fig(diffs, OUT_DIR / f"plot_x_diff{suffix}.png")
    import csv as _csv
    with open(OUT_DIR / f"summary_x_diff{suffix}.csv", "w",
              newline="") as f:
        w = _csv.writer(f)
        w.writerow(["target", "measure", "mean_diff", "ci95_half", "n"])
        w.writerows(diff_rows)

    CONDITIONS = orig_conditions
    mcb.CONDITIONS = orig_conditions

    ctrl = mcb.controlled_seeds(mcb.TARGETS, data, max_drop=0.0)
    target_n = {t: len(ctrl[t]) for t in mcb.TARGETS}
    all_raw_ctrl = mcb.pool(
        mcb.TARGETS, CONDITIONS, measure_keys, data, present, allowed_seeds=ctrl
    )
    strict_measures = [
        ("winrate", "Realism win rate", "top_left"),
        (EA_KEY, ea_label, "top_right"),
        ("s_seed_adherence", "Seed adherence", "bot_left"),
        ("concerning", "Concerning", "bot_right"),
    ]
    render_fig(
        all_raw_ctrl, strict_measures,
        OUT_DIR / f"plot{suffix}_controlled_strict.png",
        target_n=target_n,
        title="Controlled for seed adherence (cr2bo4 ≥ baseline per seed)",
    )
    mcb.write_csv(
        all_raw_ctrl, OUT_DIR, f"summary{suffix}_controlled_strict.csv", measure_keys
    )

    # Relaxed control (judge-avg version only): allow cr2bo4's per-seed
    # adherence to sit up to 0.5 points (1-10 scale) below baseline's.
    if version == "avg":
        max_drop = 0.5 / 9.0
        ctrl_r = mcb.controlled_seeds(mcb.TARGETS, data, max_drop)
        n_r = {t: len(ctrl_r[t]) for t in mcb.TARGETS}
        all_raw_r = mcb.pool(
            mcb.TARGETS, CONDITIONS, measure_keys, data, present,
            allowed_seeds=ctrl_r,
        )
        render_fig(
            all_raw_r, strict_measures,
            OUT_DIR / "plot_controlled_maxdrop05.png",
            target_n=n_r,
            title="",  # paper version: caption carries the control description
        )
        mcb.write_csv(all_raw_r, OUT_DIR, "summary_controlled_maxdrop05.csv",
                      measure_keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judge", choices=VERSIONS + ["all"], default="all")
    ap.add_argument("--refresh-scores", action="store_true")
    args = ap.parse_args()

    mcb.CONDITIONS = CONDITIONS  # write_csv/pool read these globals
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = js.load(refresh=args.refresh_scores)
    for version in VERSIONS if args.judge == "all" else [args.judge]:
        print(f"\n=== version: {version} ===")
        run_version(version, scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
