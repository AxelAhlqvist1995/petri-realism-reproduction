#!/usr/bin/env python3
"""Appendix-B logprob C&R results in the `cr_comparison/plot.png` style:
the METHOD decides the bar colour (baseline blue / realism filter green /
cr11 orange), targets are the x-tick groups, no hatching.

Same 3-panel layout as the original cr_improvement_combined plot. The bottom
legend lists the methods.

Output is overwritten in-place at `plots/` next to this script on every run
(appendix logprob C&R results; targets ordered haiku, sonnet, opus).
"""

from __future__ import annotations

import csv as _csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Thicker hatch lines so the pattern stays visible at paper rendering size.
plt.rcParams["hatch.linewidth"] = 1.0

_HERE = Path(__file__).parent
_NO_STOPPING = _HERE.parents[2] / "compare_protocols_no_stopping"
for p in (_NO_STOPPING, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import plot_crs_scaling as _crs  # noqa: E402

_pa = _crs._pa
_plot = _crs._plot

# ── Inject averaged win rate ─────────────────────────────────────────────────

_orig_loader = _pa._load_csv_rows_walkup


def _load_csv_rows_with_avg_win_rate(path: Path) -> list[dict[str, str]]:
    rows = _orig_loader(path)
    for row in rows:
        vals: list[float] = []
        for col in ("realism_win_rate", "realism_win_rate_cc"):
            raw = row.get(col, "").strip()
            if not raw:
                vals = []
                break
            try:
                v = float(raw)
            except ValueError:
                vals = []
                break
            if v != v:
                vals = []
                break
            vals.append(v)
        if len(vals) == 2:
            row["realism_win_rate_avg"] = f"{sum(vals) / len(vals)}"
    return rows


_pa._load_csv_rows_walkup = _load_csv_rows_with_avg_win_rate


# ── Config ───────────────────────────────────────────────────────────────────

# _crs.ROOT is CWD-relative; resolve against the repo root so this script
# runs from anywhere.
ROOT = _HERE.parents[4] / "outputs" / "ablation_protocol_no_stopping"

# Method colours match cr_comparison/plot.png (cr rung takes the orange).
CONDITIONS: list[tuple[str, str, str]] = [
    ("baseline", "Baseline", "tab:blue"),
    ("realism_filter", "Realism filter", "tab:green"),
    ("cr11", "cr11", "tab:orange"),
]

TARGETS = ["haiku-4.5", "sonnet-4.6", "opus-4.7"]
AUDITORS = _pa.AUDITORS

_PREFILL_ONLY: set[int] = {5, 6, 24, 88, 97, 105, 106, 107}
EXCLUDE_SEEDS: dict[str, set[int]] = {
    "sonnet-4.6": _PREFILL_ONLY,
    "opus-4.7": _PREFILL_ONLY,
}

SEED_DATASETS = _crs.SEED_DATASETS
SEED_DATASET: str | None = "filtered"

MEASURES = [
    ("realism_win_rate_avg", "Realism win rate", "↑"),
    ("Has_EA", "Eval awareness", "↓"),
    ("concerning_norm", "Concerning", "↑"),
]
# The adherence-controlled companion figure (paper fig 17) adds the fourth
# panel; grid order matches the cr4-selector figure (adherence bottom-right).
MEASURES4 = [
    MEASURES[0],
    MEASURES[1],
    ("s_seed_adherence_norm", "Seed adherence", "↑"),
    MEASURES[2],
]
ADHERENCE_COLUMN = "s_seed_adherence_norm"
ADHERENCE_BASELINE_COND = "baseline"
ADHERENCE_TEST_COND = "cr11"
ADHERENT_FIG_SIZE = (14, 8.4)
ADHERENT_YLABEL_FONTSIZE = 18
SEED_COLUMN = _pa.SEED_COLUMN
CONFIDENCE_LEVEL = _pa.CONFIDENCE_LEVEL
PLOTS_ROOT = _HERE / "plots"

PER_MEASURE_EXCLUDE = {"Ne_when_EA", "target_turns_before_first_ea"}

FIG_SIZE = (16, 6.6)
DPI = 200
BAR_WIDTH = 0.22
TARGET_GAP = 0.40
PANEL_LAYOUT: dict[str, str] = {
    "realism_win_rate_avg": "left",
    "Has_EA": "top_right",
    "concerning_norm": "bot_right",
}

PANEL_YMAX: dict[str, float] = {
    "realism_win_rate_avg": 0.9,
}

# Per-measure y-label font size — the left "Realism win rate" panel is taller
# than the right-column panels, so it can carry a larger label.
PANEL_YLABEL_FONTSIZE: dict[str, float] = {
    "realism_win_rate_avg": 22,
    "Has_EA": 18,
    "concerning_norm": 18,
}
DEFAULT_YLABEL_FONTSIZE = 16

# Target → color (mirrors the seed adherence + scaling plot palette).
TARGET_COLORS: dict[str, str] = {
    "haiku-4.5": "tab:orange",
    "sonnet-4.6": "tab:blue",
    "opus-4.7": "tab:green",
}

# Condition → hatch pattern (rendered in black via edgecolor). Repeating the
# pattern character increases hatch density.
CONDITION_HATCHES: dict[str, str] = {
    "baseline": "",
    "realism_filter": "//",
    "cr11": "xx",
}


# ── Per-target pooling + rendering ───────────────────────────────────────────


def _pool_one_target(
    target: str,
    conditions: list[tuple[str, str, str]],
    auditors: list[str],
    condition_csvs: list[dict[tuple[str, str], Path]],
    per_target_allowed: dict[str, set[int] | None],
    per_measure_target_allowed: dict[str, dict[str, set[int] | None]],
    measures: list[tuple[str, str, str]],
    z: float,
) -> dict[tuple[str, str], tuple[float, float, int]]:
    raw: dict[tuple[str, str], tuple[float, float, int]] = {}
    for column, _display, _arrow in measures:
        if column in per_measure_target_allowed:
            allowed = per_measure_target_allowed[column].get(target)
        else:
            allowed = per_target_allowed.get(target)
        for c_idx, (cond_path, _label, _color) in enumerate(conditions):
            vals: list[float] = []
            for a in auditors:
                csv_path = condition_csvs[c_idx].get((target, a))
                if csv_path is None:
                    continue
                rows = _pa._load_csv_rows_walkup(csv_path)
                if allowed is not None:
                    kept = []
                    for row in rows:
                        s = row.get(SEED_COLUMN, "").strip()
                        if not s:
                            continue
                        try:
                            if int(float(s)) in allowed:
                                kept.append(row)
                        except ValueError:
                            continue
                    rows = kept
                vals.extend(_plot.extract_measure_values(rows, column))
            mean, err = _plot.mean_and_err(vals, z)
            raw[(column, cond_path)] = (mean, err, len(vals))
    return raw


def _per_seed_adherence(
    target: str,
    cond_path: str,
    auditors: list[str],
    condition_csvs: list[dict[tuple[str, str], Path]],
) -> dict[int, float]:
    """Mean s_seed_adherence_norm per seed for one (target, condition)."""
    import statistics
    from collections import defaultdict

    cond_idx = next(
        (i for i, (cp, _l, _c) in enumerate(CONDITIONS) if cp == cond_path), None
    )
    if cond_idx is None:
        return {}
    per_seed: dict[int, list[float]] = defaultdict(list)
    for a in auditors:
        csv_path = condition_csvs[cond_idx].get((target, a))
        if csv_path is None:
            continue
        for row in _pa._load_csv_rows_walkup(csv_path):
            raw = row.get(SEED_COLUMN, "").strip()
            if not raw:
                continue
            try:
                seed = int(float(raw))
            except ValueError:
                continue
            vals = _plot.extract_measure_values([row], ADHERENCE_COLUMN)
            if vals:
                per_seed[seed].append(vals[0])
    return {s: statistics.mean(v) for s, v in per_seed.items() if v}


def _seeds_where_cr11_ge_baseline(
    target: str,
    auditors: list[str],
    condition_csvs: list[dict[tuple[str, str], Path]],
) -> set[int]:
    """Seeds where cr11's mean seed adherence >= baseline's (per seed)."""
    base = _per_seed_adherence(
        target, ADHERENCE_BASELINE_COND, auditors, condition_csvs
    )
    test = _per_seed_adherence(
        target, ADHERENCE_TEST_COND, auditors, condition_csvs
    )
    return {s for s in base.keys() & test.keys() if test[s] >= base[s]}


def _target_display(t: str) -> str:
    if t.startswith("gpt"):
        return t.upper()
    return " ".join(p[:1].upper() + p[1:] for p in t.split("-"))


def _render_one_panel(
    ax,
    measure: tuple[str, str, str],
    targets: list[str],
    conditions: list[tuple[str, str, str]],
    all_raw: dict[str, dict[tuple[str, str], tuple[float, float, int]]],
    drawn_cond_handles: dict[str, object],
    drawn_target_handles: dict[str, object],
    hide_xticklabels: bool = False,
    show_n: bool = False,
    ylabel_fontsize: float | None = None,
) -> None:
    column, display, _arrow = measure
    n_cond = len(conditions)
    n_tgt = len(targets)

    cluster_w = n_cond * BAR_WIDTH
    x = np.arange(n_tgt) * (cluster_w + TARGET_GAP)

    ax.set_facecolor("#f0f0f0")

    max_top = 0.0

    for c_idx, (cond_path, label, color) in enumerate(conditions):
        offset = (c_idx - (n_cond - 1) / 2) * BAR_WIDTH
        for t_idx, target in enumerate(targets):
            mean, err, _n = all_raw[target][(column, cond_path)]
            if np.isnan(mean):
                continue
            ax.bar(
                [x[t_idx] + offset],
                [mean],
                width=BAR_WIDTH,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                yerr=[err],
                capsize=3,
                error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "black"},
                zorder=3,
            )
            max_top = max(max_top, mean + err)
        if label not in drawn_cond_handles:
            drawn_cond_handles[label] = plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=color,
                edgecolor="black",
            )

    ax.set_xticks(x)
    if hide_xticklabels:
        ax.set_xticklabels([""] * len(targets))
    else:
        xtick_labels = []
        for t in targets:
            lab = _target_display(t)
            if show_n:
                # THIS panel's N (per-measure intersections can differ).
                ns = [n for m, _e, n in
                      (all_raw[t][(column, cp)] for cp, _l, _c in conditions)
                      if not np.isnan(m)]
                if ns:
                    lab += f"\nN={max(ns)}"
            xtick_labels.append(lab)
        ax.set_xticklabels(xtick_labels, fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel(
        display,
        fontsize=ylabel_fontsize
        if ylabel_fontsize is not None
        else PANEL_YLABEL_FONTSIZE.get(column, DEFAULT_YLABEL_FONTSIZE),
    )
    ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ymax_override = PANEL_YMAX.get(column)
    ax.set_ylim(
        0.0, ymax_override if ymax_override is not None else max_top * 1.12 + 0.02
    )


def render_combined(
    targets: list[str],
    conditions: list[tuple[str, str, str]],
    auditors: list[str],
    condition_csvs: list[dict[tuple[str, str], Path]],
    per_target_allowed: dict[str, set[int] | None],
    per_measure_target_allowed: dict[str, dict[str, set[int] | None]],
    measures: list[tuple[str, str, str]],
    output_dir: Path,
    z: float,
    file_stem: str,
) -> None:
    all_raw: dict[str, dict[tuple[str, str], tuple[float, float, int]]] = {}
    for target in targets:
        all_raw[target] = _pool_one_target(
            target=target,
            conditions=conditions,
            auditors=auditors,
            condition_csvs=condition_csvs,
            per_target_allowed=per_target_allowed,
            per_measure_target_allowed=per_measure_target_allowed,
            measures=measures,
            z=z,
        )

    fig = plt.figure(figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.25, wspace=0.20)
    panel_axes = {
        "left": fig.add_subplot(gs[:, 0]),
        "top_right": fig.add_subplot(gs[0, 1]),
        "bot_right": fig.add_subplot(gs[1, 1]),
    }

    drawn_cond_handles: dict[str, object] = {}
    drawn_target_handles: dict[str, object] = {}
    for measure in measures:
        slot = PANEL_LAYOUT.get(measure[0])
        if slot is None or slot not in panel_axes:
            continue
        ax = panel_axes[slot]
        _render_one_panel(
            ax=ax,
            measure=measure,
            targets=targets,
            conditions=conditions,
            all_raw=all_raw,
            drawn_cond_handles=drawn_cond_handles,
            drawn_target_handles=drawn_target_handles,
            hide_xticklabels=(slot == "top_right"),
        )

    cond_handles = [
        drawn_cond_handles[c[1]] for c in conditions if c[1] in drawn_cond_handles
    ]
    cond_labels = [c[1] for c in conditions if c[1] in drawn_cond_handles]
    fig.legend(
        cond_handles,
        cond_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=16,
        ncol=len(cond_handles),
    )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    out_png = output_dir / f"{file_stem}.png"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_png}")

    csv_out = output_dir / f"{file_stem}.csv"
    with open(csv_out, "w", newline="") as f:
        w = _csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "measure",
                "display",
                "condition",
                "n",
                "mean",
                "ci_half",
            ],
        )
        w.writeheader()
        for target in targets:
            raw = all_raw[target]
            for column, display, _arrow in measures:
                for cond_path, _label, _color in conditions:
                    mean, err, n = raw[(column, cond_path)]
                    w.writerow(
                        {
                            "target": target,
                            "measure": column,
                            "display": display,
                            "condition": cond_path,
                            "n": n,
                            "mean": mean,
                            "ci_half": err,
                        }
                    )
    print(f"Saved {csv_out}")


def render_adherent_grid(
    targets: list[str],
    conditions: list[tuple[str, str, str]],
    all_raw: dict[str, dict[tuple[str, str], tuple[float, float, int]]],
    output_dir: Path,
    file_stem: str,
) -> None:
    """Paper fig 17: equal 2x2 grid (realism win rate / eval awareness /
    concerning / seed adherence), method-coloured bars in the cr4-selector
    figure's style, per-panel target+N labels (shown because the figure is
    adherence-controlled and per-measure Ns can differ)."""
    fig, axes = plt.subplots(2, 2, figsize=ADHERENT_FIG_SIZE)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.34, wspace=0.20)
    drawn_cond: dict[str, object] = {}
    drawn_tgt: dict[str, object] = {}
    for measure, (r, c) in zip(MEASURES4, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        _render_one_panel(
            ax=axes[r, c],
            measure=measure,
            targets=targets,
            conditions=conditions,
            all_raw=all_raw,
            drawn_cond_handles=drawn_cond,
            drawn_target_handles=drawn_tgt,
            hide_xticklabels=False,
            show_n=True,
            ylabel_fontsize=ADHERENT_YLABEL_FONTSIZE,
        )
    cond_handles = [
        drawn_cond[c[1]] for c in conditions if c[1] in drawn_cond
    ]
    cond_labels = [c[1] for c in conditions if c[1] in drawn_cond]
    fig.legend(
        cond_handles,
        cond_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=15,
        ncol=len(cond_handles),
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    out_png = output_dir / f"{file_stem}.png"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_png}")

    with open(output_dir / f"{file_stem}.csv", "w", newline="") as f:
        w = _csv.DictWriter(
            f,
            fieldnames=["target", "measure", "display", "condition", "n",
                        "mean", "ci_half"],
        )
        w.writeheader()
        for target in targets:
            raw = all_raw[target]
            for column, display, _arrow in MEASURES4:
                for cond_path, _label, _color in conditions:
                    mean, err, n = raw[(column, cond_path)]
                    w.writerow({
                        "target": target, "measure": column,
                        "display": display, "condition": cond_path,
                        "n": n, "mean": mean, "ci_half": err,
                    })
    print(f"Saved {output_dir / f'{file_stem}.csv'}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    _plot.SEED_COLUMN = SEED_COLUMN
    _plot.SEED_FILTER = None

    z = _plot.get_z_score(CONFIDENCE_LEVEL)

    condition_csvs: list[dict[tuple[str, str], Path]] = []
    plain_conditions: list[tuple[str, str, str]] = []
    for cond_path, label, color in CONDITIONS:
        found = _pa._discover_gpt54_csvs(ROOT / cond_path)
        found = {k: v for k, v in found.items() if k[0] in TARGETS}
        condition_csvs.append(found)
        plain_conditions.append((cond_path, label, color))
        print(f"{cond_path} (under {ROOT}): {len(found)} (target, auditor) cells")

    if AUDITORS is not None:
        auditors = list(AUDITORS)
    else:
        auditor_sets = [{a for (_t, a) in d} for d in condition_csvs if d]
        auditors = sorted(set.intersection(*auditor_sets)) if auditor_sets else []
    print(f"Using auditors: {auditors}")
    print(f"Using targets:  {TARGETS}")

    per_target_allowed: dict[str, set[int] | None] = {}
    print()
    print("Per-target seed intersection across non-empty conditions (CSV rows):")
    for t in TARGETS:
        per_cond_seeds: list[set[int]] = []
        for c_idx, (cond_path, _label, _color) in enumerate(CONDITIONS):
            seeds_here: set[int] = set()
            for a in auditors:
                csv_path = condition_csvs[c_idx].get((t, a))
                if csv_path is None:
                    continue
                seeds_here |= _pa._seeds_in_csv(csv_path)
            if seeds_here:
                per_cond_seeds.append(seeds_here)
                tag = ""
            else:
                tag = " (skipped — no cell)"
            print(f"  {t} / {cond_path}: {len(seeds_here)} seeds{tag}")
        inter = set.intersection(*per_cond_seeds) if per_cond_seeds else set()
        excluded = EXCLUDE_SEEDS.get(t, set()) & inter
        if excluded:
            inter = inter - excluded
            print(
                f"  → {t} intersection: {len(inter)} seeds (excluded {sorted(excluded)})"
            )
        else:
            print(f"  → {t} intersection: {len(inter)} seeds")
        if SEED_DATASET is not None:
            ds = SEED_DATASETS[SEED_DATASET]
            before = len(inter)
            inter = inter & ds
            print(
                f"  → {t} after '{SEED_DATASET}' dataset filter: {len(inter)} / {before} seeds"
            )
        per_target_allowed[t] = inter
    print()

    def _seeds_with_value(rows: list[dict[str, str]], column: str) -> set[int]:
        out: set[int] = set()
        for row in rows:
            raw = row.get(SEED_COLUMN, "").strip()
            if not raw:
                continue
            try:
                sid = int(float(raw))
            except ValueError:
                continue
            if _plot.extract_measure_values([row], column):
                out.add(sid)
        return out

    per_measure_target_allowed: dict[str, dict[str, set[int] | None]] = {}
    print(
        "Per-measure, per-target seed intersection (seeds producing a value in every non-empty condition):"
    )
    for column, display, _arrow in MEASURES4:
        if column in PER_MEASURE_EXCLUDE:
            print(f"  {display:<24} — using default per-target intersection")
            continue
        per_measure_target_allowed[column] = {}
        for t in TARGETS:
            per_cond_seeds: list[set[int]] = []
            for c_idx in range(len(CONDITIONS)):
                seeds_here: set[int] = set()
                for a in auditors:
                    csv_path = condition_csvs[c_idx].get((t, a))
                    if csv_path is None:
                        continue
                    rows = _pa._load_csv_rows_walkup(csv_path)
                    seeds_here |= _seeds_with_value(rows, column)
                if seeds_here:
                    per_cond_seeds.append(seeds_here)
            inter = set.intersection(*per_cond_seeds) if per_cond_seeds else set()
            inter = inter - EXCLUDE_SEEDS.get(t, set())
            if SEED_DATASET is not None:
                inter = inter & SEED_DATASETS[SEED_DATASET]
            per_measure_target_allowed[column][t] = inter
            print(f"  {display:<24} {t}: {len(inter)} seeds")
    print()

    output_dir = PLOTS_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    render_combined(
        targets=TARGETS,
        conditions=plain_conditions,
        auditors=auditors,
        condition_csvs=condition_csvs,
        per_target_allowed=per_target_allowed,
        per_measure_target_allowed=per_measure_target_allowed,
        measures=MEASURES,
        output_dir=output_dir,
        z=z,
        file_stem="cr_improvement_combined_targets",
    )

    # ── Adherence-controlled companion (paper fig 17) ────────────────────────
    # Per target keep only seeds where cr11's seed adherence >= baseline's.
    print()
    print("Adherence control (cr11 >= baseline per seed):")
    keep_by_target: dict[str, set[int]] = {}
    per_target_allowed_a: dict[str, set[int] | None] = {}
    for t in TARGETS:
        keep_by_target[t] = _seeds_where_cr11_ge_baseline(
            t, auditors, condition_csvs
        )
        base_allowed = per_target_allowed.get(t) or set()
        per_target_allowed_a[t] = base_allowed & keep_by_target[t]
        print(f"  {t}: {len(per_target_allowed_a[t])} of "
              f"{len(base_allowed)} seeds kept")
    per_measure_a: dict[str, dict[str, set[int] | None]] = {
        column: {
            t: (per_target.get(t) or set()) & keep_by_target[t]
            for t in TARGETS
        }
        for column, per_target in per_measure_target_allowed.items()
    }
    all_raw_a = {
        t: _pool_one_target(
            target=t,
            conditions=plain_conditions,
            auditors=auditors,
            condition_csvs=condition_csvs,
            per_target_allowed=per_target_allowed_a,
            per_measure_target_allowed=per_measure_a,
            measures=MEASURES4,
            z=z,
        )
        for t in TARGETS
    }
    render_adherent_grid(
        TARGETS, plain_conditions, all_raw_a, output_dir,
        "cr_improvement_combined_targets_adherent",
    )


if __name__ == "__main__":
    main()
