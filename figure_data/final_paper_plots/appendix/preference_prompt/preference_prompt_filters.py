#!/usr/bin/env python3
"""Compact aggregate-only comparison for the realism paper.

Compares 4 preference-prompt ablation conditions on 4 measures:
  - Realism win rate (sourced from realism_win_rate/full_*.json)
  - Verbalized Evaluation Awareness (EA_Count)
  - Messages before awareness (Ne_when_EA)
  - Concerning (concerning, the v1 multi-judge measure — NOT concerning_v2)

All measures are pooled across (target, auditor) cells per condition, then
normalized to the first condition (=1.0) for a shared y-axis. Bars show the
normalized mean ± 95% CI; absolute means/CIs/n are NOT annotated on bars
(those are intended for the figure caption) but are printed to stdout.

Reuses CSV/JSON-discovery and stats helpers from the sibling `plot.py`.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PLOT_HELPERS_DIR = (Path(__file__).parent / ".." / "decide_pm_prompt").resolve()
if str(_PLOT_HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_HELPERS_DIR))
import plot as _plot  # noqa: E402

# ── Config ───────────────────────────────────────────────────────────────────

ROOT = Path("outputs/decide_pm_bon")

# (subfolder_name, display_label, color). First entry is the baseline for
# normalization (=1.0). Order: Baseline, Basic, Target perspective, Realism filter,
# Realism filter logprob.
_TAB10 = plt.get_cmap("tab10").colors
CONDITIONS = [
    ("baseline", "Baseline", _TAB10[0]),
    ("prompt_basic", "Basic", _TAB10[2]),
    ("prompt_selfp_basic", "Target perspective", _TAB10[9]),
    ("prompt_score_rf", "Realism filter", _TAB10[3]),
    ("prompt_full_aud_rf", "Realism filter logprob", _TAB10[1]),
]

TARGETS = ["haiku-4.5", "sonnet-4.6"]
AUDITORS: list[str] | None = None

SEED_FILTER: list[int] | None = [
    56,
    61,
    65,
    70,
    78,
    80,
    87,
    93,
    102,
    161,
    172,
    173,
    175,
]
SEED_COLUMN = "Seed Index"

# (key, display_label, arrow). Sentinel "win_rate" sources from
# realism_win_rate/full_*.json. `concerning` is the v1 (multi-judge) measure.
MEASURES = [
    ("win_rate", "Realism win rate", "↑"),
    ("EA_Count", "Evaluation awareness", "↓"),
    ("Ne_when_EA", "Messages before awareness", "↑"),
    ("concerning", "Concerning", "↑"),
]

CONFIDENCE_LEVEL = 0.95
PLOTS_ROOT = Path(__file__).parent / "plots"
RUN_NAME: str | None = "filters"

FIG_SIZE = (14, 6)
DPI = 200
BAR_WIDTH = 0.18
# Spacing between measure groups (1.0 = bars touch the next group; >1 adds a gap).
MEASURE_GAP = 1.15


# ── Data loading ─────────────────────────────────────────────────────────────


def _load_win_rate_values(csv_path: Path) -> tuple[list[float], int]:
    """Return (per-needle win rates pooled across haystacks, num_needles).

    `num_needles` comes from `summary.num_needles` in the win-rate JSON, i.e. the
    number of distinct needle audit transcripts judged in this cell. Pooling
    across cells, the caller sums these to report a per-condition needle count.
    """
    wr_json = _plot.find_win_rate_json(csv_path)
    if wr_json is None:
        return [], 0
    try:
        with open(wr_json, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return [], 0
    results = payload.get("results", [])
    per_hs = _plot.per_needle_win_rates(results)
    pooled: list[float] = []
    for hs in _plot.HAYSTACK_SOURCES:
        pooled.extend(per_hs.get(hs, []))
    num_needles = int(payload.get("summary", {}).get("num_needles", 0))
    return pooled, num_needles


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    _plot.SEED_FILTER = SEED_FILTER
    _plot.SEED_COLUMN = SEED_COLUMN

    z = _plot.get_z_score(CONFIDENCE_LEVEL)

    condition_csvs: list[dict[tuple[str, str], Path]] = []
    print("\n── Discovered cells per condition ─────────────────────────────────")
    for cond_name, label, _color in CONDITIONS:
        found = _plot.discover_csvs(ROOT / cond_name)
        found = {k: v for k, v in found.items() if k[0] in TARGETS}
        condition_csvs.append(found)
        # Surface the actual preference_prompt_name from the first config we
        # encounter, so the user can verify the folder→prompt mapping.
        sample_prompt = ""
        for _k, p in sorted(found.items()):
            cfg = p.parent / "config.yaml"
            if cfg.exists():
                for line in cfg.read_text().splitlines():
                    if "preference_prompt_name" in line:
                        sample_prompt = line.strip()
                        break
            if sample_prompt:
                break
        print(f"  [{label}] {cond_name}: {len(found)} cells  ({sample_prompt})")

    if AUDITORS is not None:
        auditors = list(AUDITORS)
    else:
        auditor_sets = [{a for (_t, a) in d} for d in condition_csvs]
        auditors = sorted(set.intersection(*auditor_sets)) if auditor_sets else []
    print(f"  Using auditors: {auditors}")
    print(f"  Using targets:  {TARGETS}")

    # Pool per-measure, per-condition values.
    pooled: dict[tuple[str, str], list[float]] = {}
    # For win_rate, `n` reported in the printed caption is the total number of
    # *needles* (summed across cells), not the number of (needle, haystack)
    # comparisons used for SE/CI.
    needle_counts: dict[tuple[str, str], int] = {}
    for column, _display, _arrow in MEASURES:
        for c_idx, (cond_name, _label, _color) in enumerate(CONDITIONS):
            vals: list[float] = []
            n_count = 0
            for t in TARGETS:
                for a in auditors:
                    csv_path = condition_csvs[c_idx].get((t, a))
                    if csv_path is None:
                        continue
                    if column == "win_rate":
                        v, nn = _load_win_rate_values(csv_path)
                        vals.extend(v)
                        n_count += nn
                    else:
                        rows = _plot.load_csv_rows(csv_path)
                        vals.extend(_plot.extract_measure_values(rows, column))
            pooled[(column, cond_name)] = vals
            needle_counts[(column, cond_name)] = n_count

    # Precompute raw means/CIs and the displayed n.
    raw: dict[tuple[str, str], tuple[float, float, int]] = {}
    for column, _display, _arrow in MEASURES:
        for cond_name, _label, _color in CONDITIONS:
            vals = pooled[(column, cond_name)]
            mean, err = _plot.mean_and_err(vals, z)
            n_display = (
                needle_counts[(column, cond_name)]
                if column == "win_rate"
                else len(vals)
            )
            raw[(column, cond_name)] = (mean, err, n_display)

    base_cond = CONDITIONS[0][0]
    base_label = CONDITIONS[0][1]

    # Output directory.
    base_name = RUN_NAME or ROOT.name or "run"
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = PLOTS_ROOT / f"{base_name}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f0f0f0")

    n_meas = len(MEASURES)
    n_cond = len(CONDITIONS)
    x = np.arange(n_meas) * MEASURE_GAP
    offsets = (np.arange(n_cond) - (n_cond - 1) / 2) * BAR_WIDTH

    max_norm_top = 1.0
    min_norm_bot = 1.0

    for c_idx, (cond_name, label, color) in enumerate(CONDITIONS):
        norm_means: list[float] = []
        norm_errs: list[float] = []
        for column, _display, _arrow in MEASURES:
            mean, err, _n = raw[(column, cond_name)]
            base_mean, _be, _bn = raw[(column, base_cond)]
            denom = (
                base_mean
                if base_mean not in (0.0, None) and not np.isnan(base_mean)
                else np.nan
            )
            if np.isnan(mean) or np.isnan(denom):
                norm_means.append(np.nan)
                norm_errs.append(0.0)
            else:
                norm_means.append(mean / denom)
                norm_errs.append(err / abs(denom))

        ax.bar(
            x + offsets[c_idx],
            norm_means,
            width=BAR_WIDTH,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            yerr=norm_errs,
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "black"},
            label=label,
            zorder=3,
        )

        for nm, ne in zip(norm_means, norm_errs):
            if np.isnan(nm):
                continue
            top = nm + ne
            bot = nm - ne
            max_norm_top = max(max_norm_top, top)
            min_norm_bot = min(min_norm_bot, bot)

    # Baseline reference line at y = 1.
    ax.axhline(1.0, color="#555", linestyle="--", linewidth=1.0, zorder=2)

    # Font sizes: figure is 14" wide but rendered at ~5.5" (NeurIPS \linewidth),
    # so matplotlib fonts get scaled down by ~0.39. Boosting to ~22pt here makes
    # the rendered PDF text roughly match the paper's body/caption size.
    # Long measure labels are wrapped to two lines so they fit within their
    # cluster without overlapping.
    _measure_labels = {
        "Realism win rate": "Realism\nwin rate",
        "Evaluation awareness": "Evaluation\nawareness",
        "Messages before awareness": "Messages before\nawareness",
        "Concerning": "Concerning",
    }
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{_measure_labels.get(d, d)} ({arr})" for _c, d, arr in MEASURES],
        fontsize=22,
    )
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    pad = 0.15 * max(max_norm_top - min_norm_bot, 0.1)
    ax.set_ylim(max(0.0, min_norm_bot - pad), max_norm_top + 0.05)

    ax.legend(loc="upper right", frameon=False, fontsize=20, ncol=3)

    fig.tight_layout()
    out_png = output_dir / "ablation_comparison.png"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved {out_png}")

    # ── CSV dump ─────────────────────────────────────────────────────────────
    csv_out = output_dir / "ablation_data.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "measure",
                "display",
                "condition",
                "label",
                "n",
                "mean",
                "ci_half",
                "ci_low",
                "ci_high",
                "normalized_mean",
                "normalized_ci_half",
            ],
        )
        w.writeheader()
        for column, display, _arrow in MEASURES:
            base_mean, _be, _bn = raw[(column, base_cond)]
            for cond_name, label, _color in CONDITIONS:
                mean, err, n = raw[(column, cond_name)]
                denom = (
                    base_mean
                    if base_mean not in (0.0, None) and not np.isnan(base_mean)
                    else np.nan
                )
                norm_mean = (
                    mean / denom
                    if not np.isnan(mean) and not np.isnan(denom)
                    else float("nan")
                )
                norm_err = (
                    err / abs(denom)
                    if not np.isnan(err) and not np.isnan(denom)
                    else float("nan")
                )
                w.writerow(
                    {
                        "measure": column,
                        "display": display,
                        "condition": cond_name,
                        "label": label,
                        "n": n,
                        "mean": mean,
                        "ci_half": err,
                        "ci_low": mean - err,
                        "ci_high": mean + err,
                        "normalized_mean": norm_mean,
                        "normalized_ci_half": norm_err,
                    }
                )
    print(f"Saved {csv_out}")

    # ── Caption-ready absolute values ────────────────────────────────────────
    print()
    print("=" * 100)
    print(
        f"ABSOLUTE VALUES FOR CAPTION — pooled across targets × auditors, "
        f"{int(CONFIDENCE_LEVEL * 100)}% CI"
    )
    print(f"  (Reference for normalization: {base_label})")
    print("=" * 100)
    header = f"{'measure':<35}" + "".join(
        f"  {label:>26}" for _, label, _ in CONDITIONS
    )
    print(header)
    print("-" * len(header))
    for column, display, arrow in MEASURES:
        line = f"{display + ' (' + arrow + ')':<35}"
        for cond_name, _label, _color in CONDITIONS:
            mean, err, n = raw[(column, cond_name)]
            n_label = "needles" if column == "win_rate" else "transcripts"
            line += f"  {mean:>7.3f} ± {err:<6.3f} (n={n} {n_label[:1]})"[:28].ljust(28)
        print(line)
    print("=" * 100)

    # ── Pairwise diffs vs reference (also caption-friendly) ──────────────────
    other_conds = [c for c in CONDITIONS[1:]]
    print()
    print("=" * 100)
    print(
        f"DIFFERENCES vs {base_label} — {int(CONFIDENCE_LEVEL * 100)}% CI on the diff"
    )
    print("=" * 100)
    diff_header = f"{'measure':<35}" + "".join(
        f"  {'Δ(' + label + '−' + base_label + ')':>26}" for _, label, _ in other_conds
    )
    print(diff_header)
    print("-" * len(diff_header))
    for column, display, arrow in MEASURES:
        base_mean, base_err, _bn = raw[(column, base_cond)]
        line = f"{display + ' (' + arrow + ')':<35}"
        se_base = base_err / z if z else 0.0
        for cond_name, _label, _color in other_conds:
            c_mean, c_err, _cn = raw[(column, cond_name)]
            se_c = c_err / z if z else 0.0
            se_diff = (se_base**2 + se_c**2) ** 0.5
            ci_diff = z * se_diff
            diff = c_mean - base_mean
            line += f"  {diff:>+8.3f} ± {ci_diff:<7.3f}".ljust(28)
        print(line)
    print("=" * 100)


if __name__ == "__main__":
    main()
