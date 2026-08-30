#!/usr/bin/env python3
"""APPENDIX FIGURE — logprob-selector scaling with all three targets overlaid.

The logprob twin of `../../plot_scaling_all_targets.py` (same styling via its
monkeypatches): cost-vs-quality grid (realism win rate, eval awareness,
concerning, seed adherence), one colour per target (Sonnet 4.6 / Opus 4.7 /
Haiku 4.5, auditor opus-4.7), data from `outputs/ablation_protocol_no_stopping`
through the `logprob_cells.build_shim` symlink tree. Per target the full depth
ladder (baseline → cr1 → … → cr11, +cr22 sonnet / +cr16 opus) — depth ONLY,
no combination ladder; every cr rung is labelled on every target and a single
Baselines callout sits above the topmost (haiku) baseline dot. Costs are
the ANALYTICAL model (app:cost-model), normalised per
target by that target's baseline. Judge panels read each cell's
`0_transcripts.csv` (opus-5 primary judge; the only judge on these cells).

Everything lands in `scaling_all_targets_logprob/`:

  plot.png / summary.csv / config.json / has_ea.png   main 2x2
  plot_2panel.png                                     1x2 win rate + EA
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FPP = HERE.parents[1]  # final_paper_plots (judge_scores.py etc.)
CWR = HERE.parents[2] / "cost_winrate_reasoning"
for p in (CWR, FPP, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import logprob_cells as lp  # noqa: E402
import plot_scaling_all_targets as sat  # noqa: E402  (paper-styling patches)
import plot_cost_grid_combined_targets as base  # noqa: E402

OUT_DIR = HERE / "scaling_all_targets_logprob"
TMP_PLOTS = HERE / ".tmp_plots"

TARGETS = ["opus-4.7", "haiku-4.5", "sonnet-4.6"]
# Same palette as the cr_improvement_logprob bars (fig 12): haiku orange,
# sonnet blue (the standard sonnet-4.6 hue across the scaling figures),
# opus green.
TARGET_COLOR = {
    "opus-4.7": "tab:green",
    "haiku-4.5": "tab:orange",
    "sonnet-4.6": "tab:blue",
}

# ── Point the combined-targets base at the logprob cells ─────────────────────

# Judge panels read straight from the cell CSVs: with no keys registered,
# sat's patched score_by_seed always falls through to the real loader.
sat.JUDGE_KEYS = []

base.AUDITOR = lp.AUDITOR
base.TARGETS = TARGETS
base.TARGET_COLOR = TARGET_COLOR
base.TARGET_DISPLAY = dict(lp.TARGET_DISPLAY)
base.TARGET_DEPTH = {t: list(lp.TARGET_DEPTH[t]) for t in TARGETS}
base.TARGET_COMBO = {t: [] for t in TARGETS}  # depth ladders only
base.COMBO_ANCHOR = "cr2"
base.WINRATE_ANNOTATE_TARGETS = list(TARGETS)
base.WINRATE_ANNOTATE_FOLDERS = ["cr1", "cr2", "cr4", "cr8", "cr11",
                                 "cr16", "cr22"]
base.WINRATE_BASELINE_TARGET = {
    "winrate": "haiku-4.5",
    "eval_awareness": "haiku-4.5",
    "concerning": "haiku-4.5",
    "s_seed_adherence": "sonnet-4.6",
}
# Rung labels on every panel, not just realism win rate.
base.ANNOTATE_PANEL_KEYS = ("winrate", "eval_awareness", "concerning",
                            "s_seed_adherence")
base.cost_by_seed = lp.analytical_cost_by_seed
base.baseline_cost_by_seed = lp.analytical_cost_by_seed

# ── Restrict every target to the random scaling subset ──────────────────────
# Resolved per target: haiku keeps the prefill-only seeds (54 total), the
# no-prefill targets have them stripped. Haiku's shallow rungs were run on
# the full seed set, so without this filter its curve would average over a
# different (much larger) seed population than sonnet/opus.
_CPNS = HERE.parents[2] / "compare_protocols_no_stopping"
if str(_CPNS) not in sys.path:
    sys.path.insert(0, str(_CPNS))
from plot_cost_winrate_scaling import (  # noqa: E402
    _load_seed_categories,
    _resolve_category_seeds,
)

_CATS = _load_seed_categories()
_RANDOM = {
    t: _resolve_category_seeds(t, "random", _CATS) or set() for t in TARGETS
}
for _t in TARGETS:
    print(f"[subset] {_t}: random subset of {len(_RANDOM[_t])} seeds")


def _target_of(cell_dir) -> str | None:
    for part in Path(cell_dir).parts:
        if part.startswith("target_"):
            return part.split("target_", 1)[1]
    return None


def _subset_filter(fn):
    def wrapped(cell_dir, *args, **kwargs):
        out = fn(cell_dir, *args, **kwargs)
        allowed = _RANDOM.get(_target_of(cell_dir))
        if allowed:
            out = {sd: v for sd, v in out.items() if sd in allowed}
        return out
    return wrapped


base.winrate_by_seed = _subset_filter(base.winrate_by_seed)
base.score_by_seed = _subset_filter(base.score_by_seed)
base.cost_by_seed = _subset_filter(lp.analytical_cost_by_seed)
base.baseline_cost_by_seed = _subset_filter(lp.analytical_cost_by_seed)

# base.main() filters its shared legend through a LOCAL "reasoning targets"
# order, which would drop opus-4.7 / haiku-4.5. Replace the handles wholesale
# (sat's own wrapper still prepends the bold "Target model:" entry).
_SAT_FIG_LEGEND = base.plt.Figure.legend


def _lp_fig_legend(self, *args, **kwargs):
    if kwargs.get("handles") is not None:
        kwargs["handles"] = [
            base.plt.Line2D(
                [0], [0], marker="o", color=TARGET_COLOR[t], markersize=15,
                linewidth=0, label=lp.TARGET_DISPLAY[t],
            )
            for t in ["haiku-4.5", "sonnet-4.6", "opus-4.7"]
        ]
        if "ncol" in kwargs:
            kwargs["ncol"] = len(kwargs["handles"])
    return _SAT_FIG_LEGEND(self, *args, **kwargs)


base.plt.Figure.legend = _lp_fig_legend

# Track which panel is being rendered so the annotate hook can shrink the
# rung labels outside the (large, uncluttered) win-rate panel.
_CURRENT_PANEL_KEY = "winrate"
_SAT_RENDER_PANEL = base.render_panel


def _lp_render_panel(ax, panel, *args, **kwargs):
    global _CURRENT_PANEL_KEY
    _CURRENT_PANEL_KEY = panel["key"]
    return _SAT_RENDER_PANEL(ax, panel, *args, **kwargs)


base.render_panel = _lp_render_panel

_AX_ANNOTATE = base.plt.Axes.annotate


def _lp_annotate(self, text, *args, **kwargs):
    if text != "Baselines" and _CURRENT_PANEL_KEY == "eval_awareness":
        # All three curves collapse to ~0 there; rung labels are unreadable,
        # so the eval-awareness panel keeps only the Baselines callout.
        return None
    # "Baselines" keeps the base offset (30, 34): starts above the dot and
    # runs right, exactly like the sonnet-types figures. The seed-adherence
    # baseline dot sits near the panel top, so there the label drops just
    # low enough to stay inside the axes.
    if text == "Baselines" and _CURRENT_PANEL_KEY == "s_seed_adherence":
        kwargs["xytext"] = (30, 20)
    if text == "Baselines" and _CURRENT_PANEL_KEY == "winrate":
        # Lowered from the base (30, 34) so the callout hugs the Haiku
        # baseline dot instead of floating toward the cr1 label.
        kwargs["xytext"] = (30, 24)
    if text == "cr11":
        # cr8/cr11 dots sit close on every ladder; cr11 goes to the right
        # of its dot.
        kwargs["xytext"] = (8, 7)
        kwargs["ha"] = "left"
    elif text in ("cr16", "cr22"):
        # Last rungs (opus cr16, sonnet cr22): to the right, clear of the
        # right-shifted cr11 label.
        kwargs["xytext"] = (8, 7)
        kwargs["ha"] = "left"
    return _AX_ANNOTATE(self, text, *args, **kwargs)


base.plt.Axes.annotate = _lp_annotate

# (layout suffix, panel keys or None for all four) — matches the main figure.
LAYOUTS = [
    ("", None),  # main 2x2
    ("_2panel", {"winrate", "eval_awareness"}),  # 1x2 alternative
]


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()

    shim = lp.build_shim(
        {t: base.TARGET_DEPTH[t] + base.TARGET_COMBO[t] for t in TARGETS}
    )
    base.REPO_ROOT = shim
    base.PLOTS_ROOT = TMP_PLOTS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for laysuffix, panel_filter in LAYOUTS:
        sat._PANEL_FILTER = panel_filter
        base.PANELS = (
            sat._PRISTINE_PANELS
            if panel_filter is None
            else [p for p in sat._PRISTINE_PANELS if p["key"] in panel_filter]
        )
        rc = base.main()
        if rc != 0:
            return rc
        src = TMP_PLOTS / "cost_grid_combined_targets"
        if panel_filter is None:
            moves = [("plot", "png"), ("has_ea", "png"),
                     ("summary", "csv"), ("config", "json")]
        else:
            moves = [("plot", "png")]
        for name, ext in moves:
            f = src / f"{name}.{ext}"
            if f.is_file():
                f.replace(OUT_DIR / f"{name}{laysuffix}.{ext}")
        shutil.rmtree(src, ignore_errors=True)
        print(f"[done] -> {OUT_DIR}/plot{laysuffix}.png")
    shutil.rmtree(TMP_PLOTS, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
