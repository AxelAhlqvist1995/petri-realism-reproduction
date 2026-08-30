#!/usr/bin/env python3
"""FINAL PAPER FIGURE — scaling with all four targets overlaid.

Cost-vs-quality grid (realism win rate, eval awareness, concerning, seed
adherence), one colour per target (Sonnet 4.6 / Opus 4.8 / GPT-5.5 /
Gemini 3.5 Flash, auditor sonnet-4.6). Per target: baseline → cr1 → cr2
(depth ladder) and cr2 → cr2bo2 → cr2bo4 (combination ladder, darker tone),
costs normalised by that target's baseline.

One folder per paper figure: everything lands in `scaling_all_targets/`.

  plot.png / summary.csv / config.json / has_ea.png   main 2x2 (judge avg)
  plot_opus5.png / ...                                opus-5-only judge
  plot_gpt56.png / ...                                gpt-5.6-only judge
  plot_2panel*.png                                    1x2 with realism win
                                                      rate + eval awareness

Paper styling (matching scaling_types_sonnet): plain measure labels without
judge tags — the filename names the judge — axis titles 24pt, ticks 17pt,
point annotations 19pt, shared dot legend below the grid.

Judge-dependent panels are served from `judge_scores.py` views with the
fairness rule: per target and measure, a seed only counts if the serving
judge scored it for EVERY rung (baseline, cr1, cr2, cr2bo2, cr2bo4); `avg`
averages the two judges where both are complete on a seed and falls back to
the single complete judge otherwise. `eval_awareness` / `concerning` are
FULL-judge measures with no gpt-5.6 pass on cr1/cr2/cr2bo2 for the non-sonnet
targets, so those panels fall back to opus-5 there in the gpt56/avg versions
(notes printed per run). Win rate / cost / Has_EA are judge-independent.

Loading/rendering reuses
`cost_winrate_reasoning/plot_cost_grid_combined_targets.py` (its imported
`score_by_seed` / `render_panel` are monkeypatched).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CWR = HERE.parent / "cost_winrate_reasoning"
for p in (CWR, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import judge_scores as js  # noqa: E402
import plot_cost_grid_combined_targets as base  # noqa: E402

OUT_DIR = HERE / "scaling_all_targets"
TMP_PLOTS = HERE / "plots"
VERSIONS = ["avg", "opus5", "gpt56"]

JUDGE_KEYS = ["eval_awareness", "concerning", "s_seed_adherence"]
FOLDERS = ["baseline", "cr1", "cr2", "cr2bo2", "cr2bo4"]

# (layout suffix, panel keys or None for all four)
LAYOUTS = [
    ("", None),  # main 2x2
    ("_2panel", {"winrate", "eval_awareness"}),  # 1x2 alternative
]
TWO_PANEL_FIGSIZE = (17, 6.6)
YLABEL_FONTSIZE = 24
TICK_FONTSIZE = 17
ANNOTATION_FONTSIZE = 19

_ORIG_SCORE_BY_SEED = base.score_by_seed
_ORIG_RENDER_PANEL = base.render_panel
_ORIG_SUBPLOTS = base.plt.subplots
_ORIG_TIGHT_LAYOUT = base.plt.Figure.tight_layout
_PRISTINE_PANELS = list(base.PANELS)

# {target: {key: {folder: {seed: value01}}}} — set per version.
_CUR_VIEWS: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
_PANEL_FILTER: set[str] | None = None


def _parse_cell(cell_dir: Path) -> tuple[str, str]:
    target = next(
        p.split("target_", 1)[1] for p in cell_dir.parts if p.startswith("target_")
    )
    folder = "baseline" if "baseline" in cell_dir.parts else cell_dir.name
    return target, folder


def _patched_score_by_seed(cell_dir: Path, key: str, normalize: bool = False):
    if key in JUDGE_KEYS:
        target, folder = _parse_cell(cell_dir)
        # View values are already min-max normalised 0-1.
        return dict(_CUR_VIEWS.get(target, {}).get(key, {}).get(folder, {}))
    return _ORIG_SCORE_BY_SEED(cell_dir, key, normalize=normalize)


def _patched_render_panel(ax, panel, targets_data, all_norms, show_xlabel=True):
    """Paper font sizes matching scaling_types_sonnet: 24pt axis titles,
    17pt ticks, 19pt point annotations. The 1x2 layout keeps its x-labels
    (base.main() only puts them on the bottom row of the 2x2)."""
    if _PANEL_FILTER is not None:
        show_xlabel = True
    _ORIG_RENDER_PANEL(ax, panel, targets_data, all_norms, show_xlabel)
    ax.yaxis.label.set_size(YLABEL_FONTSIZE)
    ax.xaxis.label.set_size(YLABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    for txt in ax.texts:
        txt.set_fontsize(ANNOTATION_FONTSIZE)


def _patched_subplots(nrows=1, ncols=1, **kwargs):
    # base.main() hardcodes the 2x2 grid; render the filtered panels 1x2.
    if _PANEL_FILTER is not None and (nrows, ncols) == (2, 2):
        kwargs["figsize"] = TWO_PANEL_FIGSIZE
        return _ORIG_SUBPLOTS(1, 2, **kwargs)
    return _ORIG_SUBPLOTS(nrows, ncols, **kwargs)


_ORIG_FIG_LEGEND = base.plt.Figure.legend


def _patched_fig_legend(self, *args, **kwargs):
    """Prepend a bold 'Target model:' pseudo-entry to the shared legend so it
    is clear the colours denote the audited target models."""
    handles = kwargs.get("handles")
    if handles is None:
        return _ORIG_FIG_LEGEND(self, *args, **kwargs)
    blank = base.plt.Line2D([], [], linestyle="none", label="Target model:")
    kwargs["handles"] = [blank] + list(handles)
    if "ncol" in kwargs:
        kwargs["ncol"] += 1
    leg = _ORIG_FIG_LEGEND(self, *args, **kwargs)
    leg.get_texts()[0].set_fontweight("bold")
    return leg


def _patched_tight_layout(self, *args, **kwargs):
    # The 1x2 figure is much shorter, so the bottom strip reserved for the
    # shared legend + x-labels needs a larger fraction of the height.
    rect = kwargs.get("rect")
    if _PANEL_FILTER is not None and rect == (0.015, 0.05, 1, 0.97):
        kwargs["rect"] = (0.015, 0.14, 1, 0.97)
    return _ORIG_TIGHT_LAYOUT(self, *args, **kwargs)


base.score_by_seed = _patched_score_by_seed
base.render_panel = _patched_render_panel
base.plt.subplots = _patched_subplots
base.plt.Figure.tight_layout = _patched_tight_layout
base.plt.Figure.legend = _patched_fig_legend


def run_version(version: str, scores: dict) -> int:
    global _CUR_VIEWS, _PANEL_FILTER
    _CUR_VIEWS = {}
    for target in base.TARGETS:
        target_data = scores.get(target, {})
        _CUR_VIEWS[target] = {}
        for key in JUDGE_KEYS:
            view, tag = js.version_view(target_data, FOLDERS, key, version)
            _CUR_VIEWS[target][key] = view
            if tag == "opus5-fallback" and version != "opus5":
                print(
                    f"[note] {version}/{target}: no gpt-5.6 scores for "
                    f"'{key}' — using opus-5."
                )

    vsuffix = "" if version == "avg" else f"_{version}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.PLOTS_ROOT = TMP_PLOTS  # module-level global read inside base.main()
    for laysuffix, panel_filter in LAYOUTS:
        _PANEL_FILTER = panel_filter
        base.PANELS = (
            _PRISTINE_PANELS
            if panel_filter is None
            else [p for p in _PRISTINE_PANELS if p["key"] in panel_filter]
        )
        rc = base.main()
        if rc != 0:
            return rc
        src = TMP_PLOTS / "cost_grid_combined_targets"
        if panel_filter is None:
            moves = [("plot", "png"), ("has_ea", "png"),
                     ("summary", "csv"), ("config", "json")]
        else:
            # Sidecars are identical to the main layout's — keep only the png.
            moves = [("plot", "png")]
        for name, ext in moves:
            f = src / f"{name}.{ext}"
            if f.is_file():
                f.replace(OUT_DIR / f"{name}{laysuffix}{vsuffix}.{ext}")
        shutil.rmtree(src, ignore_errors=True)
        print(f"[done] {version} -> {OUT_DIR}/plot{laysuffix}{vsuffix}.png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judge", choices=VERSIONS + ["all"], default="all")
    ap.add_argument(
        "--refresh-scores", action="store_true", help="re-extract judge_scores.json"
    )
    args = ap.parse_args()

    scores = js.load(refresh=args.refresh_scores)
    for version in VERSIONS if args.judge == "all" else [args.judge]:
        print(f"\n=== version: {version} ===")
        rc = run_version(version, scores)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
