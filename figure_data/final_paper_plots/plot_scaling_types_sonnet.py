#!/usr/bin/env python3
"""FINAL PAPER FIGURE — the three kinds of scaling for target sonnet-4.6.

Cost-vs-quality panels with the auto-discovered depth (cr1…cr16), breadth
(bo2…bo8) and breadth×depth (cr2bo2, cr2bo4) ladders, x-axis = measured cost
relative to baseline (log scale).

One folder per paper figure: everything lands in `scaling_types_sonnet/`.
The PAPER figure is the 1x2 layout (realism win rate + seed adherence); the
full 2x2 with eval awareness and concerning is kept as the `_4panel`
alternative.

  plot.png / summary.csv / config.json            main figure (judge avg, 1x2)
  plot_opus5.png / summary_opus5.csv / ...        opus-5-only judge
  plot_gpt56.png / summary_gpt56.csv / ...        gpt-5.6-only judge
  plot_4panel*.png / summary_4panel*.csv / ...    all four measures (2x2)

Paper styling (matching cr_comparison): no suptitle, no per-panel titles
(Ns live in the CSVs and the caption), plain measure labels without judge
tags — the filename names the judge — and a shared dot-marker legend below
the grid.

Judge-dependent panels are served from `judge_scores.py` views with the
fairness rule: per measure, a seed only counts if the serving judge scored it
for EVERY rung (baseline + all ladders); `avg` averages the two judges where
both are complete on a seed and falls back to the single complete judge
otherwise. Win rate / cost are judge-independent.

Loading/rendering reuses `cost_winrate_reasoning/plot_cost_grid_reasoning.py`
(its `score_by_seed` is monkeypatched to serve the per-version views).
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
import plot_cost_grid_reasoning as base  # noqa: E402

import matplotlib.colors as mcolors  # noqa: E402

# Paper Figure 4 palette and legend labels (identical to the old
# cost_grid_sonnet_pairwise figure): critique-refine (depth) is red,
# best-of-N (breadth) is blue, the mix ladder a darker tone of the red.


def _darken(color: str, factor: float = 0.58) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


base.GROUP_COLOR.update(
    {"depth": "tab:blue", "breath": "tab:red", "mix": _darken("tab:blue")}
)
base.GROUP_NAMES.update(
    {"depth": "Depth (critique-refine)", "mix": "Depth \N{MULTIPLICATION SIGN} breadth"}
)
LEGEND_GROUPS = ["breath", "depth", "mix"]  # baseline is labelled in-panel

TARGET = "sonnet-4.6"
AUDITOR = "sonnet-4.6"
OUT_DIR = HERE / "scaling_types_sonnet"
TMP_PLOTS = HERE / "plots"
VERSIONS = ["avg", "opus5", "gpt56"]

JUDGE_KEYS = ["eval_awareness", "concerning", "s_seed_adherence"]

# (layout suffix, panel keys or None for all)
LAYOUTS = [
    ("", {"winrate", "s_seed_adherence"}),  # paper figure: 1x2
    ("_4panel", None),  # alternative: full 2x2
]
TWO_PANEL_FIGSIZE = (17, 6.6)
YLABEL_FONTSIZE = 24
LEGEND_FONTSIZE = 21

_ORIG_SCORE_BY_SEED = base.score_by_seed
_ORIG_RENDER_PANEL = base.render_panel
_ORIG_BUILD_PANELS = base.build_panels
_ORIG_SUBPLOTS = base.plt.subplots

# Set per version/layout before base.main() runs.
_CUR_VIEWS: dict[str, dict[str, dict[int, float]]] = {}
_PANEL_FILTER: set[str] | None = None


def _cell_folder(cell_dir: Path) -> str:
    return "baseline" if "baseline" in cell_dir.parts else cell_dir.name


def _patched_score_by_seed(cell_dir: Path, key: str, normalize: bool = False):
    if key in _CUR_VIEWS:
        # View values are already min-max normalised 0-1 (the only way the
        # grid panels consume these keys).
        return dict(_CUR_VIEWS[key].get(_cell_folder(cell_dir), {}))
    return _ORIG_SCORE_BY_SEED(cell_dir, key, normalize=normalize)


def _patched_build_panels(baseline_root: Path, winrate_haystack: str = "avg"):
    panels = _ORIG_BUILD_PANELS(baseline_root, winrate_haystack)
    if _PANEL_FILTER is not None:
        panels = [p for p in panels if p["key"] in _PANEL_FILTER]
    return panels


def _patched_subplots(nrows=1, ncols=1, **kwargs):
    # base.main() hardcodes a 2x2 grid; in the two-panel layout render the
    # same panels side by side instead.
    if _PANEL_FILTER is not None and (nrows, ncols) == (2, 2):
        kwargs["figsize"] = TWO_PANEL_FIGSIZE
        return _ORIG_SUBPLOTS(1, 2, **kwargs)
    return _ORIG_SUBPLOTS(nrows, ncols, **kwargs)


ANNOTATION_FONTSIZE = 19  # point labels (cr2, bo4, Baseline, …)


def _patched_render_panel(ax, rows, common_cost, common_y, panel, trend_lines,
                          show_legend):
    """Paper styling: no per-panel '<title> (N=…)' headers, no x-label on the
    top row of the 2x2 (the single row keeps it in the 1x2), 'Baseline'
    (singular) annotation, bigger y-axis titles and point labels, and the
    Figure-4 dot legend (no Baseline entry) in the top-left corner of the
    realism-win-rate panel."""
    summary = _ORIG_RENDER_PANEL(
        ax, rows, common_cost, common_y, panel, trend_lines, show_legend
    )
    ax.set_title("")
    if _PANEL_FILTER is None and panel["key"] in ("winrate", "eval_awareness"):
        ax.set_xlabel("")  # top row of the 2x2
    ax.yaxis.label.set_size(YLABEL_FONTSIZE)
    ax.xaxis.label.set_size(YLABEL_FONTSIZE)
    for txt in ax.texts:
        if txt.get_text() == "Baselines":
            txt.set_text("Baseline")
            # Match scaling_all_targets: normal weight (base uses bold).
            txt.set_fontweight("normal")
        txt.set_fontsize(ANNOTATION_FONTSIZE)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    if panel["key"] == "winrate":
        handles = [
            base.plt.Line2D(
                [0], [0], marker="o", linewidth=0, markersize=15,
                color=base.GROUP_COLOR[g], label=base.GROUP_NAMES[g],
            )
            for g in LEGEND_GROUPS
        ]
        ax.legend(
            handles=handles, loc="upper left", fontsize=LEGEND_FONTSIZE,
            frameon=False, handlelength=1.0, handletextpad=0.5,
        )
    return summary


base.score_by_seed = _patched_score_by_seed
base.build_panels = _patched_build_panels
base.plt.subplots = _patched_subplots
base.render_panel = _patched_render_panel
# Paper styling: no figure suptitle (base.main() hardcodes one).
base.plt.Figure.suptitle = lambda self, *a, **k: None


def run_version(version: str, scores: dict) -> int:
    global _CUR_VIEWS, _PANEL_FILTER
    target_data = scores.get(TARGET, {})
    pairwise_root = (
        base.REPO_ROOT / "outputs" / "reasoning" / "pairwise"
        / f"target_{TARGET}" / f"auditor_{AUDITOR}"
    )
    folders = [m[0] for m in base.discover_methods(pairwise_root)]

    _CUR_VIEWS = {}
    for key in JUDGE_KEYS:
        view, tag = js.version_view(target_data, folders, key, version)
        _CUR_VIEWS[key] = view
        if tag == "opus5-fallback" and version != "opus5":
            print(f"[note] {version}: no gpt-5.6 scores for '{key}' — using opus-5.")

    vsuffix = "" if version == "avg" else f"_{version}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for laysuffix, panel_filter in LAYOUTS:
        _PANEL_FILTER = panel_filter
        sys.argv = [sys.argv[0], "--target", TARGET, "--out-dir", str(TMP_PLOTS)]
        rc = base.main()
        if rc != 0:
            return rc
        src = TMP_PLOTS / f"cost_grid_reasoning_{TARGET}"
        for name, ext in (("plot", "png"), ("summary", "csv"), ("config", "json")):
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
