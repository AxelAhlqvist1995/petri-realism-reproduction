#!/usr/bin/env python3
"""APPENDIX FIGURE — the three kinds of scaling for target sonnet-4.6,
logprob selector.

The logprob twin of `../../plot_scaling_types_sonnet.py` (same styling via its
monkeypatches): depth (cr1…cr22), breadth (bo2…bo16) and breadth×depth
(cr2bo2/4/8) ladders, auditor opus-4.7, data from
`outputs/ablation_protocol_no_stopping` through the `logprob_cells.build_shim`
symlink tree. The x-axis is the ANALYTICAL cost model (app:cost-model)
relative to baseline — the old runs' measured cost_logs are too sparse.
Judge panels read each cell's `0_transcripts.csv` (opus-5 primary judge; the
only judge on these cells, so no judge versions).

Everything lands in `scaling_types_sonnet_logprob/`:

  plot.png / summary.csv / config.json       paper 1x2 (win rate + adherence)
  plot_4panel.* / summary_4panel.csv / ...   full 2x2 (+ EA + concerning)
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
import plot_scaling_types_sonnet as sts  # noqa: E402  (paper-styling patches)
import plot_cost_grid_reasoning as base  # noqa: E402

TARGET = "sonnet-4.6"
OUT_DIR = HERE / "scaling_types_sonnet_logprob"
TMP_PLOTS = HERE / ".tmp_plots"

SONNET_CONDS = lp.TARGET_DEPTH[TARGET] + lp.SONNET_BREADTH + lp.SONNET_MIX

# (layout suffix, panel keys or None for all) — matches the pairwise figure.
LAYOUTS = [
    ("", {"winrate", "s_seed_adherence"}),  # paper figure: 1x2
    ("_4panel", None),  # alternative: full 2x2
]


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()

    shim = lp.build_shim({TARGET: SONNET_CONDS})
    base.REPO_ROOT = shim
    base.cost_by_seed = lp.analytical_cost_by_seed
    base.baseline_cost_by_seed = lp.analytical_cost_by_seed
    # No judge views: sts._CUR_VIEWS stays empty, so its patched score_by_seed
    # falls through to the real CSV loader (opus-5 primary judge).
    sts._CUR_VIEWS = {}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for laysuffix, panel_filter in LAYOUTS:
        sts._PANEL_FILTER = panel_filter
        sys.argv = [
            sys.argv[0],
            "--target", TARGET,
            "--auditor", lp.AUDITOR,
            "--out-dir", str(TMP_PLOTS),
        ]
        rc = base.main()
        if rc != 0:
            return rc
        src = TMP_PLOTS / f"cost_grid_reasoning_{TARGET}"
        for name, ext in (("plot", "png"), ("summary", "csv"), ("config", "json")):
            f = src / f"{name}.{ext}"
            if f.is_file():
                f.replace(OUT_DIR / f"{name}{laysuffix}.{ext}")
        shutil.rmtree(src, ignore_errors=True)
        print(f"[done] -> {OUT_DIR}/plot{laysuffix}.png")
    shutil.rmtree(TMP_PLOTS, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
