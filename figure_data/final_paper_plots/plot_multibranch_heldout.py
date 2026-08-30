#!/usr/bin/env python3
"""Appendix: multibranch held-out realism win rate figures (paper styling).

Title-less versions (Ns live in the sidecar CSVs, captions in LaTeX). All three
bar figures + the sensitivity histogram (plot_multibranch_cv_sensitivity.py) share
ONE seed set and ONE set of strategy definitions via cv_heldout_stats in
../../cost_winrate_reasoning/plot_multibranch_measure_optimization.py:

  optimize RWR / first branch / baseline / cr2bo4 : scored on the full 10/10
  held-out RWR : averaged over ALL balanced 5/5 splits (full C(10,5)^2 grid)

A seed is kept only where every compared method has a (non-refusal) value, so all
bars use the identical seed-instruction distribution.

  heldout_only.png              — Baseline / cr2bo4 / multibranch held-out RWR.
  heldout_family.png            — + optimize RWR + first branch (all seeds).
  heldout_family_controlled.png — the family restricted to seeds where cr2bo4
                                  seed adherence >= baseline (seed-adherence control).

Output: multibranch_heldout/{*.png, *.csv}
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CWR = _HERE.parent / "cost_winrate_reasoning"
if str(_CWR) not in sys.path:
    sys.path.insert(0, str(_CWR))

from plot_multibranch_measure_optimization import (  # noqa: E402
    cv_heldout_stats,
    heldout_seed_filter,
    render_variants_overlay,
)

OUT = _HERE / "multibranch_heldout"


def _write_csv(path: Path, rows: list[tuple[str, dict]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bar", "measure", "mean", "ci_half", "n"])
        for label, stats in rows:
            for mk, (m, ci, n) in stats.items():
                w.writerow([label, mk, m, ci, n])


def _family(opt, ho, first):
    return [
        ("Multibranch (optimize RWR)", opt),
        ("Multibranch (held-out RWR)", ho),
        ("Multibranch (first branch)", first),
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    ref, opt, ho, first = cv_heldout_stats()

    # heldout_only: Baseline / cr2bo4 / held-out RWR
    ho_only = [("Multibranch (held-out RWR)", ho)]
    render_variants_overlay(ref, ho_only, OUT / "heldout_only.png", title_prefix=None)
    _write_csv(OUT / "heldout_only.csv", list(ref) + ho_only)

    # heldout_family: full 5-bar family, all seeds (no seed-adherence control)
    fam = _family(opt, ho, first)
    render_variants_overlay(ref, fam, OUT / "heldout_family.png", title_prefix=None)
    _write_csv(OUT / "heldout_family.csv", list(ref) + fam)

    # controlled companion: seeds where cr2bo4 seed adherence >= baseline
    allowed = heldout_seed_filter("cr2bo4_sa_ge_baseline")
    refc, optc, hoc, firstc = cv_heldout_stats(allowed=allowed)
    famc = _family(optc, hoc, firstc)
    render_variants_overlay(
        refc, famc, OUT / "heldout_family_controlled.png", title_prefix=None
    )
    _write_csv(OUT / "heldout_family_controlled.csv", list(refc) + famc)

    # drop the previous (renamed) controlled outputs so nothing stale lingers
    for stale in ("cr2bo4_sa_ge_baseline.png", "cr2bo4_sa_ge_baseline.csv"):
        (OUT / stale).unlink(missing_ok=True)

    print(f"Wrote {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
