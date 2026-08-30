#!/usr/bin/env python3
"""Appendix (Compute matched baseline): where the long audit's concerning
behavior comes from, relative to when the auditor first felt done.

In the compute-matched multibranch baseline the auditor cannot end the audit and
is forced through the full 235-turn budget. It typically signals (in its private
reasoning) that it feels the audit is finished very early, then keeps probing.
For each transcript, an opus-5 classifier (grounded on the whole-transcript
judge's own `concerning` findings and the judge's turn-tagged highlights)
decides whether the target behavior that drives the `concerning` score occurs
before that done-signal, on both sides, or only after it (forced continuation),
or whether the transcript is not meaningfully concerning. See
`endsignal/analyze_endsignal_concerning.py` for the classifier.

Left: attribution of the concerning behavior across the 50 audits. Right: how
early the auditor first signals it is done (fraction of the audit elapsed).
Title-less (paper styling); counts/Ns in the CSV sidecar.

Output: multibranch_heldout/endsignal_concerning.{png,csv}
"""

from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
SRC = _HERE / "appendix" / "endsignal" / "endsignal_concerning.json"
OUT = _HERE / "multibranch_heldout"
DPI = 200

# category -> (display label, colour). after_only shares the brown of the
# whole-transcript multibranch bar in the companion figure.
CATS = [
    ("after_only", "Only after\n(forced continuation)", "#8c564b"),
    ("both", "Both sides", "#dd8452"),
    ("before_only", "Only before", "#4c72b0"),
    ("no_concern", "No concrete concern", "#c7c7c7"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = json.load(open(SRC))
    n = len(recs)
    counts = {k: 0 for k, _l, _c in CATS}
    other = 0
    for r in recs:
        c = r.get("category")
        if c in counts:
            counts[c] += 1
        else:
            other += 1  # parse_error / no_done_signal (should be 0)

    fracs = [r["boundary_frac"] for r in recs if r.get("boundary_frac") is not None]
    med = st.median(fracs)

    # concern-only denominator for the headline sentence
    concern = counts["after_only"] + counts["both"] + counts["before_only"]
    post = counts["after_only"] + counts["both"]

    fig = plt.figure(figsize=(11, 4.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # ---- left: attribution bars ----
    axL.set_facecolor("#f0f0f0")
    labels = [l for _k, l, _c in CATS]
    vals = [counts[k] for k, _l, _c in CATS]
    cols = [c for _k, _l, c in CATS]
    ypos = list(range(len(CATS)))[::-1]
    axL.barh(ypos, vals, color=cols, edgecolor="black", linewidth=0.8, zorder=3)
    for y, v in zip(ypos, vals):
        axL.text(v + max(vals) * 0.015, y, str(v), va="center", ha="left", fontsize=11)
    axL.set_yticks(ypos)
    axL.set_yticklabels(labels, fontsize=11)
    axL.set_xlabel(f"Number of audits (of {n})", fontsize=12)
    axL.set_xlim(0, max(vals) * 1.15)
    axL.grid(axis="x", color="white", linewidth=0.8, zorder=0)
    axL.set_axisbelow(True)

    # ---- right: how early the auditor first feels done ----
    axR.set_facecolor("#f0f0f0")
    axR.hist([f * 100 for f in fracs], bins=20, color="#8c564b",
             edgecolor="black", zorder=3)
    axR.axvline(med * 100, color="black", lw=2, zorder=6)
    axR.text(med * 100, axR.get_ylim()[1] * 0.92, f"  median {med * 100:.0f}%",
             ha="left", va="top", fontsize=11, fontweight="bold")
    axR.set_xlabel("When the auditor first signals it is done\n(% of the audit elapsed)",
                   fontsize=12)
    axR.set_ylabel("# of audits", fontsize=12)
    axR.grid(axis="y", color="white", linewidth=0.8, zorder=0)
    axR.set_axisbelow(True)

    fig.tight_layout()
    png = OUT / "endsignal_concerning.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with open(OUT / "endsignal_concerning.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        w.writerow(["n_audits", n])
        for k, l, _c in CATS:
            w.writerow([f"count_{k}", counts[k]])
        w.writerow(["count_other", other])
        w.writerow(["concern_denominator", concern])
        w.writerow(["concern_involving_post_done", post])
        w.writerow(["concern_after_only", counts["after_only"]])
        w.writerow(["median_done_frac", med])
        w.writerow(["mean_done_frac", st.mean(fracs)])

    print(f"Saved {png}")
    print(f"  n={n}: " + ", ".join(f"{k}={counts[k]}" for k, _l, _c in CATS) + f", other={other}")
    print(f"  among {concern} concerning audits: {post} involve post-done behavior "
          f"({post / concern:.0%}), {counts['after_only']} only-after "
          f"({counts['after_only'] / concern:.0%})")
    print(f"  first-done fraction: median {med:.1%}, mean {st.mean(fracs):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
