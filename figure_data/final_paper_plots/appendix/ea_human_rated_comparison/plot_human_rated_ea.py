#!/usr/bin/env python3
"""APPENDIX FIGURE — baseline vs cr2bo4 eval awareness as rated by the HUMAN
annotator (ea_gt_v3, the paper's n=65 set), per target and pooled.

Companion to the judge-validation appendix: the same 65 ratings that validate
the judges, used as the measure itself. Styling matches the main Fig. 2 bars
(baseline tab:blue, cr2bo4 tab:orange, grey panel, white grid).

Two designs, both written to summary.csv; the figure shows the first:

  paired   — per target, the UNION of seeds rated in either condition enters
             both bars. A seed rated in only one condition gets the human score
             1 (normalised 0) on the other side: transcripts were only sent
             for rating when at least one judge flagged them, so an unrated
             transcript is one no judge flagged. The imputed transcript must
             exist in the other cell (present in its 0_transcripts.csv).
  rated    — each bar is the plain mean over that cell's rated transcripts
             (44 baseline + 21 cr2bo4 = 65), no pairing, no imputation.

Scores are the rater's 1-10 eval-awareness rating normalised to [0,1] via
(s-1)/9, like every judge EA figure. Error bars: 95% CI (1.96 SEM) over seeds
(paired design) / transcripts (rated design); paired differences get their
own CI over per-seed differences.

Output (overwritten): plots/{plot.png, plot.pdf, summary.csv, paired_diff.csv}
Copy to the paper: figures/human_rated_ea_baseline_vs_cr2bo4.{png,pdf}
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
CWR = REPO_ROOT / "experiments/general_plotting/cost_winrate_reasoning"
for p in (REPO_ROOT / "experiments/util", CWR):
    sys.path.insert(0, str(p))
from ea_gt_v3 import paper_set  # noqa: E402
from plot_cost_grid_reasoning import score_by_seed, winrate_by_seed  # noqa: E402

OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)
AUDITOR = "sonnet-4.6"
TARGETS = [("sonnet-4.6", "Sonnet 4.6"), ("opus-4.8", "Opus 4.8"),
           ("gpt-5.5", "GPT-5.5"), ("gemini-3.5-flash", "Gemini 3.5 Flash")]
POOLED = ("__pooled__", "All targets\n(pooled)")
CONDS = [("baseline", "Baseline", "tab:blue"), ("cr2bo4", "cr2bo4", "tab:orange")]
BAR_WIDTH, GAP = 0.30, 0.45


def cell_path(target: str, cond: str) -> Path:
    base = REPO_ROOT / "outputs/reasoning"
    if cond == "baseline":
        return base / "baseline" / f"target_{target}" / f"auditor_{AUDITOR}"
    return base / "pairwise" / f"target_{target}" / f"auditor_{AUDITOR}" / cond


def mean_ci(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), 0.0
    if len(x) < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def main() -> int:
    # human ratings: gt[(target, cond)] = {seed: normalised score}
    gt: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for a in paper_set():
        gt[(a.target, a.method)][int(a.seed)] = (a.gt - 1) / 9
    assert sum(len(v) for v in gt.values()) == 65, gt

    rows, diffs = [], []
    stats = {}  # (target_key, cond) -> (mean, ci, n) for the paired design
    pooled_vals = {c: [] for c, _, _ in CONDS}
    pooled_d = []
    for t, _title in TARGETS:
        # seeds that exist in both cells (same existence rule as the Aug figure)
        exists = {}
        for c, _, _ in CONDS:
            cp = cell_path(t, c)
            exists[c] = set(winrate_by_seed(cp)) | set(
                score_by_seed(cp, "s_seed_adherence", normalize=True))
        union = (set(gt[(t, "baseline")]) | set(gt[(t, "cr2bo4")]))
        kept = sorted(s for s in union if all(s in exists[c] for c, _, _ in CONDS))
        dropped = sorted(union - set(kept))
        if dropped:
            print(f"[{t}] rated seeds missing from a cell CSV, dropped: {dropped}")
        vals = {c: [gt[(t, c)].get(s, 0.0) for s in kept] for c, _, _ in CONDS}
        n_imp = {c: sum(s not in gt[(t, c)] for s in kept) for c, _, _ in CONDS}
        d = np.array(vals["cr2bo4"]) - np.array(vals["baseline"])
        for c, _, _ in CONDS:
            m, e = mean_ci(vals[c])
            stats[(t, c)] = (m, e, len(kept))
            pooled_vals[c] += vals[c]
            rows.append(dict(design="paired", target=t, condition=c, n=len(kept),
                             n_rated=len(kept) - n_imp[c], n_imputed=n_imp[c],
                             mean=m, ci95=e))
            rm, re = mean_ci(list(gt[(t, c)].values()))
            rows.append(dict(design="rated", target=t, condition=c, n=len(gt[(t, c)]),
                             n_rated=len(gt[(t, c)]), n_imputed=0, mean=rm, ci95=re))
        dm, de = mean_ci(d)
        pooled_d += list(d)
        diffs.append(dict(target=t, n=len(kept), diff=dm, ci95=de,
                          n_lower=int((d < 0).sum()), n_higher=int((d > 0).sum()),
                          n_tie=int((d == 0).sum())))
    for c, _, _ in CONDS:
        m, e = mean_ci(pooled_vals[c])
        stats[(POOLED[0], c)] = (m, e, len(pooled_vals[c]))
        rows.append(dict(design="paired", target="pooled", condition=c, n=len(pooled_vals[c]),
                         n_rated=sum(len(gt[(t, c)]) for t, _ in TARGETS),
                         n_imputed=len(pooled_vals[c]) - sum(len(gt[(t, c)]) for t, _ in TARGETS),
                         mean=m, ci95=e))
        allr = [v for t, _ in TARGETS for v in gt[(t, c)].values()]
        rm, re = mean_ci(allr)
        rows.append(dict(design="rated", target="pooled", condition=c, n=len(allr),
                         n_rated=len(allr), n_imputed=0, mean=rm, ci95=re))
    pd_ = np.array(pooled_d)
    dm, de = mean_ci(pd_)
    diffs.append(dict(target="pooled", n=len(pd_), diff=dm, ci95=de,
                      n_lower=int((pd_ < 0).sum()), n_higher=int((pd_ > 0).sum()),
                      n_tie=int((pd_ == 0).sum())))

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT / "paired_diff.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diffs[0])); w.writeheader(); w.writerows(diffs)
    for r in rows:
        print(f"{r['design']:<7} {r['target']:<17} {r['condition']:<9} n={r['n']:>2} "
              f"(rated {r['n_rated']:>2}, imputed {r['n_imputed']:>2})  "
              f"{r['mean']:.3f} ± {r['ci95']:.3f}")
    for r in diffs:
        print(f"diff    {r['target']:<17} n={r['n']:>2}  {r['diff']:+.3f} ± {r['ci95']:.3f}  "
              f"lower/higher/tie = {r['n_lower']}/{r['n_higher']}/{r['n_tie']}")

    # ---- figure (paired design), Fig. 2 styling
    clusters = TARGETS + [POOLED]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f0f0f0")
    x = np.arange(len(clusters)) * (len(CONDS) * BAR_WIDTH + GAP)
    top = 0.0
    for ci, (c, label, color) in enumerate(CONDS):
        off = (ci - (len(CONDS) - 1) / 2) * BAR_WIDTH
        hs = [stats[(k, c)][0] for k, _ in clusters]
        es = [stats[(k, c)][1] for k, _ in clusters]
        top = max(top, max(h + e for h, e in zip(hs, es)))
        ax.bar(x + off, hs, width=BAR_WIDTH, color=color, edgecolor="black",
               linewidth=0.8, yerr=es, capsize=3,
               error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "black"},
               label=label, zorder=3)
    ax.axvline((x[-1] + x[-2]) / 2, color="0.6", linewidth=1.0, linestyle=":", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{title}\nN={stats[(k, 'baseline')][2]}" for k, title in clusters],
                       fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Eval awareness (human rating)", fontsize=14)
    ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, top * 1.12 + 0.02)
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"plot.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved", OUT / "plot.{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
