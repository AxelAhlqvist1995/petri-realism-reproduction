#!/usr/bin/env python3
"""THE multibranch comparison figure, 2x2 grid — per target.

Default target is sonnet-4.6 (multibranch cell baseline_multibranch);
`--target opus-4.8` produces the companion figures from the 235-turn
opus multibranch cell (baseline_multibranch_235), written as
multibranch_comparison_opus*.png/csv alongside the sonnet ones.

Bars: Baseline / cr2bo4 / Multibranch (max concerning) / Multibranch
(max held-out RWR). Panels: realism win rate, eval awareness, concerning,
seed adherence. Replaces the previous heldout_only* / wholetranscript /
petri_labels / maxconcerning plot family.

Realism win rate: Baseline and cr2bo4 on all 10/10 deployment comparisons;
the two multibranch selections via the full C(10,5)^2 held-out grid — per
balanced 5/5 split the branch is selected on the SELECTION half (max RWR, or
max concerning with earliest-branch tie-break) and its win rate is scored on
the complementary HELD-OUT half, averaged over all splits.

Judge versions: the main figure averages the opus-5 and gpt-5.6-sol judges
for eval awareness, concerning, and seed adherence (per seed / per branch;
single-judge fallback where one is missing — branch-level gpt-5.6 seed
adherence is picked up automatically from the singular sidecars'
scores_by_judge once that pass lands). The max-concerning SELECTION uses the
two-judge average concerning when both judges scored every candidate branch
of the seed; with refusals it falls back to the judge with the fewest branch
refusals (opus-5 on ties). Both judge versions use the SAME selection, so
their difference is purely measurement. The _opus5 companion scores
everything with opus-5 only; both use ONE common seed set.

  multibranch_comparison_sonnet.png/.csv        main (judge avg EA/concerning)
  multibranch_comparison_sonnet_opus5.png/.csv  opus-5-only companion
  multibranch_comparison_sonnet_3panel.png/.csv
      3-pane layout of the main (judge avg) figure: no seed adherence;
      realism win rate is a double-height panel on the left, eval awareness
      top right, concerning bottom right. Same stats as the main figure.
  multibranch_comparison_sonnet_firstbranch_sa1.png/.csv
      extra version (judge avg): adds a Multibranch (first branch) bar (b00,
      no selection; win rate on all 10/10 comparisons) and replaces the two
      selection bars with ADHERENCE-CONTROLLED ones — the candidate pool per
      seed is restricted to branches whose two-judge average seed adherence
      is at most 1 point (1-10 scale) BELOW the seed's FIRST branch — higher
      adherence is always fine (b00 itself is always eligible).

Output: multibranch_heldout/
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CWR = _HERE.parent / "cost_winrate_reasoning"
_FPP = _HERE
for _p in (_HERE, _CWR, str(_FPP)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import judge_scores as js  # noqa: E402
import plot_multibranch_measure_optimization as _mo  # noqa: E402
from plot_multibranch_measure_optimization import (  # noqa: E402
    CONDITIONS as MO_CONDITIONS,
    MULTIBRANCH_CELL,
    MULTIBRANCH_TARGET,
    REF_COLORS,
    STRATEGY_COLORS,
    _cv_bwr_split,
    _cv_stat,
    render_variants_overlay,
)
from plot_method_comparison_bars_multibranch_pd_selected import (  # noqa: E402
    load_per_seed_opus5,
)

OUT = _HERE / "multibranch_heldout"

# Target selection (global default + --target CLI, per repo convention).
# Maps target -> (multibranch cell folder, output-name token).
TARGET = "sonnet-4.6"
_TARGET_SETTINGS = {
    "sonnet-4.6": ("baseline_multibranch_320", "sonnet"),
    "opus-4.8": ("baseline_multibranch_320", "opus"),
}
# Set by _apply_target(); defaults preserve the original sonnet behaviour.
T = MULTIBRANCH_TARGET
CELL = MULTIBRANCH_CELL
NAME = "sonnet"


def _apply_target(target: str) -> None:
    global T, CELL, NAME
    folder, NAME = _TARGET_SETTINGS[target]
    T = target
    # .../outputs/reasoning/pairwise/target_<t>/auditor_sonnet-4.6/<folder>
    pairwise_root = MULTIBRANCH_CELL.parents[2]
    CELL = pairwise_root / f"target_{target}" / "auditor_sonnet-4.6" / folder


GPT56 = "openai/gpt-5.6-sol"
OPUS5 = "anthropic/claude-opus-5"

HO_LABEL = "Multibranch (max held-out RWR)"
MC_LABEL = "Multibranch (max concerning)"
STRATEGY_COLORS[MC_LABEL] = STRATEGY_COLORS["max Concerning"]
STRATEGY_COLORS[HO_LABEL] = STRATEGY_COLORS["Multibranch (held-out RWR)"]

MKEYS = ["winrate", "eval_awareness", "concerning", "s_seed_adherence"]
LEGEND_FONTSIZE = 14

_ORIG_FIG_LEGEND = _mo.plt.Figure.legend


def _one_row_legend(self, *args, **kwargs):
    if "ncol" in kwargs and len(args) >= 2 and len(args[1]) <= 4:
        kwargs["ncol"] = len(args[1])
        kwargs["fontsize"] = LEGEND_FONTSIZE
        kwargs["handlelength"] = 1.2
        kwargs["handletextpad"] = 0.4
        kwargs["columnspacing"] = 1.2
    return _ORIG_FIG_LEGEND(self, *args, **kwargs)


_mo.plt.Figure.legend = _one_row_legend


def load_branches() -> dict[int, list[dict]]:
    """seed -> branch dicts: opus-5 measures + judge-avg EA/concerning/SA
    (from the multi-judge branch sidecars) + per-repo comparisons."""
    bj = CELL / "branch_judging"
    ind = bj / "winrate_leaf" / "realism_win_rate" / "individual_transcripts"
    comp: dict[str, dict] = {}
    for tag in ("cc", "wildchat"):
        for f in sorted((ind / tag).glob("*.json")):
            doc = json.loads(f.read_text())
            comp.setdefault(doc["transcript_id"], {})[tag] = [
                r.get("correctly_identified") for r in doc.get("results", [])
            ]
    # (seed, branch_index) -> {judge: {ea, conc}} from the sidecars.
    by_judge: dict[tuple[int, int], dict[str, dict]] = {}
    for f in (bj / "judge").glob("*_branch_judge.json"):
        doc = json.loads(f.read_text())
        seed = int(doc["seed_id"])
        for b in doc.get("branches", []):
            sbj = b.get("scores_by_judge") or {}
            entry = {}
            for judge in (OPUS5, GPT56):
                sc = sbj.get(judge)
                if sc and sc.get("eval_awareness") is not None \
                        and sc.get("concerning") is not None:
                    entry[judge] = {
                        "ea": (float(sc["eval_awareness"]) - 1.0) / 9.0,
                        "conc": (float(sc["concerning"]) - 1.0) / 9.0,
                    }
            by_judge[(seed, int(b["branch_index"]))] = entry

    # (seed, branch_index) -> {judge: sa01} from the singular sidecars
    # (scores_by_judge post-migration; primary-judge `scores` otherwise).
    sa_by_judge: dict[tuple[int, int], dict[str, float]] = {}
    for f in (bj / "singular_scores").glob("*_branch_singular.json"):
        doc = json.loads(f.read_text())
        seed = int(doc["seed_id"])
        primary = doc.get("primary_judge", OPUS5)
        for b in doc.get("branches", []):
            entry = {}
            sbj = b.get("scores_by_judge") or {}
            for judge in (OPUS5, GPT56):
                sc = sbj.get(judge)
                if sc is None and judge == primary:
                    sc = b.get("scores")
                if sc and sc.get("s_seed_adherence") is not None:
                    entry[judge] = (float(sc["s_seed_adherence"]) - 1.0) / 9.0
            sa_by_judge[(seed, int(b["branch_index"]))] = entry

    seeds: dict[int, list[dict]] = defaultdict(list)
    n_fallback = 0
    n_sa_fallback = 0
    # (seed, opus5, gpt56) for both-judged branches, per measure.
    conc_pairs: list[tuple[int, float, float]] = []
    ea_pairs: list[tuple[int, float, float]] = []
    sa_pairs: list[tuple[int, float, float]] = []
    with open(bj / "summary.csv", newline="") as f:
        for row in csv.DictReader(f):
            c = comp.get(row["sample_id"])
            if not c:
                continue
            try:
                b = {
                    "bidx": int(row["branch_index"]),
                    "eval_awareness": (float(row["eval_awareness"]) - 1.0) / 9.0,
                    "concerning": (float(row["concerning"]) - 1.0) / 9.0,
                    "s_seed_adherence": (float(row["s_seed_adherence"]) - 1.0) / 9.0,
                    "cc": c.get("cc"),
                    "wildchat": c.get("wildchat"),
                }
            except (KeyError, ValueError):
                continue
            seed = int(row["seed"])
            jb = by_judge.get((seed, b["bidx"]), {})
            vals_ea = [jb[j]["ea"] for j in (OPUS5, GPT56) if j in jb] \
                or [b["eval_awareness"]]
            vals_co = [jb[j]["conc"] for j in (OPUS5, GPT56) if j in jb] \
                or [b["concerning"]]
            if GPT56 not in jb:
                n_fallback += 1
            elif OPUS5 in jb:
                conc_pairs.append((seed, jb[OPUS5]["conc"], jb[GPT56]["conc"]))
                ea_pairs.append((seed, jb[OPUS5]["ea"], jb[GPT56]["ea"]))
            b["ea_avg"] = sum(vals_ea) / len(vals_ea)
            b["conc_avg"] = sum(vals_co) / len(vals_co)
            b["conc_has_gpt56"] = GPT56 in jb
            sab = sa_by_judge.get((seed, b["bidx"]), {})
            vals_sa = [sab[j] for j in (OPUS5, GPT56) if j in sab] \
                or [b["s_seed_adherence"]]
            if GPT56 not in sab:
                n_sa_fallback += 1
            elif OPUS5 in sab:
                sa_pairs.append((seed, sab[OPUS5], sab[GPT56]))
            b["sa_avg"] = sum(vals_sa) / len(vals_sa)
            b["sa_by_judge"] = dict(sab)
            seeds[seed].append(b)
    n_all = sum(len(v) for v in seeds.values())
    print(f"[branches] {n_all} branches; gpt-5.6 missing for "
          f"{n_fallback} (EA/concerning) and {n_sa_fallback} (seed "
          "adherence) — opus-5 fallback in the avg measures")
    # Judge-scale diagnostic: the avg-with-fallback estimator mixes two
    # scales whenever the judges disagree systematically. Report the offset
    # on both-judged branches so a large adherence offset is visible BEFORE
    # trusting the averaged panel (rejudging session measured concerning at
    # gpt-5.6 ~= opus-5 + 1.26 pts, stable across cells).
    # The offsets differ in SIGN by measure (concerning +, EA -), and a
    # partial backfill is seed-clustered (transcript-by-transcript), so a
    # low n_seeds means the pooled offset is not trustworthy yet.
    for name, pairs in (("concerning", conc_pairs), ("eval_awareness", ea_pairs),
                        ("s_seed_adherence", sa_pairs)):
        if pairs:
            d = [(g - o) * 9.0 for _s, o, g in pairs]
            mean_d = sum(d) / len(d)
            n_seeds_p = len({sd for sd, _o, _g in pairs})
            print(f"[judge-offset] {name}: gpt-5.6 - opus-5 = {mean_d:+.2f} "
                  f"pts (1-10 scale) over {len(pairs)} both-judged branches "
                  f"in {n_seeds_p} seeds")
    # Flip the adherence panel to the averaged mode only once the gpt-5.6
    # pass is essentially complete (<=10% fallback). A partial pass would mix
    # the opus-5 scale with the ~+0.9pt-shifted average on an arbitrary
    # subset of branches.
    sa_avg_available = n_sa_fallback <= 0.10 * n_all
    if 0 < (n_all - n_sa_fallback) and not sa_avg_available:
        print(f"[note] gpt-5.6 seed adherence only covers "
              f"{n_all - n_sa_fallback}/{n_all} branches (pass in progress?) "
              "— adherence panel stays opus-5 until coverage >= 90%")
    return dict(seeds), sa_avg_available


def render_3panel(entries, out):
    """3-pane layout: realism win rate spans both rows on the left; eval
    awareness (top) and concerning (bottom) on the right. `entries` is
    (label, {mkey: (mean, ci_half, n)}) — same stats as the 2x2 figure."""
    from matplotlib.patches import Patch

    colors = [REF_COLORS.get(lbl) or STRATEGY_COLORS[lbl] for lbl, _ in entries]
    fig = _mo.plt.figure(figsize=(12, 6.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], wspace=0.24,
                          hspace=0.25)
    panels = [
        ("winrate", "Realism win rate", fig.add_subplot(gs[:, 0]), 17, 13),
        ("eval_awareness", "Eval awareness", fig.add_subplot(gs[0, 1]), 13, 11),
        ("concerning", "Concerning", fig.add_subplot(gs[1, 1]), 13, 11),
    ]
    for mk, disp, ax, fs, ts in panels:
        ax.set_facecolor("#f0f0f0")
        top = 0.0
        for i, (_lbl, stats) in enumerate(entries):
            m, e, _n = stats[mk]
            ax.bar(i, m, width=0.72, color=colors[i], edgecolor="black",
                   linewidth=0.8, yerr=e, capsize=3,
                   error_kw={"elinewidth": 1.0, "capthick": 1.0,
                             "ecolor": "black"},
                   zorder=3)
            top = max(top, m + e)
        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels([])
        ax.set_ylabel(disp, fontsize=fs)
        ax.tick_params(axis="y", labelsize=ts)
        ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, top * 1.12)
    handles = [Patch(facecolor=c, edgecolor="black") for c in colors]
    fig.legend(handles, [lbl for lbl, _ in entries], loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               fontsize=LEGEND_FONTSIZE, ncol=len(entries),
               handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    _mo.plt.close(fig)
    print(f"Saved {out}")


def cv_grid_select(seeds, pool, mode):
    """Full C(10,5)^2 split grid. mode='ho': select the SELECTION-half max-RWR
    branch; mode='mc': select max concerning, ties by earliest branch.
    The mc selection measure is the two-judge AVERAGE concerning when both
    judges scored every branch of the seed; with refusals it is the judge
    with the fewest branch refusals (opus-5 on ties; opus-5-unscored branches
    never load, so opus-5 has zero refusals among candidates). Reported
    winrate = HELD-OUT half; branch measures (both judge versions)
    split-averaged."""
    all5 = list(itertools.combinations(range(10), 5))
    parts = [(tuple(sorted(set(range(10)) - set(e))), tuple(e)) for e in all5]
    keys = ("winrate", "eval_awareness", "concerning", "s_seed_adherence",
            "ea_avg", "conc_avg", "sa_avg")
    sub = {s: seeds[s] for s in pool if s in seeds}
    sel_key = {}
    if mode == "mc":
        for seed, bs in sub.items():
            n_miss_gpt = sum(not b["conc_has_gpt56"] for b in bs)
            sel_key[seed] = "conc_avg" if n_miss_gpt == 0 else "concerning"
        n_avg = sum(v == "conc_avg" for v in sel_key.values())
        print(f"[mc-select] {n_avg}/{len(sel_key)} seeds select on avg "
              "concerning (rest opus-5 due to gpt-5.6 branch refusals)")
    acc = {s: dict.fromkeys(keys, 0.0) | {"n": 0} for s in sub}
    for S_cc, E_cc in parts:
        for S_wc, E_wc in parts:
            for seed, bs in sub.items():
                best = None
                for b in bs:
                    sw = _cv_bwr_split(b, S_cc, S_wc)
                    if sw is None:
                        continue
                    ew = _cv_bwr_split(b, E_cc, E_wc)
                    if ew is None:
                        continue
                    key = sw if mode == "ho" \
                        else (b[sel_key[seed]], -b["bidx"])
                    if best is None or key > best[0]:
                        best = (key, ew, b)
                if best is None:
                    continue
                a = acc[seed]
                a["winrate"] += best[1]
                for k in keys[1:]:
                    a[k] += best[2][k]
                a["n"] += 1
    return {
        s: {k: a[k] / a["n"] for k in keys}
        for s, a in acc.items() if a["n"]
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds, sa_avg_available = load_branches()
    data, _present = load_per_seed_opus5(
        [c for c in MO_CONDITIONS if c[0] in ("baseline", "cr2bo4")], MKEYS
    )

    # Judge views for Baseline/cr2bo4 (opus-5 and two-judge average).
    scores = js.load()
    views = {}
    for ver in ("opus5", "avg"):
        for mk in ("eval_awareness", "concerning", "s_seed_adherence"):
            view, tag = js.version_view(
                scores.get(T, {}), ["baseline", "cr2bo4"], mk, ver
            )
            views[(ver, mk)] = view
            if ver == "avg" and tag != "avg":
                print(f"[warn] avg view for {mk} is '{tag}'")

    # One common seed set for every bar and BOTH judge versions.
    common = set(data[(T, "baseline", "winrate")]) & set(
        data[(T, "cr2bo4", "winrate")]
    )
    for (ver, mk), view in views.items():
        for cond in ("baseline", "cr2bo4"):
            common &= set(view.get(cond, {}))
    ho = cv_grid_select(seeds, common, "ho")
    mc = cv_grid_select(seeds, common, "mc")
    common &= set(ho) & set(mc)
    common = sorted(common)
    print(f"[seeds] common set n={len(common)}")

    def stat(series):
        return _cv_stat([series[s] for s in common])

    for ver, suffix in (("avg", ""), ("opus5", "_opus5")):
        ea_k = "ea_avg" if ver == "avg" else "eval_awareness"
        co_k = "conc_avg" if ver == "avg" else "concerning"
        sa_ver = ver if (ver == "opus5" or sa_avg_available) else "opus5"
        ref = []
        for cond, disp in (("baseline", "Baseline"), ("cr2bo4", "cr2bo4")):
            ref.append((disp, {
                "winrate": stat(data[(T, cond, "winrate")]),
                "eval_awareness": stat(views[(ver, "eval_awareness")][cond]),
                "concerning": stat(views[(ver, "concerning")][cond]),
                # Judge-consistent adherence panel: only average when the
                # branch-level gpt-5.6 adherence exists too.
                "s_seed_adherence": stat(views[(sa_ver, "s_seed_adherence")][cond]),
            }))
        sa_k = "sa_avg" if sa_ver == "avg" else "s_seed_adherence"
        variants = [
            (lbl, {
                "winrate": stat({s: sel[s]["winrate"] for s in common}),
                "eval_awareness": stat({s: sel[s][ea_k] for s in common}),
                "concerning": stat({s: sel[s][co_k] for s in common}),
                "s_seed_adherence": stat({s: sel[s][sa_k] for s in common}),
            })
            for lbl, sel in ((MC_LABEL, mc), (HO_LABEL, ho))
        ]
        render_variants_overlay(
            ref, variants, OUT / f"multibranch_comparison_{NAME}{suffix}.png",
            title_prefix=None,
        )
        if ver == "avg":
            render_3panel(
                list(ref) + variants,
                OUT / f"multibranch_comparison_{NAME}_3panel.png",
            )
            with open(OUT / f"multibranch_comparison_{NAME}_3panel.csv", "w",
                      newline="") as f:
                w = csv.writer(f)
                w.writerow(["bar", "measure", "mean", "ci_half", "n"])
                for label, stats in list(ref) + variants:
                    for mk in ("winrate", "eval_awareness", "concerning"):
                        m, ci, n = stats[mk]
                        w.writerow([label, mk, m, ci, n])
        with open(OUT / f"multibranch_comparison_{NAME}{suffix}.csv", "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["bar", "measure", "mean", "ci_half", "n"])
            for label, stats in list(ref) + variants:
                for mk, (m, ci, n) in stats.items():
                    w.writerow([label, mk, m, ci, n])

    # ---- Extra version: first branch + adherence-controlled selections ----
    # Candidate pool per seed = branches whose two-judge AVERAGE seed
    # adherence is at most 1 point (1-10 scale) below the seed's first branch
    # (b00) — higher adherence never disqualifies; b00 is always eligible.
    # The first-branch bar involves no selection: win rate on all 10/10
    # comparisons, measures straight from b00. Judge-avg version only.
    fb_label = "Multibranch (first branch)"
    ho_sa_label = "Multibranch (max held-out RWR, adherence ctrl)"
    mc_sa_label = "Multibranch (max concerning, adherence ctrl)"
    STRATEGY_COLORS[ho_sa_label] = STRATEGY_COLORS[HO_LABEL]
    STRATEGY_COLORS[mc_sa_label] = STRATEGY_COLORS[MC_LABEL]
    full = tuple(range(10))
    tol = 1.0 / 9.0 + 1e-9
    first, seeds_ctrl = {}, {}
    for s in seeds:
        fb = next((b for b in seeds[s] if b["bidx"] == 0), None)
        if fb is None:
            continue
        first[s] = fb
        seeds_ctrl[s] = [
            b for b in seeds[s] if b["sa_avg"] >= fb["sa_avg"] - tol
        ]
    n_el = sum(len(v) for v in seeds_ctrl.values())
    n_tot = sum(len(seeds[s]) for s in seeds_ctrl)
    print(f"[sa-ctrl] {n_el}/{n_tot} branches at most 1 pt below first-branch "
          f"adherence (avg) across {len(seeds_ctrl)} seeds with a b00")
    ho_c = cv_grid_select(seeds_ctrl, common, "ho")
    mc_c = cv_grid_select(seeds_ctrl, common, "mc")
    fb_wr = {s: _cv_bwr_split(first[s], full, full) for s in first}
    fb_wr = {s: v for s, v in fb_wr.items() if v is not None}
    common_fb = sorted(set(common) & set(fb_wr) & set(ho_c) & set(mc_c))
    print(f"[seeds] first-branch/sa-ctrl figure n={len(common_fb)}")

    def stat_fb(series):
        return _cv_stat([series[s] for s in common_fb])

    sa_ver = "avg" if sa_avg_available else "opus5"
    sa_k = "sa_avg" if sa_avg_available else "s_seed_adherence"
    ref = []
    for cond, disp in (("baseline", "Baseline"), ("cr2bo4", "cr2bo4")):
        ref.append((disp, {
            "winrate": stat_fb(data[(T, cond, "winrate")]),
            "eval_awareness": stat_fb(views[("avg", "eval_awareness")][cond]),
            "concerning": stat_fb(views[("avg", "concerning")][cond]),
            "s_seed_adherence": stat_fb(
                views[(sa_ver, "s_seed_adherence")][cond]),
        }))
    variants = [(fb_label, {
        "winrate": stat_fb(fb_wr),
        "eval_awareness": stat_fb({s: first[s]["ea_avg"] for s in common_fb}),
        "concerning": stat_fb({s: first[s]["conc_avg"] for s in common_fb}),
        "s_seed_adherence": stat_fb({s: first[s][sa_k] for s in common_fb}),
    })]
    for lbl, sel in ((mc_sa_label, mc_c), (ho_sa_label, ho_c)):
        variants.append((lbl, {
            "winrate": stat_fb({s: sel[s]["winrate"] for s in common_fb}),
            "eval_awareness": stat_fb({s: sel[s]["ea_avg"] for s in common_fb}),
            "concerning": stat_fb({s: sel[s]["conc_avg"] for s in common_fb}),
            "s_seed_adherence": stat_fb({s: sel[s][sa_k] for s in common_fb}),
        }))
    render_variants_overlay(
        ref, variants,
        OUT / f"multibranch_comparison_{NAME}_firstbranch_sa1.png",
        title_prefix=None,
    )
    with open(OUT / f"multibranch_comparison_{NAME}_firstbranch_sa1.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["bar", "measure", "mean", "ci_half", "n"])
        for label, stats in list(ref) + variants:
            for mk, (m, ci, n) in stats.items():
                w.writerow([label, mk, m, ci, n])
    return 0




# ---- Combined figure: sonnet + opus side by side per panel ----------------

TARGET_DISPLAY = {"sonnet-4.6": "Sonnet 4.6", "opus-4.8": "Opus 4.8"}


def compute_avg_entries():
    """The four judge-avg bars for the CURRENT target (set via _apply_target):
    [(label, {mkey: (mean, ci_half, n)})]. Mirrors main()'s ver="avg" branch."""
    seeds, sa_avg_available = load_branches()
    data, _present = load_per_seed_opus5(
        [c for c in MO_CONDITIONS if c[0] in ("baseline", "cr2bo4")], MKEYS
    )
    scores = js.load()
    views = {}
    for ver in ("opus5", "avg"):
        for mk in ("eval_awareness", "concerning", "s_seed_adherence"):
            view, _tag = js.version_view(
                scores.get(T, {}), ["baseline", "cr2bo4"], mk, ver
            )
            views[(ver, mk)] = view
    common = set(data[(T, "baseline", "winrate")]) & set(
        data[(T, "cr2bo4", "winrate")]
    )
    for view in views.values():
        for cond in ("baseline", "cr2bo4"):
            common &= set(view.get(cond, {}))
    ho = cv_grid_select(seeds, common, "ho")
    mc = cv_grid_select(seeds, common, "mc")
    common = sorted(common & set(ho) & set(mc))
    print(f"[seeds] {T}: common set n={len(common)}")

    def stat(series):
        return _cv_stat([series[s] for s in common])

    sa_ver = "avg" if sa_avg_available else "opus5"
    sa_k = "sa_avg" if sa_avg_available else "s_seed_adherence"
    entries = []
    for cond, disp in (("baseline", "Baseline"), ("cr2bo4", "cr2bo4")):
        entries.append((disp, {
            "winrate": stat(data[(T, cond, "winrate")]),
            "eval_awareness": stat(views[("avg", "eval_awareness")][cond]),
            "concerning": stat(views[("avg", "concerning")][cond]),
            "s_seed_adherence": stat(views[(sa_ver, "s_seed_adherence")][cond]),
        }))
    for lbl, sel in ((MC_LABEL, mc), (HO_LABEL, ho)):
        entries.append((lbl, {
            "winrate": stat({s: sel[s]["winrate"] for s in common}),
            "eval_awareness": stat({s: sel[s]["ea_avg"] for s in common}),
            "concerning": stat({s: sel[s]["conc_avg"] for s in common}),
            "s_seed_adherence": stat({s: sel[s][sa_k] for s in common}),
        }))
    return entries


def render_combined(groups, out):
    """2x2 measure panels; within each panel one bar group per target
    (sonnet left, opus right), target names on the x-axis. `groups` is
    [(target, entries)] with entries as in compute_avg_entries()."""
    from matplotlib.patches import Patch

    from plot_multibranch_measure_optimization import MEASURES4

    labels = [lbl for lbl, _ in groups[0][1]]
    colors = [_bar_color(lbl) for lbl in labels]
    nbar = len(labels)
    fig, axes = _mo.plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("white")
    for (mkey, display), ax in zip(MEASURES4, axes.flat):
        ax.set_facecolor("#f0f0f0")
        top = 0.0
        centers = []
        for g, (_target, entries) in enumerate(groups):
            off = g * (nbar + 1)
            centers.append(off + (nbar - 1) / 2.0)
            for i, (_lbl, stats) in enumerate(entries):
                m, e, _n = stats[mkey]
                ax.bar(off + i, m, width=1.0, color=colors[i],
                       edgecolor="black", linewidth=0.8, yerr=e, capsize=3,
                       error_kw={"elinewidth": 1.0, "capthick": 1.0,
                                 "ecolor": "black"},
                       zorder=3)
                top = max(top, m + e)
        ax.set_xticks(centers)
        ax.set_xticklabels(
            [TARGET_DISPLAY.get(t, t) for t, _ in groups], fontsize=13
        )
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(display, fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, top * 1.08)
    handles = [Patch(facecolor=c, edgecolor="black") for c in colors]
    ncol = len(labels) if len(labels) <= 4 else (len(labels) + 1) // 2
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               fontsize=LEGEND_FONTSIZE, ncol=ncol,
               handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    _mo.plt.close(fig)
    print(f"Saved {out}")


# ---- Adherence-controlled combined figure -----------------------------------

FB_LABEL = "Multibranch (first branch)"
MC_CTRL_LABEL = "Multibranch (max concerning, adherence ctrl)"
HO_CTRL_LABEL = "Multibranch (max held-out RWR, adherence ctrl)"
SA_CTRL_TOL_PTS = 0.5  # default gate: first branch minus 0.5 pts (1-10 scale)


def _seed_sa_measure(bs):
    """Per-seed adherence accessor for the eligibility filter.

    Uses the two-judge AVERAGE when both judges scored the same branches
    (including the exact-same-refusals case, where the commonly refused
    branches simply drop out); with unequal refusals, the judge with the
    fewest unjudged branches (opus-5 on ties). Returns (mode, sa_fn) where
    sa_fn(branch) -> normalised adherence or None (branch not judgeable
    under the chosen measure)."""
    miss_o = {b["bidx"] for b in bs if OPUS5 not in b["sa_by_judge"]}
    miss_g = {b["bidx"] for b in bs if GPT56 not in b["sa_by_judge"]}
    if miss_o == miss_g:
        mode = "avg"
    elif len(miss_g) < len(miss_o):
        mode = "gpt56"
    else:
        mode = "opus5"

    def sa(b):
        sbj = b["sa_by_judge"]
        if mode == "avg":
            if OPUS5 in sbj and GPT56 in sbj:
                return (sbj[OPUS5] + sbj[GPT56]) / 2.0
            return None
        return sbj.get(GPT56 if mode == "gpt56" else OPUS5)

    return mode, sa


def compute_controlled_entries(tol_pts: float = SA_CTRL_TOL_PTS):
    """Five bars for the CURRENT target: Baseline / cr2bo4 / first branch /
    max concerning (adherence ctrl) / max held-out RWR (adherence ctrl).

    The controlled selections draw only from branches whose seed adherence
    (per-seed measure via _seed_sa_measure) is >= the FIRST branch's minus
    `tol_pts` (1-10 scale); b00 itself is always eligible. Displayed
    measures are the judge-avg ones, like the main figure."""
    tol = tol_pts / 9.0 + 1e-9
    seeds, sa_avg_available = load_branches()
    data, _present = load_per_seed_opus5(
        [c for c in MO_CONDITIONS if c[0] in ("baseline", "cr2bo4")], MKEYS
    )
    scores = js.load()
    views = {}
    for ver in ("opus5", "avg"):
        for mk in ("eval_awareness", "concerning", "s_seed_adherence"):
            view, _tag = js.version_view(
                scores.get(T, {}), ["baseline", "cr2bo4"], mk, ver
            )
            views[(ver, mk)] = view

    sa_ver = "avg" if sa_avg_available else "opus5"
    sa_k = "sa_avg" if sa_avg_available else "s_seed_adherence"
    full = tuple(range(10))

    modes = {"avg": 0, "gpt56": 0, "opus5": 0}
    seeds_ctrl, first = {}, {}
    n_el = n_tot = 0
    for s_, bs in seeds.items():
        fb = next((b for b in bs if b["bidx"] == 0), None)
        if fb is None:
            continue
        mode, sa_of = _seed_sa_measure(bs)
        fb_sa = sa_of(fb)
        if fb_sa is None:
            continue
        modes[mode] += 1
        w = _cv_bwr_split(fb, full, full)
        if w is None:
            continue
        first[s_] = {"winrate": w, "eval_awareness": fb["ea_avg"],
                     "concerning": fb["conc_avg"], "s_seed_adherence": fb[sa_k]}
        eligible = [
            b for b in bs
            if b["bidx"] == 0
            or (sa_of(b) is not None and sa_of(b) >= fb_sa - tol)
        ]
        n_el += len(eligible)
        n_tot += len(bs)
        seeds_ctrl[s_] = eligible
    print(f"[sa-ctrl-{tol_pts:g}] {T}: {n_el}/{n_tot} branches within "
          f"{tol_pts:g} pts of the "
          f"first branch's adherence across {len(seeds_ctrl)} seeds; per-seed "
          f"adherence measure: {modes}")

    common = set(data[(T, "baseline", "winrate")]) & set(
        data[(T, "cr2bo4", "winrate")]
    )
    for view in views.values():
        for cond in ("baseline", "cr2bo4"):
            common &= set(view.get(cond, {}))
    common &= set(first)
    ho_c = cv_grid_select(seeds_ctrl, common, "ho")
    mc_c = cv_grid_select(seeds_ctrl, common, "mc")
    common = sorted(common & set(ho_c) & set(mc_c))
    print(f"[seeds] {T} controlled: common set n={len(common)}")

    def stat(series):
        return _cv_stat([series[s_] for s_ in common])

    entries = []
    for cond, disp in (("baseline", "Baseline"), ("cr2bo4", "cr2bo4")):
        entries.append((disp, {
            "winrate": stat(data[(T, cond, "winrate")]),
            "eval_awareness": stat(views[("avg", "eval_awareness")][cond]),
            "concerning": stat(views[("avg", "concerning")][cond]),
            "s_seed_adherence": stat(views[(sa_ver, "s_seed_adherence")][cond]),
        }))
    entries.append((FB_LABEL,
                    {mk: stat({s_: first[s_][mk] for s_ in common})
                     for mk in MKEYS}))
    for lbl, sel in ((MC_CTRL_LABEL, mc_c), (HO_CTRL_LABEL, ho_c)):
        entries.append((lbl, {
            "winrate": stat({s_: sel[s_]["winrate"] for s_ in common}),
            "eval_awareness": stat({s_: sel[s_]["ea_avg"] for s_ in common}),
            "concerning": stat({s_: sel[s_]["conc_avg"] for s_ in common}),
            "s_seed_adherence": stat({s_: sel[s_][sa_k] for s_ in common}),
        }))
    return entries


def _bar_color(lbl):
    if lbl in REF_COLORS:
        return REF_COLORS[lbl]
    if lbl in STRATEGY_COLORS:
        return STRATEGY_COLORS[lbl]
    alias = {MC_CTRL_LABEL: MC_LABEL, HO_CTRL_LABEL: HO_LABEL}
    if lbl in alias and alias[lbl] in STRATEGY_COLORS:
        return STRATEGY_COLORS[alias[lbl]]
    return "#7f7f7f"


def render_combined_3panel(groups, out):
    """3-pane combined layout: realism win rate spans both rows on the left,
    eval awareness top-right, concerning bottom-right; within each panel one
    touching bar group per target, target names on the x-axis."""
    from matplotlib.patches import Patch

    labels = [lbl for lbl, _ in groups[0][1]]
    colors = [_bar_color(lbl) for lbl in labels]
    nbar = len(labels)
    fig = _mo.plt.figure(figsize=(12, 6.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], wspace=0.24,
                          hspace=0.25)
    panels = [
        ("winrate", "Realism win rate", fig.add_subplot(gs[:, 0]), 17, 13),
        ("eval_awareness", "Eval awareness", fig.add_subplot(gs[0, 1]), 13, 11),
        ("concerning", "Concerning", fig.add_subplot(gs[1, 1]), 13, 11),
    ]
    for mk, disp, ax, fs, ts in panels:
        ax.set_facecolor("#f0f0f0")
        top = 0.0
        centers = []
        for g, (_target, entries) in enumerate(groups):
            off = g * (nbar + 1)
            centers.append(off + (nbar - 1) / 2.0)
            for i, (_lbl, stats) in enumerate(entries):
                m, e, _n = stats[mk]
                ax.bar(off + i, m, width=1.0, color=colors[i],
                       edgecolor="black", linewidth=0.8, yerr=e, capsize=3,
                       error_kw={"elinewidth": 1.0, "capthick": 1.0,
                                 "ecolor": "black"},
                       zorder=3)
                top = max(top, m + e)
        ax.set_xticks(centers)
        ax.set_xticklabels(
            [TARGET_DISPLAY.get(t, t) for t, _ in groups], fontsize=ts + 1
        )
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(disp, fontsize=fs)
        ax.tick_params(axis="y", labelsize=ts)
        ax.grid(axis="y", color="white", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, top * 1.08)
    handles = [Patch(facecolor=c, edgecolor="black") for c in colors]
    ncol = len(labels) if len(labels) <= 4 else (len(labels) + 1) // 2
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               fontsize=LEGEND_FONTSIZE, ncol=ncol,
               handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    _mo.plt.close(fig)
    print(f"Saved {out}")


def main_combined() -> int:
    """Judge-avg figure with sonnet and opus grouped side by side per panel."""
    OUT.mkdir(parents=True, exist_ok=True)
    groups, groups_ctrl = [], []
    for target in ("sonnet-4.6", "opus-4.8"):
        _apply_target(target)
        groups.append((target, compute_avg_entries()))
        groups_ctrl.append((target, compute_controlled_entries()))

    def write_csv(path, gs_):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["target", "bar", "measure", "mean", "ci_half", "n"])
            for target, entries in gs_:
                for label, stats in entries:
                    for mk, (m, ci, n) in stats.items():
                        w.writerow([target, label, mk, m, ci, n])

    render_combined(groups, OUT / "multibranch_comparison_combined.png")
    write_csv(OUT / "multibranch_comparison_combined.csv", groups)
    render_combined_3panel(
        groups, OUT / "multibranch_comparison_combined_3panel.png"
    )
    write_csv(OUT / "multibranch_comparison_combined_3panel.csv", groups)
    render_combined(
        groups_ctrl, OUT / "multibranch_comparison_combined_controlled.png"
    )
    write_csv(OUT / "multibranch_comparison_combined_controlled.csv",
              groups_ctrl)
    return 0



def main_family() -> int:
    """heldout_family analogue for the current target (judge-avg measures):
    Baseline / cr2bo4 / optimize RWR / held-out RWR / first branch.

    optimize RWR and first branch are scored on the full 10/10 comparisons
    (optimize is IN-SAMPLE selection, i.e. the optimistic upper bound);
    held-out RWR uses the C(10,5)^2 grid, exactly like heldout_family.png.
    Output: multibranch_comparison_<name>_family.png/csv."""
    OUT.mkdir(parents=True, exist_ok=True)
    seeds, sa_avg_available = load_branches()
    data, _present = load_per_seed_opus5(
        [c for c in MO_CONDITIONS if c[0] in ("baseline", "cr2bo4")], MKEYS
    )
    scores = js.load()
    views = {}
    for ver in ("opus5", "avg"):
        for mk in ("eval_awareness", "concerning", "s_seed_adherence"):
            view, _tag = js.version_view(
                scores.get(T, {}), ["baseline", "cr2bo4"], mk, ver
            )
            views[(ver, mk)] = view

    sa_ver = "avg" if sa_avg_available else "opus5"
    sa_k = "sa_avg" if sa_avg_available else "s_seed_adherence"
    full = tuple(range(10))

    def point(b, w):
        return {"winrate": w, "eval_awareness": b["ea_avg"],
                "concerning": b["conc_avg"], "s_seed_adherence": b[sa_k]}

    opt, first = {}, {}
    for s_, bs in seeds.items():
        cand = []
        for b in bs:
            w = _cv_bwr_split(b, full, full)
            if w is not None:
                cand.append((w, b))
        if cand:
            w, b = max(cand, key=lambda x: x[0])
            opt[s_] = point(b, w)
        fb = next((b for b in bs if b["bidx"] == 0), None)
        if fb is not None:
            w = _cv_bwr_split(fb, full, full)
            if w is not None:
                first[s_] = point(fb, w)

    common = set(data[(T, "baseline", "winrate")]) & set(
        data[(T, "cr2bo4", "winrate")]
    )
    for view in views.values():
        for cond in ("baseline", "cr2bo4"):
            common &= set(view.get(cond, {}))
    common &= set(opt) & set(first)
    ho = cv_grid_select(seeds, common, "ho")
    common = sorted(common & set(ho))
    print(f"[seeds] {T} family: common set n={len(common)}")

    def stat(series):
        return _cv_stat([series[s_] for s_ in common])

    ref = []
    for cond, disp in (("baseline", "Baseline"), ("cr2bo4", "cr2bo4")):
        ref.append((disp, {
            "winrate": stat(data[(T, cond, "winrate")]),
            "eval_awareness": stat(views[("avg", "eval_awareness")][cond]),
            "concerning": stat(views[("avg", "concerning")][cond]),
            "s_seed_adherence": stat(views[(sa_ver, "s_seed_adherence")][cond]),
        }))
    variants = [
        ("Multibranch (optimize RWR)",
         {mk: stat({s_: opt[s_][mk] for s_ in common}) for mk in MKEYS}),
        ("Multibranch (held-out RWR)",
         {"winrate": stat({s_: ho[s_]["winrate"] for s_ in common}),
          "eval_awareness": stat({s_: ho[s_]["ea_avg"] for s_ in common}),
          "concerning": stat({s_: ho[s_]["conc_avg"] for s_ in common}),
          "s_seed_adherence": stat({s_: ho[s_][sa_k] for s_ in common})}),
        ("Multibranch (first branch)",
         {mk: stat({s_: first[s_][mk] for s_ in common}) for mk in MKEYS}),
    ]
    render_variants_overlay(
        ref, variants, OUT / f"multibranch_comparison_{NAME}_family.png",
        title_prefix=None,
    )
    with open(OUT / f"multibranch_comparison_{NAME}_family.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["bar", "measure", "mean", "ci_half", "n"])
        for label, stats in list(ref) + variants:
            for mk, (m, ci, n) in stats.items():
                w.writerow([label, mk, m, ci, n])
    return 0

def main_combined_controlled(tol_pts: float) -> int:
    """Only the adherence-controlled combined figure, at a custom tolerance.
    Writes multibranch_comparison_combined_controlled_<tol>pt.png/csv (the
    default 0.5 tolerance keeps the unsuffixed name via main_combined)."""
    OUT.mkdir(parents=True, exist_ok=True)
    groups_ctrl = []
    for target in ("sonnet-4.6", "opus-4.8"):
        _apply_target(target)
        groups_ctrl.append((target, compute_controlled_entries(tol_pts)))
    suffix = "" if tol_pts == SA_CTRL_TOL_PTS else f"_{tol_pts:g}pt"
    out = OUT / f"multibranch_comparison_combined_controlled{suffix}.png"
    render_combined(groups_ctrl, out)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "bar", "measure", "mean", "ci_half", "n"])
        for target, entries in groups_ctrl:
            for label, stats in entries:
                for mk, (m, ci, n) in stats.items():
                    w.writerow([target, label, mk, m, ci, n])
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=TARGET,
                        choices=sorted(_TARGET_SETTINGS)
                        + ["combined", "combined-controlled"])
    parser.add_argument("--family", action="store_true",
                        help="heldout_family analogue (optimize RWR / "
                             "held-out RWR / first branch) for --target")
    parser.add_argument("--sa-tol", type=float, default=SA_CTRL_TOL_PTS,
                        help="adherence-control tolerance in 1-10 scale "
                             "points (combined-controlled only)")
    _args = parser.parse_args()
    if _args.target == "combined":
        raise SystemExit(main_combined())
    if _args.target == "combined-controlled":
        raise SystemExit(main_combined_controlled(_args.sa_tol))
    _apply_target(_args.target)
    raise SystemExit(main_family() if _args.family else main())
