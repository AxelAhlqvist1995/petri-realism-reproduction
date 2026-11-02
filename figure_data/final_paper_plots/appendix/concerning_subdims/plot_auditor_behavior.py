#!/usr/bin/env python3
"""PAPER FIGURE — auditor-behaviour judge ratings, baseline vs cr2bo4.

  auditor_behavior/auditor_behavior_horizontal_bars_three_models[_<judge>].{png,pdf}
  auditor_behavior/auditor_behavior_shifts[_<judge>].csv

Judge versions (``--judge``):

  gpt55july  the original 2026-07-10 gpt-5.5 run (auditor_behavior_scores_gpt55judge.csv,
             extracted from s3://petrievalbucket/runs/260709_auditor_behavior_rejudge/;
             all Sonnet seeds, 50 Opus / 46 GPT seeds)
  opus5      ratings/ratings_opus5.jsonl  (experiments/util/rejudge_auditor_behavior.py)
  gpt56      ratings/ratings_gpt56.jsonl  (same runner, openai/gpt-5.6-sol)
  avg        per-transcript mean of opus5 and gpt56 where both judged it, the single
             judge where only one did (judge refusals) -- the paper's main-text
             two-judge convention. DEFAULT; also written without a suffix (the
             file the paper includes).

Seeds are restricted to those with a successful rating for both conditions
(and, for gpt55july, present in rows.csv). Scores are normalised (s-1)/9;
error bars are 95% CIs (1.96 SEM) over seeds; diffs are paired by seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

from bar_style import TARGETS, mean_ci, three_panel_bars

HERE = Path(__file__).resolve().parent
OUT = HERE / "auditor_behavior"
RATINGS = OUT / "ratings"
OUT.mkdir(exist_ok=True)

# judge key -> label used in the figure (same order as the original figure)
DIMS = [("harmful_intent", "user_harmful_intent"),
        ("pressure_to_take_action", "user_pressure_to_take_action"),
        ("system_prompt_pressure", "system_prompt_pressure"),
        ("auditor_persistence_adaptation", "auditor_persistence"),
        ("audit_buildup", "audit_buildup"),
        ("action_affordances", "consequential_action_opportunities"),
        ("scenario_detail", "scenario_detail"),
        ("audit_success", "seed_adherence")]
KEYS = [k for k, _ in DIMS]
COND = {"Baseline": "baseline", "Pairwise CR": "cr2bo4"}
JUDGES = ["avg", "opus5", "gpt56", "gpt55july"]


def load_july() -> pd.DataFrame:
    ab = pd.read_csv(HERE / "auditor_behavior_scores_gpt55judge.csv")
    ab = ab[ab.parsed & ~ab.error].copy()
    ab["cell"] = ab.condition.map(COND)
    # if a seed was judged in several eval files keep the last successful one
    ab = ab.drop_duplicates(["target", "cell", "seed"], keep="last")
    local = pd.read_csv(HERE / "rows.csv")[["target", "cell", "seed"]].drop_duplicates()
    ab = ab.merge(local, on=["target", "cell", "seed"])  # only locally present transcripts
    return ab[["target", "cell", "seed"] + KEYS]


def load_jsonl(slug: str) -> pd.DataFrame:
    path = RATINGS / f"ratings_{slug}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path} -- run experiments/util/rejudge_auditor_behavior.py "
                 f"-j {'anthropic/claude-opus-5' if slug == 'opus5' else 'openai/gpt-5.6-sol'}")
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("ok"):
                    rows.append({"target": r["target"], "cell": r["cell"], "seed": int(r["seed"]),
                                 **{k: r["scores"][k] for k in KEYS}})
    df = pd.DataFrame(rows).drop_duplicates(["target", "cell", "seed"], keep="last")
    print(f"[{slug}] {len(df)} rated transcripts")
    return df


def load(judge: str) -> pd.DataFrame:
    if judge == "gpt55july":
        return load_july()
    if judge in ("opus5", "gpt56"):
        return load_jsonl(judge)
    a, b = load_jsonl("opus5"), load_jsonl("gpt56")
    m = a.merge(b, on=["target", "cell", "seed"], how="outer", suffixes=("_a", "_b"))
    for k in KEYS:
        m[k] = m[[f"{k}_a", f"{k}_b"]].mean(axis=1)  # mean of the judges present
    both = m[[f"{KEYS[0]}_a", f"{KEYS[0]}_b"]].notna().all(axis=1).sum()
    print(f"[avg] {len(m)} transcripts: {both} with both judges, {len(m) - both} with one")
    return m[["target", "cell", "seed"] + KEYS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", choices=JUDGES, default="avg")
    args = ap.parse_args()
    ab = load(args.judge)
    stats, rows, titles = {}, [], []
    for t, title in TARGETS:
        s = ab[ab.target == t]
        piv = {c: s[s.cell == c].set_index("seed")[KEYS] for c in COND.values()}
        seeds = piv["baseline"].index.intersection(piv["cr2bo4"].index)
        b = (piv["baseline"].loc[seeds] - 1) / 9
        c = (piv["cr2bo4"].loc[seeds] - 1) / 9
        print(f"[seeds] {t:<11} judged b/c = {len(piv['baseline'])}/{len(piv['cr2bo4'])}, paired n={len(seeds)}")
        titles.append((t, f"{title} (n={len(seeds)})"))
        for k, lab in DIMS:
            mb, eb = mean_ci(b[k]); mc, ec = mean_ci(c[k]); md, ed = mean_ci(c[k] - b[k])
            stats[(t, "baseline", lab)] = (mb, eb); stats[(t, "cr2bo4", lab)] = (mc, ec)
            rows.append(dict(target=t, dimension=lab, judge_key=k, judge=args.judge, n=len(seeds), baseline=mb,
                             baseline_ci95=eb, cr2bo4=mc, cr2bo4_ci95=ec, diff=md, diff_ci95=ed,
                             significant=abs(md) > ed))
    suffixes = [f"_{args.judge}"] + ([""] if args.judge == "avg" else [])
    for suf in suffixes:
        with open(OUT / f"auditor_behavior_shifts{suf}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        three_panel_bars(stats, [lab for _, lab in DIMS], titles,
                         str(OUT / f"auditor_behavior_horizontal_bars_three_models{suf}"),
                         figsize=(11, 6.4), sharex=True)


if __name__ == "__main__":
    main()
