#!/usr/bin/env python3
"""Extract per-judge rubric (full-judge) dimension scores for baseline vs cr2bo4.

Writes rows.csv: one row per (target, cell, seed, judge) with every rubric
dimension, plus the opus-5 judge summary/justification text for reading.
"""
import csv, glob, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
TARGETS = ["sonnet-4.6", "opus-4.8", "gpt-5.5", "gemini-3.5-flash"]
CELLS = {"baseline": "outputs/reasoning/baseline/target_{t}/auditor_sonnet-4.6",
         "cr2bo4": "outputs/reasoning/pairwise/target_{t}/auditor_sonnet-4.6/cr2bo4"}
SEED_RE = re.compile(r"transcript_(\d+)_")
def canon(j):
    if "opus-5" in j: return "opus5"
    if "5.6" in j: return "gpt56"
    return None
rows=[]; texts=[]
for t in TARGETS:
    for cell, pat in CELLS.items():
        for fp in sorted(glob.glob(str(ROOT/pat.format(t=t)/"transcript_*.json"))):
            seed=int(SEED_RE.match(Path(fp).name).group(1))
            d=json.load(open(fp)); md=d["metadata"]
            sbj=(md.get("run_config") or {}).get("scores_by_judge") or {}
            jo=md.get("judge_output") or {}
            cur=jo.get("scores") or {}
            cj=canon(str(cur.get("judge_model_effective") or md["run_config"].get("judge_model")))
            per={}
            if cj: per[cj]=cur
            for j,sc in sbj.items():
                c=canon(j)
                if c and c not in per: per[c]=sc
            for j,sc in per.items():
                r={"target":t,"cell":cell,"seed":seed,"judge":j,"file":fp}
                for k,v in sc.items():
                    if isinstance(v,(int,float)) and not isinstance(v,bool) and "/" not in k: r[k]=v
                rows.append(r)
            texts.append({"target":t,"cell":cell,"seed":seed,"file":fp,
                          "seed_instruction":md.get("seed_instruction"),
                          "summary":jo.get("summary"),"justification":jo.get("justification")})
keys=sorted({k for r in rows for k in r})
with open(HERE/"rows.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
json.dump(texts,open(HERE/"judge_texts.json","w"),indent=1)
print(len(rows),"rows;",len(texts),"transcripts")
