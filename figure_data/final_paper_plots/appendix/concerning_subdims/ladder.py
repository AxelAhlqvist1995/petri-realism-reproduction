#!/usr/bin/env python3
"""Per-dimension means across the scaling ladder for all four targets (avg of opus-5 & gpt-5.6 rubric judges)."""
import glob, json, re, sys
from pathlib import Path
import pandas as pd, numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[4]
TARGETS=["sonnet-4.6","opus-4.8","gpt-5.5","gemini-3.5-flash"]
FOLDERS=["baseline","cr1","cr2","cr2bo2","cr2bo4"]
DIMS=["concerning","unprompted_sycophancy","user_over_sysprompt","cooperation_with_human_misuse","input_hallucination",
      "unprompted_initiative","unfaithful_thinking","unprompted_deception_toward_user","unprompted_leaking","unprompted_whistleblowing",
      "cooperation_with_harmful_sysprompt","manipulated_by_developer","unprompted_self_serving_bias"]
def cell(t,f):
    if f=="baseline": return ROOT/f"outputs/reasoning/baseline/target_{t}/auditor_sonnet-4.6"
    return ROOT/f"outputs/reasoning/pairwise/target_{t}/auditor_sonnet-4.6/{f}"
def canon(j):
    if "opus-5" in j: return "opus5"
    if "5.6" in j: return "gpt56"
rows=[]
for t in TARGETS:
    for f in FOLDERS:
        for fp in glob.glob(str(cell(t,f)/"transcript_*.json")):
            seed=int(re.match(r"transcript_(\d+)_",Path(fp).name).group(1))
            md=json.load(open(fp))["metadata"]
            sbj=(md.get("run_config") or {}).get("scores_by_judge") or {}
            per={}
            cur=(md.get("judge_output") or {}).get("scores") or {}
            cj=canon(str(cur.get("judge_model_effective") or md["run_config"].get("judge_model") or ""))
            if cj: per[cj]=cur
            for j,sc in sbj.items():
                c=canon(j)
                if c and c not in per: per[c]=sc
            for j,sc in per.items():
                rows.append({"target":t,"folder":f,"seed":seed,"judge":j,**{d:sc.get(d) for d in DIMS}})
df=pd.DataFrame(rows)
df.to_csv(HERE/"ladder_rows.csv",index=False)
avg=df.groupby(["target","folder","seed"])[DIMS].mean().reset_index()
# paired diff vs baseline per folder
out=[]
for t in TARGETS:
    b=avg[(avg.target==t)&(avg.folder=="baseline")].set_index("seed")[DIMS]
    for f in FOLDERS[1:]:
        c=avg[(avg.target==t)&(avg.folder==f)].set_index("seed")[DIMS]
        s=b.index.intersection(c.index)
        if len(s)==0: continue
        d=(c.loc[s]-b.loc[s])
        se=d.std()/np.sqrt(len(s))
        for dim in DIMS:
            out.append({"target":t,"folder":f,"n":len(s),"dim":dim,"diff":d[dim].mean(),"se":se[dim],"sig":abs(d[dim].mean())>1.96*se[dim]})
o=pd.DataFrame(out)
o.to_csv(HERE/"ladder_diffs.csv",index=False)
for dim in ["concerning","unprompted_sycophancy","user_over_sysprompt","cooperation_with_human_misuse","input_hallucination","unprompted_initiative","unfaithful_thinking"]:
    print(f"\n=== {dim}: paired diff vs baseline (* = |diff|>1.96 SE) ===")
    p=o[o.dim==dim].pivot(index="target",columns="folder",values="diff").reindex(TARGETS)[FOLDERS[1:]]
    sg=o[o.dim==dim].pivot(index="target",columns="folder",values="sig").reindex(TARGETS)[FOLDERS[1:]]
    print((p.round(2).astype(str)+sg.replace({True:"*",False:""})).to_string())
    print("n:",o[o.dim==dim].pivot(index="target",columns="folder",values="n").reindex(TARGETS)[FOLDERS[1:]].iloc[:, :].to_dict("index"))

print("\n=== Score-level shift for sycophancy / user_over_sysprompt (baseline -> cr2bo4), per target: counts of transcripts by score bucket (avg judge) ===")
for dim in ["unprompted_sycophancy","user_over_sysprompt","cooperation_with_human_misuse","input_hallucination","unprompted_initiative"]:
    print("\n--",dim)
    for t in TARGETS[:3]:
        for f in ["baseline","cr2bo4"]:
            v=avg[(avg.target==t)&(avg.folder==f)][dim].dropna()
            b=pd.cut(v,[0,1.4,2.9,4.9,10],labels=["=1","1.5-2.5","3-4.5",">=5"]).value_counts().reindex(["=1","1.5-2.5","3-4.5",">=5"])
            print(f"  {t:12s} {f:8s} n={len(v):3d}  "+"  ".join(f"{k}:{int(x):3d}" for k,x in b.items()))
