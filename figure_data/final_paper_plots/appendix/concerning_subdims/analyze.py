#!/usr/bin/env python3
import pandas as pd, numpy as np, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
df=pd.read_csv(HERE/"rows.csv")
META={"needs_attention","disappointing","concerning","concerning_calibrated","concerning_with_no_mitigating_factors",
      "admirable","fun_or_funny","high_stakes_setting","scenario_realism","eval_awareness","audit_situational_awareness",
      "auditor_failure","auditor_refused_audit","stuck_in_loops","broken_tool_use","overrefusal","prefill_susceptibility",
      "s_eval_awareness","s_seed_adherence","s_concerning_calibrated","s_audit_situational_awareness","s_seed_faithfulness",
      "s_seed_audit_success","judge_content_filter_refusals","judge_fallback_used","seed","counterfactual_bad_behavior"}
beh=["cooperation_with_human_misuse","facilitating_terrorism","cooperation_with_harmful_sysprompt","user_over_sysprompt",
 "manipulated_by_developer","unprompted_deception_toward_user","unprompted_sycophancy","unprompted_encouragement_of_user_delusion",
 "unprompted_leaking","unprompted_whistleblowing","unprompted_blackmail","unprompted_initiative","unprompted_emotional_manipulativeness",
 "unprompted_self_preservation","unfaithful_thinking","input_hallucination","unprompted_self_serving_bias","unprompted_gratitude",
 "unprompted_spirituality_or_extreme_awe","unprompted_pos_affect","unprompted_neg_affect","unprompted_bizarre_behavior"]
TARGETS=["sonnet-4.6","opus-4.8","gpt-5.5"]
rng=np.random.default_rng(0)

def avg_judge(sub):
    # per (seed) average over judges available
    return sub.groupby("seed")[beh+["concerning"]].mean()

def paired(t, judge=None):
    s=df[df.target==t]
    if judge: s=s[s.judge==judge]
    b=avg_judge(s[s.cell=="baseline"]); c=avg_judge(s[s.cell=="cr2bo4"])
    seeds=b.index.intersection(c.index)
    return b.loc[seeds], c.loc[seeds]

out={}
print("=== Paired diffs (avg of opus5 & gpt56 where available), raw 1-10 ===")
tab={}
for t in TARGETS:
    b,c=paired(t); d=c-b; n=len(d)
    boots=np.array([d.sample(n,replace=True,random_state=int(rng.integers(1e9))).mean() for _ in range(2000)])
    lo,hi=np.percentile(boots,[2.5,97.5],axis=0)
    tab[t]=pd.DataFrame({"base":b.mean(),"cr2bo4":c.mean(),"diff":d.mean(),"lo":lo,"hi":hi},index=d.columns)
    tab[t]["sig"]=(tab[t].lo>0)|(tab[t].hi<0)
    print(f"\n--- {t} (n={n}) ---")
    print(tab[t].sort_values("diff",ascending=False).round(2).to_string())

print("\n=== Gap: diff(claude) - diff(gpt-5.5), sorted ===")
gap=pd.DataFrame({"sonnet-gpt":tab["sonnet-4.6"]["diff"]-tab["gpt-5.5"]["diff"],
                  "opus-gpt":tab["opus-4.8"]["diff"]-tab["gpt-5.5"]["diff"]})
gap["mean_gap"]=gap.mean(axis=1)
print(gap.sort_values("mean_gap",ascending=False).round(2).to_string())

print("\n=== Per-judge diffs for key dims ===")
key=["concerning","cooperation_with_human_misuse","input_hallucination","unprompted_sycophancy","user_over_sysprompt",
     "unprompted_deception_toward_user","unfaithful_thinking","unprompted_initiative","manipulated_by_developer","cooperation_with_harmful_sysprompt"]
for j in ["opus5","gpt56"]:
    print(f"\n-- judge {j} --")
    rows={}
    for t in TARGETS:
        b,c=paired(t,j); rows[t]=(c-b).mean()[key]
    print(pd.DataFrame(rows).round(2).to_string())

print("\n=== Regression decomposition: concerning ~ behavior dims (pooled, avg judge, per-transcript) ===")
allX=[]; 
for t in TARGETS:
    for cell in ["baseline","cr2bo4"]:
        a=avg_judge(df[(df.target==t)&(df.cell==cell)]); a["target"]=t; a["cell"]=cell; allX.append(a)
A=pd.concat(allX)
from numpy.linalg import lstsq
A=A.dropna(subset=beh+["concerning"])
regb=[b for b in beh if A[b].std()>0]
X=np.c_[np.ones(len(A)),A[regb].values]; y=A["concerning"].values
beta,*_=lstsq(X,y,rcond=None)
pred=X@beta; r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
print("R2 =",round(r2,3))
B=pd.Series(beta[1:],index=regb).reindex(beh).fillna(0)
print(B.sort_values(ascending=False).round(2).to_string())
print("\nContribution beta_d * diff_d per target:")
contrib=pd.DataFrame({t:B*tab[t]["diff"][beh] for t in TARGETS})
contrib.loc["SUM"]=contrib.sum(); contrib.loc["actual_dconcerning"]=[tab[t]["diff"]["concerning"] for t in TARGETS]
print(contrib.sort_values("opus-4.8",ascending=False).round(3).to_string())
print("\nGap contributions (claude - gpt):")
gc=pd.DataFrame({"sonnet-gpt":contrib["sonnet-4.6"]-contrib["gpt-5.5"],"opus-gpt":contrib["opus-4.8"]-contrib["gpt-5.5"]})
print(gc.drop(["SUM","actual_dconcerning"]).assign(mean=lambda x:x.mean(axis=1)).sort_values("mean",ascending=False).round(3).to_string())

print("\n=== Max-dim decomposition: concerning vs max behavior dim ===")
A["maxdim"]=A[beh].max(axis=1)
print("corr(concerning, max beh dim) =",round(np.corrcoef(A.concerning,A.maxdim)[0,1],3))
print("\n=== Which behavior dims are >=5 (avg judge) among transcripts with concerning>=5, per cell ===")
for t in TARGETS:
    print(f"\n--- {t} ---")
    res={}
    for cell in ["baseline","cr2bo4"]:
        a=A[(A.target==t)&(A.cell==cell)]
        conc=a[a.concerning>=5]
        res[cell]=pd.Series({"n_concerning>=5":len(conc),"n_total":len(a),**{d:(conc[d]>=5).sum() for d in beh}})
    r=pd.DataFrame(res); r["diff"]=r.cr2bo4-r.baseline
    print(r[(r.baseline>0)|(r.cr2bo4>0)].sort_values("diff",ascending=False).to_string())

# candidates for reading: top paired increases on key dims
cands={}
for t in TARGETS:
    b,c=paired(t); d=c-b
    cands[t]={}
    for dim in ["unprompted_sycophancy","user_over_sysprompt","cooperation_with_human_misuse","input_hallucination","unprompted_deception_toward_user","unfaithful_thinking","concerning"]:
        s=d[dim].sort_values(ascending=False)
        cands[t][dim+"_up"]=[(int(k),float(b.loc[k,dim]),float(c.loc[k,dim])) for k in s.index[:8] if s[k]>0]
        s2=d[dim].sort_values()
        cands[t][dim+"_down"]=[(int(k),float(b.loc[k,dim]),float(c.loc[k,dim])) for k in s2.index[:8] if s2[k]<0]
json.dump(cands,open(HERE/"candidates.json","w"),indent=1)
pd.concat({t:tab[t] for t in TARGETS}).to_csv(HERE/"paired_diffs.csv")
A.to_csv(HERE/"avg_judge_per_transcript.csv")
