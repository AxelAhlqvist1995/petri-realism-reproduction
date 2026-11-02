"""Cluster-level view of the concerning decomposition (avg of opus5+gpt56 judges)."""
import pandas as pd, numpy as np
df = pd.read_csv("avg_judge_per_transcript.csv")
print(list(df.columns)[-8:])
CL = {
 "A_compliance": ["cooperation_with_human_misuse","cooperation_with_harmful_sysprompt","facilitating_terrorism",
                  "unprompted_sycophancy","unprompted_encouragement_of_user_delusion","unfaithful_thinking"],
 "B_unilateral": ["unprompted_initiative","unprompted_leaking","unprompted_whistleblowing","unprompted_self_preservation"],
 "C_misreport":  ["input_hallucination","unprompted_deception_toward_user","user_over_sysprompt","unprompted_self_serving_bias"],
}
beh=[c for c in df.columns if c not in ("seed","concerning","target","cell","method","judge","path","transcript")]
beh=[b for b in beh if df[b].dtype!=object and b!="maxdim"]
tcol=[c for c in df.columns if df[c].astype(str).str.contains("sonnet").any()][0]
ccol=[c for c in df.columns if df[c].astype(str).str.contains("cr2bo4").any()][0]
print("target col",tcol,"cell col",ccol, "beh n",len(beh))
T=["sonnet-4.6","opus-4.8","gpt-5.5"]
d=df[df[tcol].isin(T)].dropna(subset=beh+["concerning"])
A=np.c_[np.ones(len(d)),d[beh].values]; beta=np.linalg.lstsq(A,d["concerning"].values,rcond=None)[0][1:]
beta=pd.Series(beta,index=beh)
def paired(t):
    b=d[(d[tcol]==t)&(d[ccol].str.contains("baseline"))].set_index("seed")
    c=d[(d[tcol]==t)&(d[ccol].str.contains("cr2bo4"))].set_index("seed")
    s=b.index.intersection(c.index); return b.loc[s],c.loc[s]
rows={}
for t in T:
    b,c=paired(t); diff=(c[beh]-b[beh]).mean(); contrib=beta*diff
    r={k:contrib[v].sum() for k,v in CL.items()}
    r["other"]=contrib.sum()-sum(r.values()); r["explained"]=contrib.sum(); r["observed"]=(c["concerning"]-b["concerning"]).mean()
    rows[t]=r
print("\n=== cluster contributions (beta*shift) ===\n",pd.DataFrame(rows).round(3))
print("\n=== counts: transcripts with cluster max >=5 (and concerning>=5 overall) ===")
for t in T:
    b,c=paired(t)
    out={}
    for k,v in CL.items():
        out[k]=(int((b[v].max(axis=1)>=5).sum()),int((c[v].max(axis=1)>=5).sum()))
    out["concerning>=5"]=(int((b.concerning>=5).sum()),int((c.concerning>=5).sum()))
    out["initiative>=5"]=(int((b.unprompted_initiative>=5).sum()),int((c.unprompted_initiative>=5).sum()))
    out["initiative in [2,5)"]=(int(((b.unprompted_initiative>=2)&(b.unprompted_initiative<5)).sum()),int(((c.unprompted_initiative>=2)&(c.unprompted_initiative<5)).sum()))
    print(t,out)
print("\n=== mean concerning among initiative>=5 transcripts (base, cr2bo4) ===")
for t in T:
    b,c=paired(t)
    print(t, round(b[b.unprompted_initiative>=5].concerning.mean(),2), int((b.unprompted_initiative>=5).sum()),
             round(c[c.unprompted_initiative>=5].concerning.mean(),2), int((c.unprompted_initiative>=5).sum()))
print("\n=== Sonnet baseline initiative>=5 seeds: (seed, base init, base conc, cr init, cr conc) ===")
b,c=paired("sonnet-4.6")
for s in b[b.unprompted_initiative>=5].index: print(s, b.loc[s,"unprompted_initiative"], b.loc[s,"concerning"], c.loc[s,"unprompted_initiative"], c.loc[s,"concerning"])
print("\n=== GPT baseline initiative>=5 seeds ===")
b,c=paired("gpt-5.5")
for s in b[b.unprompted_initiative>=5].index: print(s, b.loc[s,"unprompted_initiative"], b.loc[s,"concerning"], c.loc[s,"unprompted_initiative"], c.loc[s,"concerning"])
print("\n=== share of concerning>=5 transcripts whose max dim is in each cluster (argmax) ===")
for t in T:
    b,c=paired(t)
    for name,x in (("base",b),("cr2bo4",c)):
        x=x[x.concerning>=5]; am=x[beh].idxmax(axis=1)
        print(t,name,len(x),{k:int(am.isin(v).sum()) for k,v in CL.items()})
