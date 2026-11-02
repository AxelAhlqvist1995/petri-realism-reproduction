#!/usr/bin/env python3
"""Build LaTeX tables: (1) ladder of paired dimension shifts, (2) regression decomposition of the concerning shift."""
import pandas as pd, numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent
NAMES={"concerning":"Concerning","unprompted_sycophancy":"Unprompted sycophancy","user_over_sysprompt":"User over sysprompt",
 "cooperation_with_human_misuse":"Coop.\\ with human misuse","input_hallucination":"Input hallucination","unprompted_initiative":"Unprompted initiative",
 "unfaithful_thinking":"Unfaithful thinking","unprompted_deception_toward_user":"Deception toward user","unprompted_leaking":"Unprompted leaking",
 "unprompted_whistleblowing":"Unprompted whistleblowing","cooperation_with_harmful_sysprompt":"Coop.\\ with harmful sysprompt",
 "manipulated_by_developer":"Manipulated by developer","unprompted_self_serving_bias":"Self-serving bias",
 "unprompted_encouragement_of_user_delusion":"Encouragement of delusion","unprompted_self_preservation":"Self-preservation"}
NAMES={k:("Concerning" if k=="concerning" else "\\subdim{"+k+"}") for k in list(NAMES)+["facilitating_terrorism"]}
TN={"sonnet-4.6":"Sonnet 4.6","opus-4.8":"Opus 4.8","gpt-5.5":"GPT-5.5","gemini-3.5-flash":"Gemini 3.5 Flash"}
o=pd.read_csv(HERE/"ladder_diffs.csv")
FOLD=["cr1","cr2","cr2bo2","cr2bo4"]
dims=["concerning","unprompted_sycophancy","user_over_sysprompt","cooperation_with_human_misuse","input_hallucination","unprompted_initiative","unfaithful_thinking"]
L=[]
L.append(r"\begin{tabular}{llrrrr}"); L.append(r"\toprule")
L.append(r"Dimension & Target & $cr1$ & $cr2$ & $cr2bo2$ & $cr2bo4$ \\"); L.append(r"\midrule")
for dim in dims:
    for i,t in enumerate(TN):
        cells=[]
        for f in FOLD:
            r=o[(o.dim==dim)&(o.target==t)&(o.folder==f)]
            if r.empty: cells.append("--"); continue
            v=r["diff"].iloc[0]; s="^{*}" if r["sig"].iloc[0] else ""
            cells.append(f"${v:+.2f}{s}$")
        L.append((NAMES[dim] if i==0 else "")+" & "+TN[t]+" & "+" & ".join(cells)+r" \\")
    if dim!=dims[-1]: L.append(r"\addlinespace")
L.append(r"\bottomrule"); L.append(r"\end{tabular}")
(HERE/"table_ladder.tex").write_text("\n".join(L)+"\n")

# decomposition table from analyze.py output (recompute here for self-containment)
df=pd.read_csv(HERE/"rows.csv")
beh=["cooperation_with_human_misuse","facilitating_terrorism","cooperation_with_harmful_sysprompt","user_over_sysprompt",
 "manipulated_by_developer","unprompted_deception_toward_user","unprompted_sycophancy","unprompted_encouragement_of_user_delusion",
 "unprompted_leaking","unprompted_whistleblowing","unprompted_blackmail","unprompted_initiative","unprompted_emotional_manipulativeness",
 "unprompted_self_preservation","unfaithful_thinking","input_hallucination","unprompted_self_serving_bias","unprompted_gratitude",
 "unprompted_spirituality_or_extreme_awe","unprompted_pos_affect","unprompted_neg_affect","unprompted_bizarre_behavior"]
T=["sonnet-4.6","opus-4.8","gpt-5.5"]
A=df.groupby(["target","cell","seed"])[beh+["concerning"]].mean().reset_index().dropna()
A=A[A.target.isin(["sonnet-4.6","opus-4.8","gpt-5.5"])&A.cell.isin(["baseline","cr2bo4"])]
regb=[b for b in beh if A[b].std()>0]
X=np.c_[np.ones(len(A)),A[regb].values]; y=A.concerning.values
beta=np.linalg.lstsq(X,y,rcond=None)[0]; B=pd.Series(beta[1:],index=regb)
pred=X@beta; r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
diffs={}; dconc={}
for t in T:
    b=A[(A.target==t)&(A.cell=="baseline")].set_index("seed"); c=A[(A.target==t)&(A.cell=="cr2bo4")].set_index("seed")
    s=b.index.intersection(c.index); d=(c.loc[s,regb+["concerning"]]-b.loc[s,regb+["concerning"]])
    diffs[t]=d[regb].mean(); dconc[t]=d.concerning.mean()
contrib=pd.DataFrame({t:B*diffs[t] for t in T})
contrib["gap"]=(contrib["sonnet-4.6"]+contrib["opus-4.8"])/2-contrib["gpt-5.5"]
GROUPS=[("Going along with a request made to look legitimate",
         ["cooperation_with_human_misuse","cooperation_with_harmful_sysprompt","manipulated_by_developer","facilitating_terrorism",
          "unprompted_sycophancy","unprompted_encouragement_of_user_delusion","unfaithful_thinking"]),
        ("Drastic unilateral action",
         ["unprompted_initiative","unprompted_whistleblowing","unprompted_leaking","unprompted_self_preservation"]),
        ("Misreporting inside long agentic workflows",
         ["input_hallucination","unprompted_deception_toward_user","user_over_sysprompt","unprompted_self_serving_bias"])]
NAMES["facilitating_terrorism"]="Facilitating terrorism"
def row(name,r,bold=False):
    f=(lambda x:f"$\\mathbf{{{x:+.2f}}}$") if bold else (lambda x:f"${x:+.2f}$")
    return f"{name} & {f(r['sonnet-4.6'])} & {f(r['opus-4.8'])} & {f(r['gpt-5.5'])} & {f(r['gap'])} \\\\"
L=[r"\begin{tabular}{lrrrr}",r"\toprule",r"Dimension & Sonnet 4.6 & Opus 4.8 & GPT-5.5 & Claude$-$GPT gap \\",r"\midrule"]
used=[]
for gname,dims in GROUPS:
    dims=[d for d in dims if d in contrib.index]; used+=dims
    L.append("\\multicolumn{5}{l}{\\textit{"+gname+"}} \\\\")
    for d in dims: L.append(r"\quad "+row(NAMES.get(d,d),contrib.loc[d]))
    L.append(row(r"\quad \textbf{Group total}",contrib.loc[dims].sum(),bold=True))
    L.append(r"\addlinespace")
rest=contrib.drop(used).sum()
L.append(row("All other dimensions",rest))
L.append(r"\midrule")
tot=contrib.sum()
L.append(row("Explained shift",tot))
g=(dconc['sonnet-4.6']+dconc['opus-4.8'])/2-dconc['gpt-5.5']
L.append(f"Observed concerning shift & ${dconc['sonnet-4.6']:+.2f}$ & ${dconc['opus-4.8']:+.2f}$ & ${dconc['gpt-5.5']:+.2f}$ & ${g:+.2f}$ \\\\")
L+= [r"\bottomrule",r"\end{tabular}"]
(HERE/"table_decomposition.tex").write_text("\n".join(L)+"\n")
print("R2",round(r2,3)); print(open(HERE/"table_decomposition.tex").read()); print(open(HERE/"table_ladder.tex").read())
