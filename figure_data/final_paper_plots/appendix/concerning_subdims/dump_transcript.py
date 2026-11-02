#!/usr/bin/env python3
"""Usage: dump_transcript.py TARGET CELL SEED [--full]
TARGET in sonnet-4.6|opus-4.8|gpt-5.5 ; CELL in baseline|cr2bo4.
Prints seed instruction, per-judge rubric scores (key dims), Opus-5 judge summary+justification,
and (with --full) the final_branch_transcript text (target-visible transcript)."""
import glob, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
t, cell, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
full = "--full" in sys.argv
pat = {"baseline": f"outputs/reasoning/baseline/target_{t}/auditor_sonnet-4.6",
       "cr2bo4": f"outputs/reasoning/pairwise/target_{t}/auditor_sonnet-4.6/cr2bo4"}[cell]
fp = glob.glob(str(ROOT / pat / f"transcript_{seed}_*.json"))[0]
d = json.load(open(fp)); md = d["metadata"]; jo = md.get("judge_output") or {}
KEY = ["concerning","cooperation_with_human_misuse","unprompted_sycophancy","user_over_sysprompt","input_hallucination",
       "unprompted_initiative","unfaithful_thinking","unprompted_deception_toward_user","unprompted_leaking","s_seed_adherence"]
print(f"### {t} / {cell} / seed {seed}\n{fp}\n")
print("SEED INSTRUCTION:\n", md.get("seed_instruction"), "\n")
sbj = (md.get("run_config") or {}).get("scores_by_judge") or {}
for j, sc in sbj.items():
    if "opus-5" in j or "5.6" in j:
        print(f"SCORES [{j}]:", {k: sc.get(k) for k in KEY})
print("\nJUDGE SUMMARY (opus-5):\n", jo.get("summary"), "\n")
print("JUDGE JUSTIFICATION (opus-5):\n", jo.get("justification"), "\n")
if full:
    print("=" * 80, "\nFINAL BRANCH TRANSCRIPT:\n", md.get("final_branch_transcript"))
