# Source map

Every file in this repo is extracted verbatim from the experiment codebase (a fork of
[Petri 2](https://alignment.anthropic.com/2026/petri-v2/)) used for the paper
*Improving Evaluation Realism with Inference-Time Compute and Deployment Scaffolds*.
This table records where each asset comes from, so readers can line the frozen copies
up against future Petri releases.

Paths below are relative to the Petri 2 fork; `scorers/prompts.py` means
`src/petri/scorers/prompts.py`.

| File in this repo | Source constant | Source location |
|---|---|---|
| `prompts/concerning_judge/full_judge_prompt_assembled.txt` | `JUDGE_PROMPT` + all 40 `DIMENSIONS`, assembled by `_build_judge_prompt` (see below) | `scorers/prompts.py`, `scorers/judge.py` |
| `prompts/singular_judges/unified_judge_system.txt` | `UNIFIED_JUDGE_SYSTEM` | `scorers/prompts.py` |
| `prompts/singular_judges/sub_judge_user.txt` | `SUB_JUDGE_USER` | `scorers/prompts.py` |
| `prompts/singular_judges/sub_judge_user_with_seed.txt` | `SUB_JUDGE_USER_WITH_SEED` | `scorers/prompts.py` |
| `prompts/singular_judges/dimension_eval_awareness.txt` | `DIMENSIONS["eval_awareness"]` (= `SINGULAR_DIMENSIONS["s_eval_awareness"]`) | `scorers/prompts.py` |
| `prompts/singular_judges/dimension_s_seed_adherence.txt` | `SEED_AWARE_DIMENSIONS["s_seed_adherence"]` | `scorers/prompts.py` |
| `prompts/singular_judges/dimension_concerning_calibrated.txt` | `DIMENSIONS["concerning_calibrated"]` (= `SINGULAR_DIMENSIONS["s_concerning_calibrated"]`) | `scorers/prompts.py` |
| `prompts/singular_judges/dimension_audit_situational_awareness.txt` | `DIMENSIONS["audit_situational_awareness"]` (= `SINGULAR_DIMENSIONS["s_audit_situational_awareness"]`) | `scorers/prompts.py` |
| `prompts/singular_judges/dimension_eval_awareness_seeded_v3_addendum.txt` | `_EVAL_AWARENESS_SEEDED_V3_ADDENDUM` (appended to `DIMENSIONS["eval_awareness"]` to form `SINGULAR_SEED_DIMENSIONS["s_eval_awareness_seeded_v3"]`) | `scorers/prompts.py` |
| `prompts/realism_win_rate/realism_ranking_prompt.txt` | `REALISM_RANKING_PROMPT` | `experiments/transcript_realism/prompts.py` |
| `prompts/detailed_eval_awareness/system.txt` | `EVAL_AWARENESS_SYSTEM` | `scorers/eval_awareness.py` |
| `prompts/detailed_eval_awareness/user_template.txt` | `EVAL_AWARENESS_USER` | `scorers/eval_awareness.py` |
| `prompts/detailed_eval_awareness/followup_explanation.txt` | `FOLLOWUP_PROMPT` | `experiments/DEA_prompt/rescore_detailed_eval_awareness.py` |

### Critique refinement prompt templates

The critique refinement / BoN prompts are `.txt` templates in the fork, loaded by name by
`src/petri/solvers/bon.py`; the name-to-role bindings live in `experiments/audit/defaults.yaml`
(copied here as `config/defaults.yaml`, e.g. `bon.feedback_prompt_name: "natural"`,
`pairwise.preference_prompt_name: "basic_shared_prefix_no_prefill"`,
`pairwise.feedback_prompt_name: "natural_pairwise"`).

| Directory in this repo | Source directory (fork) |
|---|---|
| `prompts/critique_refinement/preference_transcript/` | `experiments/prerun/testrun/prompts/preference/transcript/` |
| `prompts/critique_refinement/preference_pairwise/` | `experiments/prerun/testrun/prompts/preference/pairwise/` |
| `prompts/critique_refinement/feedback/` | `experiments/prerun/testrun/prompts/preference/feedback/` |
| `prompts/critique_refinement/auditor_context/` | `experiments/prerun/testrun/prompts/auditor_feedback/` |

All files are byte-for-byte copies. Only the paper-relevant subset of each directory is
included (the `_prefill` variants of the preference prompts and the non-paper experimental
feedback prompts are omitted).

### Prompts transcribed from the paper

Two prompt sets were not preserved in the experiment fork's code tree and are reproduced
from the paper's appendices instead:

| File in this repo | Source |
|---|---|
| `prompts/auditor_behavior_judge/system.txt`, `user_template.txt` | Paper appendix "Auditor behavior judge prompt" (verbatim blocks) |
| `prompts/environment_realism/realism_filter_prompt.txt`, `isolating_environment_realism_prompt.txt` | Paper appendix "Isolating environment realism prompts" (de-LaTeXed) |

In the auditor-behavior judge prompt, the paper's verbatim block renders the XML-tag
examples inside `<transcript_format>` as HTML entities (`&lt;user index="1"&gt;`); these have
been converted back to literal angle brackets here.

### Seeds, figure data, code

| Item in this repo | Source |
|---|---|
| `seeds/seed_instructions.json` | `SEED_INSTRUCTIONS`, `src/petri/tasks/petri.py` (canary GUID from the module docstring preserved) |
| `seeds/single_branch_seed_instructions.json` | `SINGLE_BRANCH_SEED_INSTRUCTIONS`, `src/petri/tasks/single_branch_seeds.py` |
| `seeds/paper_seed_sets.json` | `seed_instruction_ids` from the run configs named in each entry's `source_config`, expanded with `expand_seed_ids` from `experiments/audit/run.py` |
| `config/defaults.yaml` | `experiments/audit/defaults.yaml` (verbatim) |
| `figure_data/final_paper_plots/` | `experiments/general_plotting/final_paper_plots/` minus PNGs, caches, and two withheld data files (`appendix/endsignal/endsignal_concerning.json` — verbatim transcript excerpts; `appendix/ea_judge_vs_human_grids/plots/pairs.csv` — per-transcript human annotations) |
| `data/wildchat/wildchat_{harmful,vulnerable,mixed}` | `dataset/realism_win_rate/` in the fork (byte-for-byte; WildChat data, ODC-BY — see `data/wildchat/README.md`) |
| `code/realism_win_rate.py` | `experiments/transcript_realism/realism_win_rate.py` (verbatim) |
| `code/openai_safety.py` | `src/petri/openai_safety.py` (verbatim) |

## How the full judge prompt was assembled

The full rubric judge prompt (used for the paper's *concerning* score) is not stored
assembled anywhere in the codebase. `alignment_judge` builds it at runtime from the
`JUDGE_PROMPT` template via `_build_judge_prompt` in `src/petri/scorers/judge.py`:

```python
dimensions_text = "\n\n".join(
    f"### {name.replace('_', ' ').title()}\n{description}"
    for name, description in DIMENSIONS.items()
)
scores_list = "\n".join(
    f"{i + 1}. {name}: [score]/{max_score}"   # max_score = 10
    for i, name in enumerate(DIMENSIONS.keys())
)
prompt = JUDGE_PROMPT.format(dimensions_text=dimensions_text, scores_list=scores_list)
```

`full_judge_prompt_assembled.txt` is the output of exactly this procedure over the
40-entry `DIMENSIONS` dict. Note the literal `{{}}/10` placeholders inside dimension
descriptions are preserved: `str.format` does not recurse into substituted values, so
this is byte-for-byte what the judge model receives as its system prompt.
