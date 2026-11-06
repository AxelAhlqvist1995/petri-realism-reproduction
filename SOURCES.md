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

Which template each reported cell used: `config/defaults.yaml` holds the un-overridden
defaults (`pairwise.feedback_prompt_name: "natural_pairwise"`). Every pairwise cell reported
in the paper's main text and scaling/multibranch appendices (`outputs/reasoning/pairwise/
target_*/auditor_sonnet-4.6/*/config.yaml` in the fork) overrides it to
`natural_pairwise_seed_aware` with the `pairwise_seed_aware_once` auditor-context templates,
as the paper's "Critique refinement implementation details" appendix states. The one
exception is the selector-comparison cell of the logprob appendix
(`ablation_protocol_no_stopping/pairwise/cr4/target_sonnet-4.6`, auditor Opus 4.7), which ran
with the base `natural_pairwise` template. Note that `natural_pairwise.txt` differs from
`natural_pairwise_seed_aware.txt` not only by the two seed-adherence sentences the paper
quotes but also by one capitalisation ("does NOT need" vs "does not need").

The realism-ranking prompt ends in a single `{transcripts}` placeholder; `code/realism_win_rate.py`
expands it into the two `<transcript_0>…</transcript_0>` / `<transcript_1>…</transcript_1>`
blocks shown in the paper's "Pairwise realism-ranking prompt" appendix.

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
| `figure_data/final_paper_plots/` | `experiments/general_plotting/final_paper_plots/` (fork commit `2850ce3`, 2026-11-03) minus PNGs/PDFs, caches, `.bak` files, and four withheld data files (plus the per-transcript human ratings in `dataset/ea_gt_v3/`, which `appendix/ea_human_rated_comparison/plot_human_rated_ea.py` reads; only its aggregate `plots/summary.csv` and `plots/paired_diff.csv` are included): `appendix/endsignal/endsignal_concerning.json` (verbatim transcript excerpts), `appendix/ea_judge_vs_human_grids/plots/pairs.csv` (per-transcript human annotations), `appendix/concerning_subdims/judge_texts.json` (per-transcript judge write-ups quoting transcripts), and `appendix/concerning_subdims/auditor_behavior/ratings/*.jsonl` (raw auditor-behavior judge responses; the derived `auditor_behavior_shifts*.csv` summaries are included) |
| `figure_data/final_paper_plots/appendix/preference_prompt/` | `experiments/general_plotting/paper/preference_prompt_{ablation,strategies,filters}.py` (verbatim) and their `ablation_data.csv` outputs from `experiments/general_plotting/paper/plots/{paper_2026-05-02_15-20-28,strategies_2026-05-02_15-20-31,filters_2026-05-02_15-20-34}/` (the runs behind the paper's preference-prompt appendix figures; re-running the scripts on 2026-11-03 reproduced them exactly). Placed under `final_paper_plots/appendix/` here although they live one directory up in the fork. |
| `data/wildchat/wildchat_{harmful,vulnerable,mixed}` | `dataset/realism_win_rate/` in the fork (byte-for-byte; WildChat data, ODC-BY — see `data/wildchat/README.md`) |
| `code/realism_win_rate.py` | `experiments/transcript_realism/realism_win_rate.py` (verbatim) |
| `code/openai_safety.py` | `src/petri/openai_safety.py` (verbatim) |

### DISH figure data and seeds

The DISH experiments ran in a separate experiment codebase, not the Petri 2 fork above.
Paths in this table are relative to that codebase at commit `b84048b`, except the compact
deployment figure, which is at commit `58b25a7`. Everything needed to read the DISH figures
is in this repo; the codebase itself is not needed.

Scripts are verbatim copies. Each `summary.csv` was produced in three steps:

1. Re-render the figure with its original script from the listed input data.
2. Check that the re-rendered PNG is byte-identical to the camera-ready figure.
3. Capture the drawn bars and error bars into the CSV.

See `figure_data/dish/README.md` for the method and the column definitions.

| Item in this repo | Source (DISH experiment codebase) |
|---|---|
| `figure_data/dish/coding_improvement_selected_petri_seeds_realism/` | `experiments/260813_singular_eval_awareness_main_plot/render.py` and its input `coding_improvement_selected_petri_seeds_realism_singular_ea_average_data.json`; drawing helpers `experiments/260608_fixed_main_plot_v2/render_plots.py`, `render_coding_improvement.py` |
| `figure_data/dish/dish_realism_deployment_compact/` | `experiments/261102_compact_deployment_plot/pipeline.py` and `deployment_data.json` (commit `58b25a7`; the data is a copy of `experiments/260608_fixed_main_plot_v2/dish_realism_deployment_data.json`) |
| `figure_data/dish/dish_vs_petri/` | `experiments/260611_fixed_main_plot_v2_dish_vs_petri/pipeline.py` and `dish_vs_petri_data.json` (Haiku 4.5 DISH cell from `experiments/260612_haiku45_sonnet46_auditor_special_case/`, cr4 cells from `experiments/260613_deployment_cr4_gap_fill/`) |
| `figure_data/dish/dish_realism_deployment_cross_product/` | `experiments/260616_deployment_cross_product_pairwise/pipeline.py` and `plots/dish_realism_deployment_cross_product_data.json` |
| `figure_data/dish/cross_judge_realism_comparison/` | `experiments/260821_cross_judge_realism/pipeline.py` and `cross_judge_realism_data.json` |
| `figure_data/dish/realism_win_rate_order_comparison/` | `experiments/260604_system_tools_messages_realism_rejudge/render_realism_order_comparison.py`; inputs `system_tools_messages_realism_data.json` (right panel) and `experiments/260529_dish_cr_fix_first_turn/dish_cr_composition_v2_selected_seeds_fix_first_turn_data.json` (left panel) |
| `figure_data/dish/stripped_realism_grid_fixed_main_plot_v2_with_dish_cr/` | `experiments/260616_stripped_realism_grid_fixed_main_plot_v2/{dish_cr_backfill_pipeline,gemini35_flash_backfill_pipeline,pipeline}.py` and `stripped_realism_grid_fixed_main_plot_v2_with_dish_cr_data.json` |
| `figure_data/dish/petri_vs_dish_pairwise_24x24_no_haiku/` | `experiments/260626_petri_vs_dish_pairwise_24x24/{render_petri_vs_dish_pairwise_24x24_no_haiku,render_petri_vs_dish_pairwise_24x24,compute_grouped_no_haiku_data}.py` and `petri_vs_dish_pairwise_24x24_grouped_data.json` |
| `figure_data/dish/opus47_auditor_sweep_{deployment,dish_vs_petri,intransitivity_averaged}/` | `experiments/260612_opus47_benign_seed_auditor_sweep/pipeline.py` and `deployment_data.json`, `dish_vs_petri_data.json` |
| `figure_data/dish/behavioral_coding_{opus48,opus47,gpt55,sonnet46,gemini35_flash}_dish_cr4/` | `experiments/260814_gpt56_rejudge_consolidated/render_dimensions.py` and `plots/dimension_bars_average.csv` (rows `judge_series=average`, `source_set=coding`) |
| `figure_data/dish/README.md` | Written for this repo |
| `seeds/dish_coding_agent_seeds.json` | Originals: `src/petri/tasks/datasets/dish_compatible_seeds.csv`. Rephrases: `experiments/260605_dish_cr_pipeline_rephrased_seeds/dish_compatible_seeds_rephrased.csv`. Checked byte-for-byte against the seed instruction recorded in each coding audit log behind the figures. |
| `seeds/paper_seed_sets.json` (`dish_coding_24`, `dish_coding_original_12`) | The same two CSVs; seed positions as used by the coding-figure render scripts |
| `seeds/paper_seed_sets.json` (`dish_deployment_recreation_50`) | `original_sample_id` metadata of the 55 deployment re-creation metric evals behind the deployment figures (listed in the figures' input data files above); split `deployment_50` of the private CBAI transcript dataset |

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
