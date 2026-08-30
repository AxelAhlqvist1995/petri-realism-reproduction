# Reproduction assets for "Improving Evaluation Realism with Inference-Time Compute and Deployment Scaffolds"

Frozen prompts and small reproduction assets for the paper
*Improving Evaluation Realism with Inference-Time Compute and Deployment Scaffolds*.

The experiments were run on a fork of [Petri 2](https://alignment.anthropic.com/2026/petri-v2/).
Because the upstream [Petri](https://github.com/safety-research/petri) repository is a moving
target, this repo pins the exact prompt text behind the paper's quantitative measures in one
frozen, citable location. `SOURCES.md` maps every file back to the constant or module it was
extracted from.

The critique refinement / best-of-N protocol itself is released separately as
[petri-bon](https://github.com/AxelAhlqvist1995/petri-bon), a standalone package that ports the
paper's protocols to the current Petri release.

## Contents

### `prompts/concerning_judge/`

- `full_judge_prompt_assembled.txt` — the complete system prompt of Petri's rubric judge
  (~70k chars), which produces the paper's **concerning** score along with the other 39
  rubric dimensions. Petri assembles this prompt at runtime from a template plus a dimensions
  dict; it is not stored assembled anywhere in the codebase, so we ship the assembled result.
  See `SOURCES.md` for the exact assembly procedure.

### `prompts/singular_judges/`

The paper's per-dimension ("singular") judges score one dimension per judge call instead of
the full 40-dimension rubric (paper appendix "Prompts for measures"):

- `unified_judge_system.txt` — shared system prompt (`{transcript_xml}` placeholder).
- `sub_judge_user.txt` — user message template for a single dimension
  (`{dimension_name}`, `{dimension_description}` placeholders).
- `sub_judge_user_with_seed.txt` — variant that additionally provides the auditor's seed
  instruction (`{seed_instruction}`), used for seed-aware dimensions.
- `dimension_eval_awareness.txt` — the **eval-awareness** dimension text.
- `dimension_s_seed_adherence.txt` — the **seed adherence** dimension text (seed-aware).
- `dimension_concerning_calibrated.txt` — the calibrated concerning dimension, also scored
  as a singular judge.
- `dimension_audit_situational_awareness.txt` — the audit situational-awareness dimension.
- `dimension_eval_awareness_seeded_v3_addendum.txt` — the seeded-v3 addendum appended to the
  eval-awareness dimension for the seed-aware EA judge variant evaluated in the paper.

### `prompts/realism_win_rate/`

- `realism_ranking_prompt.txt` — the pairwise ranking prompt behind the paper's **realism
  win rate**: a judge is shown one audit transcript and one deployment transcript and ranks
  them from least to most realistic (paper appendix "Pairwise realism-ranking prompt").

### `prompts/critique_refinement/`

The prompt templates of the critique refinement / best-of-N protocol (paper appendices
"Preference prompt", "Critique and refine", "Auditor context between rounds"). In the fork
these are `.txt` templates loaded by name; the roles they play are bound in
`config/defaults.yaml`.

- `preference_transcript/` — the logprob preference prompts that score a single candidate
  transcript as audit (A) vs deployment (D), including the structured prompt
  (`basic_structured_full_no_prefill.txt`, the paper's "Structured prompt") and its
  task/env/user ablations.
- `preference_pairwise/` — the pairwise champion-vs-challenger selection prompts used by the
  main-text protocol (`basic_shared_prefix_no_prefill.txt` is the default;
  `audit_framed_shared_prefix_no_prefill.txt` is the audit-framed variant).
- `feedback/` — the critique ("feedback") prompts that turn preference outcomes into
  revision guidance: `natural.txt` (logprob), `natural_pairwise.txt` (pairwise),
  `natural_score.txt` (score variant), plus the seed-adherence mitigation variants
  (`*_seedpush*`, `*_seed_aware`, `*_no_rewrite`) used in the ablations.
- `auditor_context/` — the messages inserted into the auditor's context between refinement
  rounds (`preference_round1.txt` / `preference_round_n.txt` and the pairwise/realism
  equivalents).

### `prompts/auditor_behavior_judge/`

The gpt-5.5 auditor-behavior judge (paper appendix "Auditor behavior judge prompt"): rates
eight audit characteristics (harmful_intent, scenario_detail, pressure_to_take_action,
audit_buildup, audit_success, auditor_persistence_adaptation, action_affordances,
system_prompt_pressure) on auditor-created content only. `system.txt` is the judge prompt,
`user_template.txt` the per-audit message. Transcribed from the paper appendix; see
`SOURCES.md`.

### `prompts/environment_realism/`

The two prompts tuned for the isolating-environment-realism experiments (paper appendix
"Isolating environment realism prompts"): `realism_filter_prompt.txt` scores individual
tool-call results as real vs LLM-simulated; `isolating_environment_realism_prompt.txt` is
the pairwise real-vs-simulated-environment prompt. Transcribed from the paper appendix.

### `seeds/`

The seed instruction lists and the exact seed ID sets behind each experiment cell
(main-text 149, logprob-appendix 157, random subset 54/50, unrealistic 13). Contains the
upstream Petri benchmark canary GUID — see `seeds/README.md`.

### `config/defaults.yaml`

The fork's shared experiment defaults: judge/preference/realism model choices, turn limits,
auditor token settings, and the prompt-name bindings for the critique refinement protocol.

### `figure_data/final_paper_plots/`

The plotting scripts, numeric summary CSVs, and per-seed judge scores
(`judge_scores.json`, extracted by `judge_scores.py`) behind the paper's figures, including
its README documenting judge versions and the strict-subset fairness rule. PNGs are omitted
(regenerable); the scripts read the full experiment outputs, which are not distributed, so
they serve as provenance rather than standalone runnable code. Two data files were withheld:
one JSON containing verbatim transcript excerpts and one CSV of per-transcript human
annotations.

### `data/wildchat/`

The 20 WildChat conversations (10 harmful, 10 vulnerable) used as deployment transcripts in
the realism win rate haystack, redistributed with attribution under WildChat's ODC-BY
license so the measure can be reproduced without re-deriving the exact subset. See
`data/wildchat/README.md` for provenance.

### `code/`

- `realism_win_rate.py` — the needle-in-haystack realism win rate implementation used in the
  paper (verbatim from the fork). See `code/README.md` for its data requirements.
- `openai_safety.py` — the one small fork-internal module it imports.

## What is deliberately not here

- **No audit transcripts and no Claude Code deployment transcripts.** The CBAI deployment
  transcripts were donated for internal use only and will not be released. (The WildChat
  portion of the realism win rate haystack *is* included, under `data/wildchat/`.)
- No datasets or experiment outputs (hundreds of MB in the fork).
- Not a runnable Petri fork — for running audits, use upstream Petri plus
  [petri-bon](https://github.com/AxelAhlqvist1995/petri-bon).
