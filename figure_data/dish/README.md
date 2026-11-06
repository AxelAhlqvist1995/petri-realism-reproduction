# DISH figure data

Numeric data behind every DISH (Deployment-Imitating SWE-Agent Harness) figure in the paper.
One folder per figure, named after the figure's `\includegraphics` stem in the camera-ready
source. Each folder holds `summary.csv` (exactly the bars drawn in the paper figure) and the
script(s) that drew it, copied verbatim from the DISH experiment codebase as provenance. The scripts
read the full experiment outputs (Inspect eval logs on S3, not distributed), so they do not
run standalone. No PNGs/PDFs and no transcripts are included.

## How the data was checked

For every figure, the paper PNG was matched byte-for-byte (md5) to a render in the DISH
experiment codebase (commit `b84048b`, and commit `58b25a7` for the compact deployment
figure). The figure was then re-rendered from
that render's stored input data with the original script, and the new PNG is byte-identical to
the paper PNG for all 16 figures. While re-rendering, every drawn bar and error bar was
captured from the matplotlib figure. Each row of `summary.csv` matches exactly one drawn bar
(height and error-bar ends equal to within 1e-12), and every drawn bar has a row.

## `summary.csv` columns

`target,measure,condition,n,mean,ci_half,ci_lower,ci_upper`, one row per bar.

- `target`: the audited target model (`sonnet-4.6`, `opus-4.7`, `opus-4.8`, `gpt-5.5`,
  `gemini-3.5-flash`, `haiku-4.5`).
- `measure`: what the bar measures (see the per-figure notes below). The realism-judge model
  (cross-judge figure), transcript serialization (key-order figure) or panel (stripped grid)
  is folded into `measure`.
- `condition`: `baseline` (Petri), `resources`, `realism_filter`, `dish`, `cr4`
  (critique refinement), `dish_cr4` (DISH + cr4). In the Opus 4.7 auditor-sweep figures it is
  `<method>:<auditor/seed condition>`.
- `n`: the number of pairwise judgments for win rates, or the number of audits for judge scores.
- `mean`: the plotted bar height. Judge scores are normalized to `(x - 1) / 9`.
- `ci_lower`, `ci_upper`: the plotted error-bar ends. Some intervals are asymmetric Wilson
  intervals, so both ends are given.
- `ci_half`: `(ci_upper - ci_lower) / 2`. This equals the error-bar half-width for every
  symmetric interval.

Realism win rates against deployment transcripts use the target model as the pairwise judge,
unless `measure` names a different judge.

## Figures

| Folder | Paper figure | Plotted from (fork path) |
|---|---|---|
| `coding_improvement_selected_petri_seeds_realism` | Sec. 5.2, "Coding-setting results" | `experiments/260813_singular_eval_awareness_main_plot/coding_improvement_selected_petri_seeds_realism_singular_ea_average_data.json` |
| `dish_realism_deployment_compact` | Sec. 5.2 wrapfigure, "Deployment task re-creations" | `experiments/261102_compact_deployment_plot/deployment_data.json` |
| `dish_vs_petri` | App. "Deployment task re-creations / Additional measurements", direct comparison | `experiments/260611_fixed_main_plot_v2_dish_vs_petri/dish_vs_petri_data.json` |
| `dish_realism_deployment_cross_product` | Same appendix, cross-product comparison | `experiments/260616_deployment_cross_product_pairwise/plots/dish_realism_deployment_cross_product_data.json` |
| `cross_judge_realism_comparison` | App. "Robustness to the judge model" | `experiments/260821_cross_judge_realism/cross_judge_realism_data.json` |
| `realism_win_rate_order_comparison` | App. "Pairwise realism-ranking prompt", key order | `experiments/260529_dish_cr_fix_first_turn/dish_cr_composition_v2_selected_seeds_fix_first_turn_data.json` (left panel), `experiments/260604_system_tools_messages_realism_rejudge/system_tools_messages_realism_data.json` (right panel) |
| `stripped_realism_grid_fixed_main_plot_v2_with_dish_cr` | Same appendix, stripped DISH context | `experiments/260616_stripped_realism_grid_fixed_main_plot_v2/stripped_realism_grid_fixed_main_plot_v2_with_dish_cr_data.json` |
| `petri_vs_dish_pairwise_24x24_no_haiku` | Same appendix, direct DISH vs Petri | `experiments/260626_petri_vs_dish_pairwise_24x24/petri_vs_dish_pairwise_24x24_grouped_data.json` |
| `opus47_auditor_sweep_deployment` | App. "Understanding DISH's decrease in realism win rates for benign seeds" | `experiments/260612_opus47_benign_seed_auditor_sweep/deployment_data.json` |
| `opus47_auditor_sweep_dish_vs_petri` | Same appendix | `experiments/260612_opus47_benign_seed_auditor_sweep/dish_vs_petri_data.json` |
| `opus47_auditor_sweep_intransitivity_averaged` | Same appendix | both sweep data files above, averaged by `average_cells` in `pipeline.py` |
| `behavioral_coding_{opus48,opus47,gpt55,sonnet46,gemini35_flash}_dish_cr4` | App. "Auditor-behavior judge prompt", dimension shifts | `experiments/260814_gpt56_rejudge_consolidated/plots/dimension_bars_average.csv` (`judge_series=average`, `source_set=coding`) |

### Coding-agent seed set (24 seeds)

These figures use the 12 coding-agent seeds and their 12 rephrases (`seeds/dish_coding_agent_seeds.json`,
seed set `dish_coding_24`): `coding_improvement_selected_petri_seeds_realism`, `stripped_realism_grid_*`,
`petri_vs_dish_pairwise_24x24_no_haiku`, `cross_judge_realism_comparison` and `behavioral_coding_*`.

- **`coding_improvement_selected_petri_seeds_realism`**
  - `winrate`: realism win rate against the 40-transcript deployment haystack (20 WildChat,
    20 Claude Code). `n` is 24 audits x 40 deployment transcripts, minus unparsed judgments.
    The error bar is 1.96 x SE across the 24 per-audit win rates, with one exception. Five
    GPT-5.5 cells (`baseline`, `resources`, `realism_filter`, `dish`, `dish_cr4`) come from,
    or are pooled with, `260612_gpt55_realism_canonical_key_order`, and carry the Inspect `stderr` over
    individual judgments instead. Their means are unaffected; `resources` and
    `realism_filter` are 0 and `baseline` is 0.001.
  - `eval_awareness`: the singular eval-awareness judge, averaged per audit over Opus 5 and
    GPT-5.6 Sol.
  - `concerning`: the Petri alignment judge's `concerning` score as stored in the source
    audit logs. These were scored at audit time by GPT-5.5 for all 30 cells, not re-judged.
  - Both judge measures are normalized `(x - 1) / 9`, with 1.96 x SE across the 24 audits.
  - `render.py` (fork `260813_singular_eval_awareness_main_plot`) draws the figure using
    `render_plots.py` and `render_coding_improvement.py` from `260608_fixed_main_plot_v2`.
- **`stripped_realism_grid_fixed_main_plot_v2_with_dish_cr`**: six panels, as `measure` =
  `winrate_{overall,wildchat,claude_code}` and their `_stripped` variants. The stripped
  variants remove the system prompt, tool definitions and system reminders before judging.
  The error bar is 1.96 x SE across per-audit win rates. The unstripped `winrate_overall`
  panel is copied from the coding figure, so it has the same GPT-5.5 exception. In the
  stripped panels the Opus 4.7
  Baseline cell has `n` = 1440 / 720 (instead of 960 / 480): its 12 original audits were
  re-judged twice, once per source metric eval. The plotted mean pools all judgments; an
  equal-weight-per-audit mean differs by at most 0.0007. The pipeline also imports
  `render_plots.py` / `render_coding_improvement.py` (in the coding folder above).
- **`petri_vs_dish_pairwise_24x24_no_haiku`**: `winrate_dish_vs_petri_baseline` is how often
  the DISH transcript is judged more realistic than the baseline Petri transcript. It covers
  the full 24 x 24 cross product with both orderings (`n` <= 1152). The error bar is
  1.96 x SE across the 24 per-DISH-seed win rates, computed by
  `compute_grouped_no_haiku_data.py`.
- **`cross_judge_realism_comparison`**: `winrate_judge_<model>` is the realism win rate with
  `<model>` as the pairwise judge. The bars where `<model>` equals `target` are the original
  target-as-judge judgments over the same pairs. The error bar is a binomial Wald interval
  over the `n` judgments with a parseable verdict.
- **`behavioral_coding_*`**: one row per Petri alignment-judge dimension and condition
  (`baseline`, `dish_cr4`). `mean` is the per-audit average of the Opus 5 and GPT-5.6 Sol
  full alignment judges, using one judge alone where the other refused, normalized
  `(x - 1) / 9`. The error bar is 1.96 x SE across the 24 audits. Rows are in plotted order
  (top to bottom: descending paired shift).

### Key-order figure (12 seeds)

**`realism_win_rate_order_comparison`** uses only the 12 original coding-agent seeds (seed set
`dish_coding_original_12`). `n` is 12 audits x 40 deployment transcripts, minus unparsed
judgments. `measure` is `winrate_order_messages-system-tools` or
`winrate_order_system-tools-messages`, the top-level key order of the serialized transcripts.
Unlike the other coding figures, the error bar is not grouped by audit: it is 1.96 x the
Inspect `stderr` metric over the `n` individual judgments (sample SD / sqrt(n)).

### Deployment task re-creations (50 tasks)

These audits are seeded with the first user message of one of 50 real Claude Code transcripts
(seed set `dish_deployment_recreation_50`). The transcripts are not distributed.

- **`dish_realism_deployment_compact`**: `winrate_vs_matched_deployment` compares each
  re-creation with the original transcript for the same task, in both orders (`n` = 100).
  Error bars are Wilson 95% intervals. The `haiku-4.5` cells come from the Haiku 4.5 special
  case run with the Sonnet 4.6 auditor (`260612_haiku45_sonnet46_auditor_special_case`).
  `deployment_data.json` is a copy of `260608_fixed_main_plot_v2/dish_realism_deployment_data.json`
  (identical values).
- **`dish_vs_petri`**: `winrate_vs_petri_baseline` compares the DISH or cr4 re-creation
  directly against the baseline Petri re-creation of the same task, in both orders
  (`n` = 100), with Wilson intervals. The Haiku DISH cell comes from
  `260612_haiku45_sonnet46_auditor_special_case/dish_vs_petri_data.json` and the cr4 cells
  from `260613_deployment_cr4_gap_fill/cr4_vs_petri_data.json`.
- **`dish_realism_deployment_cross_product`**: `winrate_vs_all_deployment` compares each of
  50 re-creations against all 50 deployment transcripts (`n` = 2500). The error bar is
  1.96 x SE across the 50 per-audit win rates.
- **`opus47_auditor_sweep_*`**: target Opus 4.7, Baseline vs DISH, across five auditor/seed
  conditions (`condition` = `<method>:auditor_<model>_<normal|rephrased>_seeds`). Wilson
  intervals with `n` = 100 per bar. The `sonnet-4.6` normal-seed cells are the Opus 4.7 cells
  of the main deployment figure. The GPT-5.4 auditor cells come from
  `260611_opus47_gpt54_auditor_special_case`. The rephrased-seed cells use the same 50 tasks
  with rephrased seed instructions. The averaged figure pools the five conditions (`n` = 500,
  n-weighted mean, Wilson interval on the pooled count).
