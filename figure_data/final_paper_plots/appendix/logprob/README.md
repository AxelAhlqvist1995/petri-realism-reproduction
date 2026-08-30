# Logprob-appendix final paper plots

The three figures for the paper's "Critique Refinement using Logprobs"
appendix (`app:cr-logprobs`), each the logprob twin of a main-text figure in
`../../` — same layout, palette and styling, different data tree:

| Script | Output folder | Main-text twin |
|---|---|---|
| `plot_cr_comparison_logprob.py` | `cr_comparison_logprob/` | `plot_cr_comparison.py` — Baseline / Realism filter / cr11 bars for Sonnet 4.6, Opus 4.7, Haiku 4.5 (win rate left, singular EA top-right, concerning bottom-right) + `plot_controlled_strict.png` (cr11 adherence ≥ baseline per seed) |
| `plot_scaling_types_sonnet_logprob.py` | `scaling_types_sonnet_logprob/` | `plot_scaling_types_sonnet.py` — depth (cr1…cr22) / breadth (bo2…bo16) / mix (cr2bo2/4/8) ladders on Sonnet 4.6; paper 1x2 (win rate + adherence) + `plot_4panel.png` |
| `plot_scaling_all_targets_logprob.py` | `scaling_all_targets_logprob/` | `plot_scaling_all_targets.py` — all three targets overlaid, full depth ladders + sonnet combo ladder, 2x2 (+ `plot_2panel.png`, `has_ea.png`) |

## Data

Cells: `outputs/ablation_protocol_no_stopping/<cond>/target_<T>/auditor_opus-4.7`
(the 30 cells rejudged with claude-opus-5 on 2026-08-27; gpt-5.4's original
judging is preserved in `judge_output_history` / the per-judge store).

- **Judge measures** (`concerning`, `eval_awareness`, `s_eval_awareness`,
  `s_seed_adherence`): each cell's `0_transcripts.csv` — opus-5 is the primary
  judge and the ONLY judge on these cells, so unlike the main figures there
  are no `--judge` versions.
- **Realism win rate**: `realism_win_rate/summary.json` per cell (wildchat+cc
  mean via the shared loader). Judge-independent.
- **Cost axis**: the ANALYTICAL cost model
  (`experiments_logic.cost_model.predict_audit_cost`, the paper's
  app:cost-model), exactly like the original appendix figures — these old
  runs' measured `cost_logs` are too sparse to use.

`logprob_cells.py` holds the cell paths, the analytical cost adapter, and
`build_shim()`, which bridges the old layout (`<cond>/target/auditor`) to the
reasoning layout (`…/pairwise/target/auditor/<cond>`) with a symlink tree
(`.shim/`, rebuilt per run) so the shared grid machinery in
`../../../cost_winrate_reasoning/` runs on these cells unchanged. The two
scaling scripts import their main-text twins for the paper styling
monkeypatches — run each script in its own process (the patches stack).

Run from the checkout holding `outputs/` (or set `PETRI_REPO_ROOT`):

```bash
.venv/bin/python experiments/general_plotting/final_paper_plots/appendix/logprob/plot_cr_comparison_logprob.py
.venv/bin/python experiments/general_plotting/final_paper_plots/appendix/logprob/plot_scaling_types_sonnet_logprob.py
.venv/bin/python experiments/general_plotting/final_paper_plots/appendix/logprob/plot_scaling_all_targets_logprob.py
```
