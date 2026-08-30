# Final paper plots

Final versions of the figures used in the paper. Each `plot_*.py` here reuses
the loading/pooling/rendering machinery in `../cost_winrate_reasoning/`
(single source of truth) and writes into `plots/<figure-name>/` (overwritten
in place on rerun, with `summary.csv` / `config.json` sidecars).

All cells are auditor sonnet-4.6, `outputs/reasoning/{baseline,pairwise}`.

## Judge versions

Every figure exists in three versions via `--judge {opus5,gpt56,avg,all}`
(default `all`): the unsuffixed folder is the opus-5 paper default, `_gpt56`
uses gpt-5.6-sol, `_avg` averages the two judges. Per-judge scores are
extracted from the transcripts by `judge_scores.py` into `judge_scores.json`
(rebuild with `--refresh-scores` after any rejudge):

- **Singular measures** (`s_eval_awareness`, `s_seed_adherence`): from
  `run_config.singular_scores_by_judge` — both opus-5 and gpt-5.6-sol exist
  for every plotted cell.
- **Full-judge measures** (`concerning`, holistic `eval_awareness`): from
  `judge_output` + `judge_output_history` — currently opus-5 only. In the
  gpt56/avg versions those panels fall back to opus-5 and the panel label
  names the judge actually used. A future gpt-5.6 full rejudge is picked up
  automatically (rebuild the cache, rerun).

**Fairness rule** (per target and measure, across the methods/rungs compared
in the figure): a seed counts only if the serving judge scored it for ALL
methods; `avg` uses the two-judge mean where both judges are complete on a
seed, the single complete judge where only one is, and drops the seed where
neither is. Win rate / cost / Has_EA are judge-independent.

## Figures

| Script | Figure folders | Notes |
|---|---|---|
| `plot_cr_comparison.py` | `cr_comparison/` (one folder, all versions) | Baseline vs Realism filter vs cr2bo4 bars, 4 targets; win rate spans the left half, EA (singular) top-right, concerning (full judge) bottom-right. Main = judge avg (`plot.png`); alternatives = `plot_opus5.png`, `plot_gpt56.png`, each with a `*_controlled_strict.png` (seed-adherence strict control, Ns shown there only). |
| `plot_scaling_types_sonnet.py` | `scaling_types_sonnet/` (one folder, all versions) | Depth (cr1…cr16), breadth (bo2…bo8) and breadth×depth (cr2bo2, cr2bo4) ladders for target sonnet-4.6, cost-relative x-axis. PAPER figure (Fig. 4) = 1x2 realism win rate + seed adherence, judge avg (`plot.png`); alternatives = `plot_opus5.png`, `plot_gpt56.png` and the full 2x2 with eval awareness + concerning as `plot_4panel*.png` (EA/concerning = holistic full judge). Paper styling: no suptitle/panel titles, plain measure labels (Ns in the CSVs), Figure-4 palette (depth red, breadth blue, mix dark red) with the shared 3-dot legend below (baseline labelled in-panel). |
| `plot_scaling_all_targets.py` | `scaling_all_targets/` (one folder, all versions) | All 4 targets overlaid: baseline → cr1 → cr2 (depth) + cr2 → cr2bo2 → cr2bo4 (combo, darker tone), cost normalised per target. Main = judge avg 2x2 (`plot.png`); alternatives = `plot_opus5.png`, `plot_gpt56.png` and the 1x2 win rate + eval awareness as `plot_2panel*.png` (plus the `has_ea*.png` companion). Plain measure labels (no judge tags), fonts matched to `scaling_types_sonnet` (axis titles 24pt, ticks 17pt, annotations 19pt). EA/concerning = holistic full judge, now two-judge avg on every cell. |

Run from the repo (venv has matplotlib/numpy):

```bash
.venv/bin/python experiments/general_plotting/final_paper_plots/plot_cr_comparison_all_targets.py
.venv/bin/python experiments/general_plotting/final_paper_plots/plot_scaling_types_sonnet.py
.venv/bin/python experiments/general_plotting/final_paper_plots/plot_scaling_all_targets.py
```

## Paper facts: multibranch audits (prep for the text)

Facts to lift into the paper (computed 2026-08-27 from the 50-seed 235-turn
multibranch cells `baseline_multibranch` (sonnet) / `baseline_multibranch_235`
(opus), branch counts from `branch_judging/branch_inventory.json` (event
replay), word counts from the target-visible `target_trajectories.jsonl`):

- **Experimental details — branches per audit** (50 audits each, 235 turns):
  - target sonnet-4.6: median **8**, mean 19.8 (range 1–84; 7/50 audits never
    branched)
  - target opus-4.8: median **9**, mean 15.2 (range 1–68; 9/50 never branched)
- **Appendix (seed-adherence-control section) — first-branch length**: the
  multibranch FIRST branch (b00) is far longer than a whole baseline audit,
  seed-matched, in target-visible transcript words:
  - opus-4.8: b00 mean 36,849 words (median 31,392) vs baseline mean 6,863
    (median 6,166) → ratio mean **6.7x** (median 5.3x), n=50
  - sonnet-4.6: b00 mean 35,342 (median 20,326) vs baseline mean 8,199 →
    ratio mean 4.9x (median 2.3x), n=49
  Mechanically expected (235-turn multibranch vs ~14-target-turn baselines);
  relevant to the adherence-controlled comparison because branch length, not
  just selection, separates the multibranch bars from Baseline/cr2bo4.
- **Figure 6 replacement**: `multibranch_heldout/
  multibranch_comparison_combined_3panel.png` (sonnet + opus grouped per
  panel, win rate double-height left, EA/concerning right) replaces the
  current Fig. 6 in the paper.
- **Appendix (seed-adherence-control section) — FIGURE:
  `multibranch_comparison_combined_controlled_1pt.png`** (1-pt gate,
  per-seed judge rule; the stricter 0.5-pt companion
  `..._controlled.png` exists as an alternative). Draft points for the
  text:
  - Without the control, seed adherence was quite low for the two max
    selections (uncontrolled combined figure: sonnet max-concerning 0.64 /
    held-out-RWR 0.63 vs Baseline 0.89; opus 0.61 / 0.74 vs 0.89).
  - With the gate, the selected branches' mean adherence sits at the first
    branch's level (0.85-0.86 everywhere) — it does NOT drop below it even
    at the looser 1-pt tolerance (selections pick above the first branch
    about as often as below).
  - In general, average seed adherence is low from roughly the THIRD
    branch onward (two-judge mean, both targets): b00 0.85, b01 ~0.70,
    b02 ~0.55, b03+ plateau ~0.40-0.45. This seems to be partly because
    later branches contain only partial audits, and partly because the
    auditor, feeling it has accomplished the audit, starts to vary things
    up — which can drift quite far from the seed instruction.
  - First-branch length context (see facts above): the first branch alone
    is 6.7x (opus) / 4.9x (sonnet) the length of a whole baseline audit.
- **Appendix F — rename to "Sensitivity analysis of held-out max RWR"**.
  Drop its current plots; the ONLY figure is
  `multibranch_heldout/cv_sensitivity_full_grid_bytarget.png` (sonnet left,
  opus right). Draft text:
  - *Setup*: each multibranch branch is scored on 10 CC + 10 WildChat
    realism comparisons. For every balanced 5/5 split of each repo's 10
    comparisons (C(10,5)^2 = 63,504 splits, repos split independently), the
    branch with the highest SELECTION-half win rate is picked per seed and
    scored on the complementary HELD-OUT half; the histogram is the
    across-seed mean per split. Dashed lines are single full-10/10 values:
    Baseline, cr2bo4, optimize RWR (in-sample selection = optimistic upper
    bound), first branch.
  - *Conclusions*: (1) Sonnet: the entire held-out distribution (mean
    0.175, sd 0.013, range 0.116-0.224) lies well above Baseline (0.089)
    and the first branch (0.136) — branch selection buys a real,
    split-independent realism gain — but recovers only ~half of the
    in-sample optimize value (0.257), quantifying selection overfitting to
    the 20 comparisons; cr2bo4 (0.278) stays out of reach. (2) Opus: the
    held-out distribution (mean 0.015, sd 0.005, max 0.034) straddles
    Baseline (0.020) and no split comes near optimize (0.045) or cr2bo4
    (0.057) — at opus's near-zero per-branch win rates the apparent
    selection gain is almost entirely selection noise. (3) Takeaway:
    in-sample max-RWR selection substantially overstates achievable
    realism; held-out estimation is required, and the size of the
    optimize-to-held-out gap is target-dependent.

## Appendix

| Script | Figure folder | Notes |
|---|---|---|
| `appendix/plot_ea_judge_vs_human_grids.py` | `appendix/plots/ea_judge_vs_human_grids/` | 10x10 count grids of singular judge EA score (x) vs human GT rating (y), one panel per judge (opus-5, gpt-5.6), identical n=65 transcript set from `dataset/ea_gt_v3` (rated, non-skipped), Spearman rho in titles. |
| `appendix/plot_multibranch_heldout.py` | `appendix/plots/multibranch_heldout/` | Title-less multibranch held-out realism win rate. All three bar figures + the histogram below share ONE seed set and ONE set of strategy definitions via `cv_heldout_stats` in `../../cost_winrate_reasoning/plot_multibranch_measure_optimization.py`: optimize RWR / first branch / baseline / cr2bo4 scored on the full 10/10; held-out RWR averaged over the full C(10,5)²=63,504 split grid. A seed is kept only where every compared method has a non-refusal value (identical seed-instruction distribution across bars; a seed whose realism win rate is all-refusals for any method is dropped for all). `heldout_only.png` = Baseline / cr2bo4 / held-out RWR; `heldout_family.png` = + optimize RWR + first branch (all seeds, no control); `heldout_family_controlled.png` = the family restricted to seeds where cr2bo4 seed adherence ≥ baseline. Ns in the `*.csv` sidecars. |
| `appendix/plot_multibranch_wholetranscript.py` | `appendix/plots/multibranch_heldout/` | Title-less whole-transcript measures (`wholetranscript_measures.png`): judges the ENTIRE compute-matched multibranch audit as one transcript (opus-5) vs single-branch Baseline / cr2bo4 / first branch (b00). No realism win rate; concerning is the big left panel, eval awareness + seed adherence stacked right. Whole-transcript multibranch scores via `load_per_seed_opus5(..., "multibranch")`; per-measure fairness intersection. Paper Fig. 24. |
| `appendix/plot_multibranch_endsignal.py` (+ `appendix/endsignal/analyze_endsignal_concerning.py`, `appendix/endsignal/STUDY.md`) | `appendix/plots/multibranch_heldout/` | Title-less figure (`endsignal_concerning.png`): how much of the compute-matched multibranch audit's whole-transcript `concerning` signal is elicited *after* the auditor first (verbally) feels the audit is done. **NOT in the current paper** (removed at authors' request); documented in `appendix/endsignal/STUDY.md`. The auditor has no `end_conversation` tool, so its first "wrap up"/"conclude the audit" phrase in its private reasoning marks the boundary (deterministic; median 8% of the audit). `analyze_endsignal_concerning.py` is a paid Opus 5 pass (~$1-2, resumable) that classifies each of the 50 transcripts before-only / both / after-only / no-concern, grounded on the whole-transcript judge's own `concerning` findings + turn-tagged highlights → `endsignal/endsignal_concerning.json`. Left panel = attribution counts; right = boundary-timing histogram. Result: of the 40 concerning audits, 80% involve post-done behavior, 33% exclusively. Run the analyzer first (`.venv/bin/python .../endsignal/analyze_endsignal_concerning.py`), then the plot. |
| `appendix/plot_multibranch_cv_sensitivity.py` | `appendix/plots/multibranch_heldout/` | Title-less sensitivity histogram — the same info as `heldout_family.png` plus how the choice of held-out comparisons affects the estimate: held-out RWR is a DISTRIBUTION over the full 63,504-split grid (purple, matching its bar), while optimize / first / baseline / cr2bo4 are single values on the full 10/10 (dashed vertical lines in matching colours). Same seed set, labels and colours as the bar family (via `cv_heldout_hist`). Its held-out mean == the held-out bar. Stats in `cv_sensitivity_full_grid.csv`. |
| `appendix/seed_adherence_target_swap/plot_seed_adherence_target_swap.py` | `appendix/seed_adherence_target_swap/` | Two single-panel seed-adherence bar charts extending the channel ablations (paper Figs. 18/19) with a TARGET-SWAP arm: the channel ablations hold the target fixed and move the critique-refine machinery, the swap does the reverse. `gpt55_vs_opus48.png` — target Opus 4.8 (0.74) vs target GPT-5.5 (0.49), then selector→Opus (0.52) and target→Opus (0.50) change nothing while feedback→Opus recovers most of the gap (0.70), so the collapse travels with the feedback model. `opus48_vs_sonnet46.png` — same treatment for Opus 4.8 vs Sonnet 4.6, which shows no gap (0.73 vs 0.74); not expected in the paper. All arms cr8, auditor Sonnet 4.6, 10-seed channel-ablation set (N=9 on the second figure, whose swap arm lost seed 52 to an auditor refusal). **Sized 7.0x5.4in for a half-width wrap-around slot** — at `0.5\linewidth` the house fonts land at the same on-page size as the full-width figures, so don't shrink them. Baselines (0.83 GPT-5.5 / 0.84 Opus 4.8) are deliberately left to the body text. Realism win rate is not plotted but is still written to each `_summary.csv`. `--judge {avg,opus5,gpt56,all}`, default `avg`. |
