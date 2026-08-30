#!/usr/bin/env python3
"""Shared cell paths + loaders for the logprob-appendix final paper plots.

The logprob-selector experiments (paper appendix "Critique Refinement using
Logprobs") live under `outputs/ablation_protocol_no_stopping/<cond>/
target_<T>/auditor_opus-4.7` — condition FIRST, unlike the reasoning tree the
final-paper machinery expects (`outputs/reasoning/{baseline,pairwise}/
target_<T>/auditor_<A>/<cond>`). `build_shim()` bridges the two layouts with a
symlink tree, so `plot_cost_grid_reasoning.py` / `plot_cost_grid_combined_
targets.py` (and their paper-styling wrappers) run on these cells unchanged.

Costs: the old runs have only sparse/partial `cost_logs`, so — exactly like
the original appendix figures — the x-axis uses the analytical cost model
(`experiments_logic.cost_model.predict_audit_cost`, the paper's app:cost-model)
via `analytical_cost_by_seed()`. Judge measures (concerning, eval_awareness,
s_eval_awareness, s_seed_adherence) come from each cell's `0_transcripts.csv`,
which since the 2026-08-27 rejudge carries claude-opus-5 as the primary judge
(gpt-5.4 preserved in `judge_output_history` / per-judge columns).
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _resolve_repo_root() -> Path:
    """The checkout holding `outputs/`. Defaults to this script's repo; a
    worktree without outputs falls back to $PETRI_REPO_ROOT or the main
    checkout the worktree belongs to."""
    if env := os.environ.get("PETRI_REPO_ROOT"):
        return Path(env).expanduser().resolve()
    own = HERE.parents[4]
    if (own / "outputs" / "ablation_protocol_no_stopping").is_dir():
        return own
    for anc in own.parents:  # …/.claude/worktrees/<name> -> main checkout
        if (anc / "outputs" / "ablation_protocol_no_stopping").is_dir():
            return anc
    return own


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_logic.cost_model import predict_audit_cost  # noqa: E402

LP_ROOT = REPO_ROOT / "outputs" / "ablation_protocol_no_stopping"
AUDITOR = "opus-4.7"

TARGETS = ["sonnet-4.6", "opus-4.7", "haiku-4.5"]
TARGET_DISPLAY = {
    "sonnet-4.6": "Sonnet 4.6",
    "opus-4.7": "Opus 4.7",
    "haiku-4.5": "Haiku 4.5",
}

# Depth / breadth / mix rungs available per target (all rejudged with opus-5).
TARGET_DEPTH = {
    "sonnet-4.6": ["cr1", "cr2", "cr4", "cr8", "cr11", "cr22"],
    "opus-4.7": ["cr1", "cr2", "cr4", "cr8", "cr11", "cr16"],
    "haiku-4.5": ["cr1", "cr2", "cr4", "cr8", "cr11"],
}
SONNET_BREADTH = ["bo2", "bo4", "bo8", "bo16"]
SONNET_MIX = ["cr2bo2", "cr2bo4", "cr2bo8"]


def cell(target: str, cond: str) -> Path:
    return LP_ROOT / cond / f"target_{target}" / f"auditor_{AUDITOR}"


# ── Analytical cost (same model as the original appendix figures) ─────────────

_RE_CRBO = re.compile(r"^cr(\d+)bo(\d+)$")
_RE_CR = re.compile(r"^cr(\d+)$")
_RE_BO = re.compile(r"^bo(\d+)$")


def protocol_params(method: str) -> tuple[int, int] | None:
    """(n_parallel, cr_rounds) from a condition folder name."""
    if method in ("baseline", "realism_filter"):
        return (1, 0)
    if m := _RE_CRBO.match(method):
        return (int(m.group(2)), int(m.group(1)))
    if m := _RE_CR.match(method):
        return (1, int(m.group(1)))
    if m := _RE_BO.match(method):
        return (int(m.group(1)), 0)
    return None


def seeds_in_cell(cell_dir: Path) -> set[int]:
    csv_path = cell_dir / "0_transcripts.csv"
    if not csv_path.is_file():
        return set()
    out: set[int] = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("Seed Index") or "").strip()
            if not raw:
                continue
            try:
                out.add(int(float(raw)))
            except ValueError:
                continue
    return out


def analytical_cost_by_seed(cell_dir: Path, needed_seeds=None) -> dict[int, float]:
    """{seed: predicted audit cost USD} — the same number for every seed of a
    cell, which is what makes the cost axis apples-to-apples. Accepts real
    cell paths or `build_shim()` symlinks (resolved before parsing)."""
    real = Path(os.path.realpath(cell_dir))
    # real: .../ablation_protocol_no_stopping/<method>/target_<T>/auditor_<A>
    method = real.parent.parent.name
    target = next(
        (p[len("target_"):] for p in real.parts if p.startswith("target_")),
        "unknown",
    )
    nr = protocol_params(method)
    if nr is None:
        print(f"[warn] {real}: method {method!r} doesn't parse — no cost",
              file=sys.stderr)
        return {}
    cost = predict_audit_cost(
        auditor=AUDITOR, target=target,
        n_parallel=nr[0], cr_rounds=nr[1], include_feedback=True,
    )
    return {s: cost for s in seeds_in_cell(real)}


# ── Reasoning-layout shim ─────────────────────────────────────────────────────


def build_shim(targets_conds: dict[str, list[str]], shim_root: Path | None = None) -> Path:
    """Create a `<shim>/outputs/reasoning/...` symlink tree over the logprob
    cells so the reasoning-tree plot machinery can read them:

        outputs/reasoning/baseline/target_<T>/auditor_opus-4.7  -> baseline cell
        outputs/reasoning/pairwise/target_<T>/auditor_opus-4.7/<cond> -> cond cell

    Rebuilt from scratch on every call (symlinks only, nothing copied).
    Returns the shim root, to be assigned to the base module's REPO_ROOT.
    """
    shim_root = shim_root or (HERE / ".shim")
    if shim_root.exists():
        shutil.rmtree(shim_root)
    for target, conds in targets_conds.items():
        base_link = (
            shim_root / "outputs" / "reasoning" / "baseline"
            / f"target_{target}" / f"auditor_{AUDITOR}"
        )
        base_real = cell(target, "baseline")
        if base_real.is_dir():
            base_link.parent.mkdir(parents=True, exist_ok=True)
            base_link.symlink_to(base_real)
        else:
            print(f"[warn] missing baseline cell {base_real}", file=sys.stderr)
        pair_dir = (
            shim_root / "outputs" / "reasoning" / "pairwise"
            / f"target_{target}" / f"auditor_{AUDITOR}"
        )
        pair_dir.mkdir(parents=True, exist_ok=True)
        for cond in conds:
            real = cell(target, cond)
            if not real.is_dir():
                print(f"[warn] missing cell {real}", file=sys.stderr)
                continue
            (pair_dir / cond).symlink_to(real)
    return shim_root
