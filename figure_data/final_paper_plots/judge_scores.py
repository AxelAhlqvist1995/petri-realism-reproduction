#!/usr/bin/env python3
"""Per-judge score extraction + fair multi-judge views for the final paper plots.

The cell CSVs (`0_transcripts.csv`) only carry the *current* judge's scores
(claude-opus-5). This module reads the transcripts directly to recover every
judge's scores:

  - FULL judge measures (`concerning`, `eval_awareness`): the current
    `metadata.judge_output.scores` plus every entry in
    `metadata.judge_output_history` (keyed by judge model).
  - SINGULAR measures (`s_eval_awareness`, `s_seed_adherence`):
    `metadata.run_config.singular_scores_by_judge` (per-judge dict).

Judges are canonicalised by substring: "opus-5" -> `opus5`,
"5.6" (gpt-5.6-sol) -> `gpt56`. Other judges (gpt-5.4/5.5) are ignored.

Run as a script to (re)build `judge_scores.json` next to this file:

    python judge_scores.py [--refresh]

The JSON layout is {target: {folder: {measure: {judge: {seed: raw_1_10}}}}}.

`version_view()` then builds fair per-method score maps for a plot version
(`opus5` / `gpt56` / `avg`): for a given measure, a seed is only kept if a
judge scored it for ALL compared methods; the `avg` version averages the two
judges where both are complete on a seed, uses the single complete judge where
only one is, and drops the seed where neither is. If the requested judge has
no scores at all for a measure (today: gpt-5.6 has no full-judge pass), the
view falls back to opus-5 and says so via the returned source tag, so the
panel can be labelled honestly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CACHE_PATH = HERE / "judge_scores.json"

sys.path.insert(0, str(REPO_ROOT))
from experiments.util import singular_scores as ss  # noqa: E402

AUDITOR = "sonnet-4.6"
TARGETS = ["sonnet-4.6", "opus-4.8", "gpt-5.5", "gemini-3.5-flash"]
# Folders per target that any final paper plot uses. Sonnet additionally has
# the full depth/breadth ladders for the scaling-types figure.
BASE_FOLDERS = ["baseline", "realism_filter", "cr1", "cr2", "cr2bo2", "cr2bo4"]
SONNET_EXTRA = ["cr4", "cr8", "cr16", "bo2", "bo4", "bo8"]

MEASURES_FULL = ["concerning", "eval_awareness"]
MEASURES_SINGULAR = ["s_eval_awareness", "s_seed_adherence"]
ALL_MEASURES = MEASURES_FULL + MEASURES_SINGULAR

JUDGES = ["opus5", "gpt56"]

_SEED_RE = re.compile(r"transcript_(\d+)_")


def canonical_judge(name: str) -> str | None:
    if "opus-5" in name:
        return "opus5"
    if "5.6" in name:
        return "gpt56"
    return None


def cell_dir(target: str, folder: str) -> Path:
    if folder == "baseline":
        return (
            REPO_ROOT / "outputs" / "reasoning" / "baseline"
            / f"target_{target}" / f"auditor_{AUDITOR}"
        )
    if folder == "realism_filter":
        return (
            REPO_ROOT / "outputs" / "reasoning" / "realism_filter"
            / f"target_{target}" / f"auditor_{AUDITOR}"
        )
    return (
        REPO_ROOT / "outputs" / "reasoning" / "pairwise"
        / f"target_{target}" / f"auditor_{AUDITOR}" / folder
    )


def _num(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_cell(cell: Path) -> dict[str, dict[str, dict[int, float]]]:
    """{measure: {judge: {seed: raw 1-10 value}}} for one cell."""
    out: dict[str, dict[str, dict[int, float]]] = {
        m: {j: {} for j in JUDGES} for m in ALL_MEASURES
    }
    for fp in sorted(cell.glob("transcript_*.json")):
        m_seed = _SEED_RE.match(fp.name)
        if not m_seed:
            continue
        seed = int(m_seed.group(1))
        try:
            md = json.loads(fp.read_text())["metadata"]
        except (OSError, json.JSONDecodeError, KeyError):
            print(f"[warn] unreadable transcript {fp}", file=sys.stderr)
            continue
        rc = md.get("run_config") or {}

        # Full judge: current output wins for its judge, history fills others.
        entries: dict[str, dict] = {}
        jo = md.get("judge_output") or {}
        sc = jo.get("scores") or {}
        eff = str(sc.get("judge_model_effective") or rc.get("judge_model") or "")
        cj = canonical_judge(eff)
        if cj:
            entries[cj] = sc
        for jname, payload in (md.get("judge_output_history") or {}).items():
            c = canonical_judge(str(jname))
            if c and c not in entries:
                entries[c] = (payload or {}).get("scores") or {}
        for measure in MEASURES_FULL:
            for c, scores in entries.items():
                val = _num(scores.get(measure))
                if val is not None:
                    out[measure][c][seed] = val

        # Per-judge store. Read through the shared accessor rather than a literal
        # key: the store moved from run_config.singular_scores_by_judge (v1, which
        # held singular measures only) to run_config.scores_by_judge (v2, which
        # holds the rubric measures too), and the accessor resolves either. A
        # literal v1 key silently yields NOTHING once the migration has run.
        store = ss.scores_by_judge(rc)
        for jname, scores in store.items():
            c = canonical_judge(str(jname))
            if not c:
                continue
            for measure in MEASURES_SINGULAR:
                val = _num((scores or {}).get(measure))
                if val is not None:
                    out[measure][c][seed] = val
        # Rubric measures also resolve per judge once a secondary rubric pass has
        # run (e.g. gpt-5.6 via `rejudge_all_singular.py --rubric-only`). The
        # store is authoritative where it has a value: a secondary judge never
        # writes judge_output.scores, so the block above cannot see it.
        for jname, scores in store.items():
            c = canonical_judge(str(jname))
            if not c:
                continue
            for measure in MEASURES_FULL:
                val = _num((scores or {}).get(measure))
                if val is not None:
                    out[measure][c][seed] = val
    return out


def build_cache() -> dict:
    data: dict = {}
    for target in TARGETS:
        folders = BASE_FOLDERS + (SONNET_EXTRA if target == "sonnet-4.6" else [])
        data[target] = {}
        for folder in folders:
            cell = cell_dir(target, folder)
            if not cell.is_dir():
                print(f"[skip] {target}/{folder} missing", file=sys.stderr)
                continue
            scores = extract_cell(cell)
            data[target][folder] = {
                m: {j: {str(s): v for s, v in d.items()} for j, d in by_j.items()}
                for m, by_j in scores.items()
            }
            counts = "  ".join(
                f"{m}=o5:{len(scores[m]['opus5'])}/g56:{len(scores[m]['gpt56'])}"
                for m in ALL_MEASURES
            )
            print(f"[cell] {target:<17}{folder:<9} {counts}")
    return data


def load(refresh: bool = False) -> dict:
    """{target: {folder: {measure: {judge: {int seed: raw value}}}}}"""
    if refresh or not CACHE_PATH.is_file():
        data = build_cache()
        CACHE_PATH.write_text(json.dumps(data))
        print(f"Wrote {CACHE_PATH}")
    else:
        data = json.loads(CACHE_PATH.read_text())
    return {
        t: {
            f: {
                m: {j: {int(s): v for s, v in d.items()} for j, d in by_j.items()}
                for m, by_j in by_m.items()
            }
            for f, by_m in by_f.items()
        }
        for t, by_f in data.items()
    }


# ── Fair multi-judge views ────────────────────────────────────────────────────


def _complete_seeds(per_method: dict[str, dict[int, float]]) -> set[int]:
    """Seeds scored for every method (any empty method makes the set empty)."""
    sets = [set(d) for d in per_method.values()]
    return set.intersection(*sets) if sets else set()


def version_view(
    target_data: dict,
    methods: list[str],
    measure: str,
    version: str,
    normalize: bool = True,
) -> tuple[dict[str, dict[int, float]], str]:
    """({method: {seed: value}}, source_tag) for one (measure, version).

    A seed is kept only when the serving judge scored the measure for ALL
    `methods` (so every method pools over the same seed distribution). For
    `avg`, both-complete seeds get the two-judge mean, single-judge-complete
    seeds fall back to that judge alone, and seeds complete for neither are
    dropped. `source_tag` is the judge actually used ("opus5", "gpt56",
    "avg", or "opus5-fallback" when the requested judge has no scores at all
    for this measure). With ``normalize`` the raw 1-10 values are rescaled to
    0-1 via (raw-1)/9.
    """
    methods = [m for m in methods if m in target_data]
    o5 = {m: target_data[m].get(measure, {}).get("opus5", {}) for m in methods}
    g56 = {m: target_data[m].get(measure, {}).get("gpt56", {}) for m in methods}
    g56_empty = all(not d for d in g56.values())

    if version == "opus5" or (version in ("gpt56", "avg") and g56_empty):
        common = _complete_seeds(o5)
        out = {m: {s: o5[m][s] for s in common} for m in methods}
        tag = "opus5" if version == "opus5" else "opus5-fallback"
    elif version == "gpt56":
        common = _complete_seeds(g56)
        out = {m: {s: g56[m][s] for s in common} for m in methods}
        tag = "gpt56"
    elif version == "avg":
        o5_ok = _complete_seeds(o5)
        g56_ok = _complete_seeds(g56)
        out = {m: {} for m in methods}
        for s in o5_ok | g56_ok:
            for m in methods:
                if s in o5_ok and s in g56_ok:
                    out[m][s] = (o5[m][s] + g56[m][s]) / 2.0
                elif s in o5_ok:
                    out[m][s] = o5[m][s]
                else:
                    out[m][s] = g56[m][s]
        tag = "avg"
    else:
        raise ValueError(f"unknown version {version!r}")

    if normalize:
        out = {m: {s: (v - 1.0) / 9.0 for s, v in d.items()} for m, d in out.items()}
    return out, tag


VERSION_LABEL = {
    "opus5": "opus-5",
    "gpt56": "gpt-5.6",
    "avg": "judge avg",  # avg(opus-5, gpt-5.6); kept short for axis labels
    "opus5-fallback": "opus-5",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="rebuild the cache")
    args = ap.parse_args()
    load(refresh=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
