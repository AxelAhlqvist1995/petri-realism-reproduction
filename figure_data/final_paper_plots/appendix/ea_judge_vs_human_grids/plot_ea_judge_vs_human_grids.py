#!/usr/bin/env python3
"""APPENDIX FIGURE — judge vs human eval-awareness annotation grids.

Two 10x10 count grids (human GT rating 1-10 on the y-axis, judge score 1-10
on the x-axis), one per singular judge: opus-5 and gpt-5.6(-sol). A cell
counts transcripts with that (judge score, human rating) pair; the diagonal
is exact agreement.

Data: the ea_gt_v3 human annotation set (dataset/ea_gt_v3) — every rating
file with a `gt_eval_awareness` and no skip_reason, resolved through
blind_map.json to its transcript under outputs/. Judge scores are each
transcript's CURRENT singular `s_eval_awareness` from the per-judge store
(`run_config.scores_by_judge`, read via experiments/util/singular_scores.py).
Both grids use the identical transcript set (n printed in the titles), so
the two judges are directly comparable. Spearman rho (tie-aware) per judge
is shown in the panel titles.

Output (overwritten): plots/{plot.png, pairs.csv,
grid_opus5.csv, grid_gpt56.csv}.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
GT_DIR = REPO_ROOT / "dataset" / "ea_gt_v3"
OUT_DIR = HERE / "plots"

sys.path.insert(0, str(REPO_ROOT))
from experiments.util import singular_scores as ss  # noqa: E402

JUDGES = [("opus5", "opus-5", "opus-5"), ("gpt56", "5.6", "gpt-5.6")]


def load_triples() -> list[tuple[str, int, int, int]]:
    """[(rel_path, human, opus5, gpt56)] for every usable rated transcript."""
    blind_map = json.loads((GT_DIR / "blind_map.json").read_text())
    triples = []
    for path in sorted((GT_DIR / "ratings").glob("t*_ea_gt_v3.json")):
        r = json.loads(path.read_text())
        entry = blind_map.get(r.get("source_path", "")) or blind_map.get(
            path.name.replace("_ea_gt_v3.json", ".json")
        )
        if entry is None or r.get("skip_reason") or r.get("gt_eval_awareness") is None:
            continue
        rel = entry["original"]["rel_path"]
        fp = REPO_ROOT / "outputs" / rel
        rc = json.loads(fp.read_text())["metadata"].get("run_config") or {}
        store = ss.scores_by_judge(rc)
        vals = {}
        for key, pat, _label in JUDGES:
            vals[key] = next(
                (
                    v.get("s_eval_awareness")
                    for j, v in store.items()
                    if pat in j and v.get("s_eval_awareness") is not None
                ),
                None,
            )
        if any(v is None for v in vals.values()):
            print(f"[skip] {rel}: missing judge score {vals}")
            continue
        triples.append(
            (rel, int(r["gt_eval_awareness"]), int(vals["opus5"]), int(vals["gpt56"]))
        )
    return triples


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        r = np.empty(len(x))
        r[order] = np.arange(len(x), dtype=float)
        out = r.copy()
        for v in np.unique(x):
            m = x == v
            out[m] = r[m].mean()
        return out

    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main() -> int:
    triples = load_triples()
    n = len(triples)
    human = np.array([t[1] for t in triples], dtype=float)
    judge_vals = {"opus5": np.array([t[2] for t in triples], dtype=float),
                  "gpt56": np.array([t[3] for t in triples], dtype=float)}

    grids = {}
    for key, _pat, _label in JUDGES:
        grid = np.zeros((10, 10), dtype=int)  # [human-1, judge-1]
        for h, j in zip(human.astype(int), judge_vals[key].astype(int)):
            grid[h - 1, j - 1] += 1
        grids[key] = grid

    vmax = max(g.max() for g in grids.values())
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4))
    fig.patch.set_facecolor("white")
    for ax, (key, _pat, label) in zip(axes, JUDGES):
        grid = grids[key]
        rho = spearman(human, judge_vals[key])
        im = ax.imshow(
            grid, origin="lower", cmap="Blues", vmin=0, vmax=vmax,
            extent=(0.5, 10.5, 0.5, 10.5),
        )
        # Diagonal = exact agreement.
        ax.plot([0.5, 10.5], [0.5, 10.5], color="tab:orange", linewidth=1.5,
                alpha=0.8, zorder=2)
        for h in range(10):
            for j in range(10):
                c = grid[h, j]
                if c:
                    ax.text(
                        j + 1, h + 1, str(c), ha="center", va="center",
                        fontsize=12,
                        color="white" if c > 0.6 * vmax else "black",
                        zorder=3,
                    )
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(1, 11))
        ax.tick_params(labelsize=12)
        ax.set_xlabel(f"{label} singular eval awareness", fontsize=15)
        ax.set_ylabel("Human eval awareness", fontsize=15)
        ax.set_title(f"{label} (singular) vs human  (ρ={rho:.2f}, n={n})",
                     fontsize=15)
    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label("Transcripts", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "plot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    with open(OUT_DIR / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "human", "opus5", "gpt56"])
        w.writerows(triples)
    for key, _pat, _label in JUDGES:
        with open(OUT_DIR / f"grid_{key}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["human\\judge"] + list(range(1, 11)))
            for h in range(10):
                w.writerow([h + 1] + list(grids[key][h]))
    print(f"Wrote {OUT_DIR}/ (pairs.csv, grid_*.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
