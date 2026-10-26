"""Aggregate the shot-count x exemplar-draw x model sweep (scripts/hendrix_shot_sweep.sh).

Recomputes every metric from the stored ``preds_*.jsonl`` (not the metrics
JSON), so all rows share one definition:

- hard labels: ``pred_gender`` in {male, female}; anything else is an abstention
  and is excluded from Acc / BAcc / F1-macro;
- AUC: every user with a logprob ``score`` (abstentions included), with a 95%
  bootstrap CI;
- across exemplar draws (seeds): mean +- sample std of each metric;
- 8 -> 16 shots within a seed is a paired comparison (the 16 exemplars start
  with the same 8), reported as a paired-bootstrap CI of the AUC difference.

The existing GLM-4 runs of 2026-09-21 are picked up as well: ``..._shots8``
= seed 2024, 32 / 128 shots without a seed suffix = seed 2024.
2 shots: random draws ``..._shots2_s{seed}`` (scripts/hendrix_shot_fill.sh)
when any exist for that model x template, else the fixed ``few_shot`` /
``staab_few_shot`` fixture (marked ``fixed`` in the table).

Usage:
    uv run python -m src.report.shot_sweep_table \
        --llm-dir runs/EXP-004_lightgcn_d32_311/llm_aia \
        --out-dir runs/EXP-004_lightgcn_d32_311/shot_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

MODELS = {"glm4_9b": "GLM-4-9B", "llama31_8b": "Llama-3.1-8B", "qwen25_14b": "Qwen2.5-14B"}
TEMPLATES = {"plain": ("zero_shot", "few_shot", "many_shot"), "staab": ("staab", "staab_few_shot", "staab_many_shot")}
SEEDS = (2024, 2025, 2026)
SHOTS = (0, 2, 8, 16, 32, 64, 128)
N_BOOT = 2000
N_USERS = 500


def _preds_path(llm_dir: Path, model: str, template: str, shots: int, seed: int,
                prefix: str = "lightgcn") -> Path | None:
    zero, two, many = TEMPLATES[template]
    base = f"{prefix}_{model}_500"
    if shots == 0:
        cand = [f"preds_{zero}_{base}.jsonl"]
    elif shots == 2 and not any(llm_dir.glob(f"preds_{many}_{base}_shots2_s*.jsonl")):
        cand = [f"preds_{two}_{base}.jsonl"] if seed == 2024 else []
    else:
        cand = [f"preds_{many}_{base}_shots{shots}_s{seed}.jsonl"]
        if seed == 2024:  # 2026-09-21 runs predate the seed suffix
            cand.append(f"preds_{many}_{base}_shots{shots}.jsonl")
    for c in cand:
        if (llm_dir / c).exists():
            return llm_dir / c
    return None


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    y = np.array([r["gender"] == "male" for r in rows], dtype=int)
    pred = np.array([r.get("pred_gender") or "" for r in rows], dtype=object)
    score = np.array([np.nan if r.get("score") is None else float(r["score"]) for r in rows])
    return y, pred, score


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC (ties = 1/2), identical to roc_auc_score but ~50x faster in a bootstrap loop."""
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _boot_auc(y: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> list[float]:
    aucs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(_auc(y[idx], s[idx]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def _metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray, seed: int = 2024) -> dict[str, Any]:
    lab = np.isin(pred, ["male", "female"])
    yp = (pred[lab] == "male").astype(int)
    ok = ~np.isnan(score)
    out: dict[str, Any] = {
        "n_users": int(len(y)),
        "n_labels": int(lab.sum()),
        "n_abstain": int((~lab).sum()),
        "n_scored": int(ok.sum()),
        "acc": float((yp == y[lab]).mean()) if lab.any() else None,
        "bacc": float(balanced_accuracy_score(y[lab], yp)) if lab.any() else None,
        "f1_macro": float(f1_score(y[lab], yp, average="macro")) if lab.any() else None,
        "pred_male_share": float(yp.mean()) if lab.any() else None,
        "auc": float(roc_auc_score(y[ok], score[ok])),
    }
    out["auc_ci95"] = _boot_auc(y[ok], score[ok], np.random.default_rng(seed))
    return out


def _paired_delta(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, seed: int) -> dict[str, Any]:
    """AUC(b) - AUC(a) on users scored in both runs, paired bootstrap."""
    ok = ~np.isnan(s_a) & ~np.isnan(s_b)
    y, s_a, s_b = y[ok], s_a[ok], s_b[ok]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            d.append(_auc(y[idx], s_b[idx]) - _auc(y[idx], s_a[idx]))
    return {
        "n": int(len(y)),
        "delta_auc": float(roc_auc_score(y, s_b) - roc_auc_score(y, s_a)),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
    }


def _mean_sd(vals: list[float | None]) -> tuple[float | None, float | None]:
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None)


def collect(llm_dir: Path, prefix: str = "lightgcn") -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    loaded: dict[tuple, tuple] = {}
    for model in MODELS:
        for template in TEMPLATES:
            for shots in SHOTS:
                for seed in SEEDS if shots else (2024,):
                    p = _preds_path(llm_dir, model, template, shots, seed, prefix)
                    if p is None:
                        continue
                    y, pred, score = _load(p)
                    if len(y) < N_USERS:  # still being written by a running job
                        print(f"skip incomplete {p.name}: {len(y)}/{N_USERS} users")
                        continue
                    loaded[(model, template, shots, seed)] = (y, score)
                    runs.append({
                        "model": model, "template": template, "shots": shots,
                        "seed": seed if shots else None, "file": p.name,
                        "exemplars": "fixed" if p.name.startswith(f"preds_{TEMPLATES[template][1]}_") else "drawn",
                        **_metrics(y, pred, score),
                    })

    cells: list[dict[str, Any]] = []
    keys = sorted({(r["model"], r["template"], r["shots"]) for r in runs},
                  key=lambda k: (list(MODELS).index(k[0]), k[1], k[2]))
    for model, template, shots in keys:
        rs = [r for r in runs if (r["model"], r["template"], r["shots"]) == (model, template, shots)]
        cell: dict[str, Any] = {"model": model, "template": template, "shots": shots, "n_draws": len(rs),
                                "seeds": [r["seed"] for r in rs],
                                "fixed_exemplars": any(r["exemplars"] == "fixed" for r in rs)}
        for m in ("auc", "bacc", "f1_macro", "n_abstain", "pred_male_share"):
            cell[f"{m}_mean"], cell[f"{m}_sd"] = _mean_sd([r[m] for r in rs])
        cell["auc_per_draw"] = [r["auc"] for r in rs]
        cells.append(cell)

    paired: list[dict[str, Any]] = []
    for model in MODELS:
        for template in TEMPLATES:
            for seed in SEEDS:
                a, b = loaded.get((model, template, 8, seed)), loaded.get((model, template, 16, seed))
                if a is None or b is None:
                    continue
                paired.append({"model": model, "template": template, "seed": seed,
                               "comparison": "16 - 8 shots", **_paired_delta(a[0], a[1], b[1], seed)})
    return {"runs": runs, "cells": cells, "paired_8_vs_16": paired}


def _f(x: float | None, sd: float | None = None) -> str:
    if x is None:
        return "—"
    return f"{x:.3f}" if sd is None else f"{x:.3f} ± {sd:.3f}"


def to_markdown(res: dict[str, Any]) -> str:
    out = ["| Model | Template | Shots | Draws | AUC (mean ± sd) | AUC per draw | BAcc (mean ± sd) | F1-macro | Abstentions (mean) |",
           "| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |"]
    for c in res["cells"]:
        per = " / ".join(f"{a:.3f}" for a in c["auc_per_draw"])
        out.append(
            f"| {MODELS[c['model']]} | {c['template']} | {c['shots']}{' (fixed)' if c['fixed_exemplars'] else ''} | {c['n_draws']} | "
            f"{_f(c['auc_mean'], c['auc_sd'])} | {per} | {_f(c['bacc_mean'], c['bacc_sd'])} | "
            f"{_f(c['f1_macro_mean'])} | {c['n_abstain_mean']:.0f} |"
        )
    out += ["", "Paired 8 → 16 shots, same exemplar draw (the 16 start with the same 8):", "",
            "| Model | Template | Seed | ΔAUC (16 − 8) | 95% CI |", "| --- | --- | ---: | ---: | --- |"]
    for p in res["paired_8_vs_16"]:
        out.append(f"| {MODELS[p['model']]} | {p['template']} | {p['seed']} | {p['delta_auc']:+.3f} | "
                   f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] |")
    return "\n".join(out) + "\n"


def plot(res: dict[str, Any], path: Path, classifier_auc: float = 0.700) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"glm4_9b": "#2a78d6", "llama31_8b": "#eb6834", "qwen25_14b": "#1baf7a"}
    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e6e5df"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, template in zip(axes, TEMPLATES):
        for model, name in MODELS.items():
            cs = sorted((c for c in res["cells"] if c["model"] == model and c["template"] == template),
                        key=lambda c: c["shots"])
            if not cs:
                continue
            x = [c["shots"] for c in cs]
            dx = {"glm4_9b": -0.12, "llama31_8b": 0.0, "qwen25_14b": 0.12}[model]  # de-overlap error bars
            xpos = [(np.log2(s) + 1 if s else 0) + dx for s in x]  # 0,2,8,16,32,64,128 -> 0,2,4,5,6,7,8
            m = [c["auc_mean"] for c in cs]
            sd = [c["auc_sd"] or 0 for c in cs]
            ax.errorbar(xpos, m, yerr=sd, color=colors[model], lw=2, marker="o", ms=7,
                        mec="white", mew=1.5, capsize=3, label=name)
            ax.annotate(name, (xpos[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=ink)
        ax.axhline(0.5, color=muted, lw=1, ls="--")
        ax.axhline(classifier_auc, color=muted, lw=1, ls=":")
        ax.text(9.7, 0.5, "chance ", fontsize=8, color=muted, va="bottom", ha="right")
        ax.text(9.7, classifier_auc, f"LR on same lists ({classifier_auc:.3f}) ", fontsize=8,
                color=muted, va="bottom", ha="right")
        ticks = list(SHOTS)
        ax.set_xticks([np.log2(s) + 1 if s else 0 for s in ticks], [str(t) for t in ticks])
        ax.set_xlim(-0.4, 9.8)
        ax.set_title(f"{template} template", fontsize=10, color=ink, loc="left")
        ax.set_xlabel("in-context exemplars", fontsize=9, color=muted)
        ax.grid(axis="y", color=grid, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    axes[0].set_ylabel("AUC (mean ± sd over 3 exemplar draws; 0-shot: 1 run)", fontsize=9, color=muted)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.suptitle("LLM list-only gender attack vs number of exemplars (LightGCN, 500 test users)",
                 fontsize=11, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm-dir", type=Path, default=Path("runs/EXP-004_lightgcn_d32_311/llm_aia"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/EXP-004_lightgcn_d32_311/shot_sweep"))
    p.add_argument("--prefix", default="lightgcn", help="tag prefix of the recommender: lightgcn, bpr")
    p.add_argument("--classifier-auc", type=float, default=0.700,
                   help="reference line: LR multi-hot AUC on the same lists (LightGCN 0.700, BPR 0.718)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res = collect(args.llm_dir, args.prefix)
    (args.out_dir / "shot_sweep.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res)
    (args.out_dir / "shot_sweep.md").write_text(md)
    plot(res, args.out_dir / "auc_vs_shots.png", args.classifier_auc)
    print(md)


if __name__ == "__main__":
    main()
