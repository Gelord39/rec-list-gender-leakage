"""Validation run: does exemplar separation cause a better many-shot LLM attacker?

Reads the manifest written by ``src.llm.exemplar_quality`` (low / mid / high
separation seeds, 16 shots, fixed 12M/4F mix) and the matching
``preds_many_shot_lightgcn_{model}_500_shots16_s{seed}.jsonl`` files.

Reports, per model: AUC / BAcc per draw, group means, a permutation test of
mean AUC (high - low; draws are the unit, 10k label shuffles), and the
Spearman correlation between separation and AUC over all 15 draws.

Usage:
    uv run python -m src.report.exemplar_quality_table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.report.shot_sweep_table import MODELS, _load, _metrics

GROUPS = ("low", "mid", "high")


def perm_test(a: list[float], b: list[float], n: int = 10_000, seed: int = 2024) -> float:
    """Two-sided permutation p-value for mean(b) - mean(a)."""
    rng = np.random.default_rng(seed)
    x = np.array(a + b)
    obs = np.mean(b) - np.mean(a)
    k = len(a)
    hits = 0
    for _ in range(n):
        rng.shuffle(x)
        hits += abs(x[k:].mean() - x[:k].mean()) >= abs(obs) - 1e-12
    return (hits + 1) / (n + 1)


def collect(manifest: dict[str, Any], llm_dir: Path, prefix: str = "lightgcn") -> dict[str, Any]:
    draws = [(g, d["seed"], d["separation"]) for g in GROUPS for d in manifest["groups"][g]]
    runs, summary = [], []
    for model in MODELS:
        per = []
        for g, seed, sep in draws:
            f = llm_dir / f"preds_many_shot_{prefix}_{model}_500_shots16_s{seed}.jsonl"
            if not f.exists():
                continue
            y, pred, score = _load(f)
            if len(y) < 500:
                print(f"skip incomplete {f.name}: {len(y)}/500")
                continue
            r = {"model": model, "group": g, "seed": seed, "separation": sep, **_metrics(y, pred, score)}
            runs.append(r)
            per.append(r)
        if not per:
            continue
        auc = {g: [r["auc"] for r in per if r["group"] == g] for g in GROUPS}
        bacc = {g: [r["bacc"] for r in per if r["group"] == g] for g in GROUPS}
        s = {"model": model, "n_draws": len(per)}
        for g in GROUPS:
            s[f"auc_{g}_mean"] = float(np.mean(auc[g])) if auc[g] else None
            s[f"auc_{g}_sd"] = float(np.std(auc[g], ddof=1)) if len(auc[g]) > 1 else None
            s[f"bacc_{g}_mean"] = float(np.mean(bacc[g])) if bacc[g] else None
        if auc["high"] and auc["low"]:
            s["auc_high_minus_low"] = float(np.mean(auc["high"]) - np.mean(auc["low"]))
            s["perm_p_high_vs_low"] = perm_test(auc["low"], auc["high"])
        rho, p = spearmanr([r["separation"] for r in per], [r["auc"] for r in per])
        s["spearman_sep_auc"], s["spearman_p"] = float(rho), float(p)
        summary.append(s)
    return {"manifest": manifest, "runs": runs, "summary": summary}


def to_markdown(res: dict[str, Any]) -> str:
    out = ["| Model | AUC low sep | AUC mid sep | AUC high sep | Δ high − low | perm. p | Spearman(sep, AUC), 15 draws | BAcc low / mid / high |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    f = lambda m, s: "—" if m is None else (f"{m:.3f}" if s is None else f"{m:.3f} ± {s:.3f}")
    g = lambda x: "—" if x is None else f"{x:.3f}"
    for s in res["summary"]:
        out.append(
            f"| {MODELS[s['model']]} | {f(s['auc_low_mean'], s['auc_low_sd'])} | {f(s['auc_mid_mean'], s['auc_mid_sd'])} | "
            f"{f(s['auc_high_mean'], s['auc_high_sd'])} | {s.get('auc_high_minus_low', float('nan')):+.3f} | "
            f"{s.get('perm_p_high_vs_low', float('nan')):.3f} | {s['spearman_sep_auc']:+.2f} (p={s['spearman_p']:.3f}) | "
            f"{g(s['bacc_low_mean'])} / {g(s['bacc_mid_mean'])} / {g(s['bacc_high_mean'])} |")
    out += ["", "| Model | Group | Seed | Separation | AUC | BAcc | Predicted male |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in res["runs"]:
        out.append(f"| {MODELS[r['model']]} | {r['group']} | {r['seed']} | {r['separation']:+.3f} | {r['auc']:.3f} | "
                   f"{r['bacc']:.3f} | {r['pred_male_share']:.0%} |")
    return "\n".join(out) + "\n"


def plot(res: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"glm4_9b": "#2a78d6", "llama31_8b": "#eb6834", "qwen25_14b": "#1baf7a"}
    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e6e5df"
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model, name in MODELS.items():
        rs = [r for r in res["runs"] if r["model"] == model]
        if not rs:
            continue
        x = np.array([r["separation"] for r in rs])
        yv = np.array([r["auc"] for r in rs])
        ax.scatter(x, yv, s=46, color=colors[model], edgecolor="white", linewidth=1.5, label=name, zorder=3)
        b = np.polyfit(x, yv, 1)
        xs = np.linspace(x.min(), x.max(), 10)
        ax.plot(xs, np.polyval(b, xs), color=colors[model], lw=2, alpha=0.8)
    ax.axhline(0.5, color=muted, lw=1, ls="--")
    ax.text(ax.get_xlim()[1], 0.5, "chance ", fontsize=8, color=muted, va="bottom", ha="right")
    ax.set_xlabel("exemplar separation (mean OOF P(male): male − female exemplars)", fontsize=9, color=muted)
    ax.set_ylabel("AUC, 500 test users", fontsize=9, color=muted)
    ax.set_title("16-shot LLM attack vs exemplar separation (12M/4F draws, plain template)",
                 fontsize=10, color=ink, loc="left")
    ax.grid(axis="y", color=grid, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path("runs/EXP-004_lightgcn_d32_311")
    p.add_argument("--manifest", type=Path, default=base / "exemplar_quality/manifest_16shot.json")
    p.add_argument("--llm-dir", type=Path, default=base / "llm_aia")
    p.add_argument("--out-dir", type=Path, default=base / "exemplar_quality")
    p.add_argument("--prefix", default="lightgcn", help="tag prefix of the recommender: lightgcn, bpr")
    args = p.parse_args()
    res = collect(json.loads(args.manifest.read_text()), args.llm_dir, args.prefix)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "exemplar_quality.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res)
    (args.out_dir / "exemplar_quality.md").write_text(md)
    if res["runs"]:
        plot(res, args.out_dir / "auc_vs_separation.png")
    print(md)


if __name__ == "__main__":
    main()
