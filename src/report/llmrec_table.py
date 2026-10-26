"""Markdown tables for EXP-006 (LLM as the recommender, Stairway setup).

Reads ``llmrec_report.json`` (list diagnostics + utility), ``attack_grid_500/summary.json``
(classifier attackers), ``compare_*.json`` (neutral vs the second variant) and, if present,
``llm_aia/metrics_*.json`` (T4, one row per LLM attacker) under ``runs/EXP-006_llmrec_311``.

  uv run python -m src.report.llmrec_table [--run-root runs/EXP-006_llmrec_311] [--setting ""|_hist50]
  # 2026-09-28 extension: neutral vs gender-only prompt, 20 requested -> 10 kept
  uv run python -m src.report.llmrec_table --variant-b gender --setting _req20 \
      --attackers glm4_9b,llama31_8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = {"glm4_9b": "GLM-4-9B", "llama31_8b": "Llama-3.1-8B"}
INPUTS = [
    ("history_multi_hot", "TH", "Full train history (reference)"),
    ("prompt_history_multi_hot", "TH-prompt", "Items shown in the LLMRec prompt (reference)"),
    ("multi_hot", "T2", "Multi-hot over list items"),
    ("tfidf_multi_hot", "T2", "TF-IDF multi-hot"),
    ("genre_histogram", "T3", "Genre histogram (18-d)"),
    ("title_text_embedding", "T3", "Title text embedding (MiniLM)"),
]


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _auc(summary: dict, inp: str, variant: str = "lr_balanced") -> float | None:
    for r in summary["table"]:
        if r["input"] == inp and r["variant"] == variant:
            return r["auc"]
    return None


def _best(summary: dict, inp: str) -> float | None:
    v = [r["auc"] for r in summary["table"] if r["input"] == inp]
    return max(v) if v else None


def _f(x: float | None, fmt: str = "{:.3f}") -> str:
    return "—" if x is None else fmt.format(x)


def tables(root: Path, setting: str, vb: str = "sensitive", attackers: tuple[str, ...] = ("glm4_9b",)) -> str:
    out: list[str] = []
    variants = ("neutral", vb)
    runs = [(v, m, ver) for m in MODELS for v in variants for ver in ("", "_unseen")]

    out += ["**List diagnostics and utility** (3,867 users; HR/P/NDCG@10 on the 695 users with test items)", "",
            "| Model | Prompt | List version | Match rate | Mean length | <5 items | Already watched (raw) | Distinct items | Top-50 share | HR@10 | NDCG@10 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for v, m, ver in runs:
        rep = _load(root / f"{v}{setting}_{m}{ver}" / "llmrec_report.json")
        if rep is None:
            continue
        o, c, u = rep["diagnostics"]["overall"], rep["diagnostics"]["concentration"], rep["rec_metrics"]
        out.append(f"| {MODELS[m]} | {v} | {'unseen' if ver else 'raw'} | {o['match_rate']:.2f} | "
                   f"{o['mean_rec_len']:.1f} | {o['share_under_5']:.1%} | {o['raw_seen_rate']:.1%} | "
                   f"{c['distinct_items']} | {c['top50_share']:.1%} | {u['hit@10']:.3f} | {u['ndcg@10']:.3f} |")

    for ver, label in (("_unseen", "unseen lists (train items removed, comparable with EXP-003/004)"),
                       ("", "raw lists (Stairway's output as is)")):
        out += ["", f"**Gender AUC, {label}** — LR balanced (best of the 9 classifier variants in brackets); "
                    "500 fixed test users, 3,367-user train pool", "",
                "| Tier | Input | " + " | ".join(f"{MODELS[m]} {v}" for m in MODELS for v in variants) + " |",
                "| --- | --- | " + " | ".join("---:" for _ in range(4)) + " |"]
        sums = {(m, v): _load(root / f"{v}{setting}_{m}{ver}" / "attack_grid_500" / "summary.json")
                for m in MODELS for v in variants}
        for inp, tier, desc in INPUTS:
            cells = []
            for m in MODELS:
                for v in variants:
                    s = sums[(m, v)]
                    cells.append("—" if s is None else f"{_f(_auc(s, inp))} ({_f(_best(s, inp))})")
            out.append(f"| {tier} | {desc} | " + " | ".join(cells) + " |")
        # T4, one row per LLM attacker
        for att in attackers:
            cells = []
            for m in MODELS:
                for v in variants:
                    parts = []
                    for style, tag in (("zero_shot", f"{att}_500"), ("staab_many_shot", f"{att}_500_shots8")):
                        met = _load(root / f"{v}{setting}_{m}{ver}" / "llm_aia" / f"metrics_{style}_{tag}.json")
                        parts.append(_f(met["overall"]["auc"]) if met else "—")
                    cells.append(" / ".join(parts))
            if any(c != "— / —" for c in cells):
                out.append(f"| T4 | {MODELS[att]} attacker on the titles (zero-shot / 8-shot staab) | "
                           + " | ".join(cells) + " |")

    out += ["", f"**Neutral → {vb}, paired on the same 500 users** (LR balanced; ΔAUC with 95% paired-bootstrap CI) "
                "and how much the list content moves (Jaccard between a user's two lists, all 3,867 users)", "",
            "| Model | List version | Multi-hot ΔAUC | TF-IDF ΔAUC | Genre ΔAUC | MiniLM ΔAUC | Jaccard male | Jaccard female | Romance share Δ (F / M) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in MODELS:
        for ver in ("_unseen", ""):
            cmp_ = _load(root / f"compare_neutral{setting}_{m}{ver}__vs__{vb}{setting}_{m}{ver}.json")
            if cmp_ is None:
                continue
            d = {(r["input"], r["variant"]): r for r in cmp_["auc_deltas"]}

            def cell(inp: str) -> str:
                r = d.get((inp, "lr_balanced"))
                return "—" if r is None else (f"{r['auc_a']:.3f}→{r['auc_b']:.3f} ({r['delta_auc_b_minus_a']:+.3f} "
                                              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}])")

            sh = cmp_["content_shift"]["by_gender"]
            out.append(f"| {MODELS[m]} | {'unseen' if ver else 'raw'} | {cell('multi_hot')} | {cell('tfidf_multi_hot')} | "
                       f"{cell('genre_histogram')} | {cell('title_text_embedding')} | {sh['male']['mean_jaccard']:.2f} | "
                       f"{sh['female']['mean_jaccard']:.2f} | {sh['female']['genre_share_b_minus_a']['Romance']:+.3f} / "
                       f"{sh['male']['genre_share_b_minus_a']['Romance']:+.3f} |")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, default=Path("runs/EXP-006_llmrec_311"))
    p.add_argument("--setting", default="", help="'' for Stairway's 10-item history, '_hist50' for the extension")
    p.add_argument("--variant-b", default="sensitive", help="compared with neutral: sensitive (Stairway) or gender")
    p.add_argument("--attackers", default="glm4_9b", help="T4 attacker slugs, comma-separated")
    args = p.parse_args()
    md = tables(args.run_root, args.setting, args.variant_b, tuple(args.attackers.split(",")))
    name = f"llmrec_tables{args.setting}" + ("" if args.variant_b == "sensitive" else f"_{args.variant_b}")
    (args.run_root / f"{name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
