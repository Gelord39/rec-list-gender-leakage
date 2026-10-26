"""Build one shared results table: utility + all attribute-inference attacks.

Meeting 2026-08-17 action item: "include result in same table" — pulls
together the embedding attacks (MLP / logistic regression / SVM, from
``src/attack/embedding_aia.py``) and the LLM list attacks (zero_shot /
few_shot / staab / staab_few_shot, from ``src/llm/list_aia.py``) for one run
directory, alongside the recommender's utility numbers.

LLM list-attack files are tagged (``preds_{style}_{tag}.jsonl`` /
``metrics_{style}_{tag}.json``, see ``src/llm/list_aia.py``); pass
``--llm-tag`` to pick which batch to include. A handful of pre-tagging files
(``metrics_{style}.json``, no tag suffix) may still exist from before
2026-08-23 — those used the original synthetic few-shot examples; pass
``--legacy-tag none`` to skip pulling them in as a second comparison block,
or leave the default to include them.

Example:
  uv run python -m src.report.build_table --run-dir runs/EXP-003_bpr_d32_311 --llm-tag real_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_EMBEDDING_ATTACKS = {
    "mlp": "attack_metrics.json",
    "logistic_regression": "attack_metrics_logistic_regression.json",
    "svm": "attack_metrics_svm.json",
}

_LLM_STYLES = ["zero_shot", "few_shot", "staab", "staab_few_shot"]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _llm_rows(run_dir: Path, tag: str | None, label_suffix: str) -> list[tuple]:
    rows = []
    for style in _LLM_STYLES:
        fname = f"metrics_{style}.json" if tag is None else f"metrics_{style}_{tag}.json"
        data = _load_json(run_dir / "llm_aia" / fname)
        if not data:
            continue
        overall = data["overall"]
        rows.append(
            (
                f"LLM list — {style}{label_suffix}",
                "rec list only",
                overall.get("bacc"),
                overall.get("accuracy"),
                overall.get("n"),
                data["baselines"]["chance_bacc"],
            )
        )
    return rows


def build_table(run_dir: Path, llm_tag: str | None, legacy_tag: str | None) -> str:
    run_dir = Path(run_dir)
    exp_id = run_dir.name

    rec = _load_json(run_dir / "rec_metrics.json")
    lines: list[str] = [f"# Results — {exp_id}", ""]

    lines.append("## Recommendation utility (test set)")
    lines.append("")
    if rec and rec.get("test_result"):
        tr = rec["test_result"]
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        for k in ["hit@5", "hit@10", "ndcg@5", "ndcg@10", "recall@5", "recall@10"]:
            if k in tr:
                lines.append(f"| {k} | {tr[k]:.4f} |")
    else:
        lines.append("_rec_metrics.json not found or has no test_result._")
    lines.append("")

    lines.append("## Sensitive-attribute leakage — all attackers")
    lines.append("")
    lines.append(
        "| Attacker | Input | BAcc | AUC / Accuracy | n | Majority/chance BAcc |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")

    rows: list[tuple] = []

    for name, fname in _EMBEDDING_ATTACKS.items():
        data = _load_json(run_dir / "embedding_aia" / fname)
        if not data:
            continue
        agg = data["aggregate"]
        auc_mean = agg.get("auc", {}).get("mean") if "auc" in agg else None
        rows.append(
            (
                f"Embedding — {name}",
                "user embedding",
                agg["bacc"]["mean"],
                auc_mean,
                data["n_users"],
                data["baselines"]["majority_bacc"],
            )
        )

    rows.extend(_llm_rows(run_dir, llm_tag, ""))
    if legacy_tag != "none":
        legacy_rows = _llm_rows(run_dir, None, " [legacy: synthetic few-shot, pre-2026-08-23]")
        rows.extend(legacy_rows)

    ctf = _load_json(run_dir / "liu_ctf.summary.json")
    if ctf:
        rows.append(
            (
                "LLM CTF (Liu et al.) — unfair rate",
                "history + rec list",
                ctf.get("unfair_rate"),
                None,
                ctf.get("n"),
                float("nan"),
            )
        )

    for label, inp, bacc, other, n, base in rows:
        bacc_s = f"{bacc:.4f}" if isinstance(bacc, (int, float)) else "—"
        other_s = f"{other:.4f}" if isinstance(other, (int, float)) else "—"
        base_s = f"{base:.4f}" if isinstance(base, (int, float)) and base == base else "—"
        n_s = str(n) if n is not None else "—"
        lines.append(f"| {label} | {inp} | {bacc_s} | {other_s} | {n_s} | {base_s} |")

    lines.append("")
    lines.append(
        "chance BAcc = 0.5 for balanced binary gender classification; "
        "majority BAcc = 0.5 by construction (balanced accuracy of an "
        "always-predict-majority classifier). \"AUC / Accuracy\" column is "
        "AUC for embedding attackers, plain accuracy for LLM list attackers "
        "(no probability score available from those)."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Build unified results table for one run dir")
    p.add_argument(
        "--run-dir", type=Path, default=root / "runs" / "EXP-003_bpr_d32_311"
    )
    p.add_argument(
        "--llm-tag",
        default="real_v1",
        help="Tag suffix for LLM list-attack files (metrics_{style}_{tag}.json)",
    )
    p.add_argument(
        "--legacy-tag",
        default="none",
        choices=["include", "none"],
        help="Whether to also include pre-tagging (synthetic few-shot) results for comparison",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    table = build_table(run_dir, args.llm_tag, args.legacy_tag)
    out_path = args.out or (run_dir / "results_table.md")
    out_path = out_path if out_path.is_absolute() else root / out_path
    out_path.write_text(table, encoding="utf-8")
    print(table)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
