"""Build the master attribute-inference results table (English, Markdown).

Meeting 2026-09-14 action item "big table": one table holding every attacker
tried so far, on the one protocol where they are actually comparable — the
fixed single train/test split with the SAME 500 test users for the classifier
attackers (``src/attack/list_representation_aia.py``) and the LLM list attacker
(``src/llm/list_aia.py``).

Rows are grouped by threat model, i.e. what the attacker is assumed to see:

  T0  the recommender's internal user embedding
  T1  the rec list + the recommender's item embedding table
  T2  the rec list's item IDs only
  T3  the rec list + public item metadata (genres, titles)
  T4  the rec list's titles as plain text

and carry a *label budget*: how many users with a known gender the attacker
needs before it can attack anyone. The classifier attackers need the whole
train pool; a zero-shot LLM needs none; a few-shot LLM needs its exemplars.
This is what separates the two literatures the project builds on — RAID /
Slokom assume an auxiliary labelled set, Staab et al.'s LLM adversary does not.

Emits three tables: the master table (balanced attackers only), the class-balance
appendix (default vs balanced, same estimator otherwise), and the classifier
robustness grid (AUC spread across classifiers per input).

Example:
  uv run python -m src.report.build_master_table \\
    --grid-summary runs/EXP-004_lightgcn_d32_311/attack_grid_500/summary.json \\
    --llm-dir runs/EXP-004_lightgcn_d32_311/llm_aia --llm-tag lightgcn_llama32_500_lp \\
    --run-label "EXP-004 (LightGCN d32, global_311)" \\
    --out runs/EXP-004_lightgcn_d32_311/attack_grid_500/master_table.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

# input -> (tier, human-readable "what the attacker sees", display name)
INPUTS: dict[str, tuple[str, str, str]] = {
    "true_user_embedding": ("T0", "model's user embedding", "User embedding (upper bound)"),
    "ld_mean_pool":        ("T1", "list + item embedding table", "List-derived, mean-pool"),
    "ld_rank_pool":        ("T1", "list + item embedding table", "List-derived, rank-pool"),
    "inversion":           ("T1", "list + item embedding table", "Recommender inversion u-hat"),
    "multi_hot":           ("T2", "list item IDs", "Multi-hot over list items"),
    "tfidf_multi_hot":     ("T2", "list item IDs", "TF-IDF multi-hot"),
    "genre_histogram":     ("T3", "list + public metadata", "Genre histogram (18-d)"),
    "title_text_embedding":("T3", "list + public metadata", "Title text embedding (MiniLM)"),
}
CLASSIFIER_NAMES = {"lr": "LR", "mlp": "MLP", "svm_rbf": "SVM (RBF)", "svm_linear": "SVM (linear)"}
CLASSIFIER_ORDER = ["lr", "mlp", "svm_rbf", "svm_linear"]
LLM_STYLES = ["zero_shot", "staab", "few_shot", "staab_few_shot"]
LLM_LABEL_BUDGET = {"zero_shot": 0, "staab": 0, "few_shot": 2, "staab_few_shot": 2}


def _fmt(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _ci(ci: list[float] | None) -> str:
    return "" if not ci else f" [{ci[0]:.3f}, {ci[1]:.3f}]"


def llm_rows(llm_dir: Path, tag: str, model_name: str) -> list[dict[str, Any]]:
    """Recompute LLM metrics from the per-user predictions (metrics json has no f1_macro)."""
    rows = []
    for style in LLM_STYLES:
        path = llm_dir / f"preds_{style}_{tag}.jsonl"
        if not path.exists():
            continue
        recs = [json.loads(line) for line in path.open(encoding="utf-8")]
        kept = [r for r in recs if r.get("pred_gender") in ("male", "female")]
        if not kept:
            continue
        y = np.array([r["gender"] == "male" for r in kept])
        p = np.array([r["pred_gender"] == "male" for r in kept])
        scored = [r for r in kept if r.get("score") is not None]
        auc = (roc_auc_score(np.array([r["gender"] == "male" for r in scored]),
                             np.array([r["score"] for r in scored]))
               if len(scored) > 1 and len(set(r["gender"] for r in scored)) > 1 else None)
        rows.append({
            "tier": "T4", "sees": "list titles (text)",
            "input": "Raw list titles", "classifier": f"{model_name} / {style}",
            "labels": LLM_LABEL_BUDGET[style], "n": len(kept), "n_total": len(recs),
            "refused": len(recs) - len(kept),
            "accuracy": float((p == y).mean()),
            "bacc": float(balanced_accuracy_score(y, p)),
            "f1_macro": float(f1_score(y, p, average="macro")),
            "recall_female": float(recall_score(y, p, pos_label=False)),
            "recall_male": float(recall_score(y, p, pos_label=True)),
            "auc": auc, "auc_ci95": None, "pred_male_share": float(p.mean()),
        })
    return rows


def grid_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    n_train = summary["protocol"]["n_train"]
    rows = []
    for r in summary["table"]:
        if r["input"] not in INPUTS:
            continue
        tier, sees, display = INPUTS[r["input"]]
        rows.append({
            "tier": tier, "sees": sees, "input": display, "input_key": r["input"],
            "classifier_key": r["classifier"], "variant": r["variant"],
            "classifier": CLASSIFIER_NAMES.get(r["classifier"], r["classifier"]),
            "balance": r["balance"], "labels": n_train, "n": summary["protocol"]["n_test"],
            **{k: r.get(k) for k in ("accuracy", "bacc", "f1_macro", "auc", "auc_ci95",
                                     "recall_male", "recall_female", "pred_male_share")},
        })
    return rows


def _sort_key(r: dict[str, Any]) -> tuple:
    return (r["tier"], -(r["auc"] or 0))


def majority_row(grid_dir: Path, n_test: int) -> dict[str, Any] | None:
    """Always-predict-the-majority-class baseline, from any attacker's saved y_true."""
    src = grid_dir / "attack_metrics_true_user_embedding.json"
    if not src.exists():
        return None
    payload = json.loads(src.read_text(encoding="utf-8"))
    preds = next(iter(payload["variants"].values()))["predictions"]
    y = np.array([r["y_true"] == "male" for r in preds])
    p = np.full(len(y), y.mean() >= 0.5)
    return {
        "tier": "—", "sees": "nothing", "input": "Majority class", "classifier": "—",
        "labels": 0, "n": n_test, "accuracy": float((p == y).mean()), "bacc": 0.5,
        "f1_macro": float(f1_score(y, p, average="macro")),
        "recall_female": float(recall_score(y, p, pos_label=False)),
        "recall_male": float(recall_score(y, p, pos_label=True)),
        "auc": None, "auc_ci95": None, "pred_male_share": float(p.mean()),
    }


def build(summary: dict[str, Any], llm: list[dict[str, Any]], run_label: str,
          grid_dir: Path | None = None) -> str:
    g = grid_rows(summary)
    proto = summary["protocol"]
    n_test = proto["n_test"]
    main = [r for r in g if r["variant"] in ("lr_balanced", "mlp_balanced",
                                             "svm_rbf_balanced", "svm_linear_balanced")]
    out: list[str] = []
    out.append(f"### Table 1 — Master table: all attribute-inference attackers tried so far\n")
    llm_note = (
        " The LLM has no training step to balance; its AUC comes from the answer token's "
        "log-probabilities (score = P(male) / (P(male) + P(female)))."
        if llm else
        " T4 (LLM on the raw list titles) has not been run for this recommender yet."
    )
    out.append(
        f"{run_label}. Fixed single train/test split: the same {n_test} test users for every "
        f"attacker, {proto['n_train']}-user training pool. Target attribute: gender "
        f"(binary). All classifiers use balanced class weights (see Table 2)." + llm_note +
        f" \"Labels\" = how many users with a known gender the attacker needs before it can "
        f"attack anybody.\n"
    )
    out.append("Tiers — **T0** model's user embedding · **T1** list + the model's item embedding table · "
               "**T2** list item IDs only · **T3** list + public item metadata · **T4** list titles as plain text. "
               f"n = {n_test} test users for every classifier row; the LLM rows give their own n "
               "(refusals are dropped before scoring).\n")
    out.append("| Tier | Input representation | Attacker | Labels | Acc | BAcc | F1-macro | Recall (F) | AUC [95% CI] |")
    out.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    baseline = majority_row(grid_dir, n_test) if grid_dir else None
    for r in sorted(main + llm, key=_sort_key) + ([baseline] if baseline else []):
        name = r["classifier"] if r["tier"] != "T4" else f"{r['classifier']} (n={r['n']})"
        out.append(
            f"| {r['tier']} | {r['input']} | {name} | "
            f"{r['labels']:,} | {_fmt(r['accuracy'])} | {_fmt(r['bacc'])} | "
            f"{_fmt(r['f1_macro'])} | {_fmt(r['recall_female'])} | {_fmt(r['auc'])}{_ci(r.get('auc_ci95'))} |"
        )
    out.append("")
    refused = [r for r in llm if r["refused"]]
    if refused:
        out.append("LLM rows drop hard safety refusals before scoring: "
                   + "; ".join(f"{r['classifier'].split(' / ')[1]} {r['refused']}/{r['n_total']}"
                               for r in refused) + ".\n")

    out.append("### Table 2 — Effect of class balancing (same estimator, nothing else changed)\n")
    out.append(
        "`default` = library defaults, no class weighting. `balanced` = balanced class weights "
        "and nothing else (LR/SVM via `class_weight`, MLP via `sample_weight`, which "
        "`MLPClassifier` takes instead — it has no `class_weight` parameter). "
        "`lr_balanced_cv` additionally tunes C by 5-fold CV on the train pool and is reported "
        "separately because it changes two things at once (this is the variant the 2026-09-14 "
        "numbers used).\n"
    )
    out.append("| Input representation | Attacker | BAcc def → bal | ΔBAcc | AUC def → bal | ΔAUC | Predicted male def → bal |")
    out.append("| --- | --- | --- | ---: | --- | ---: | --- |")
    by = {(r["input_key"], r["variant"]): r for r in g}
    for key in INPUTS:
        for clf in CLASSIFIER_ORDER:
            d, b = by.get((key, f"{clf}_default")), by.get((key, f"{clf}_balanced"))
            if not (d and b):
                continue
            out.append(
                f"| {INPUTS[key][2]} | {CLASSIFIER_NAMES[clf]} | "
                f"{_fmt(d['bacc'])} → {_fmt(b['bacc'])} | {b['bacc'] - d['bacc']:+.3f} | "
                f"{_fmt(d['auc'])} → {_fmt(b['auc'])} | {b['auc'] - d['auc']:+.3f} | "
                f"{d['pred_male_share']:.0%} → {b['pred_male_share']:.0%} |"
            )
        cvr = by.get((key, "lr_balanced_cv"))
        if cvr:
            out.append(f"| {INPUTS[key][2]} | LR, balanced + C tuned | — → {_fmt(cvr['bacc'])} | — | "
                       f"— → {_fmt(cvr['auc'])} | — | — → {cvr['pred_male_share']:.0%} |")
    out.append("")

    out.append("### Table 3 — Does the choice of classifier matter? (balanced variants, AUC)\n")
    out.append("| Input representation | " + " | ".join(CLASSIFIER_NAMES[c] for c in CLASSIFIER_ORDER) + " | spread |")
    out.append("| --- | " + " | ".join("---:" for _ in CLASSIFIER_ORDER) + " | ---: |")
    spreads = []
    for key in INPUTS:
        vals = [by.get((key, f"{c}_balanced"), {}).get("auc") for c in CLASSIFIER_ORDER]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        spread = max(present) - min(present)
        spreads.append(spread)
        out.append(f"| {INPUTS[key][2]} | " + " | ".join(_fmt(v) for v in vals) + f" | {spread:.3f} |")
    if spreads:
        all_auc = [r["auc"] for r in main if r["auc"] is not None]
        out.append("")
        out.append(
            f"Largest AUC spread across classifiers for a fixed input: **{max(spreads):.3f}**. "
            f"AUC spread across inputs (balanced attackers, all classifiers): "
            f"**{max(all_auc) - min(all_auc):.3f}**. The input representation, not the "
            f"classifier, is what decides how much gender leaks."
        )
    out.append("")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Build the master AIA results table (Markdown, English)")
    p.add_argument("--grid-summary", type=Path, required=True,
                   help="summary.json from src.attack.list_representation_aia")
    p.add_argument("--llm-dir", type=Path, default=None, help="run's llm_aia/ directory")
    p.add_argument("--llm-tag", default=None, help="LLM run tag, e.g. lightgcn_llama32_500_lp")
    p.add_argument("--llm-model", default="llama3.2")
    p.add_argument("--run-label", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    summary = json.loads(args.grid_summary.read_text(encoding="utf-8"))
    llm = llm_rows(args.llm_dir, args.llm_tag, args.llm_model) if args.llm_dir and args.llm_tag else []
    md = build(summary, llm, args.run_label, grid_dir=args.grid_summary.parent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
