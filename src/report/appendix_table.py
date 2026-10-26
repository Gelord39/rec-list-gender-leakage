"""One table with every recommender x attacker tier (appendix of the 2026-10-05 deck).

Columns: BPR, LightGCN, and the LLM recommender (Stairway LLMRec, `req20` unseen lists)
GLM-4-9B / Llama-3.1-8B x neutral / gender prompt. Rows: gender AUC per attacker tier
(LR with balanced class weights for TH-T3; GLM-4-9B as the T4 attacker on every column),
then effectiveness (HR@10, NDCG@10 on the 695 users with test items). Every number is
read from runs/; "—" = the setting does not exist (an LLM recommender has no embeddings).
Each attack cell also carries balanced accuracy (BAcc): the classifier's own 0.5
threshold, or the LLM's stated answer over the users it labels male or female.
``--latex`` also writes the report's appendix table (docs/report_tables/full_auc.tex).

    uv run python -m src.report.appendix_table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

BPR = Path("runs/EXP-003_bpr_d32_311")
LGCN = Path("runs/EXP-004_lightgcn_d32_311")
LLMREC = Path("runs/EXP-006_llmrec_311")
COLS = [("BPR", BPR, "bpr"), ("LightGCN", LGCN, "lightgcn"),
        ("GLM neutral", LLMREC / "neutral_req20_glm4_9b_unseen", None),
        ("GLM gender", LLMREC / "gender_req20_glm4_9b_unseen", None),
        ("Llama neutral", LLMREC / "neutral_req20_llama31_8b_unseen", None),
        ("Llama gender", LLMREC / "gender_req20_llama31_8b_unseen", None)]
GRID_ROWS = [("TH", "Full watch history", "history_multi_hot"),
             ("TH", "Items shown in the LLMRec prompt", "prompt_history_multi_hot"),
             ("T0", "User embedding", "true_user_embedding"),
             ("T1", "Recommender inversion", "inversion"),
             ("T1", "Mean-pooled item embeddings", "ld_mean_pool"),
             ("T2", "Multi-hot over list items", "multi_hot"),
             ("T3", "Genre histogram", "genre_histogram"),
             ("T3", "Title text (MiniLM)", "title_text_embedding")]


def _grid(run: Path, inp: str) -> tuple[float, float] | None:
    """(AUC, BAcc) of the class-balanced LR on one input, or None."""
    if inp == "history_multi_hot" and not (run / "attack_grid_500/summary.json").exists():
        return None
    for r in json.loads((run / "attack_grid_500/summary.json").read_text())["table"]:
        if r["input"] == inp and r["variant"] == "lr_balanced":
            return r["auc"], r["bacc"]
    if inp == "history_multi_hot":  # CF runs: the history attacker lives in history_aia/
        t = json.loads((run / "history_aia/summary_binary.json").read_text())["table"]
        r = next(r for r in t if r["key"] == "lr_balanced")
        return r["auc"], r["bacc"]
    return None


def _llm(path: Path) -> tuple[float, float] | None:
    """(AUC over users with a score, BAcc over users the LLM labels male or female)."""
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    scored = [r for r in rows if r.get("score") is not None]
    auc = roc_auc_score([r["gender"] == "male" for r in scored], [r["score"] for r in scored])
    lab = [r for r in rows if r.get("pred_gender") in ("male", "female")]
    bacc = balanced_accuracy_score([r["gender"] for r in lab], [r["pred_gender"] for r in lab])
    return float(auc), float(bacc)


def _effectiveness(run: Path, prefix: str | None) -> tuple[float, float]:
    if prefix:
        t = json.loads((run / "rec_metrics.json").read_text())["test_result"]
        return t["hit@10"], t["ndcg@10"]
    m = json.loads((run / "llmrec_report.json").read_text())["rec_metrics"]
    return m["hit@10"], m["ndcg@10"]


def build() -> dict:
    rows = []
    for tier, label, inp in GRID_ROWS:
        rows.append({"tier": tier, "label": label, "values": [_grid(run, inp) for _, run, _ in COLS]})
    for style, tag_cf, tag_rec, label in (
            ("zero_shot", "glm4_9b_500", "glm4_9b_500", "LLM reads titles, 0 labels"),
            ("staab_many_shot", "glm4_9b_500_shots8", "glm4_9b_500_shots8", "LLM reads titles, 8 labels")):
        vals = []
        for _, run, prefix in COLS:
            if prefix:
                f = run / f"llm_aia/preds_{style}_{prefix}_{tag_cf}.jsonl"
                if not f.exists():  # BPR's 8-shot runs carry the seed suffix
                    f = run / f"llm_aia/preds_{style}_{prefix}_{tag_cf}_s2024.jsonl"
            else:
                f = run / f"llm_aia/preds_{style}_{tag_rec}.jsonl"
            vals.append(_llm(f))
        rows.append({"tier": "T4", "label": label, "values": vals})
    eff = [_effectiveness(run, prefix) for _, run, prefix in COLS]
    return {"columns": [c for c, _, _ in COLS], "auc_rows": rows,
            "hr10": [e[0] for e in eff], "ndcg10": [e[1] for e in eff]}


def to_markdown(t: dict) -> str:
    f = lambda v, d=2: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    g = lambda v: "—" if v is None else f"{v[0]:.2f} ({v[1]:.2f})"  # noqa: E731
    out = ["AUC (BAcc)", "", "| Tier | Input | " + " | ".join(t["columns"]) + " |",
           "| --- | --- | " + " | ".join("---:" for _ in t["columns"]) + " |"]
    for r in t["auc_rows"]:
        out.append(f"| {r['tier']} | {r['label']} | " + " | ".join(g(v) for v in r["values"]) + " |")
    out.append("| — | HR@10 | " + " | ".join(f(v, 3) for v in t["hr10"]) + " |")
    out.append("| — | NDCG@10 | " + " | ".join(f(v, 3) for v in t["ndcg10"]) + " |")
    return "\n".join(out) + "\n"


def to_latex(t: dict) -> str:
    g = lambda v: "--" if v is None else f"{v[0]:.2f} ({v[1]:.2f})"  # noqa: E731
    cols = ["BPR", "LightGCN", "GLM neutral", "GLM gender", "Llama neutral", "Llama gender"]
    out = [r"\begin{table*}[t]", r"\centering\small",
           r"\caption{Gender leakage of every recommender and attacker tier on the 500 test users, as AUC with "
           r"balanced accuracy (BAcc) in brackets, and effectiveness at the bottom. TH--T3: class-balanced LR, "
           r"BAcc at its own threshold. T4: GLM-4-9B reading the list titles, zero-shot or with 8 labelled "
           r"exemplars (Staab template, one draw); its BAcc uses the stated answer and excludes refusals and "
           r"abstentions (Table~\ref{tab:nonanswers}). --: the setting does not exist.}",
           r"\label{tab:full-auc}",
           r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
           "Tier & Input & " + " & ".join(cols) + r" \\", r"\midrule"]
    for r in t["auc_rows"]:
        out.append(f"{r['tier']} & {r['label']} & " + " & ".join(g(v) for v in r["values"]) + r" \\")
    out.append(r"\midrule")
    out.append("-- & HR@10 & " + " & ".join(f"{v:.3f}" for v in t["hr10"]) + r" \\")
    out.append("-- & NDCG@10 & " + " & ".join(f"{v:.3f}" for v in t["ndcg10"]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path("docs/slides_2026-10-05"))
    p.add_argument("--latex", type=Path, default=None, help="also write the LaTeX table to this file")
    args = p.parse_args()
    t = build()
    if args.latex:
        args.latex.write_text(to_latex(t) + "\n")
    (args.out_dir / "appendix_table.json").write_text(json.dumps(t, indent=2))
    md = to_markdown(t)
    (args.out_dir / "appendix_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
