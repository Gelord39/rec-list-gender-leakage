"""Differential gender leakage by true age bucket (scripted version of the 2026-09-08 table).

Slices the stored per-user predictions of one recommender's 500 fixed test users
by their true age bucket (18-24 / 25-49 / >=50) — no new training or inference:
  - classifier rows: ``attack_grid_500/attack_metrics_{input}.json``, one variant
    (default ``lr_balanced``, the master-table headline classifier)
  - LLM rows: ``llm_aia/preds_{style}_{tag}.jsonl`` for each ``--llm`` style:tag

Reports BAcc (hard labels; LLM abstentions dropped) and AUC (every scored user) per
bucket, with the bucket's n. AUC needs both genders in the bucket; with n ~70-85
per bucket the CIs are wide, so the 95% bootstrap CI is reported as well.

Usage:
    uv run python -m src.report.age_slice_table --run-dir runs/EXP-003_bpr_d32_311 \\
        --llm zero_shot:bpr_glm4_9b_500 --llm staab_few_shot:bpr_glm4_9b_500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

BUCKETS = ("18-24", "25-49", ">=50")
INPUTS = {
    "true_user_embedding": "T0 user embedding",
    "ld_mean_pool": "T1 list-derived, mean-pool",
    "inversion": "T1 recommender inversion",
    "multi_hot": "T2 multi-hot",
    "genre_histogram": "T3 genre histogram",
    "title_text_embedding": "T3 MiniLM titles",
}
N_BOOT = 1000


def _ci(y: np.ndarray, s: np.ndarray, seed: int = 2024) -> list[float] | None:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def _slice(rows: list[dict[str, Any]], age: dict[str, str]) -> dict[str, Any]:
    """rows: user_id, y_true, y_pred (male/female/None), score (higher = male, may be None)."""
    out = {}
    for b in ("all", *BUCKETS):
        r = [x for x in rows if b == "all" or age[x["user_id"]] == b]
        lab = [x for x in r if x["y_pred"] in ("male", "female")]
        sc = [x for x in r if x["score"] is not None]
        y_s = np.array([x["y_true"] == "male" for x in sc])
        s = np.array([x["score"] for x in sc], dtype=float)
        auc = roc_auc_score(y_s, s) if len(set(y_s)) == 2 else None
        out[b] = {
            "n": len(r), "n_labelled": len(lab),
            "bacc": float(balanced_accuracy_score([x["y_true"] for x in lab], [x["y_pred"] for x in lab]))
            if lab else None,
            "auc": float(auc) if auc is not None else None,
            "auc_ci95": _ci(y_s, s) if auc is not None else None,
        }
    return out


def classifier_rows(grid_dir: Path, variant: str) -> dict[str, list[dict[str, Any]]]:
    res = {}
    for inp in INPUTS:
        f = grid_dir / f"attack_metrics_{inp}.json"
        if not f.exists():
            continue
        v = json.loads(f.read_text())["variants"][variant]
        preds = v["predictions"]
        # the stored score is a decision value; orient it so that higher = male
        y = np.array([p["y_true"] == "male" for p in preds])
        s = np.array([p["score"] for p in preds], dtype=float)
        sign = 1.0 if abs(roc_auc_score(y, s) - v["metrics"]["auc"]) < 1e-6 else -1.0
        assert abs(roc_auc_score(y, sign * s) - v["metrics"]["auc"]) < 1e-6
        res[INPUTS[inp]] = [{**p, "score": sign * p["score"]} for p in preds]
    return res


def llm_rows(llm_dir: Path, style: str, tag: str) -> list[dict[str, Any]]:
    rows = []
    for line in (llm_dir / f"preds_{style}_{tag}.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        rows.append({"user_id": r["user_id"], "y_true": r["gender"],
                     "y_pred": r.get("pred_gender"), "score": r.get("score")})
    return rows


def _fmt(c: dict[str, Any]) -> str:
    if c["auc"] is None:
        return "—"
    b = f"{c['bacc']:.3f}" if c["bacc"] is not None else "—"
    return f"{b} / {c['auc']:.3f}"


def to_markdown(res: dict[str, Any], age_n: dict[str, int]) -> str:
    head = "| Attacker | all | " + " | ".join(f"{b} (n={age_n[b]})" for b in BUCKETS) + " | AUC range |"
    lines = [f"Gender attack, BAcc / AUC by true age bucket — {res['run_dir']} ({res['variant']} for classifiers)", "",
             head, "| --- | --- | " + " | ".join("---" for _ in BUCKETS) + " | ---: |"]
    for name, sl in res["rows"].items():
        aucs = [sl[b]["auc"] for b in BUCKETS if sl[b]["auc"] is not None]
        rng = f"{max(aucs) - min(aucs):.3f}" if aucs else "—"
        n_note = "" if sl["all"]["n_labelled"] == sl["all"]["n"] else f" (labels n={sl['all']['n_labelled']})"
        lines.append(f"| {name}{n_note} | {_fmt(sl['all'])} | " + " | ".join(_fmt(sl[b]) for b in BUCKETS)
                     + f" | {rng} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--variant", default="lr_balanced")
    p.add_argument("--llm", action="append", default=[], help="style:tag, repeatable")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <run-dir>/age_slices")
    args = p.parse_args()

    test = [json.loads(line) for line in (args.run_dir / "lists.jsonl").open(encoding="utf-8")]
    age = {r["user_id"]: r["age"] for r in test}
    age_n = {b: sum(a == b for a in age.values()) for b in BUCKETS}

    rows = classifier_rows(args.run_dir / "attack_grid_500", args.variant)
    for st in args.llm:
        style, tag = st.split(":", 1)
        rows[f"T4 LLM {style} ({tag})"] = llm_rows(args.run_dir / "llm_aia", style, tag)

    res = {"run_dir": str(args.run_dir), "variant": args.variant,
           "rows": {k: _slice(v, age) for k, v in rows.items()}}
    out = args.out_dir or args.run_dir / "age_slices"
    out.mkdir(parents=True, exist_ok=True)
    (out / "age_slices.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res, age_n)
    (out / "age_slices.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
