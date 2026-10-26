"""Exemplar-set quality for many-shot LLM attacks: label/list-signal separation.

Hypothesis from the 2026-09-27 shot sweep: an exemplar draw helps the LLM when
each exemplar's gender label *agrees* with the gender signal in its rec list.
Measured as

    separation = mean P(male | list) over the male exemplars
               - mean P(male | list) over the female exemplars

with P(male | list) from a logistic regression on the rec-list multi-hot
(balanced class weights), predicted **out-of-fold** (5-fold on the train pool)
so no exemplar is scored by a model that saw its own label.

``select`` scores many random draws (the same sampler and seeds that
``list_aia --shot-seed`` uses), restricts them to a fixed gender mix, and
writes a manifest of high / mid / low separation seeds for a validation run.

Usage:
    uv run python -m src.llm.exemplar_quality \
        --pool runs/EXP-004_lightgcn_d32_311/lists_train_pool.jsonl \
        --n-shots 16 --n-female 4 --per-group 5 \
        --out runs/EXP-004_lightgcn_d32_311/exemplar_quality/manifest_16shot.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.llm.list_aia import sample_shot_examples


def oof_p_male(pool: list[dict[str, Any]], seed: int = 2024) -> dict[str, float]:
    """Out-of-fold P(male) for every train-pool user from an LR on the rec-list multi-hot."""
    items = sorted({i for r in pool for i in r["rec_item_ids"]})
    col = {i: k for k, i in enumerate(items)}
    rows, cols = [], []
    for n, r in enumerate(pool):
        for i in r["rec_item_ids"]:
            rows.append(n)
            cols.append(col[i])
    X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(pool), len(items)))
    y = np.array([r["gender"] == "male" for r in pool], dtype=int)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced")
    p = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]
    return {r["user_id"]: float(pn) for r, pn in zip(pool, p)}


def separation(examples: list[dict[str, Any]], p_male: dict[str, float]) -> float | None:
    p = np.array([p_male[e["user_id"]] for e in examples])
    m = np.array([e["gender"] == "male" for e in examples])
    if m.all() or not m.any():
        return None
    return float(p[m].mean() - p[~m].mean())


def select(pool_path: Path, n_shots: int, n_female: int, per_group: int,
           n_draws: int, seed_offset: int) -> dict[str, Any]:
    pool = [json.loads(line) for line in pool_path.open(encoding="utf-8")]
    p_male = oof_p_male(pool)
    draws = []
    for k in range(n_draws):
        seed = seed_offset + k
        ex = sample_shot_examples(pool_path, n_shots, seed=seed)
        nf = sum(e["gender"] == "female" for e in ex)
        s = separation(ex, p_male)
        draws.append({"seed": seed, "n_female": nf, "separation": s})
    eligible = sorted((d for d in draws if d["n_female"] == n_female and d["separation"] is not None),
                      key=lambda d: d["separation"])
    mid = len(eligible) // 2
    groups = {
        "low": eligible[:per_group],
        "mid": eligible[mid - per_group // 2: mid - per_group // 2 + per_group],
        "high": eligible[-per_group:],
    }
    all_sep = np.array([d["separation"] for d in draws if d["separation"] is not None])
    return {
        "n_shots": n_shots, "n_female": n_female, "n_draws_scored": n_draws, "seed_offset": seed_offset,
        "n_eligible": len(eligible),
        "separation_all_draws": {"mean": float(all_sep.mean()), "sd": float(all_sep.std()),
                                 "p5": float(np.percentile(all_sep, 5)), "p95": float(np.percentile(all_sep, 95))},
        "groups": groups,
        "note": "separation = mean OOF P(male) of male exemplars - that of female exemplars; "
                "LR on rec-list multi-hot, balanced, 5-fold on the train pool",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", type=Path, default=Path("runs/EXP-004_lightgcn_d32_311/lists_train_pool.jsonl"))
    p.add_argument("--n-shots", type=int, default=16)
    p.add_argument("--n-female", type=int, default=4, help="only draws with exactly this many female exemplars")
    p.add_argument("--per-group", type=int, default=5)
    p.add_argument("--n-draws", type=int, default=2000)
    p.add_argument("--seed-offset", type=int, default=10_000,
                   help="draw seeds are seed_offset + k; disjoint from the 2024-2026 sweep")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    res = select(args.pool, args.n_shots, args.n_female, args.per_group, args.n_draws, args.seed_offset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    for g, ds in res["groups"].items():
        print(g, [(d["seed"], round(d["separation"], 3)) for d in ds])
    print(f"eligible draws: {res['n_eligible']}/{res['n_draws_scored']}", res["separation_all_draws"])


if __name__ == "__main__":
    main()
