"""List-derived embedding attribute inference attack (pool item embeddings).

Not RAID/Slokom/Staab protocol — a design of this project (2026-08-31, see
CLAUDE.md "List-derived embedding attacker"). Motivation: the embedding
attacker (`src/attack/embedding_aia.py`, full trained user embedding) and the
LLM list attacker (`src/llm/list_aia.py`, top-k titles only) aren't directly
comparable — they differ in both input (full trained embedding vs list-only)
and attacker type (classifier vs LLM). This module builds a same-input
counterpart to the embedding attacker: pool the model's own item embeddings
for each user's top-k recommended items (the exact items given to the LLM),
then run the pooled vector through the identical MLP/LR/SVM harness used for
the real user embedding.

Two pooling schemes (2026-09-07, added `rank`):
- `mean`: plain average — every rank treated equally, discards list order.
- `rank`: weighted average using the same discount NDCG@k already uses
  elsewhere in this project (`1/log2(rank+1)`, rank 1-indexed) — the top
  item counts more than the 10th, on the theory that a model's most
  confident recommendation carries more attribute signal than its least. Not
  from RAID/Slokom/Staab; reuses this project's own NDCG discount rather than
  inventing an unrelated weighting scheme.

Example (EXP-004, LightGCN):
  uv run python -m src.attack.list_embedding_aia \\
    --config configs/lightgcn_ml1m_311.yaml --model LightGCN \\
    --checkpoint runs/EXP-004_lightgcn_d32_311/checkpoint/LightGCN-Aug-30-2026_23-00-46.pth \\
    --lists runs/EXP-004_lightgcn_d32_311/lists.jsonl \\
    --run-dir runs/EXP-004_lightgcn_d32_311 \\
    --pooling mean --classifier all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.attack.embedding_aia import _CLASSIFIER_BUILDERS, run_classifier_attack
from src.export.export_lists import _load_model
from src.train.train_recbole import get_user_item_embeddings


def _pool(vecs: list[np.ndarray], pooling: str) -> np.ndarray:
    if pooling == "mean":
        return np.mean(vecs, axis=0)
    if pooling == "rank":
        # rank 1-indexed, same discount as NDCG@k: 1/log2(rank+1)
        weights = np.array([1.0 / np.log2(rank + 1) for rank in range(1, len(vecs) + 1)])
        weights = weights / weights.sum()
        return np.average(vecs, axis=0, weights=weights)
    raise ValueError(f"Unknown pooling {pooling!r}; choose from 'mean', 'rank'")


def build_list_embeddings(
    lists_path: Path,
    item_weight: np.ndarray,
    dataset,
    *,
    pooling: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool each user's top-k rec-list item embeddings into one vector.

    ``pooling``: 'mean' (plain average) or 'rank' (NDCG-style 1/log2(rank+1)
    weighted average, rank-1 item counts more). See module docstring.

    Returns (X, y_raw, user_ids) — same contract as
    ``embedding_aia.load_attack_table`` — so it drops straight into
    ``run_classifier_attack``.
    """
    iid_field = dataset.iid_field
    rows = [json.loads(line) for line in Path(lists_path).open(encoding="utf-8")]

    X: list[np.ndarray] = []
    y_raw: list[str] = []
    user_ids: list[str] = []
    for row in rows:
        vecs = []
        for tok in row["rec_item_ids"]:
            try:
                iid = dataset.token2id(iid_field, tok)
            except Exception:
                continue
            vecs.append(item_weight[iid])
        if not vecs:
            continue
        X.append(_pool(vecs, pooling))
        y_raw.append(row["gender"])
        user_ids.append(row["user_id"])

    return np.array(X), np.array(y_raw), np.array(user_ids)


def attack_from_lists(
    config_file: Path,
    checkpoint: Path,
    model_name: str,
    lists_path: Path,
    run_dir: Path,
    *,
    pooling: str = "mean",
    classifier: str = "all",
    n_splits: int = 5,
    seed: int = 2024,
    device: str | None = None,
) -> dict[str, dict[str, Any]]:
    run_dir = Path(run_dir)
    _, dataset, model, _ = _load_model(
        config_file, checkpoint, device or "cpu", model_name=model_name
    )
    _, item_weight = get_user_item_embeddings(model)

    X, y_raw, _ = build_list_embeddings(lists_path, item_weight, dataset, pooling=pooling)

    out_dir = run_dir / "list_embedding_aia"
    out_dir.mkdir(parents=True, exist_ok=True)
    classifiers = sorted(_CLASSIFIER_BUILDERS) if classifier == "all" else [classifier]
    results: dict[str, dict[str, Any]] = {}
    for clf in classifiers:
        result = run_classifier_attack(X, y_raw, classifier=clf, n_splits=n_splits, seed=seed)
        result["input"] = f"list_derived_embedding_{pooling}_pool"
        result["lists_path"] = str(Path(lists_path).resolve())
        suffix = "" if pooling == "mean" else f"_{pooling}"
        out_path = out_dir / f"attack_metrics_list_embedding{suffix}_{clf}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["paths"] = {"attack_metrics": str(out_path.resolve())}
        results[clf] = result
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="List-derived embedding AIA (mean-pool rec-list item embeddings)"
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", required=True, help="RecBole model class name, e.g. BPR, LightGCN")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--lists", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--classifier", choices=[*sorted(_CLASSIFIER_BUILDERS), "all"], default="all"
    )
    p.add_argument(
        "--pooling",
        choices=["mean", "rank"],
        default="mean",
        help="mean = plain average; rank = NDCG-style 1/log2(rank+1) weighted average",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    results = attack_from_lists(
        args.config,
        args.checkpoint,
        args.model,
        args.lists,
        args.run_dir,
        pooling=args.pooling,
        classifier=args.classifier,
        n_splits=args.n_splits,
        seed=args.seed,
        device=args.device,
    )
    for clf, result in results.items():
        agg = result["aggregate"]
        auc_s = f"{agg['auc']['mean']:.4f}" if agg["auc"]["mean"] is not None else "n/a"
        print(
            f"[{clf}] BAcc {agg['bacc']['mean']:.4f} ± {agg['bacc']['std']:.4f} | "
            f"F1-macro {agg['f1_macro']['mean']:.4f} ± {agg['f1_macro']['std']:.4f} | "
            f"AUC {auc_s} | n_users {result['n_users']}"
        )


if __name__ == "__main__":
    main()
