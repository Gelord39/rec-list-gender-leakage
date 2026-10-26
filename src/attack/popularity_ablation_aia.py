"""Is the gender signal in a rec list carried by the popular items or the tail?

Meeting 2026-09-14 action item: "AIA based on non-hot item in reclist", following
the observation that across all 3,867 users' top-10 lists LightGCN recommends only
358 distinct movies and the 50 most-recommended ones fill ~88% of all list slots.
If the head items are nearly the same for everybody they cannot discriminate users,
so the signal should sit in the few unusual recommendations.

"Hot" = the ``--top-n`` most frequently recommended items, counted over the TRAIN
POOL lists only (never the test lists). ``--top-n 10`` is the default because it is
the only cut that leaves essentially every user with something: dropping the top 10
empties 0.2% of LightGCN users, dropping the top 50 empties 63.5%, at which point a
tail-only attack is undefined for most of the test set.

Four arms, all on the same users and the same fixed split as
``list_representation_aia.py``:

  full             the whole top-10 list (reference)
  tail_only        head items removed
  head_only        only head items kept
  random_ablation  per user, the SAME NUMBER of slots removed as tail_only removed
                   for that user, chosen uniformly at random (``--n-seeds`` draws)

The random arm is the control that makes the result interpretable: it holds "how
much was removed" fixed and varies only "was what we removed popular". Without it,
"tail_only keeps its AUC" cannot be distinguished from "six items were enough".

Two representations, both fed to the same classifier grid as the master table:
``multi_hot`` (0/1 over the train-pool item vocabulary) and ``mean_pool`` (mean of
the remaining items' embeddings, the representation the popularity direction was
shown to dominate — mean cosine 0.944 between two users' pooled vectors).

Example:
  uv run python -m src.attack.popularity_ablation_aia \\
    --config configs/lightgcn_ml1m_311.yaml --model LightGCN \\
    --checkpoint runs/EXP-004_lightgcn_d32_311/checkpoint/LightGCN-*.pth \\
    --test-lists runs/EXP-004_lightgcn_d32_311/lists.jsonl \\
    --train-pool-lists runs/EXP-004_lightgcn_d32_311/lists_train_pool.jsonl \\
    --out-dir runs/EXP-004_lightgcn_d32_311/popularity_ablation_500
"""

from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from src.attack.list_representation_aia import (
    CLASSIFIERS, _evaluate, _git_commit, _read_lists,
)
from src.export.export_lists import _load_model
from src.train.train_recbole import get_user_item_embeddings

ARMS = ("full", "tail_only", "head_only", "random_ablation")


def head_items(train_rows: list[dict[str, Any]], top_n: int) -> list[Any]:
    """The `top_n` most frequently recommended items, counted on the train pool only."""
    freq = collections.Counter(i for r in train_rows for i in r["rec_item_ids"])
    return [i for i, _ in freq.most_common(top_n)]


def apply_arm(lists: list[list[Any]], head: set[Any], arm: str,
              rng: np.random.Generator | None = None) -> list[list[Any]]:
    if arm == "full":
        return lists
    if arm == "tail_only":
        return [[i for i in row if i not in head] for row in lists]
    if arm == "head_only":
        return [[i for i in row if i in head] for row in lists]
    if arm == "random_ablation":
        assert rng is not None
        out = []
        for row in lists:
            n_drop = sum(1 for i in row if i in head)  # same count tail_only removes
            keep = rng.permutation(len(row))[: len(row) - n_drop]
            out.append([row[k] for k in sorted(keep)])
        return out
    raise ValueError(arm)


def evaluable_mask(lists: list[list[Any]], head: set[Any]) -> np.ndarray:
    """Users with at least one item under EVERY arm, so all arms share a population."""
    return np.array([
        any(i not in head for i in row) and any(i in head for i in row) and len(row) > 0
        for row in lists
    ])


def run(config_file: Path, checkpoint: Path, model_name: str, test_lists_path: Path,
        train_pool_lists_path: Path, out_dir: Path, *, top_n: int = 10, seed: int = 2024,
        n_boot: int = 1000, n_seeds: int = 10, device: str | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, ds, model, _ = _load_model(config_file, checkpoint, device or "cpu", model_name=model_name)
    _, item_emb = map(np.asarray, get_user_item_embeddings(model))

    tr_rows, te_rows = _read_lists(train_pool_lists_path), _read_lists(test_lists_path)
    head_tokens = head_items(tr_rows, top_n)
    head = set(head_tokens)

    L_tr = [list(r["rec_item_ids"]) for r in tr_rows]
    L_te = [list(r["rec_item_ids"]) for r in te_rows]
    m_tr, m_te = evaluable_mask(L_tr, head), evaluable_mask(L_te, head)
    coverage = {
        "n_train_pool_total": len(L_tr), "n_train_pool_kept": int(m_tr.sum()),
        "n_test_total": len(L_te), "n_test_kept": int(m_te.sum()),
        "dropped_because_an_arm_would_be_empty": int((~m_te).sum()),
    }
    L_tr = [r for r, k in zip(L_tr, m_tr) if k]
    L_te = [r for r, k in zip(L_te, m_te) if k]
    y_tr = np.array([r["gender"] == "male" for r, k in zip(tr_rows, m_tr) if k])
    y_te = np.array([r["gender"] == "male" for r, k in zip(te_rows, m_te) if k])
    test_ids = [str(r["user_id"]) for r, k in zip(te_rows, m_te) if k]

    vocab = {i: k for k, i in enumerate(sorted({i for row in L_tr for i in row}))}
    iid = lambda tok: ds.token2id(ds.iid_field, tok)

    def multi_hot(rows: list[list[Any]]) -> csr_matrix:
        r, c = zip(*[(n, vocab[i]) for n, row in enumerate(rows) for i in row if i in vocab])
        return csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows), len(vocab)))

    def mean_pool(rows: list[list[Any]]) -> np.ndarray:
        return np.stack([item_emb[[iid(i) for i in row]].mean(0) for row in rows])

    REPR = {"multi_hot": (multi_hot, False), "mean_pool": (mean_pool, True)}
    results: dict[str, dict[str, Any]] = {}
    base: dict[str, dict[str, np.ndarray]] = {}

    for rep, (build, dense) in REPR.items():
        results[rep] = {}
        for arm in ("full", "tail_only", "head_only"):
            A_tr, A_te = apply_arm(L_tr, head, arm), apply_arm(L_te, head, arm)
            res, scores = _evaluate(build(A_tr), y_tr, build(A_te), y_te, test_ids,
                                    dense=dense, seed=seed, n_boot=n_boot,
                                    base_scores=base.get(rep), classifiers=CLASSIFIERS)
            if arm == "full":
                base[rep] = scores
            results[rep][arm] = {v: r["metrics"] for v, r in res.items()}

        # random ablation: LR only, averaged over n_seeds draws
        draws: list[dict[str, float]] = []
        for s in range(n_seeds):
            rng = np.random.default_rng(seed + s)
            A_tr = apply_arm(L_tr, head, "random_ablation", rng)
            A_te = apply_arm(L_te, head, "random_ablation", rng)
            res, _ = _evaluate(build(A_tr), y_tr, build(A_te), y_te, test_ids, dense=dense,
                               seed=seed, n_boot=0, base_scores=None, classifiers=("lr",))
            draws.append(res["lr_balanced"]["metrics"])
        results[rep]["random_ablation"] = {
            "lr_balanced": {k: {"mean": float(np.mean([d[k] for d in draws])),
                                "sd": float(np.std([d[k] for d in draws]))}
                            for k in ("bacc", "auc", "accuracy", "f1_macro")},
            "n_seeds": n_seeds, "classifier_note": "LR only (control arm)",
        }

    payload = {
        "protocol": {
            "model": model_name, "top_n_hot": top_n,
            "hot_defined_by": "recommendation frequency over the train-pool lists only",
            "head_item_tokens": head_tokens, "seed": seed, "n_bootstrap": n_boot,
            "n_random_seeds": n_seeds, "coverage": coverage,
            "test_lists": str(Path(test_lists_path).resolve()),
            "train_pool_lists": str(Path(train_pool_lists_path).resolve()),
            "checkpoint": str(Path(checkpoint).resolve()),
            "generated_at_iso": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "note": "delta_auc_vs_ld_mean_pool_ci95 in the arm results is the paired "
                    "bootstrap CI of AUC vs the SAME representation's `full` arm "
                    "(key name inherited from list_representation_aia._bootstrap_auc)",
        },
        "results": results,
    }
    (out_dir / "popularity_ablation.json").write_text(json.dumps(payload, indent=2) + "\n",
                                                      encoding="utf-8")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Head vs tail popularity ablation for list-based gender AIA")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test-lists", type=Path, required=True)
    p.add_argument("--train-pool-lists", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    out = run(a.config, a.checkpoint, a.model, a.test_lists, a.train_pool_lists, a.out_dir,
              top_n=a.top_n, seed=a.seed, n_boot=a.n_bootstrap, n_seeds=a.n_seeds, device=a.device)
    cov = out["protocol"]["coverage"]
    print(f"\ncoverage: test {cov['n_test_kept']}/{cov['n_test_total']}, "
          f"train pool {cov['n_train_pool_kept']}/{cov['n_train_pool_total']}\n")
    for rep, arms in out["results"].items():
        print(f"=== {rep} ===")
        for arm in ARMS:
            r = arms[arm]
            if arm == "random_ablation":
                m = r["lr_balanced"]
                print(f"  {arm:16} LR  BAcc {m['bacc']['mean']:.3f} ± {m['bacc']['sd']:.3f} | "
                      f"AUC {m['auc']['mean']:.3f} ± {m['auc']['sd']:.3f}")
            else:
                m = r["lr_balanced"]
                ci = m["auc_ci95"]
                d = m.get("delta_auc_vs_ld_mean_pool_ci95")
                ds = f" | ΔAUC vs full [{d[0]:+.3f}, {d[1]:+.3f}]" if d else ""
                print(f"  {arm:16} LR  BAcc {m['bacc']:.3f} | AUC {m['auc']:.3f} "
                      f"[{ci[0]:.3f}, {ci[1]:.3f}]{ds}")


if __name__ == "__main__":
    main()
