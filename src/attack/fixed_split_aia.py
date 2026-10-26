"""Fixed single train/test split attacker evaluation — cross-attacker comparability.

Motivation (2026-09-07, professor's suggestion, relayed via meeting): the
classifier attackers (`embedding_aia.py`, `list_embedding_aia.py`) use
stratified 5-fold CV, while the LLM list attacker (`list_aia.py`) evaluates
once on a fixed set of test users with fixed few-shot exemplars. CV's
variance-reduction rationale (average over many retrainings) doesn't apply to
a frozen LLM conditioned on 2 fixed exemplars, so 5-fold CV isn't the right
unit for comparing *across* attacker types. This module instead does ONE
train / ONE test split — using the SAME test users as the LLM's
`lists.jsonl` — for both the full user-embedding attacker and the
list-derived (mean-pool) embedding attacker.

This is a *second*, separate protocol used only for cross-attacker
comparability analysis. The existing 5-fold, full-population numbers in
`embedding_aia.py` remain the headline leakage numbers and are untouched by
this module.

Every test user gets one saved prediction per attacker/classifier
(`predictions` in the output JSON), so results can be joined against the
LLM's own per-user predictions (`llm_aia/preds_{style}_{tag}.jsonl`) for
paired, per-user, differential-leakage-by-slice analysis — not just
aggregate-metric comparison.

Example (EXP-004, LightGCN, 500 shared test users):
  # 1) fixed test set (already have this from the LLM pilot, or generate it):
  uv run python -m src.export.export_lists \\
    --config configs/lightgcn_ml1m_311.yaml --model LightGCN \\
    --checkpoint runs/EXP-004_lightgcn_d32_311/checkpoint/LightGCN-*.pth \\
    --users-csv data/ml-1m-311/splits/users.csv \\
    --items-csv data/ml-1m-311/splits/items.csv \\
    --train-csv data/ml-1m-311/splits/train.csv \\
    --out runs/EXP-004_lightgcn_d32_311/lists_test500.jsonl \\
    --topk 10 --history-size 0 --max-users 500

  # 2) train-pool lists (for the list-derived attacker's training rows),
  #    disjoint from the fixed test set:
  uv run python -m src.export.export_lists \\
    --config configs/lightgcn_ml1m_311.yaml --model LightGCN \\
    --checkpoint runs/EXP-004_lightgcn_d32_311/checkpoint/LightGCN-*.pth \\
    --users-csv data/ml-1m-311/splits/users.csv \\
    --items-csv data/ml-1m-311/splits/items.csv \\
    --train-csv data/ml-1m-311/splits/train.csv \\
    --out runs/EXP-004_lightgcn_d32_311/lists_train_pool.jsonl \\
    --topk 10 --history-size 0 --max-users 100000 \\
    --exclude-lists-jsonl runs/EXP-004_lightgcn_d32_311/lists_test500.jsonl

  # 3) run both attackers, fixed split:
  uv run python -m src.attack.fixed_split_aia \\
    --config configs/lightgcn_ml1m_311.yaml --model LightGCN \\
    --checkpoint runs/EXP-004_lightgcn_d32_311/checkpoint/LightGCN-*.pth \\
    --users-csv data/ml-1m-311/splits/users.csv \\
    --run-dir runs/EXP-004_lightgcn_d32_311 \\
    --test-lists runs/EXP-004_lightgcn_d32_311/lists_test500.jsonl \\
    --train-pool-lists runs/EXP-004_lightgcn_d32_311/lists_train_pool.jsonl \\
    --out-dir runs/EXP-004_lightgcn_d32_311/fixed_split_500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.attack.embedding_aia import _CLASSIFIER_BUILDERS, _auc_score, load_attack_table
from src.attack.list_embedding_aia import build_list_embeddings
from src.export.export_lists import _load_model
from src.train.train_recbole import get_user_item_embeddings


def split_by_user_ids(
    X: np.ndarray, y_raw: np.ndarray, user_ids: np.ndarray, test_user_ids: set[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition (X, y_raw, user_ids) into train/test by a fixed test-user-id set.

    Returns (X_train, y_train, ids_train, X_test, y_test, ids_test).
    """
    is_test = np.array([u in test_user_ids for u in user_ids])
    return (
        X[~is_test],
        y_raw[~is_test],
        user_ids[~is_test],
        X[is_test],
        y_raw[is_test],
        user_ids[is_test],
    )


def run_fixed_split_attack(
    X_train: np.ndarray,
    y_raw_train: np.ndarray,
    X_test: np.ndarray,
    y_raw_test: np.ndarray,
    test_user_ids: np.ndarray,
    *,
    classifier: str = "mlp",
    seed: int = 2024,
) -> dict[str, Any]:
    """Train once on (X_train, y_raw_train), evaluate once on (X_test, y_raw_test).

    Same classifier/scaler settings as
    ``embedding_aia.run_classifier_attack``, just one split instead of 5-fold
    CV. Returns aggregate metrics AND a per-user prediction list so results
    can be joined against other attackers' predictions on the same test
    users.
    """
    if classifier not in _CLASSIFIER_BUILDERS:
        raise ValueError(
            f"Unknown classifier {classifier!r}; choose from {sorted(_CLASSIFIER_BUILDERS)}"
        )
    build_estimator, extra_protocol = _CLASSIFIER_BUILDERS[classifier]

    le = LabelEncoder()
    le.fit(np.concatenate([y_raw_train, y_raw_test]))
    y_train = le.transform(y_raw_train)
    y_test = le.transform(y_raw_test)

    values, counts = np.unique(y_train, return_counts=True)
    majority = values[counts.argmax()]
    majority_bacc = balanced_accuracy_score(y_test, np.full_like(y_test, majority))
    majority_f1_micro = f1_score(y_test, np.full_like(y_test, majority), average="micro")

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (classifier, build_estimator(seed)),
        ]
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    auc = _auc_score(clf, X_test, y_test)

    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_test)[:, 1].tolist()
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test).tolist()
    else:
        scores = [None] * len(y_test)

    predictions = [
        {
            "user_id": str(uid),
            "y_true": str(le.inverse_transform([yt])[0]),
            "y_pred": str(le.inverse_transform([yp])[0]),
            "score": s,
        }
        for uid, yt, yp, s in zip(test_user_ids, y_test, pred, scores)
    ]

    return {
        "attribute": "gender",
        "classifier": classifier,
        "label_classes": le.classes_.tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "embedding_dim": int(X_train.shape[1]),
        "protocol": {
            "split": "fixed_single_split",
            "scaler": "StandardScaler",
            "seed": seed,
            **extra_protocol,
        },
        "baselines": {
            "majority_bacc": float(majority_bacc),
            "majority_f1_micro": float(majority_f1_micro),
            "chance_bacc_balanced": 0.5,
            "chance_auc": 0.5,
        },
        "metrics": {
            "bacc": float(balanced_accuracy_score(y_test, pred)),
            "f1_micro": float(f1_score(y_test, pred, average="micro")),
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
            "accuracy": float((pred == y_test).mean()),
            "auc": auc,
        },
        "predictions": predictions,
    }


def _read_user_ids(lists_path: Path) -> set[str]:
    return {json.loads(line)["user_id"] for line in Path(lists_path).open(encoding="utf-8")}


def run_fixed_split_pipeline(
    config_file: Path,
    checkpoint: Path,
    model_name: str,
    users_csv: Path,
    run_dir: Path,
    test_lists_path: Path,
    train_pool_lists_path: Path,
    out_dir: Path,
    *,
    classifiers: list[str] | None = None,
    poolings: list[str] | None = None,
    seed: int = 2024,
    device: str | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run the full-embedding attacker and the list-derived-embedding attacker
    (one or more pooling schemes) under the fixed single-split protocol, on
    the same test users, and save results.

    Returns {"embedding": {classifier: result}, "list_embedding_<pooling>": {classifier: result}, ...}.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classifiers = classifiers or sorted(_CLASSIFIER_BUILDERS)
    poolings = poolings or ["mean"]
    test_user_ids = _read_user_ids(test_lists_path)

    # --- full trained user embedding, fixed split (pooling-independent) ---
    X, y_raw, user_ids = load_attack_table(
        run_dir / "user_embeddings.npy",
        run_dir / "user_ids.npy",
        Path(users_csv),
        attribute="gender",
    )
    X_train, y_train, _, X_test, y_test, ids_test = split_by_user_ids(
        X, y_raw, user_ids, test_user_ids
    )
    if len(X_test) != len(test_user_ids):
        raise ValueError(
            f"{len(test_user_ids)} test user_ids requested but only {len(X_test)} "
            "found in the run's user_embeddings — check users-csv/run-dir match the "
            "same population as test_lists_path."
        )

    results: dict[str, dict[str, dict[str, Any]]] = {}
    embedding_results: dict[str, dict[str, Any]] = {}
    for clf in classifiers:
        result = run_fixed_split_attack(
            X_train, y_train, X_test, y_test, ids_test, classifier=clf, seed=seed
        )
        result["input"] = "full_user_embedding"
        out_path = out_dir / f"attack_metrics_embedding_{clf}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        embedding_results[clf] = result
    results["embedding"] = embedding_results

    # --- list-derived embedding, fixed split, one or more pooling schemes ---
    _, dataset, model, _ = _load_model(
        config_file, checkpoint, device or "cpu", model_name=model_name
    )
    _, item_weight = get_user_item_embeddings(model)

    for pooling in poolings:
        X_tr_list, y_tr_list, _ = build_list_embeddings(
            train_pool_lists_path, item_weight, dataset, pooling=pooling
        )
        X_te_list, y_te_list, ids_te_list = build_list_embeddings(
            test_lists_path, item_weight, dataset, pooling=pooling
        )

        list_embedding_results: dict[str, dict[str, Any]] = {}
        for clf in classifiers:
            result = run_fixed_split_attack(
                X_tr_list, y_tr_list, X_te_list, y_te_list, ids_te_list, classifier=clf, seed=seed
            )
            result["input"] = f"list_derived_embedding_{pooling}_pool"
            suffix = "" if pooling == "mean" else f"_{pooling}"
            out_path = out_dir / f"attack_metrics_list_embedding{suffix}_{clf}.json"
            out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            list_embedding_results[clf] = result
        results[f"list_embedding_{pooling}"] = list_embedding_results

    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Fixed single train/test split attacker evaluation, for cross-attacker "
            "comparability with the LLM list attacker (not the 5-fold headline numbers)"
        )
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", required=True, help="RecBole model class name, e.g. BPR, LightGCN")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--users-csv", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--test-lists",
        type=Path,
        required=True,
        help="lists.jsonl defining the fixed shared test set (same one used for the LLM attacker)",
    )
    p.add_argument(
        "--train-pool-lists",
        type=Path,
        required=True,
        help="lists.jsonl covering the training pool (disjoint from --test-lists)",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--classifier",
        choices=[*sorted(_CLASSIFIER_BUILDERS), "all"],
        default="all",
    )
    p.add_argument(
        "--pooling",
        choices=["mean", "rank", "both"],
        default="mean",
        help="mean = plain average; rank = NDCG-style weighted average; both = run and compare",
    )
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    classifiers = None if args.classifier == "all" else [args.classifier]
    poolings = ["mean", "rank"] if args.pooling == "both" else [args.pooling]
    results = run_fixed_split_pipeline(
        args.config,
        args.checkpoint,
        args.model,
        args.users_csv,
        args.run_dir,
        args.test_lists,
        args.train_pool_lists,
        args.out_dir,
        classifiers=classifiers,
        poolings=poolings,
        seed=args.seed,
        device=args.device,
    )
    for input_kind, by_clf in results.items():
        for clf, result in by_clf.items():
            m = result["metrics"]
            auc_s = f"{m['auc']:.4f}" if m["auc"] is not None else "n/a"
            print(
                f"[{input_kind}/{clf}] BAcc {m['bacc']:.4f} | F1-macro {m['f1_macro']:.4f} | "
                f"AUC {auc_s} | n_train {result['n_train']} | n_test {result['n_test']}"
            )


if __name__ == "__main__":
    main()
