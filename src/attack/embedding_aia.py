"""Attribute inference attacks on user embeddings (RAID-style Phase A).

Protocol (docs/experiment_setup.md §6.1):
- Input: user embedding y_u
- Target: gender (binary)
- Primary attacker: MLP with one hidden layer of size 100
- Robustness baselines (meeting 2026-08-17, per Slokom et al. 2021 §4.3.1,
  who found LR the strongest and SVM the second-strongest classifier for
  gender inference on recommender data): logistic regression, SVM
- Evaluation: stratified 5-fold CV (~80/20 each fold); same harness for all
  three classifiers so results stay comparable to each other. Reports mean
  BAcc, F1-micro/macro, accuracy, and AUC (AUC because it's Slokom et al.'s
  headline metric, Table 4) — not their exact protocol (10-fold, L2-norm,
  GridSearchCV), which is tuned for a raw rating matrix, not dense embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# classifier name -> (sklearn estimator factory(seed), protocol metadata)
ClassifierFactory = Callable[[int], Any]

_CLASSIFIER_BUILDERS: dict[str, tuple[ClassifierFactory, dict[str, Any]]] = {
    "mlp": (
        lambda seed: MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=seed,
        ),
        {"hidden_size": 100, "activation": "relu", "solver": "adam", "max_iter": 500},
    ),
    "logistic_regression": (
        lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        {"max_iter": 1000},
    ),
    "svm": (
        lambda seed: SVC(kernel="rbf", probability=False, random_state=seed),
        {"kernel": "rbf"},
    ),
}


def _auc_score(clf: Pipeline, X_te: np.ndarray, y_te: np.ndarray) -> float | None:
    """Binary AUC using predict_proba if available, else decision_function.

    Returns None if the fold's test set has only one class (AUC undefined).
    """
    if len(np.unique(y_te)) < 2:
        return None
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_te)[:, 1]
    else:
        scores = clf.decision_function(X_te)
    return float(roc_auc_score(y_te, scores))


def load_attack_table(
    embeddings_path: Path,
    user_ids_path: Path,
    users_csv: Path,
    attribute: str = "gender",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align embeddings with attribute labels by user_id.

    Returns:
        X: (n, d) embeddings
        y_raw: (n,) raw attribute labels (e.g. male/female)
        user_ids: (n,) original user id strings
    """
    emb = np.load(embeddings_path)
    user_ids = np.load(user_ids_path, allow_pickle=True).astype(str)
    users = pd.read_csv(users_csv)
    users["user_id"] = users["user_id"].astype(str)

    if attribute not in users.columns:
        raise ValueError(f"Attribute {attribute!r} not in {users_csv}")

    attr = users.set_index("user_id")[attribute]
    missing = [u for u in user_ids if u not in attr.index]
    if missing:
        raise ValueError(f"{len(missing)} embedding users lack attribute labels")

    y_raw = attr.loc[user_ids].to_numpy()
    if emb.shape[0] != len(user_ids):
        raise ValueError("embeddings and user_ids length mismatch")
    return emb, y_raw, user_ids


def run_classifier_attack(
    X: np.ndarray,
    y_raw: np.ndarray,
    *,
    classifier: str = "mlp",
    n_splits: int = 5,
    seed: int = 2024,
) -> dict[str, Any]:
    """Stratified K-fold attack with any classifier from ``_CLASSIFIER_BUILDERS``.

    Same CV protocol for every classifier (StratifiedKFold, StandardScaler,
    same seed/folds) so results are comparable across ``mlp`` /
    ``logistic_regression`` / ``svm``. Reports BAcc, F1-micro/macro, accuracy,
    and AUC per fold (AUC is Slokom et al. 2021's headline metric, Table 4).
    """
    if classifier not in _CLASSIFIER_BUILDERS:
        raise ValueError(
            f"Unknown classifier {classifier!r}; choose from {sorted(_CLASSIFIER_BUILDERS)}"
        )
    build_estimator, extra_protocol = _CLASSIFIER_BUILDERS[classifier]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Majority baseline on full label distribution (same for every fold report)
    values, counts = np.unique(y, return_counts=True)
    majority = values[counts.argmax()]
    majority_bacc = balanced_accuracy_score(y, np.full_like(y, majority))
    majority_f1_micro = f1_score(y, np.full_like(y, majority), average="micro")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows: list[dict[str, Any]] = []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (classifier, build_estimator(seed + fold)),
            ]
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        row = {
            "fold": fold,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "bacc": float(balanced_accuracy_score(y[te], pred)),
            "f1_micro": float(f1_score(y[te], pred, average="micro")),
            "f1_macro": float(f1_score(y[te], pred, average="macro")),
            "accuracy": float((pred == y[te]).mean()),
            "auc": _auc_score(clf, X[te], y[te]),
        }
        fold_rows.append(row)

    def _agg(key: str) -> dict[str, float]:
        vals = np.array(
            [r[key] for r in fold_rows if r[key] is not None], dtype=float
        )
        if vals.size == 0:
            return {"mean": None, "std": None}
        return {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}

    return {
        "attribute": "gender",
        "classifier": classifier,
        "label_classes": le.classes_.tolist(),
        "label_counts": {
            str(c): int((y_raw == c).sum()) for c in le.classes_
        },
        "n_users": int(len(y)),
        "embedding_dim": int(X.shape[1]),
        "protocol": {
            "n_splits": n_splits,
            "approx_train_test": "80/20 per fold (StratifiedKFold)",
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
        "folds": fold_rows,
        "aggregate": {
            "bacc": _agg("bacc"),
            "f1_micro": _agg("f1_micro"),
            "f1_macro": _agg("f1_macro"),
            "accuracy": _agg("accuracy"),
            "auc": _agg("auc"),
        },
    }


def run_mlp_attack(
    X: np.ndarray,
    y_raw: np.ndarray,
    *,
    n_splits: int = 5,
    hidden_size: int = 100,
    seed: int = 2024,
    max_iter: int = 500,
) -> dict[str, Any]:
    """Backward-compatible MLP-only entry point (used by the analysis notebook).

    ``hidden_size``/``max_iter`` overrides are accepted for compatibility but
    only the defaults (100 / 500, matching ``_CLASSIFIER_BUILDERS["mlp"]``)
    are exercised in practice — pass ``classifier="mlp"`` to
    ``run_classifier_attack`` directly if you need different hyperparameters.
    """
    return run_classifier_attack(X, y_raw, classifier="mlp", n_splits=n_splits, seed=seed)


def _out_name_for(classifier: str) -> str:
    return "attack_metrics.json" if classifier == "mlp" else f"attack_metrics_{classifier}.json"


def attack_run_dir(
    run_dir: Path,
    users_csv: Path,
    *,
    attribute: str = "gender",
    classifier: str = "mlp",
    n_splits: int = 5,
    seed: int = 2024,
    out_name: str | None = None,
) -> dict[str, Any]:
    """Run one classifier's attack over a run directory and write its metrics file.

    Written to ``run_dir/embedding_aia/`` (2026-09-08: moved off the run
    directory's top level, which was getting cluttered once
    ``list_embedding_aia.py`` and ``fixed_split_aia.py`` started writing
    their own files there too). ``mlp`` writes ``attack_metrics.json``
    (unchanged filename, for backward compatibility with existing
    runs/notebooks); ``logistic_regression`` and ``svm`` write
    ``attack_metrics_{classifier}.json`` alongside it.
    """
    run_dir = Path(run_dir)
    X, y_raw, _ = load_attack_table(
        run_dir / "user_embeddings.npy",
        run_dir / "user_ids.npy",
        Path(users_csv),
        attribute=attribute,
    )
    result = run_classifier_attack(X, y_raw, classifier=classifier, n_splits=n_splits, seed=seed)
    result["run_dir"] = str(run_dir.resolve())
    result["source_rec_metrics"] = str((run_dir / "rec_metrics.json").resolve())
    out_dir = run_dir / "embedding_aia"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or _out_name_for(classifier))
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["paths"] = {"attack_metrics": str(out_path.resolve())}
    return result


def attack_run_dir_all_classifiers(
    run_dir: Path,
    users_csv: Path,
    *,
    attribute: str = "gender",
    n_splits: int = 5,
    seed: int = 2024,
) -> dict[str, dict[str, Any]]:
    """Run mlp + logistic_regression + svm attacks over one run dir."""
    return {
        clf: attack_run_dir(
            run_dir, users_csv, attribute=attribute, classifier=clf, n_splits=n_splits, seed=seed
        )
        for clf in _CLASSIFIER_BUILDERS
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Attribute inference attack on exported user embeddings")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=root / "runs" / "EXP-003_bpr_d32_311",
    )
    parser.add_argument(
        "--users-csv",
        type=Path,
        default=root / "data" / "ml-1m-311" / "splits" / "users.csv",
    )
    parser.add_argument("--attribute", default="gender")
    parser.add_argument(
        "--classifier",
        choices=[*sorted(_CLASSIFIER_BUILDERS), "all"],
        default="all",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    classifiers = sorted(_CLASSIFIER_BUILDERS) if args.classifier == "all" else [args.classifier]
    for clf in classifiers:
        result = attack_run_dir(
            args.run_dir,
            args.users_csv,
            attribute=args.attribute,
            classifier=clf,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        agg = result["aggregate"]
        auc_s = f"{agg['auc']['mean']:.4f}" if agg["auc"]["mean"] is not None else "n/a"
        print(
            f"[{clf}] BAcc {agg['bacc']['mean']:.4f} ± {agg['bacc']['std']:.4f} | "
            f"F1-micro {agg['f1_micro']['mean']:.4f} ± {agg['f1_micro']['std']:.4f} | "
            f"AUC {auc_s} | majority BAcc {result['baselines']['majority_bacc']:.4f}"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
