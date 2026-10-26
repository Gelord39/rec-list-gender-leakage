"""Gender AIA from the user's raw interaction history — the Slokom et al. (2021) attacker.

PerBlur's gender classifier (github.com/SlokomManel/PerBlur, `GenderClassification.py`
/ `Classifiers.py`) runs on the user-item matrix itself: one row per user over all
items, L2-normalised, fed to logistic regression (l2, C=1), a linear SVM or an MLP,
scored by cross-validated AUC. This module reproduces that attacker on our data and
puts it on the SAME fixed split as every other attacker (train pool vs the 500 shared
test users), so it can sit in one table with the embedding / list-derived / LLM rows.

Why it matters: the full user embedding is a 32-d *compression of the history learned
by the recommender*, whereas this attacker reads the history directly and needs no
access to the trained model at all. It is therefore the natural upper bound for the
history row, and the matched classifier counterpart to an LLM that is shown the
user's watched titles.

Two weightings:
  - ``binary``: 1 if the item is in the user's train interactions (our pipeline
    binarises at rating >= 3, so this is what our splits carry)
  - ``rating``: the raw 1-5 rating from the original ml-1m ratings.dat, matching
    Slokom's ``load_user_item_matrix_1m_all`` exactly (needs --ratings-dat)
Rows are L2-normalised in both cases, as in PerBlur.

Example:
  uv run python -m src.attack.history_aia \\
    --train-csv data/ml-1m-311/splits/train.csv \\
    --users-csv data/ml-1m-311/splits/users.csv \\
    --test-lists runs/EXP-004_lightgcn_d32_311/lists.jsonl \\
    --train-pool-lists runs/EXP-004_lightgcn_d32_311/lists_train_pool.jsonl \\
    --out-dir runs/EXP-004_lightgcn_d32_311/history_aia
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

# PerBlur uses LR(l2, C=1), SVC(kernel="linear"), MLP(n_hidden=500); the rbf SVM and
# the 100-unit MLP are our own harness's settings, kept for continuity.
ESTIMATORS = {
    "lr": lambda seed, bal: LogisticRegression(
        penalty="l2", C=1.0, max_iter=5000, class_weight=bal, random_state=seed),
    "svm_linear": lambda seed, bal: SVC(
        kernel="linear", probability=False, class_weight=bal, random_state=seed),
    "svm_rbf": lambda seed, bal: SVC(
        kernel="rbf", probability=False, class_weight=bal, random_state=seed),
    "mlp500": lambda seed, bal: MLPClassifier(
        hidden_layer_sizes=(500,), max_iter=500, random_state=seed),
    "mlp100": lambda seed, bal: MLPClassifier(
        hidden_layer_sizes=(100,), max_iter=500, random_state=seed),
}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=Path(__file__).resolve().parents[2],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _user_ids(lists_path: Path) -> list[str]:
    return [json.loads(line)["user_id"] for line in Path(lists_path).open(encoding="utf-8")]


def build_history_matrix(
    train_csv: Path, users: list[str], *, weights: str = "binary",
    ratings_dat: Path | None = None,
) -> tuple[csr_matrix, list[str]]:
    """User x item matrix over the training interactions, L2-normalised per user."""
    rows = list(csv.DictReader(Path(train_csv).open(encoding="utf-8")))
    items = sorted({r["item_id"] for r in rows})
    col = {it: k for k, it in enumerate(items)}
    row_of = {u: i for i, u in enumerate(users)}

    value = {}
    if weights == "rating":
        if ratings_dat is None:
            raise ValueError("--ratings-dat is required for --weights rating")
        for line in Path(ratings_dat).open(encoding="latin-1"):
            u, i, r, _ = line.rstrip("\n").split("::")
            value[(u, i)] = float(r)

    r_idx, c_idx, data = [], [], []
    for rec in rows:
        u, it = rec["user_id"], rec["item_id"]
        if u not in row_of or it not in col:
            continue
        r_idx.append(row_of[u])
        c_idx.append(col[it])
        data.append(value.get((u, it), 1.0) if weights == "rating" else 1.0)
    X = csr_matrix((data, (r_idx, c_idx)), shape=(len(users), len(items)))
    return normalize(X), items  # PerBlur: sklearn.preprocessing.normalize (L2 rows)


def _scores(clf, X) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def _bootstrap_ci(y: np.ndarray, s: np.ndarray, n_boot: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def run(
    train_csv: Path, users_csv: Path, test_lists: Path, train_pool_lists: Path,
    out_dir: Path, *, weights: str = "binary", ratings_dat: Path | None = None,
    seed: int = 2024, n_boot: int = 1000,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gender = {r["user_id"]: r["gender"] for r in csv.DictReader(Path(users_csv).open(encoding="utf-8"))}

    tr_users, te_users = _user_ids(train_pool_lists), _user_ids(test_lists)
    X, items = build_history_matrix(train_csv, tr_users + te_users, weights=weights,
                                    ratings_dat=ratings_dat)
    n_tr = len(tr_users)
    X_tr, X_te = X[:n_tr], X[n_tr:]
    y_tr = np.array([gender[u] == "male" for u in tr_users])
    y_te = np.array([gender[u] == "male" for u in te_users])

    table, results = [], {}
    for name, make in ESTIMATORS.items():
        for variant, bal in (("default", None), ("balanced", "balanced")):
            if name.startswith("mlp") and bal:      # MLPClassifier has no class_weight
                continue
            clf = make(seed, bal).fit(X_tr, y_tr)
            pred, score = clf.predict(X_te), _scores(clf, X_te)
            key = f"{name}_{variant}"
            row = {
                "classifier": name, "variant": variant,
                "bacc": float(balanced_accuracy_score(y_te, pred)),
                "f1_macro": float(f1_score(y_te, pred, average="macro")),
                "accuracy": float((pred == y_te).mean()),
                "auc": float(roc_auc_score(y_te, score)),
                "auc_ci95": _bootstrap_ci(y_te, score, n_boot, seed),
                "pred_male_share": float(pred.mean()),
            }
            table.append({"key": key, **row})
            results[key] = {
                **row,
                "predictions": [
                    {"user_id": u, "y_true": "male" if t else "female",
                     "y_pred": "male" if p else "female", "score": float(s)}
                    for u, t, p, s in zip(te_users, y_te, pred, score)
                ],
            }
            print(f"[{key:<22}] BAcc {row['bacc']:.3f} | AUC {row['auc']:.3f} "
                  f"CI [{row['auc_ci95'][0]:.3f}, {row['auc_ci95'][1]:.3f}] | pred male {row['pred_male_share']:.0%}",
                  flush=True)

    protocol = {
        "attacker": "raw interaction history (Slokom et al. 2021 / PerBlur setup)",
        "weights": weights, "row_normalisation": "L2 (sklearn.preprocessing.normalize)",
        "split": "fixed_single_split", "attribute": "gender",
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "n_items": len(items),
        "seed": seed, "n_bootstrap": n_boot,
        "train_csv": str(Path(train_csv).resolve()),
        "test_lists": str(Path(test_lists).resolve()),
        "train_pool_lists": str(Path(train_pool_lists).resolve()),
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    for key, res in results.items():
        (out_dir / f"attack_metrics_history_{weights}_{key}.json").write_text(
            json.dumps({"protocol": protocol, **res}, indent=2) + "\n", encoding="utf-8")
    (out_dir / f"summary_{weights}.json").write_text(
        json.dumps({"protocol": protocol, "table": table}, indent=2) + "\n", encoding="utf-8")
    return {"protocol": protocol, "table": table}


def main() -> None:
    p = argparse.ArgumentParser(description="Gender AIA from raw interaction history (PerBlur-style)")
    p.add_argument("--train-csv", type=Path, default=Path("data/ml-1m-311/splits/train.csv"))
    p.add_argument("--users-csv", type=Path, default=Path("data/ml-1m-311/splits/users.csv"))
    p.add_argument("--test-lists", type=Path, required=True)
    p.add_argument("--train-pool-lists", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--weights", choices=["binary", "rating"], default="binary")
    p.add_argument("--ratings-dat", type=Path, default=Path("dataset/ml-1m/ratings.dat"))
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    args = p.parse_args()
    run(args.train_csv, args.users_csv, args.test_lists, args.train_pool_lists, args.out_dir,
        weights=args.weights, ratings_dat=args.ratings_dat, seed=args.seed, n_boot=args.n_bootstrap)


if __name__ == "__main__":
    main()
