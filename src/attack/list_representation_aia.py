"""List-representation attackers: can other list representations beat mean-pooling?

Motivation (2026-09-13/14 pilot): the list-derived (mean-pool) attacker in
`fixed_split_aia.py` sits at BAcc ~0.52 / AUC ~0.68 on EXP-004. Diagnostics
showed (a) BAcc is depressed by threshold collapse on the 70/30 class split
(the classifier predicts male for ~96% of test users), and (b) mean-pooling
averages item vectors onto a shared popularity direction. This module compares,
under the SAME fixed single-split protocol as `fixed_split_aia.py` (train on the
train-pool lists, test on the shared 500-user `lists.jsonl`):

  - ``history_multi_hot``: the user's raw train interactions (0/1 over all items,
    L2-normalised) — the Slokom et al. (2021) / PerBlur attacker. Needs no access
    to the trained recommender and is the true upper bound: it is the data the
    user embedding was learned *from*. Binary only; the rating-weighted variant
    of the original paper lives in ``history_aia.py`` and changes AUC by <0.005,
    which is why our implicit-feedback binarisation is kept.
  - ``true_user_embedding``: the model's scoring user embedding (a 32-d learned
    compression of that same history)
  - ``ld_mean_pool``: current list-derived attacker (mean of rec-list item embeddings)
  - ``multi_hot``: 0/1 vector over items seen in train-pool lists (no embeddings)
  - ``tfidf_multi_hot``: same, IDF-weighted (down-weights popular items)
  - ``genre_histogram``: 18-dim share of ML-1M genres in the list (each movie
    contributes 1, split evenly across its genre tags; divided by k). Public
    metadata only, interpretable — LR coefficients per genre are saved.
  - ``title_text_embedding``: mean of sentence embeddings of the rec-list
    titles (default ``sentence-transformers/all-MiniLM-L6-v2``, titles as-is
    incl. year). Recommender-independent semantic item representation.
    Needs the optional ``sentence-transformers`` package; pass
    ``--no-text-embedding`` to skip (recorded in ``summary.json``).
  - ``inversion``: recommender inversion, per user
        u_hat = argmax_u  sum_{i in list} log softmax(u . I)_i  -  lam * ||u||^2
    lam chosen by 5-fold CV AUC on the train pool (labels never touch the test set)

Each input is evaluated with two logistic-regression variants:
  - ``default``: C=1, no class weighting — matches the production fixed-split LR
  - ``balanced_cv``: class_weight="balanced", C chosen by 5-fold CV AUC on the train pool

Every result includes a 95% bootstrap CI on test AUC and a *paired* bootstrap CI
of the AUC difference vs ``ld_mean_pool`` (same variant, same resampled users),
because at n_test=500 an AUC difference of ~0.025 is only ~1 standard error.

Novel to this project (not RAID / Slokom / Staab protocol) — log in
`docs/experiment_setup.md §11 Traceability` if adopted.

Example (EXP-004):
  uv run python -m src.attack.list_representation_aia \\
    --config configs/lightgcn_ml1m_311.yaml --model LightGCN \\
    --checkpoint runs/EXP-004_lightgcn_d32_311/checkpoint/LightGCN-*.pth \\
    --test-lists runs/EXP-004_lightgcn_d32_311/lists.jsonl \\
    --train-pool-lists runs/EXP-004_lightgcn_d32_311/lists_train_pool.jsonl \\
    --out-dir runs/EXP-004_lightgcn_d32_311/list_repr_500
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
import torch
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from src.attack.history_aia import build_history_matrix
from src.export.export_lists import _load_model
from src.train.train_recbole import get_user_item_embeddings

CS_GRID = [0.001, 0.01, 0.1, 1, 10]
CLASSIFIERS = ("lr", "mlp", "svm_rbf", "svm_linear")
LAMBDA_GRID = [0.001, 0.01, 0.1]
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _read_lists(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def _bootstrap_auc(
    y: np.ndarray, score: np.ndarray, base_score: np.ndarray | None, n_boot: int, seed: int
) -> dict[str, Any]:
    """95% bootstrap CI of AUC, and (if base_score given) of the paired AUC difference."""
    if n_boot <= 0:  # caller does not want CIs (e.g. a control arm averaged over seeds)
        return {}
    rng = np.random.default_rng(seed)
    aucs, deltas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        a = roc_auc_score(y[idx], score[idx])
        aucs.append(a)
        if base_score is not None:
            deltas.append(a - roc_auc_score(y[idx], base_score[idx]))
    out: dict[str, Any] = {"auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]}
    if base_score is not None:
        out["delta_auc_vs_ld_mean_pool_ci95"] = [
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ]
    return out


def _make_estimators(seed: int, cv: StratifiedKFold, classifiers: tuple[str, ...]) -> dict[str, Any]:
    """The (classifier x balance) grid.

    ``default`` = library defaults, no class weighting. ``balanced`` = the same
    estimator with balanced class weights and NOTHING else changed, so the
    balance axis is not confounded with hyper-parameter tuning. ``lr_balanced_cv``
    is kept as a third LR variant only for continuity with the 2026-09-14 run,
    which tuned C and switched on class weights in one step.

    MLPClassifier has no ``class_weight`` parameter; its balanced variant passes
    ``sample_weight=compute_sample_weight("balanced", y)`` to ``fit`` instead
    (equivalent re-weighting of the loss).
    """
    est: dict[str, Any] = {}
    if "lr" in classifiers:
        est["lr_default"] = LogisticRegression(max_iter=5000)
        est["lr_balanced"] = LogisticRegression(max_iter=5000, class_weight="balanced")
        est["lr_balanced_cv"] = LogisticRegressionCV(
            Cs=CS_GRID, cv=cv, scoring="roc_auc", class_weight="balanced", max_iter=5000
        )
    if "mlp" in classifiers:
        # same architecture as src/attack/embedding_aia.py (RAID-style)
        est["mlp_default"] = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)
        est["mlp_balanced"] = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)
    if "svm_rbf" in classifiers:
        est["svm_rbf_default"] = SVC(kernel="rbf", random_state=seed)
        est["svm_rbf_balanced"] = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
    if "svm_linear" in classifiers:
        est["svm_linear_default"] = LinearSVC(max_iter=5000, random_state=seed)
        est["svm_linear_balanced"] = LinearSVC(max_iter=5000, class_weight="balanced", random_state=seed)
    return est


def _decision_scores(clf: Any, X) -> np.ndarray:
    """Higher = more male. MLPClassifier has no decision_function, use P(male)."""
    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(X))
    return np.asarray(clf.predict_proba(X))[:, 1]


def _evaluate(
    X_tr, y_tr: np.ndarray, X_te, y_te: np.ndarray, test_ids: list[str], *,
    dense: bool, seed: int, n_boot: int, base_scores: dict[str, np.ndarray] | None,
    classifiers: tuple[str, ...] = CLASSIFIERS,
    feature_names: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit every (classifier x balance) variant; return per-variant results and test scores."""
    if dense:
        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    results, scores = {}, {}
    for variant, clf in _make_estimators(seed, cv, classifiers).items():
        if variant == "mlp_balanced":
            clf.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
        else:
            clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        score = _decision_scores(clf, X_te)
        scores[variant] = score
        res: dict[str, Any] = {
            "classifier": variant.rsplit("_", 1)[0] if variant.endswith(("_default", "_balanced")) else "lr",
            "balance": "balanced" if "balanced" in variant else "default",
            "C": float(clf.C_[0]) if hasattr(clf, "C_") else None,
            "class_weight": getattr(clf, "class_weight", None),
            "metrics": {
                "bacc": float(balanced_accuracy_score(y_te, pred)),
                "f1_macro": float(f1_score(y_te, pred, average="macro")),
                "accuracy": float((pred == y_te).mean()),
                "auc": float(roc_auc_score(y_te, score)),
                "recall_male": float(recall_score(y_te, pred, pos_label=True)),
                "recall_female": float(recall_score(y_te, pred, pos_label=False)),
                "pred_male_share": float(pred.mean()),
                **_bootstrap_auc(
                    y_te, score, None if base_scores is None else base_scores.get(variant), n_boot, seed
                ),
            },
            "predictions": [
                {"user_id": u, "y_true": "male" if t else "female",
                 "y_pred": "male" if p else "female", "score": float(s)}
                for u, t, p, s in zip(test_ids, y_te, pred, score)
            ],
        }
        if feature_names is not None and variant.startswith("lr_"):
            # On StandardScaler-transformed features; positive = pushes towards male
            res["coefficients_standardized"] = dict(zip(feature_names, map(float, clf.coef_[0])))
        results[variant] = res
    return results, scores


def rank_pool(item_emb: np.ndarray, L: np.ndarray) -> np.ndarray:
    """NDCG-style rank-weighted mean of the rec-list item embeddings (1/log2(rank+1)).

    Same discount as `src/attack/list_embedding_aia.py`'s `rank` pooling, so the
    two modules produce the same feature; the rank-1 item counts more than the 10th.
    """
    w = np.array([1.0 / np.log2(r + 1) for r in range(1, L.shape[1] + 1)])
    w = w / w.sum()
    return np.einsum("nkd,k->nd", item_emb[L], w)


def genre_histograms(rows: list[dict[str, Any]], items_csv: Path) -> tuple[np.ndarray, list[str]]:
    """Share of each genre in a user's rec list (each movie splits 1 across its genre tags)."""
    item_genres = {
        r["item_id"]: r["genres"].split()
        for r in csv.DictReader(Path(items_csv).open(encoding="utf-8"))
    }
    names = sorted({g for gs in item_genres.values() for g in gs})
    col = {g: k for k, g in enumerate(names)}
    X = np.zeros((len(rows), len(names)))
    for n, row in enumerate(rows):
        for tok in row["rec_item_ids"]:
            gs = item_genres.get(str(tok), [])
            for g in gs:
                X[n, col[g]] += 1.0 / len(gs)
        X[n] /= len(row["rec_item_ids"])
    return X, names


def title_text_embeddings(rows: list[dict[str, Any]], title_vec: dict[str, np.ndarray]) -> np.ndarray:
    """Mean of the rec-list titles' sentence embeddings."""
    return np.stack([np.mean([title_vec[t] for t in row["rec_titles"]], axis=0) for row in rows])


def encode_titles(titles: list[str], model_name: str) -> dict[str, np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "title_text_embedding needs `sentence-transformers` (uv add sentence-transformers), "
            "or pass --no-text-embedding"
        ) from exc
    model = SentenceTransformer(model_name, device="cpu")
    vecs = model.encode(titles, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    return dict(zip(titles, np.asarray(vecs)))


def _centred_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def _topk_overlap(vecs: np.ndarray, item_emb: np.ndarray, lists: np.ndarray) -> float:
    """Mean fraction of the observed top-k list reproduced by top-k(vec . I) ([PAD] masked, no history exclusion)."""
    scores = vecs @ item_emb.T
    scores[:, 0] = -np.inf
    k = lists.shape[1]
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    return float(np.mean([len(set(t) & set(l)) / k for t, l in zip(top, lists)]))


def reconstruction_diagnostics(
    X_tr: np.ndarray, X_te: np.ndarray, u_tr: np.ndarray, u_te: np.ndarray,
    y_tr: np.ndarray, y_te: np.ndarray, *, alpha: float = 1.0,
) -> dict[str, Any]:
    """How well a list-derived vector recovers the true user embedding u on test users.

    Direct centred cosine (same 32-d space), then a ridge map X -> u fit on the
    train pool: linear R^2, centred cosine after mapping, Pearson correlation of
    the true-u LR attacker's gender score on mapped vs true u, that attacker's
    transfer AUC, and re-identification of each user's own u among all test users.
    """
    ridge = Ridge(alpha=alpha).fit(X_tr, u_tr)
    u_map = ridge.predict(X_te)
    r2 = 1 - ((u_te - u_map) ** 2).sum() / ((u_te - u_te.mean(0)) ** 2).sum()
    sc = StandardScaler().fit(u_tr)
    lr = LogisticRegression(max_iter=5000).fit(sc.transform(u_tr), y_tr)
    f_true = lr.decision_function(sc.transform(u_te))
    f_map = lr.decision_function(sc.transform(u_map))
    a = u_map - u_map.mean(0)
    b = u_te - u_te.mean(0)
    sim = (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (b / np.linalg.norm(b, axis=1, keepdims=True)).T
    ranks = (sim > np.diag(sim)[:, None]).sum(1) + 1
    return {
        "centred_cosine_direct_mean": float(_centred_cosine(X_te, u_te).mean()) if X_te.shape[1] == u_te.shape[1] else None,
        "ridge_alpha": alpha,
        "linear_r2_ridge_to_u": float(r2),
        "centred_cosine_after_ridge_mean": float(_centred_cosine(u_map, u_te).mean()),
        "gender_score_pearson_with_true_u": float(np.corrcoef(f_map, f_true)[0, 1]),
        "transfer_auc_true_u_attacker": float(roc_auc_score(y_te, f_map)),
        "reid_top1": float((ranks == 1).mean()),
        "reid_median_rank": float(np.median(ranks)),
        "reid_n_candidates": int(len(u_te)),
    }


def invert_users(item_emb: np.ndarray, lists: np.ndarray, lam: float, *, steps: int = 400, lr: float = 0.05, seed: int = 2024) -> np.ndarray:
    """Recommender inversion: MAP-style estimate of each user's embedding from their top-k list.

    Maximises sum_{i in list} log softmax(u . I)_i - lam ||u||^2 per user
    (full-batch Adam, all users at once). Index 0 is RecBole's [PAD] item and
    is masked out of the softmax. Uses only the item table and the list —
    same threat model as the list-derived attacker.
    """
    torch.manual_seed(seed)
    items = torch.tensor(item_emb, dtype=torch.float32)
    mask = torch.zeros(items.shape[0])
    mask[0] = -1e9
    L = torch.tensor(lists)
    U = torch.zeros(len(lists), items.shape[1], requires_grad=True)
    opt = torch.optim.Adam([U], lr=lr)
    for _ in range(steps):
        logp = torch.log_softmax(U @ items.T + mask, dim=1).gather(1, L).sum(1)
        loss = -logp.mean() + lam * (U ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return U.detach().numpy()


def run_list_representation_pipeline(
    config_file: Path, checkpoint: Path, model_name: str,
    test_lists_path: Path, train_pool_lists_path: Path, out_dir: Path,
    *, seed: int = 2024, n_boot: int = 1000, device: str | None = None,
    items_csv: Path = Path("data/ml-1m-311/splits/items.csv"),
    train_csv: Path = Path("data/ml-1m-311/splits/train.csv"),
    text_model: str | None = DEFAULT_TEXT_MODEL,
    classifiers: tuple[str, ...] = CLASSIFIERS,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, ds, model, _ = _load_model(config_file, checkpoint, device or "cpu", model_name=model_name)
    user_emb, item_emb = map(np.asarray, get_user_item_embeddings(model))

    tr, te = _read_lists(train_pool_lists_path), _read_lists(test_lists_path)
    item_idx = lambda rows: np.array([[ds.token2id(ds.iid_field, t) for t in r["rec_item_ids"]] for r in rows])
    user_vec = lambda rows: np.stack([user_emb[ds.token2id(ds.uid_field, r["user_id"])] for r in rows])
    y_tr = np.array([r["gender"] == "male" for r in tr])
    y_te = np.array([r["gender"] == "male" for r in te])
    test_ids = [str(r["user_id"]) for r in te]
    L_tr, L_te = item_idx(tr), item_idx(te)
    common = dict(seed=seed, n_boot=n_boot, classifiers=classifiers)

    inputs: dict[str, dict[str, Any]] = {}

    def record(name: str, description: str, res: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
        inputs[name] = {"input": name, "description": description, **(extra or {}), "variants": res}

    res, _ = _evaluate(user_vec(tr), y_tr, user_vec(te), y_te, test_ids, dense=True, base_scores=None, **common)
    record("true_user_embedding", "model's scoring user embedding (reference upper bound)", res)

    res, base_scores = _evaluate(item_emb[L_tr].mean(1), y_tr, item_emb[L_te].mean(1), y_te, test_ids,
                                 dense=True, base_scores=None, **common)
    record("ld_mean_pool", "mean of top-k rec-list item embeddings (current LD attacker)", res)

    tr_ids = [str(r["user_id"]) for r in tr]
    H, hist_items = build_history_matrix(train_csv, tr_ids + test_ids, weights="binary")
    res, _ = _evaluate(H[: len(tr_ids)], y_tr, H[len(tr_ids) :], y_te, test_ids,
                       dense=False, base_scores=base_scores, **common)
    record("history_multi_hot",
           "raw train interactions, 0/1 over all items, L2-normalised (Slokom et al. 2021 attacker)",
           res, {"n_items": len(hist_items), "train_csv": str(Path(train_csv).resolve())})

    res, _ = _evaluate(rank_pool(item_emb, L_tr), y_tr, rank_pool(item_emb, L_te), y_te, test_ids,
                       dense=True, base_scores=base_scores, **common)
    record("ld_rank_pool", "rank-weighted (1/log2(rank+1)) mean of top-k rec-list item embeddings", res)

    vocab = {i: k for k, i in enumerate(sorted(set(L_tr.ravel().tolist())))}

    def multi_hot(L: np.ndarray) -> csr_matrix:
        r, c = zip(*[(n, vocab[i]) for n, row in enumerate(L) for i in row if i in vocab])
        return csr_matrix((np.ones(len(r)), (r, c)), shape=(len(L), len(vocab)))

    M_tr, M_te = multi_hot(L_tr), multi_hot(L_te)
    oov = int(sum(i not in vocab for i in L_te.ravel().tolist()))
    vocab_info = {"vocab_size": len(vocab), "test_list_slots_out_of_vocab": oov, "test_list_slots": int(L_te.size)}
    res, _ = _evaluate(M_tr, y_tr, M_te, y_te, test_ids, dense=False, base_scores=base_scores, **common)
    record("multi_hot", "0/1 over items appearing in train-pool lists", res, vocab_info)

    tfidf = TfidfTransformer().fit(M_tr)
    res, _ = _evaluate(tfidf.transform(M_tr), y_tr, tfidf.transform(M_te), y_te, test_ids,
                       dense=False, base_scores=base_scores, **common)
    record("tfidf_multi_hot", "multi-hot with IDF weighting (sklearn TfidfTransformer, fit on train pool)", res, vocab_info)

    G_tr, genre_names = genre_histograms(tr, items_csv)
    G_te, _ = genre_histograms(te, items_csv)
    res, _ = _evaluate(G_tr, y_tr, G_te, y_te, test_ids, dense=True, base_scores=base_scores,
                       feature_names=genre_names, **common)
    record("genre_histogram", "share of ML-1M genres in the rec list (per-movie split evenly across tags)",
           res, {"genres": genre_names})

    skipped: dict[str, str] = {}
    if text_model is None:
        skipped["title_text_embedding"] = "disabled via --no-text-embedding"
    else:
        titles = sorted({t for row in tr + te for t in row["rec_titles"]})
        title_vec = encode_titles(titles, text_model)
        T_tr, T_te = title_text_embeddings(tr, title_vec), title_text_embeddings(te, title_vec)
        res, _ = _evaluate(T_tr, y_tr, T_te, y_te, test_ids, dense=True, base_scores=base_scores, **common)
        record("title_text_embedding", "mean of sentence embeddings of rec-list titles", res,
               {"text_model": text_model, "n_unique_titles": len(titles), "embedding_dim": int(T_tr.shape[1])})

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    lam_cv = {}
    best_lam, H_tr = None, None
    for lam in LAMBDA_GRID:
        H = invert_users(item_emb, L_tr, lam, seed=seed)
        s = float(cross_val_score(LogisticRegression(max_iter=5000), StandardScaler().fit_transform(H),
                                  y_tr, cv=cv, scoring="roc_auc").mean())
        lam_cv[str(lam)] = s
        if best_lam is None or s > lam_cv[str(best_lam)]:
            best_lam, H_tr = lam, H
    H_te = invert_users(item_emb, L_te, best_lam, seed=seed)
    u_te = user_vec(te)
    u_tr = user_vec(tr)
    P_tr, P_te = item_emb[L_tr].mean(1), item_emb[L_te].mean(1)
    recon = {
        "inversion_u_hat": {
            **reconstruction_diagnostics(H_tr, H_te, u_tr, u_te, y_tr, y_te),
            "list_overlap_topk_u_hat": _topk_overlap(H_te, item_emb, L_te),
        },
        "ld_mean_pool": reconstruction_diagnostics(P_tr, P_te, u_tr, u_te, y_tr, y_te),
        "reference_true_u": {
            "list_overlap_topk_true_u_no_history_exclusion": _topk_overlap(u_te, item_emb, L_te),
            "note": "observed lists exclude seen items, so even true u without exclusion does not reproduce them fully",
        },
    }
    (out_dir / "reconstruction_diagnostics.json").write_text(json.dumps(recon, indent=2) + "\n", encoding="utf-8")
    res, _ = _evaluate(H_tr, y_tr, H_te, y_te, test_ids, dense=True, base_scores=base_scores, **common)
    record("inversion", "recommender inversion u_hat from list + item table, LR trained on u_hat", res, {
        "lambda_grid_train_cv_auc": lam_cv,
        "lambda_chosen": best_lam,
        "reconstruction_diagnostics": recon,
    })

    # upper bound first: raw history, then its learned compression, then list-derived
    order = ["history_multi_hot", "true_user_embedding"]
    inputs = {k: inputs[k] for k in order if k in inputs} | {
        k: v for k, v in inputs.items() if k not in order
    }

    protocol = {
        "split": "fixed_single_split",
        "attribute": "gender",
        "label_classes": ["female", "male"],
        "score": "LR decision_function, positive = male",
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "seed": seed,
        "cv_folds_for_hyperparams": 5,
        "classifiers": list(classifiers),
        "C_grid": CS_GRID,
        "lambda_grid": LAMBDA_GRID,
        "n_bootstrap": n_boot,
        "test_lists": str(Path(test_lists_path).resolve()),
        "train_pool_lists": str(Path(train_pool_lists_path).resolve()),
        "model": model_name,
        "items_csv": str(Path(items_csv).resolve()),
        "train_csv": str(Path(train_csv).resolve()),
        "text_model": text_model,
        "checkpoint": str(Path(checkpoint).resolve()),
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    for name, payload in inputs.items():
        payload["protocol"] = protocol
        (out_dir / f"attack_metrics_{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary = {
        "protocol": protocol,
        "skipped_inputs": skipped,
        "table": [
            {"input": name, "variant": v, "classifier": r["classifier"], "balance": r["balance"],
             **{k: r["metrics"][k] for k in
                ("accuracy", "bacc", "f1_macro", "auc", "auc_ci95",
                 "recall_male", "recall_female", "pred_male_share")},
             "delta_auc_vs_ld_mean_pool_ci95": r["metrics"].get("delta_auc_vs_ld_mean_pool_ci95"),
             "C": r["C"]}
            for name, payload in inputs.items() for v, r in payload["variants"].items()
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Compare list representations for list-derived gender AIA (fixed split)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", required=True, help="RecBole model class name, e.g. BPR, LightGCN")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test-lists", type=Path, required=True)
    p.add_argument("--train-pool-lists", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--device", default=None)
    p.add_argument("--items-csv", type=Path, default=Path("data/ml-1m-311/splits/items.csv"))
    p.add_argument("--train-csv", type=Path, default=Path("data/ml-1m-311/splits/train.csv"),
                   help="interactions for the history_multi_hot attacker")
    p.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    p.add_argument("--classifiers", default=",".join(CLASSIFIERS),
                   help="comma-separated subset of " + ",".join(CLASSIFIERS))
    p.add_argument("--no-text-embedding", action="store_true",
                   help="skip title_text_embedding (e.g. if sentence-transformers is not installed)")
    args = p.parse_args()
    summary = run_list_representation_pipeline(
        args.config, args.checkpoint, args.model, args.test_lists, args.train_pool_lists, args.out_dir,
        seed=args.seed, n_boot=args.n_bootstrap, device=args.device,
        items_csv=args.items_csv, train_csv=args.train_csv, text_model=None if args.no_text_embedding else args.text_model,
        classifiers=tuple(c.strip() for c in args.classifiers.split(",") if c.strip()),
    )
    for row in summary["table"]:
        d = row["delta_auc_vs_ld_mean_pool_ci95"]
        d_s = f" | ΔAUC vs mean-pool CI [{d[0]:+.3f}, {d[1]:+.3f}]" if d else ""
        print(f"[{row['input']}/{row['variant']}] BAcc {row['bacc']:.3f} | AUC {row['auc']:.3f} "
              f"CI [{row['auc_ci95'][0]:.3f}, {row['auc_ci95'][1]:.3f}]{d_s} | pred male {row['pred_male_share']:.0%}")


if __name__ == "__main__":
    main()
