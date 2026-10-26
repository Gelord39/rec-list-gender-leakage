"""Attribute inference on LLM-generated rec lists (EXP-006, Stairway-style LLMRec).

An LLM recommender has no user / item embeddings, so only the list-based tiers
of the master table apply: T2 (multi-hot, TF-IDF multi-hot over rec-list item
IDs), T3 (genre histogram, MiniLM title embedding). T4 (an LLM reading the
titles) is ``src.llm.list_aia`` pointed at the same ``lists.jsonl``. Two
history references are added:

  - ``history_multi_hot``   the full train history (TH, identical to EXP-003/004)
  - ``prompt_history_multi_hot``  only what the LLM recommender was shown
    (last 10 train + last 10 validation items, Stairway's prompt input) — the
    fair upper bound for an LLMRec, since its lists cannot carry more than that

Everything uses the classifier x balance grid and fixed single split of
``list_representation_aia`` (train on the 3,367-user pool, test on the fixed
500), so numbers sit next to the EXP-003/004 master tables. Lists can be shorter
than 10 (unmatched / hallucinated titles) or empty; empty lists get an all-zero
feature row and are counted in the output.

``compare`` pairs two runs on the same users (neutral vs sensitive prompt):
paired-bootstrap ΔAUC per input and classifier variant, and how much the list
content itself moves (Jaccard per user, genre shift by true gender).

  uv run python -m src.attack.llmrec_aia attack --run-dir runs/EXP-006_llmrec_311/neutral_glm4_9b
  uv run python -m src.attack.llmrec_aia compare \\
      --a runs/EXP-006_llmrec_311/neutral_glm4_9b --b runs/EXP-006_llmrec_311/sensitive_glm4_9b
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

from src.attack.history_aia import build_history_matrix
from src.attack.list_representation_aia import (
    CLASSIFIERS,
    DEFAULT_TEXT_MODEL,
    _evaluate,
    _git_commit,
    _read_lists,
    encode_titles,
)

ITEMS_CSV = Path("data/ml-1m-311/splits/items.csv")
TRAIN_CSV = Path("data/ml-1m-311/splits/train.csv")


def _multi_hot(rows: list[list[str]], vocab: dict[str, int]) -> csr_matrix:
    r, c = [], []
    for n, items in enumerate(rows):
        for i in items:
            if i in vocab:
                r.append(n)
                c.append(vocab[i])
    return csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows), len(vocab)))


def _genre_matrix(rows: list[dict[str, Any]], items_csv: Path) -> tuple[np.ndarray, list[str]]:
    """As ``list_representation_aia.genre_histograms``, but an empty list gives a zero row."""
    item_genres = {r["item_id"]: r["genres"].split() for r in csv.DictReader(Path(items_csv).open(encoding="utf-8"))}
    names = sorted({g for gs in item_genres.values() for g in gs})
    col = {g: k for k, g in enumerate(names)}
    X = np.zeros((len(rows), len(names)))
    for n, row in enumerate(rows):
        for tok in row["rec_item_ids"]:
            gs = item_genres.get(str(tok), [])
            for g in gs:
                X[n, col[g]] += 1.0 / len(gs)
        X[n] /= max(len(row["rec_item_ids"]), 1)
    return X, names


def _title_matrix(rows: list[dict[str, Any]], title_vec: dict[str, np.ndarray], dim: int) -> np.ndarray:
    return np.stack([np.mean([title_vec[t] for t in r["rec_titles"]], axis=0) if r["rec_titles"] else np.zeros(dim)
                     for r in rows])


def _prompt_history(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """user -> items shown in the LLMRec prompt (last-N train + last-10 validation)."""
    return {r["user_id"]: r["prompt_item_ids"] for r in rows}


def run_attack(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = _read_lists(run_dir / "lists_train_pool.jsonl")
    te = _read_lists(run_dir / "lists.jsonl")
    y_tr = np.array([r["gender"] == "male" for r in tr])
    y_te = np.array([r["gender"] == "male" for r in te])
    tr_ids, te_ids = [r["user_id"] for r in tr], [r["user_id"] for r in te]
    classifiers = tuple(c.strip() for c in args.classifiers.split(",") if c.strip())
    common = dict(seed=args.seed, n_boot=args.n_bootstrap, classifiers=classifiers)

    inputs: dict[str, dict[str, Any]] = {}

    def record(name: str, description: str, res: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
        inputs[name] = {"input": name, "description": description, **(extra or {}), "variants": res}
        best = max(res.values(), key=lambda r: r["metrics"]["auc"])
        print(f"[{name}] best AUC {best['metrics']['auc']:.3f} ({best['classifier']}/{best['balance']}), "
              f"lr_balanced AUC {res['lr_balanced']['metrics']['auc']:.3f}" if "lr_balanced" in res
              else f"[{name}] best AUC {best['metrics']['auc']:.3f}", flush=True)

    # --- references: full history (TH) and the LLMRec's own prompt input (TH-prompt)
    H, _ = build_history_matrix(args.train_csv, tr_ids + te_ids, weights="binary")
    res, _ = _evaluate(H[: len(tr_ids)], y_tr, H[len(tr_ids):], y_te, te_ids, dense=False, base_scores=None, **common)
    record("history_multi_hot", "full train history, 0/1, L2-normalised (TH; recommender-independent)", res)

    ph = _prompt_history(tr + te)
    pvocab = {i: k for k, i in enumerate(sorted({i for u in tr_ids for i in ph[u]}))}
    P_tr = normalize(_multi_hot([ph[u] for u in tr_ids], pvocab))
    P_te = normalize(_multi_hot([ph[u] for u in te_ids], pvocab))
    res, _ = _evaluate(P_tr, y_tr, P_te, y_te, te_ids, dense=False, base_scores=None, **common)
    record("prompt_history_multi_hot",
           "only the items in the LLMRec prompt (last N train + last 10 validation), 0/1, L2-normalised", res,
           {"vocab_size": len(pvocab)})

    # --- T2: rec-list item IDs
    L_tr, L_te = [r["rec_item_ids"] for r in tr], [r["rec_item_ids"] for r in te]
    vocab = {i: k for k, i in enumerate(sorted({i for row in L_tr for i in row}))}
    M_tr, M_te = _multi_hot(L_tr, vocab), _multi_hot(L_te, vocab)
    vocab_info = {
        "vocab_size": len(vocab),
        "test_list_slots": int(sum(map(len, L_te))),
        "test_list_slots_out_of_vocab": int(sum(i not in vocab for row in L_te for i in row)),
        "empty_lists_train": int(sum(not row for row in L_tr)),
        "empty_lists_test": int(sum(not row for row in L_te)),
    }
    res, _ = _evaluate(M_tr, y_tr, M_te, y_te, te_ids, dense=False, base_scores=None, **common)
    record("multi_hot", "0/1 over items appearing in train-pool lists", res, vocab_info)
    tfidf = TfidfTransformer().fit(M_tr)
    res, _ = _evaluate(tfidf.transform(M_tr), y_tr, tfidf.transform(M_te), y_te, te_ids,
                       dense=False, base_scores=None, **common)
    record("tfidf_multi_hot", "multi-hot with IDF weighting (fit on train pool)", res, vocab_info)

    # --- T3: public metadata / text
    G_tr, names = _genre_matrix(tr, args.items_csv)
    G_te, _ = _genre_matrix(te, args.items_csv)
    res, _ = _evaluate(G_tr, y_tr, G_te, y_te, te_ids, dense=True, base_scores=None, feature_names=names, **common)
    record("genre_histogram", "share of ML-1M genres in the rec list", res, {"genres": names})

    skipped = {}
    if args.no_text_embedding:
        skipped["title_text_embedding"] = "disabled via --no-text-embedding"
    else:
        titles = sorted({t for row in tr + te for t in row["rec_titles"]})
        title_vec = encode_titles(titles, args.text_model)
        dim = len(next(iter(title_vec.values())))
        T_tr, T_te = _title_matrix(tr, title_vec, dim), _title_matrix(te, title_vec, dim)
        res, _ = _evaluate(T_tr, y_tr, T_te, y_te, te_ids, dense=True, base_scores=None, **common)
        record("title_text_embedding", "mean of sentence embeddings of rec-list titles", res,
               {"text_model": args.text_model, "n_unique_titles": len(titles)})

    protocol = {
        "split": "fixed_single_split", "attribute": "gender", "score": "decision_function, positive = male",
        "n_train": len(tr), "n_test": len(te), "seed": args.seed, "classifiers": list(classifiers),
        "n_bootstrap": args.n_bootstrap, "run_dir": str(run_dir.resolve()),
        "generated_at_iso": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
    }
    for name, payload in inputs.items():
        payload["protocol"] = protocol
        (out_dir / f"attack_metrics_{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = {
        "protocol": protocol, "skipped_inputs": skipped,
        "table": [
            {"input": name, "variant": v, "classifier": r["classifier"], "balance": r["balance"],
             **{k: r["metrics"][k] for k in ("accuracy", "bacc", "f1_macro", "auc", "auc_ci95",
                                             "recall_male", "recall_female", "pred_male_share")}}
            for name, payload in inputs.items() for v, r in payload["variants"].items()
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ compare
def _scores(path: Path, variant: str) -> dict[str, tuple[bool, float]]:
    preds = json.loads(path.read_text())["variants"][variant]["predictions"]
    return {p["user_id"]: (p["y_true"] == "male", p["score"]) for p in preds}


def run_compare(args: argparse.Namespace) -> None:
    ga, gb = args.a / args.out_subdir, args.b / args.out_subdir
    rng_seed = args.seed
    deltas = []
    for f in sorted(ga.glob("attack_metrics_*.json")):
        if not (gb / f.name).exists():
            continue
        name = f.name[len("attack_metrics_"):-len(".json")]
        for variant in json.loads(f.read_text())["variants"]:
            sa, sb = _scores(f, variant), _scores(gb / f.name, variant)
            users = sorted(set(sa) & set(sb))
            y = np.array([sa[u][0] for u in users])
            a = np.array([sa[u][1] for u in users])
            b = np.array([sb[u][1] for u in users])
            rng = np.random.default_rng(rng_seed)
            d = []
            for _ in range(args.n_bootstrap):
                idx = rng.integers(0, len(users), len(users))
                if len(np.unique(y[idx])) == 2:
                    d.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
            deltas.append({"input": name, "variant": variant, "n": len(users),
                           "auc_a": float(roc_auc_score(y, a)), "auc_b": float(roc_auc_score(y, b)),
                           "delta_auc_b_minus_a": float(roc_auc_score(y, b) - roc_auc_score(y, a)),
                           "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]})

    # content shift on all users of both runs (test + train pool)
    rows_a = {r["user_id"]: r for f in ("lists.jsonl", "lists_train_pool.jsonl") for r in _read_lists(args.a / f)}
    rows_b = {r["user_id"]: r for f in ("lists.jsonl", "lists_train_pool.jsonl") for r in _read_lists(args.b / f)}
    users = sorted(set(rows_a) & set(rows_b))
    jac = {}
    for u in users:
        A, B = set(rows_a[u]["rec_item_ids"]), set(rows_b[u]["rec_item_ids"])
        jac[u] = len(A & B) / len(A | B) if A | B else 1.0
    G_a, names = _genre_matrix([rows_a[u] for u in users], args.items_csv)
    G_b, _ = _genre_matrix([rows_b[u] for u in users], args.items_csv)
    gender = np.array([rows_a[u]["gender"] for u in users])
    shift = {}
    for g in ("male", "female"):
        m = gender == g
        shift[g] = {
            "n": int(m.sum()),
            "mean_jaccard": float(np.mean([jac[u] for u, k in zip(users, m) if k])),
            "share_identical_lists": float(np.mean([jac[u] == 1.0 for u, k in zip(users, m) if k])),
            "genre_share_b_minus_a": {n: float(v) for n, v in zip(names, (G_b[m] - G_a[m]).mean(0))},
        }
    out = {"a": str(args.a), "b": str(args.b), "auc_deltas": deltas,
           "content_shift": {"n_users": len(users), "by_gender": shift},
           "generated_at_iso": datetime.now(timezone.utc).isoformat()}
    out_path = args.out or (args.b.parent / f"compare_{args.a.name}__vs__{args.b.name}.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for r in deltas:
        if r["variant"] in ("lr_balanced", "svm_rbf_balanced"):
            print(f"[{r['input']}/{r['variant']}] AUC {r['auc_a']:.3f} -> {r['auc_b']:.3f} "
                  f"Δ {r['delta_auc_b_minus_a']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
    for g, s in shift.items():
        top = sorted(s["genre_share_b_minus_a"].items(), key=lambda kv: -abs(kv[1]))[:4]
        print(f"{g}: Jaccard {s['mean_jaccard']:.3f}, identical {s['share_identical_lists']:.1%}, "
              f"top genre shifts {', '.join(f'{k} {v:+.3f}' for k, v in top)}")
    print(f"-> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("attack")
    a.add_argument("--run-dir", type=Path, required=True, help="e.g. runs/EXP-006_llmrec_311/neutral_glm4_9b")
    a.add_argument("--out-subdir", default="attack_grid_500")
    a.add_argument("--items-csv", type=Path, default=ITEMS_CSV)
    a.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    a.add_argument("--classifiers", default=",".join(CLASSIFIERS))
    a.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    a.add_argument("--no-text-embedding", action="store_true")
    a.add_argument("--seed", type=int, default=2024)
    a.add_argument("--n-bootstrap", type=int, default=1000)
    c = sub.add_parser("compare")
    c.add_argument("--a", type=Path, required=True, help="baseline run dir (e.g. neutral)")
    c.add_argument("--b", type=Path, required=True, help="comparison run dir (e.g. sensitive)")
    c.add_argument("--out-subdir", default="attack_grid_500")
    c.add_argument("--items-csv", type=Path, default=ITEMS_CSV)
    c.add_argument("--out", type=Path, default=None)
    c.add_argument("--seed", type=int, default=2024)
    c.add_argument("--n-bootstrap", type=int, default=1000)
    args = p.parse_args()
    run_attack(args) if args.cmd == "attack" else run_compare(args)


if __name__ == "__main__":
    main()
