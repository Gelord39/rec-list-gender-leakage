"""ML-1M preprocessing for Phase A (RAID attack + Stairway-style filters).

Protocol (see docs/experiment_setup.md):
- Drop users with missing/non-binary gender; drop age < 18 (ML-1M age code 1)
- Age buckets (Stairway): 18-24 / 25-49 / >=50
- Keep ratings >= 3 as implicit positives
- Drop items without title
- Iterative 5-core on users & items
- Split: ``loo`` (RAID) or ``global_311`` (Stairway global temporal 3:1:1)
- After split: drop users/items with <= t train interactions (t=5 if enough test users)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# ML-1M native age codes → Stairway buckets (age code 1 = under 18 → dropped)
AGE_BUCKET = {
    18: "18-24",
    25: "25-49",
    35: "25-49",
    45: "25-49",
    50: ">=50",
    56: ">=50",
}

GENDER_MAP = {"M": "male", "F": "female"}

SplitMode = Literal["loo", "global_311"]


@dataclass(frozen=True)
class PreprocessConfig:
    raw_dir: Path
    out_dir: Path
    rating_threshold: int = 3
    core_k: int = 5
    post_split_t: int = 5
    min_test_users: int = 500
    dataset_name: str = "ml-1m"
    split: SplitMode = "loo"


def _load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(
        raw_dir / "users.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1",
    )
    items = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["item_id", "movie_title", "genres"],
        encoding="latin-1",
    )
    ratings = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    return users, items, ratings


def _filter_users(users: pd.DataFrame) -> pd.DataFrame:
    out = users.copy()
    out = out[out["gender"].isin(GENDER_MAP)]
    out = out[out["age"].isin(AGE_BUCKET)]
    out["gender"] = out["gender"].map(GENDER_MAP)
    out["age"] = out["age"].map(AGE_BUCKET)
    return out[["user_id", "gender", "age"]].reset_index(drop=True)


def _filter_items(items: pd.DataFrame) -> pd.DataFrame:
    out = items.copy()
    out["movie_title"] = out["movie_title"].astype(str).str.strip()
    out = out[out["movie_title"].ne("") & out["movie_title"].ne("nan")]
    # RecBole token_seq: space-separated
    out["genres"] = out["genres"].astype(str).str.replace("|", " ", regex=False)
    return out[["item_id", "movie_title", "genres"]].reset_index(drop=True)


def _k_core(inter: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iterative user & item k-core filter."""
    cur = inter
    while True:
        uc = cur.groupby("user_id").size()
        ic = cur.groupby("item_id").size()
        keep_u = uc[uc >= k].index
        keep_i = ic[ic >= k].index
        nxt = cur[cur["user_id"].isin(keep_u) & cur["item_id"].isin(keep_i)]
        if len(nxt) == len(cur):
            return nxt.reset_index(drop=True)
        cur = nxt


def _leave_one_out(
    inter: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user temporal LOO: last → test, second-last → val, rest → train."""
    ordered = inter.sort_values(
        ["user_id", "timestamp", "item_id"], kind="mergesort"
    ).reset_index(drop=True)
    # Need ≥3 interactions per user for train/val/test; 5-core already implies ≥5
    ranks = ordered.groupby("user_id").cumcount(ascending=False)
    # ranks: 0 = last (test), 1 = second-last (val), >=2 = train
    test = ordered[ranks == 0].copy()
    val = ordered[ranks == 1].copy()
    train = ordered[ranks >= 2].copy()
    return train, val, test


def _global_temporal_311(
    inter: pd.DataFrame,
    train_prop: float = 0.6,
    test_prop: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stairway / Meng-style global timeline split (3:1:1).

    Sort all interactions by ``timestamp`` ascending, then:
    - train = earliest ``train_prop`` (default 60%)
    - test  = latest ``test_prop`` (default 20%)
    - val   = the middle remainder (default 20%)

    This matches ``theresiavr/stairway-to-fairness`` ``preprocessor.filter_data``:
    no random split, so training never sees later-time interactions.
    """
    ordered = inter.sort_values(
        ["timestamp", "user_id", "item_id"], kind="mergesort"
    ).reset_index(drop=True)
    n = len(ordered)
    train_n = int(round(n * train_prop))
    test_n = int(round(n * test_prop))
    train = ordered.iloc[:train_n].copy()
    test = ordered.iloc[n - test_n :].copy()
    val = ordered.iloc[train_n : n - test_n].copy()

    # Hard guard: no future leakage across splits
    if len(val) and train["timestamp"].max() > val["timestamp"].min():
        raise RuntimeError("Temporal leak: train max timestamp > val min timestamp")
    if len(test) and val["timestamp"].max() > test["timestamp"].min():
        raise RuntimeError("Temporal leak: val max timestamp > test min timestamp")
    if len(test) and train["timestamp"].max() > test["timestamp"].min():
        raise RuntimeError("Temporal leak: train max timestamp > test min timestamp")
    return train, val, test


def _apply_post_split_filter(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    t_init: int,
    min_test_users: int,
    *,
    require_all_splits: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Post-split frequency filter.

    Stairway uses ``count >= min_inter`` on **train** for users/items, then drops
    val/test rows whose user/item is absent from the filtered train set.
    LOO mode additionally requires each kept user to appear in train+val+test.
    """
    # Stairway: keep if train count >= t; LOO path historically used > t via setup
    # wording "remove <= t". For Stairway alignment use >= t when not require_all.
    def _keep_mask(counts: pd.Series, t: int) -> pd.Index:
        if require_all_splits:
            return counts[counts > t].index  # remove <= t
        return counts[counts >= t].index  # Stairway min_inter

    t = t_init
    while True:
        tr, va, te = train.copy(), val.copy(), test.copy()
        while True:
            uc = tr.groupby("user_id").size()
            ic = tr.groupby("item_id").size()
            keep_u = set(_keep_mask(uc, t))
            keep_i = set(_keep_mask(ic, t))
            tr2 = tr[tr["user_id"].isin(keep_u) & tr["item_id"].isin(keep_i)]
            # Stairway: val/test users&items must exist in train
            keep_u = set(tr2["user_id"].unique())
            keep_i = set(tr2["item_id"].unique())
            va2 = va[va["user_id"].isin(keep_u) & va["item_id"].isin(keep_i)]
            te2 = te[te["user_id"].isin(keep_u) & te["item_id"].isin(keep_i)]
            if require_all_splits:
                keep_u = (
                    set(tr2["user_id"].unique())
                    & set(va2["user_id"].unique())
                    & set(te2["user_id"].unique())
                )
                tr2 = tr2[tr2["user_id"].isin(keep_u)]
                va2 = va2[va2["user_id"].isin(keep_u)]
                te2 = te2[te2["user_id"].isin(keep_u)]
            if len(tr2) == len(tr) and len(va2) == len(va) and len(te2) == len(te):
                tr, va, te = tr2, va2, te2
                break
            tr, va, te = tr2, va2, te2

        n_test_users = te["user_id"].nunique()
        if n_test_users >= min_test_users or t <= 0:
            keep_u = set(tr["user_id"].unique())
            keep_i = set(tr["item_id"].unique())
            if require_all_splits:
                keep_u = keep_u & set(va["user_id"].unique()) & set(te["user_id"].unique())
                tr = tr[tr["user_id"].isin(keep_u)]
                va = va[va["user_id"].isin(keep_u)]
                te = te[te["user_id"].isin(keep_u)]
            else:
                va = va[va["user_id"].isin(keep_u) & va["item_id"].isin(keep_i)]
                te = te[te["user_id"].isin(keep_u) & te["item_id"].isin(keep_i)]
            users_f = users[users["user_id"].isin(keep_u)].reset_index(drop=True)
            items_f = items[items["item_id"].isin(keep_i)].reset_index(drop=True)
            return (
                tr.reset_index(drop=True),
                va.reset_index(drop=True),
                te.reset_index(drop=True),
                users_f,
                items_f,
                t,
            )
        t -= 1


def _write_atomic(
    out_dir: Path,
    name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
) -> Path:
    # RecBole expects: {data_path}/{dataset}/{dataset}.train.inter
    atomic = out_dir / "recbole" / name
    atomic.mkdir(parents=True, exist_ok=True)

    inter_header = "user_id:token\titem_id:token\ttimestamp:float\n"
    for split, df in [("train", train), ("valid", val), ("test", test)]:
        path = atomic / f"{name}.{split}.inter"
        with path.open("w", encoding="utf-8") as f:
            f.write(inter_header)
            for row in df.itertuples(index=False):
                f.write(f"{row.user_id}\t{row.item_id}\t{row.timestamp}\n")

    user_path = atomic / f"{name}.user"
    with user_path.open("w", encoding="utf-8") as f:
        f.write("user_id:token\tgender:token\tage:token\n")
        for row in users.itertuples(index=False):
            f.write(f"{row.user_id}\t{row.gender}\t{row.age}\n")

    item_path = atomic / f"{name}.item"
    with item_path.open("w", encoding="utf-8") as f:
        f.write("item_id:token\tmovie_title:token\tgenres:token_seq\n")
        for row in items.itertuples(index=False):
            # titles may contain tabs; replace to keep TSV valid
            title = str(row.movie_title).replace("\t", " ").replace("\n", " ")
            f.write(f"{row.item_id}\t{title}\t{row.genres}\n")

    return atomic


def _compute_stats(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    cfg: PreprocessConfig,
    post_split_t_used: int,
    n_raw: dict[str, int],
) -> dict[str, Any]:
    all_inter = pd.concat([train, val, test], ignore_index=True)
    n_users = int(users["user_id"].nunique())
    n_items = int(items["item_id"].nunique())
    n_inter = int(len(all_inter))
    sparsity = 1.0 - n_inter / (n_users * n_items) if n_users and n_items else None

    train_per_user = train.groupby("user_id").size()

    split_desc = {
        "loo": "per-user temporal leave-one-out",
        "global_311": "global temporal 3:1:1 (Stairway)",
    }[cfg.split]
    return {
        "dataset": cfg.dataset_name,
        "protocol": {
            "rating_threshold": cfg.rating_threshold,
            "core_k": cfg.core_k,
            "split": split_desc,
            "split_mode": cfg.split,
            "post_split_t_requested": cfg.post_split_t,
            "post_split_t_used": post_split_t_used,
            "post_split_rule": (
                "LOO: remove users/items with <= t train interactions; "
                "311/Stairway: keep users/items with train count >= t"
            ),
            "age_buckets": ["18-24", "25-49", ">=50"],
            "gender_values": ["male", "female"],
            "timestamp_ranges": {
                "train_min": int(train["timestamp"].min()),
                "train_max": int(train["timestamp"].max()),
                "valid_min": int(val["timestamp"].min()) if len(val) else None,
                "valid_max": int(val["timestamp"].max()) if len(val) else None,
                "test_min": int(test["timestamp"].min()) if len(test) else None,
                "test_max": int(test["timestamp"].max()) if len(test) else None,
            },
        },
        "raw_counts": n_raw,
        "final": {
            "n_users": n_users,
            "n_items": n_items,
            "n_interactions": n_inter,
            "sparsity": sparsity,
            "gender_counts": users["gender"].value_counts().to_dict(),
            "age_counts": users["age"].value_counts().to_dict(),
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "n_train_users": int(train["user_id"].nunique()),
            "n_val_users": int(val["user_id"].nunique()),
            "n_test_users": int(test["user_id"].nunique()),
            "train_interactions_per_user": {
                "mean": float(train_per_user.mean()),
                "median": float(train_per_user.median()),
                "min": int(train_per_user.min()),
                "max": int(train_per_user.max()),
            },
        },
    }


def preprocess_ml1m(cfg: PreprocessConfig) -> dict[str, Any]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    users_raw, items_raw, ratings_raw = _load_raw(cfg.raw_dir)
    n_raw = {
        "n_users": int(len(users_raw)),
        "n_items": int(len(items_raw)),
        "n_ratings": int(len(ratings_raw)),
    }

    users = _filter_users(users_raw)
    items = _filter_items(items_raw)

    inter = ratings_raw[
        (ratings_raw["rating"] >= cfg.rating_threshold)
        & ratings_raw["user_id"].isin(users["user_id"])
        & ratings_raw["item_id"].isin(items["item_id"])
    ][["user_id", "item_id", "timestamp"]].copy()

    # Deduplicate user-item pairs: keep latest timestamp
    inter = (
        inter.sort_values(["user_id", "item_id", "timestamp"], kind="mergesort")
        .drop_duplicates(["user_id", "item_id"], keep="last")
        .reset_index(drop=True)
    )

    inter = _k_core(inter, cfg.core_k)

    # Align feature tables to k-core survivors
    users = users[users["user_id"].isin(inter["user_id"])].reset_index(drop=True)
    items = items[items["item_id"].isin(inter["item_id"])].reset_index(drop=True)

    if cfg.split == "loo":
        train, val, test = _leave_one_out(inter)
        require_all = True
    elif cfg.split == "global_311":
        train, val, test = _global_temporal_311(inter)
        require_all = False
    else:
        raise ValueError(f"Unknown split mode: {cfg.split}")

    train, val, test, users, items, t_used = _apply_post_split_filter(
        train,
        val,
        test,
        users,
        items,
        t_init=cfg.post_split_t,
        min_test_users=cfg.min_test_users,
        require_all_splits=require_all,
    )

    atomic_dir = _write_atomic(
        cfg.out_dir, cfg.dataset_name, train, val, test, users, items
    )

    splits_dir = cfg.out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(splits_dir / "train.csv", index=False)
    val.to_csv(splits_dir / "valid.csv", index=False)
    test.to_csv(splits_dir / "test.csv", index=False)
    users.to_csv(splits_dir / "users.csv", index=False)
    items.to_csv(splits_dir / "items.csv", index=False)

    stats = _compute_stats(train, val, test, users, items, cfg, t_used, n_raw)
    stats_path = cfg.out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    stats["paths"] = {
        "recbole_dataset_dir": str(atomic_dir),
        "recbole_data_path": str(atomic_dir.parent),
        "splits_dir": str(splits_dir),
        "stats": str(stats_path),
    }
    return stats


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Preprocess ML-1M for RecBole")
    parser.add_argument(
        "--split",
        choices=["loo", "global_311"],
        default="loo",
        help="loo = RAID leave-one-out; global_311 = Stairway 3:1:1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: data/ml-1m (loo) or data/ml-1m-311 (global_311)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = root / "data" / ("ml-1m-311" if args.split == "global_311" else "ml-1m")
    elif not out_dir.is_absolute():
        out_dir = root / out_dir

    cfg = PreprocessConfig(
        raw_dir=root / "dataset" / "ml-1m",
        out_dir=out_dir,
        split=args.split,
    )
    stats = preprocess_ml1m(cfg)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
