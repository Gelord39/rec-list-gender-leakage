"""Pick real users' rec lists as few-shot exemplars (replaces synthetic examples).

Meeting 2026-08-17 action item: "Few-shot use real user as example."

Selection strategy (agreed 2026-08-23): random with a fixed seed among users
with a reasonably long train history (>=50 interactions by default) — a
data-quality filter, not a cherry-pick for gender-distinctiveness, so the
demonstration examples are reproducible and not outcome-biased. Still 2
examples total (1 male, 1 female) to isolate "real vs synthetic" from
"number of shots" when comparing against the old prompt.

Excludes any user already present in an evaluation `lists.jsonl` so the
exemplars never leak into the attacked set. Writes
``src/llm/few_shot_examples.json``, loaded by ``src/llm/prompts.py``.

Example:
  uv run python -m src.llm.select_few_shot_examples \\
    --config configs/bpr_ml1m_311.yaml \\
    --checkpoint runs/EXP-003_bpr_d32_311/checkpoint/BPR-Aug-17-2026_00-12-47.pth \\
    --users-csv data/ml-1m-311/splits/users.csv \\
    --items-csv data/ml-1m-311/splits/items.csv \\
    --train-csv data/ml-1m-311/splits/train.csv \\
    --exclude-lists runs/EXP-003_bpr_d32_311/lists.jsonl \\
    --out src/llm/few_shot_examples.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from src.export.export_lists import _load_model, _topk_for_users

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parent / "few_shot_examples.json"


def _load_excluded_user_ids(lists_paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in lists_paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                excluded.add(json.loads(line)["user_id"])
    return excluded


def select_examples(
    config_file: Path,
    checkpoint: Path,
    users_csv: Path,
    items_csv: Path,
    train_csv: Path,
    exclude_lists: list[Path],
    *,
    min_train_interactions: int = 50,
    topk: int = 10,
    seed: int = 2024,
    device: str | None = "cpu",
) -> list[dict]:
    root = _PROJECT_ROOT
    users_csv = users_csv if users_csv.is_absolute() else root / users_csv
    items_csv = items_csv if items_csv.is_absolute() else root / items_csv
    train_csv = train_csv if train_csv.is_absolute() else root / train_csv

    users = pd.read_csv(users_csv)
    items = pd.read_csv(items_csv)
    train = pd.read_csv(train_csv)
    users["user_id"] = users["user_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)
    train["user_id"] = train["user_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)

    title = items.set_index("item_id")["movie_title"].to_dict()
    gender = users.set_index("user_id")["gender"].to_dict()

    excluded = _load_excluded_user_ids(
        [p if p.is_absolute() else root / p for p in exclude_lists]
    )

    train_counts = train.groupby("user_id").size()
    history_by_user = train.groupby("user_id")["item_id"].apply(list).to_dict()
    history_sets = {u: set(v) for u, v in history_by_user.items()}

    eligible = [
        u
        for u in users["user_id"]
        if u not in excluded
        and u in train_counts.index
        and train_counts[u] >= min_train_interactions
    ]

    picks: dict[str, str] = {}
    for target_gender in ("male", "female"):
        pool = sorted(u for u in eligible if gender.get(u) == target_gender)
        rng = random.Random(seed)
        rng.shuffle(pool)
        if not pool:
            raise RuntimeError(f"No eligible {target_gender} users left after exclusion")
        picks[target_gender] = pool[0]

    _, dataset, model, torch_device = _load_model(
        config_file, checkpoint, device or "cpu"
    )
    topk_map = _topk_for_users(
        model, dataset, torch_device, list(picks.values()), history_sets, topk
    )

    examples = []
    for g, uid in picks.items():
        rec_titles = [title[i] for i in topk_map[uid] if i in title]
        examples.append(
            {
                "user_id": uid,
                "gender": g,
                "rec_titles": rec_titles,
                "n_train_interactions": int(train_counts[uid]),
            }
        )
    return examples


def main() -> None:
    root = _PROJECT_ROOT
    p = argparse.ArgumentParser(description="Select real-user few-shot examples")
    p.add_argument("--config", type=Path, default=Path("configs/bpr_ml1m_311.yaml"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--users-csv", type=Path, default=Path("data/ml-1m-311/splits/users.csv")
    )
    p.add_argument(
        "--items-csv", type=Path, default=Path("data/ml-1m-311/splits/items.csv")
    )
    p.add_argument(
        "--train-csv", type=Path, default=Path("data/ml-1m-311/splits/train.csv")
    )
    p.add_argument(
        "--exclude-lists",
        type=Path,
        nargs="*",
        default=[Path("runs/EXP-003_bpr_d32_311/lists.jsonl")],
        help="lists.jsonl files whose users must NOT be picked as exemplars",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_FIXTURE_PATH)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--min-train-interactions", type=int, default=50)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    examples = select_examples(
        args.config,
        args.checkpoint,
        args.users_csv,
        args.items_csv,
        args.train_csv,
        args.exclude_lists,
        min_train_interactions=args.min_train_interactions,
        topk=args.topk,
        seed=args.seed,
        device=args.device,
    )

    out_path = args.out if args.out.is_absolute() else root / args.out
    out_path.write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(examples, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
