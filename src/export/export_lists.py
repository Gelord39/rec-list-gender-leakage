"""Export top-k recommendation lists for LLM attacks / prompts.

Labels (gender/age) are stored for evaluation only — do not put them in attack prompts.

Example (EXP-003, list-only, 100 users):
  uv run python -m src.export.export_lists \\
    --config configs/bpr_ml1m_311.yaml \\
    --checkpoint runs/EXP-003_bpr_d32_311/checkpoint/BPR-Aug-17-2026_00-12-47.pth \\
    --users-csv data/ml-1m-311/splits/users.csv \\
    --items-csv data/ml-1m-311/splits/items.csv \\
    --train-csv data/ml-1m-311/splits/train.csv \\
    --out runs/EXP-003_bpr_d32_311/lists.jsonl \\
    --topk 10 --history-size 0 --max-users 100
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_model, init_seed

from src.train.train_recbole import _resolve_device, _torch_load_compat

torch.load = _torch_load_compat


def _load_model(config_file: Path, checkpoint: Path, device: str, model_name: str = "BPR"):
    root = Path(__file__).resolve().parents[2]
    config_file = config_file if config_file.is_absolute() else root / config_file
    checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint

    config = Config(model=model_name, config_file_list=[str(config_file)])
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        config["data_path"] = str((root / data_path).resolve())
    torch_device = torch.device(_resolve_device(device))
    config["device"] = torch_device

    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)
    model = get_model(config["model"])(config, train_data.dataset).to(torch_device)
    state = torch.load(checkpoint, map_location=torch_device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return config, train_data.dataset, model, torch_device


@torch.no_grad()
def _topk_for_users(
    model,
    dataset,
    device: torch.device,
    user_tokens: list[str],
    history_by_user: dict[str, set[str]],
    topk: int,
) -> dict[str, list[str]]:
    """Return {user_token: [item_token, ...]} top-k excluding train history."""
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    out: dict[str, list[str]] = {}

    for u_tok in user_tokens:
        try:
            u_id = dataset.token2id(uid_field, u_tok)
        except Exception:
            continue
        inter = Interaction({uid_field: torch.tensor([u_id], device=device)})
        scores = model.full_sort_predict(inter).view(-1)
        scores[0] = -np.inf
        for i_tok in history_by_user.get(u_tok, set()):
            try:
                i_id = dataset.token2id(iid_field, i_tok)
                scores[i_id] = -np.inf
            except Exception:
                pass
        k = min(topk, scores.numel() - 1)
        top_ids = torch.topk(scores, k=k).indices.tolist()
        out[u_tok] = [str(dataset.id2token(iid_field, i)) for i in top_ids]
    return out


def export_lists(
    config_file: Path,
    checkpoint: Path,
    users_csv: Path,
    items_csv: Path,
    train_csv: Path,
    out_path: Path,
    *,
    model_name: str = "BPR",
    topk: int = 10,
    history_size: int = 0,
    history_order: str = "recent",
    max_users: int | None = 100,
    seed: int = 2024,
    device: str | None = "mps",
    extra_exclude_user_ids: set[str] | None = None,
) -> Path:
    root = Path(__file__).resolve().parents[2]
    users_csv = users_csv if users_csv.is_absolute() else root / users_csv
    items_csv = items_csv if items_csv.is_absolute() else root / items_csv
    train_csv = train_csv if train_csv.is_absolute() else root / train_csv
    out_path = out_path if out_path.is_absolute() else root / out_path

    users = pd.read_csv(users_csv)
    items = pd.read_csv(items_csv)
    train = pd.read_csv(train_csv)
    users["user_id"] = users["user_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)
    train["user_id"] = train["user_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)

    title = items.set_index("item_id")["movie_title"].to_dict()
    gender = users.set_index("user_id")["gender"].to_dict()
    age = (
        users.set_index("user_id")["age"].to_dict()
        if "age" in users.columns
        else {}
    )

    # Chronological per user, so --history-order recent can take the last N.
    if "timestamp" in train.columns:
        train = train.sort_values(["user_id", "timestamp"], kind="stable")
    history_by_user: dict[str, list[str]] = (
        train.groupby("user_id")["item_id"].apply(list).to_dict()
    )
    history_sets = {u: set(v) for u, v in history_by_user.items()}

    from src.llm.prompts import few_shot_example_user_ids

    excluded = few_shot_example_user_ids() | (extra_exclude_user_ids or set())
    rng = random.Random(seed)
    candidates = [
        u
        for u in users["user_id"].tolist()
        if u in history_by_user and u not in excluded
    ]
    rng.shuffle(candidates)
    if max_users is not None:
        candidates = candidates[:max_users]

    _, dataset, model, torch_device = _load_model(
        config_file, checkpoint, device or "cpu", model_name=model_name
    )
    topk_map = _topk_for_users(
        model, dataset, torch_device, candidates, history_sets, topk
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for u in candidates:
            if u not in topk_map or u not in gender:
                continue
            hist_ids = history_by_user.get(u, [])
            if history_size <= 0:
                sampled: list[str] = []
                hist_titles: list[str] = []
            else:
                if len(hist_ids) < 1:
                    continue
                if len(hist_ids) <= history_size:
                    sampled = list(hist_ids)
                elif history_order == "recent":
                    sampled = hist_ids[-history_size:]  # most recent by timestamp
                else:
                    sampled = rng.sample(hist_ids, history_size)
                hist_titles = [title[i] for i in sampled if i in title]

            rec_titles = [title[i] for i in topk_map[u] if i in title]
            if not rec_titles:
                continue
            row: dict[str, Any] = {
                "user_id": u,
                "gender": gender[u],
                "age": age.get(u),
                "history_item_ids": sampled,
                "history_titles": hist_titles,
                "rec_item_ids": topk_map[u],
                "rec_titles": rec_titles,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} users → {out_path}")
    return out_path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ckpt_dir = root / "runs" / "EXP-003_bpr_d32_311" / "checkpoint"
    default_ckpt = next(ckpt_dir.glob("BPR-*.pth"), ckpt_dir / "missing.pth")

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/bpr_ml1m_311.yaml"))
    p.add_argument("--model", default="BPR", help="RecBole model class name, e.g. BPR, LightGCN")
    p.add_argument("--checkpoint", type=Path, default=default_ckpt)
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
        "--out", type=Path, default=Path("runs/EXP-003_bpr_d32_311/lists.jsonl")
    )
    p.add_argument("--topk", type=int, default=10)
    p.add_argument(
        "--history-order",
        choices=["recent", "random"],
        default="recent",
        help="which interactions to keep when a user has more than --history-size",
    )
    p.add_argument(
        "--history-size",
        type=int,
        default=0,
        help="0 = list-only export (no history titles); >0 samples train history",
    )
    p.add_argument("--max-users", type=int, default=100)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", default="mps")
    p.add_argument(
        "--exclude-lists-jsonl",
        type=Path,
        default=None,
        help=(
            "Path to another lists.jsonl whose user_ids are excluded from candidates "
            "— use to keep a train-pool export disjoint from an existing fixed test set"
        ),
    )
    args = p.parse_args()

    extra_exclude: set[str] = set()
    if args.exclude_lists_jsonl is not None:
        path = args.exclude_lists_jsonl
        path = path if path.is_absolute() else root / path
        extra_exclude = {
            json.loads(line)["user_id"] for line in path.open(encoding="utf-8")
        }

    export_lists(
        args.config,
        args.checkpoint,
        args.users_csv,
        args.items_csv,
        args.train_csv,
        args.out,
        model_name=args.model,
        topk=args.topk,
        history_size=args.history_size,
        history_order=args.history_order,
        max_users=args.max_users,
        seed=args.seed,
        device=args.device,
        extra_exclude_user_ids=extra_exclude,
    )


if __name__ == "__main__":
    main()
