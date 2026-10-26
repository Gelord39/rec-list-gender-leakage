"""EXP-005: LightGCN candidates re-ranked by a local LLM (LLMRank-style).

Follows Hou et al. 2024, *Large Language Models are Zero-Shot Rankers for
Recommender Systems* (github.com/RUCAIBox/LLMRank): prompt from `model/rf.py`
(recency-focused), parsing from `model/rank.py::parsing_output_text`,
hyperparameters from `props/RF.yaml` (max_his_len 50, temperature 0.2,
max_tokens 512) and `props/overall.yaml` (recall_budget 20).

Deliberate deviations from LLMRank (logged in docs/experiment_setup.md §11):
  - candidates are LightGCN's top-20 *without* forcing the ground truth in
    (LLMRank's has_gt=True) — a deployable pipeline, not an oracle ranking task
  - global_311 split: every test item of a user counts as relevant (same as
    RecBole full-sort eval), not a single leave-one-out next item
  - local Ollama llama3.2 instead of gpt-3.5-turbo, with a fixed Ollama seed
  - titles keep their release year (LLMRank's ml-1m.item stores it separately)
  - candidates the LLM never mentions are back-filled in LightGCN order
    (LLMRank leaves them tied at -10000)
  - besides LLMRank's first-mention parser, a second parser reads only the
    last numbered block (llama3.2 often lists candidates in its reasoning
    before giving the final ranking)

Example:
  uv run python -m src.rec.llm_rerank \\
    --checkpoint runs/EXP-004_lightgcn_d32_311/checkpoint/LightGCN-Aug-30-2026_23-00-46.pth \\
    --out-dir runs/EXP-005_lightgcn_llmrerank_311 --tag llama32
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.export.export_lists import _load_model, _topk_for_users
from src.llm.backends import OllamaBackend

ROOT = Path(__file__).resolve().parents[2]
KS = (5, 10, 20)
_YEAR = re.compile(r"^(.*) (\(\d{4}\))$")
_NUMBERED = re.compile(r"^\s*(?:[*#>-]+\s*)?\[?(\d+)[.):\]]\s+")


def llmrank_title(raw: str) -> str:
    """LLMRank's ml-1m title cleanup (", The" / ", A" suffix → prefix), year kept."""
    m = _YEAR.match(raw)
    name, year = (m.group(1), m.group(2)) if m else (raw, "")
    if name.endswith(", The"):
        name = "The " + name[:-5]
    elif name.endswith(", A"):
        name = "A " + name[:-3]
    return f"{name} {year}".strip()


def build_prompt(history_titles: list[str], candidate_titles: list[str]) -> str:
    """Verbatim port of LLMRank `RF.construct_prompt` (ml-1m branch).

    As in the original, the numbered lists are interpolated as Python list
    reprs (``['0. Title', '1. Title']``), zero-indexed.
    """
    n = len(candidate_titles)
    user_his_text = [f"{j}. {t}" for j, t in enumerate(history_titles)]
    candidate_text_order = [f"{j}. {t}" for j, t in enumerate(candidate_titles)]
    recent_item = user_his_text[-1][user_his_text[-1].find(". ") + 2 :]
    return (
        f"I've watched the following movies in the past in order:\n{user_his_text}\n\n"
        f"Now there are {n} candidate movies that I can watch next:\n{candidate_text_order}\n"
        f"Please rank these {n} movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step.\n"
        f"Note that my most recently watched movie is {recent_item}. "
        f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
    )


def _lcs_len(a: str, b: str) -> int:
    """Longest common subsequence length (stands in for `pylcs.lcs_sequence_length`)."""
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def _match_line(line: str, candidates: list[str]) -> int | None:
    """One line of LLMRank `parsing_output_text` → index of first matching candidate."""
    pr = line.find(". ")
    item_name = line[pr + 2 :] if line[:pr].isdigit() else line
    for idx, cand in enumerate(candidates):
        clean = html.unescape(cand.strip())
        if (
            clean in item_name
            or item_name in clean
            # LCS can't exceed len(item_name): skip the O(n·m) DP when it can't pass
            or (
                len(item_name) > 0.9 * len(clean)
                and _lcs_len(item_name, clean) > 0.9 * len(clean)
            )
        ):
            return idx
    return None


def parse_llmrank(response: str, candidates: list[str]) -> list[int]:
    """LLMRank's parser: candidates in order of first mention anywhere in the response."""
    order: list[int] = []
    for line in response.split("\n"):
        if len(line) < 1 or line.endswith("candidate movies:"):
            continue
        idx = _match_line(line, candidates)
        if idx is not None and idx not in order:
            order.append(idx)
    return order


def parse_final_block(response: str, candidates: list[str]) -> list[int]:
    """Candidates in order within the *last* numbered block that names any candidate.

    A block starts whenever the numbering does not increase (e.g. "1." after "20.").
    Falls back to `parse_llmrank` when no numbered block matches.
    """
    blocks: list[list[str]] = []
    prev: int | None = None
    for line in response.split("\n"):
        m = _NUMBERED.match(line)
        if not m:
            continue
        n = int(m.group(1))
        if prev is None or n <= prev:
            blocks.append([])
        blocks[-1].append(line[m.end() :].strip(" *"))
        prev = n
    for block in reversed(blocks):
        order: list[int] = []
        for text in block:
            if not text:
                continue
            idx = _match_line(text, candidates)
            if idx is not None and idx not in order:
                order.append(idx)
        if order:
            return order
    return parse_llmrank(response, candidates)


def rank_metrics(ranked: list[str], positives: set[str]) -> dict[str, float]:
    """RecBole-style Hit / Recall / NDCG (idcg over min(|positives|, k))."""
    out: dict[str, float] = {}
    for k in KS:
        hits = [r for r, item in enumerate(ranked[:k]) if item in positives]
        idcg = sum(1 / math.log2(r + 2) for r in range(min(len(positives), k)))
        out[f"hit@{k}"] = float(bool(hits))
        out[f"recall@{k}"] = len(hits) / len(positives)
        out[f"ndcg@{k}"] = sum(1 / math.log2(r + 2) for r in hits) / idcg
    return out


def load_eval_data(splits_dir: Path):
    """Chronological train+valid history, test positives, and LLMRank-cleaned titles."""
    ids = {"user_id": str, "item_id": str}
    train, valid, test = (
        pd.read_csv(splits_dir / f"{s}.csv", dtype=ids) for s in ("train", "valid", "test")
    )
    # RecBole's test-time full ranking treats train+valid as the user's history
    seen = pd.concat([train, valid], ignore_index=True).sort_values(
        "timestamp", kind="mergesort"
    )
    history = seen.groupby("user_id")["item_id"].apply(list).to_dict()
    positives = test.groupby("user_id")["item_id"].apply(set).to_dict()
    items = pd.read_csv(splits_dir / "items.csv", dtype={"item_id": str})
    titles = {i: llmrank_title(t) for i, t in zip(items["item_id"], items["movie_title"])}
    return history, positives, titles


def evaluate(
    records: list[dict[str, Any]],
    positives: dict[str, set[str]],
    titles: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Metrics for LightGCN's candidate order vs. both LLM parsers (with back-fill)."""
    variants = ("lightgcn_order", "llm_llmrank_parse", "llm_final_block_parse")
    per_user: dict[str, list[dict[str, float]]] = {v: [] for v in variants}
    n_matched: dict[str, list[int]] = {v: [] for v in variants[1:]}
    reranked: list[dict[str, Any]] = []

    for rec in records:
        cands = rec["candidate_item_ids"]
        cand_titles = [titles[i] for i in cands]
        orders = {
            "lightgcn_order": list(range(len(cands))),
            "llm_llmrank_parse": parse_llmrank(rec["response"], cand_titles),
            "llm_final_block_parse": parse_final_block(rec["response"], cand_titles),
        }
        row: dict[str, Any] = {"user_id": rec["user_id"]}
        for v, order in orders.items():
            if v in n_matched:
                n_matched[v].append(len(order))
            full = order + [i for i in range(len(cands)) if i not in order]
            ranked = [cands[i] for i in full]
            per_user[v].append(rank_metrics(ranked, positives[rec["user_id"]]))
            row[v] = ranked
        reranked.append(row)

    n = len(records)
    metrics: dict[str, Any] = {"n_users": n}
    for v in variants:
        metrics[v] = {
            key: round(sum(m[key] for m in per_user[v]) / n, 6) for key in per_user[v][0]
        }
    metrics["parse_stats"] = {
        v: {
            "mean_candidates_matched": round(sum(c) / n, 3),
            "frac_users_zero_matched": round(sum(x == 0 for x in c) / n, 4),
            # below 10, part of the top-10 is LightGCN back-fill, not LLM ranking
            "frac_users_at_least_10_matched": round(sum(x >= 10 for x in c) / n, 4),
            "frac_users_all_matched": round(
                sum(x == len(r["candidate_item_ids"]) for x, r in zip(c, records)) / n, 4
            ),
        }
        for v, c in n_matched.items()
    }
    if any("done_reason" in r for r in records):
        metrics["generation_stats"] = {
            "frac_truncated_at_max_tokens": round(
                sum(r.get("done_reason") == "length" for r in records) / n, 4
            ),
            "mean_output_tokens": round(sum(r.get("eval_count") or 0 for r in records) / n, 1),
            "mean_seconds_per_user": round(sum(r.get("seconds", 0) for r in records) / n, 2),
        }
    return metrics, reranked


def _git_commit() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def main() -> None:
    p = argparse.ArgumentParser(description="LightGCN top-k candidates re-ranked by an LLM")
    p.add_argument("--config", type=Path, default=Path("configs/lightgcn_ml1m_311.yaml"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rec-model", default="LightGCN")
    p.add_argument("--splits-dir", type=Path, default=Path("data/ml-1m-311/splits"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/EXP-005_lightgcn_llmrerank_311"))
    p.add_argument("--tag", required=True, help="output file suffix, e.g. llama32")
    p.add_argument("--recall-budget", type=int, default=20)
    p.add_argument("--max-his-len", type=int, default=50)
    p.add_argument("--llm", default="llama3.2")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="first N test users (pilot)")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="only score LightGCN's candidate order (checks metrics vs RecBole)",
    )
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = args.splits_dir if args.splits_dir.is_absolute() else ROOT / args.splits_dir

    history, positives, titles = load_eval_data(splits_dir)
    users = sorted(positives, key=int)[: args.limit]

    print(f"Loading {args.rec_model} and ranking {len(users)} test users ...")
    _, dataset, model, device = _load_model(
        args.config, args.checkpoint, args.device, model_name=args.rec_model
    )
    history_sets = {u: set(h) for u, h in history.items()}
    candidates = _topk_for_users(model, dataset, device, users, history_sets, args.recall_budget)

    if args.no_llm:
        records = [
            {"user_id": u, "candidate_item_ids": candidates[u], "response": ""} for u in users
        ]
        metrics, _ = evaluate(records, positives, titles)
        path = out_dir / "rec_metrics_candidates_only.json"
        path.write_text(json.dumps(metrics["lightgcn_order"] | {"n_users": len(users)}, indent=2))
        print(json.dumps(metrics["lightgcn_order"], indent=2), f"\n→ {path}")
        return

    backend = OllamaBackend(
        model=args.llm,
        host=args.host,
        temperature=args.temperature,
        timeout=900,
        seed=args.seed,
        num_predict=args.max_tokens,
        logprobs=False,
    )
    resp_path = out_dir / f"responses_{args.tag}.jsonl"
    done = (
        {json.loads(line)["user_id"] for line in resp_path.open(encoding="utf-8")}
        if resp_path.exists()
        else set()
    )
    todo = [u for u in users if u not in done]
    print(f"{len(done)} users already done, {len(todo)} to go → {resp_path}")

    t_start = time.time()
    with resp_path.open("a", encoding="utf-8") as f:
        for n, u in enumerate(todo, 1):
            hist_titles = [titles[i] for i in history[u][-args.max_his_len :]]
            prompt = build_prompt(hist_titles, [titles[i] for i in candidates[u]])
            t0 = time.time()
            resp = backend.generate(prompt)
            raw = resp.raw or {}
            rec = {
                "user_id": u,
                "candidate_item_ids": candidates[u],
                "prompt": prompt,
                "response": resp.text,
                "done_reason": raw.get("done_reason"),
                "prompt_tokens": raw.get("prompt_eval_count"),
                "eval_count": raw.get("eval_count"),
                "seconds": round(time.time() - t0, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                rate = (time.time() - t_start) / n
                print(f"[{n}/{len(todo)}] {rate:.1f}s/user, ETA {rate * (len(todo) - n) / 60:.0f} min", flush=True)

    user_set = set(users)
    records = [
        r
        for r in map(json.loads, resp_path.open(encoding="utf-8"))
        if r["user_id"] in user_set
    ]
    metrics, reranked = evaluate(records, positives, titles)
    with (out_dir / f"reranked_{args.tag}.jsonl").open("w", encoding="utf-8") as f:
        for row in reranked:
            f.write(json.dumps(row) + "\n")

    metrics["provenance"] = {
        "exp_id": out_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": _git_commit(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "method": "LLMRank RF prompt (Hou et al. 2024), candidates = LightGCN top-k without ground truth",
        "eval": "global_311 test users with >=1 test item; train+valid masked; all test items relevant",
        "example_prompt": records[0]["prompt"],
    }
    path = out_dir / f"rec_metrics_{args.tag}.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    summary = {k: metrics[k] for k in ("n_users", "lightgcn_order", "llm_llmrank_parse", "llm_final_block_parse")}
    print(json.dumps(summary, indent=2), f"\n→ {path}")


if __name__ == "__main__":
    main()
