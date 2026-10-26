"""Effectiveness by gender: HR@10 / NDCG@10 for women vs men, every recommender.

Group fairness of effectiveness in the Stairway sense: is the recommender as useful
for women as for men, and does the gender prompt change that? Over the users with
test items (695 on global_311).

- per recommender: HR@10 / NDCG@10 by gender, gap female - male (bootstrap CI,
  resampling within each gender)
- per neutral -> gender LLMRec pair: paired change within each gender, and the
  difference of the two changes (female change - male change)

Lists: CF = top-10 re-ranked from the checkpoint with train+valid masked (RecBole's
test protocol, reproduces rec_metrics.json; the attack lists.jsonl mask train only, so
some contain validation items), the re-ranker = parser B top-10, LLMRec = unseen lists.

    uv run python -m src.report.effectiveness_by_gender
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

TOPK = 10
RUNS = Path("runs")
LLMREC = RUNS / "EXP-006_llmrec_311"

CF = {  # name: (config, checkpoint)
    "BPR": ("configs/bpr_ml1m_311.yaml", RUNS / "EXP-003_bpr_d32_311/checkpoint/BPR-Aug-17-2026_00-12-47.pth"),
    "LightGCN": (
        "configs/lightgcn_ml1m_311.yaml",
        RUNS / "EXP-004_lightgcn_d32_311/checkpoint/LightGCN-Aug-30-2026_23-00-46.pth",
    ),
}
RERANK = ("LightGCN -> llama3.2 re-rank", RUNS / "EXP-005_lightgcn_llmrerank_311/reranked_llama32.jsonl")
LLMREC_PAIRS = [  # (label, neutral run dir, gender run dir)
    ("GLM-4-9B LLMRec", "neutral_req20_glm4_9b_unseen", "gender_req20_glm4_9b_unseen"),
    ("Llama-3.1-8B LLMRec", "neutral_req20_llama31_8b_unseen", "gender_req20_llama31_8b_unseen"),
    ("GLM-4-9B LLMRec hist50", "neutral_hist50_req20_glm4_9b_unseen", "gender_hist50_req20_glm4_9b_unseen"),
    ("Llama-3.1-8B LLMRec hist50", "neutral_hist50_req20_llama31_8b_unseen", "gender_hist50_req20_llama31_8b_unseen"),
]


def rank_metrics(rec: list[str], positives: set[str]) -> tuple[float, float]:
    rel = [1 if i in positives else 0 for i in rec[:TOPK]]
    dcg = sum(r / math.log2(k + 2) for k, r in enumerate(rel))
    idcg = sum(1 / math.log2(k + 2) for k in range(min(len(positives), TOPK)))
    return float(any(rel)), dcg / idcg


def read_lists(paths: list[Path], key: str = "rec_item_ids") -> dict[str, list[str]]:
    out = {}
    for p in paths:
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            out[r["user_id"]] = r[key]
    return out


def cf_lists(name: str, config: str, ckpt: Path, splits_dir: Path) -> dict[str, list[str]]:
    """Top-10 over the full catalogue with train+valid masked, as RecBole's test evaluation."""
    from src.export.export_lists import _load_model, _topk_for_users
    from src.rec.llm_rerank import load_eval_data

    history, positives, _ = load_eval_data(splits_dir)
    _, dataset, model, device = _load_model(Path(config), ckpt, "cpu", model_name=name)
    users = sorted(positives, key=int)
    return _topk_for_users(model, dataset, device, users, {u: set(h) for u, h in history.items()}, TOPK)


def per_user(lists: dict[str, list[str]], test_pos: dict[str, set[str]]) -> pd.DataFrame:
    """One row per user with test items; a user without a list counts as a miss."""
    rows = [(u, *rank_metrics(lists.get(u, []), pos)) for u, pos in test_pos.items()]
    return pd.DataFrame(rows, columns=["user_id", "hr", "ndcg"]).set_index("user_id")


def ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def group_summary(df: pd.DataFrame, female: np.ndarray, rng, n_boot: int) -> dict:
    out = {"n_female": int(female.sum()), "n_male": int((~female).sum())}
    f_idx, m_idx = np.flatnonzero(female), np.flatnonzero(~female)
    bf = rng.choice(f_idx, (n_boot, len(f_idx)))
    bm = rng.choice(m_idx, (n_boot, len(m_idx)))
    for metric in ("hr", "ndcg"):
        v = df[metric].to_numpy()
        out[metric] = {
            "all": float(v.mean()),
            "female": float(v[f_idx].mean()),
            "male": float(v[m_idx].mean()),
            "gap_f_minus_m": float(v[f_idx].mean() - v[m_idx].mean()),
            "gap_ci95": ci(v[bf].mean(1) - v[bm].mean(1)),
        }
    return out


def paired_change(a: pd.DataFrame, b: pd.DataFrame, female: np.ndarray, rng, n_boot: int) -> dict:
    """b - a for the same users: within each gender, and female change - male change."""
    out = {}
    f_idx, m_idx = np.flatnonzero(female), np.flatnonzero(~female)
    bf = rng.choice(f_idx, (n_boot, len(f_idx)))
    bm = rng.choice(m_idx, (n_boot, len(m_idx)))
    for metric in ("hr", "ndcg"):
        d = (b[metric] - a[metric]).to_numpy()
        df_, dm_ = d[bf].mean(1), d[bm].mean(1)
        out[metric] = {
            "change_female": float(d[f_idx].mean()),
            "change_female_ci95": ci(df_),
            "change_male": float(d[m_idx].mean()),
            "change_male_ci95": ci(dm_),
            "diff_f_minus_m": float(d[f_idx].mean() - d[m_idx].mean()),
            "diff_ci95": ci(df_ - dm_),
        }
    return out


def fmt_ci(c: list[float]) -> str:
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits-dir", type=Path, default=Path("data/ml-1m-311/splits"))
    ap.add_argument("--out-dir", type=Path, default=RUNS / "effectiveness_by_gender")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    test = pd.read_csv(args.splits_dir / "test.csv", dtype={"user_id": str, "item_id": str})
    test_pos = test.groupby("user_id").item_id.agg(set).to_dict()
    users = pd.read_csv(args.splits_dir / "users.csv", dtype={"user_id": str}).set_index("user_id")
    order = sorted(test_pos)
    female = (users.loc[order, "gender"] == "female").to_numpy()

    def run(lists):
        return per_user(lists, test_pos).loc[order]

    results = {"n_users_with_test_items": len(order), "recommenders": {}, "neutral_to_gender": {}}
    frames = {}
    for name, (config, ckpt) in CF.items():
        frames[name] = run(cf_lists(name, config, ckpt, args.splits_dir))
    frames[RERANK[0]] = run(read_lists([RERANK[1]], key="llm_final_block_parse"))
    for label, neu, gen in LLMREC_PAIRS:
        for variant, sub in (("neutral", neu), ("gender", gen)):
            d = LLMREC / sub
            frames[f"{label} · {variant}"] = run(read_lists([d / "lists.jsonl", d / "lists_train_pool.jsonl"]))

    for name, df in frames.items():
        results["recommenders"][name] = group_summary(df, female, rng, args.n_boot)
    for label, _, _ in LLMREC_PAIRS:
        a, b = frames[f"{label} · neutral"], frames[f"{label} · gender"]
        results["neutral_to_gender"][label] = paired_change(a, b, female, rng, args.n_boot)

    results["provenance"] = {
        "n_boot": args.n_boot, "seed": args.seed, "topk": TOPK,
        "lists": "CF top-10 from checkpoint, train+valid masked; re-ranker llm_final_block_parse; LLMRec *_unseen",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "effectiveness_by_gender.json").write_text(json.dumps(results, indent=2))

    n_f, n_m = int(female.sum()), int((~female).sum())
    md = [
        f"# Effectiveness by gender ({len(order)} users with test items: {n_f} women, {n_m} men)",
        "",
        "Gap = women − men; 95% bootstrap CI resampling within each gender.",
        "",
        "| Recommender | HR@10 all | HR@10 F | HR@10 M | HR gap [CI] | NDCG@10 F | NDCG@10 M | NDCG gap [CI] |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in results["recommenders"].items():
        h, n = r["hr"], r["ndcg"]
        md.append(
            f"| {name} | {h['all']:.3f} | {h['female']:.3f} | {h['male']:.3f} | {h['gap_f_minus_m']:+.3f} {fmt_ci(h['gap_ci95'])} "
            f"| {n['female']:.3f} | {n['male']:.3f} | {n['gap_f_minus_m']:+.3f} {fmt_ci(n['gap_ci95'])} |"
        )
    md += [
        "",
        "## Neutral → gender prompt, paired on the same users",
        "",
        "| LLMRec | metric | change, women [CI] | change, men [CI] | women − men [CI] |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, r in results["neutral_to_gender"].items():
        for metric, mname in (("hr", "HR@10"), ("ndcg", "NDCG@10")):
            c = r[metric]
            md.append(
                f"| {label} | {mname} | {c['change_female']:+.3f} {fmt_ci(c['change_female_ci95'])} "
                f"| {c['change_male']:+.3f} {fmt_ci(c['change_male_ci95'])} | {c['diff_f_minus_m']:+.3f} {fmt_ci(c['diff_ci95'])} |"
            )
    (args.out_dir / "effectiveness_by_gender.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
