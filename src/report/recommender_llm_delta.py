"""LLM attacker on BPR vs LightGCN lists, paired on the same 500 test users.

Tests the 2026-09-28 prediction "BPR lists have a Romance gap between men and women,
LightGCN lists do not, so the LLM attacker (largely a Romance detector) should do
better on BPR". Same attacker / style / tag on both recommenders' lists; ΔAUC
(BPR − LightGCN) with a paired bootstrap over users (AUC uses every scored user).

    uv run python -m src.report.recommender_llm_delta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PAIRS = [  # (style, tag suffix after the recommender prefix)
    ("zero_shot", "llama32_500_lp"),
    ("staab_few_shot", "llama32_500_lp"),
    ("zero_shot", "glm4_9b_500"),
    ("staab_few_shot", "glm4_9b_500"),
    ("zero_shot", "llama31_8b_500"),
    ("zero_shot", "qwen25_14b_500"),
    ("many_shot", "glm4_9b_500_shots8_s2025"),
    ("many_shot", "qwen25_14b_500_shots16_s2025"),
    ("staab_many_shot", "qwen25_14b_500_shots16_s2025"),
]


def _load(p: Path) -> dict[str, tuple[bool, float]]:
    out = {}
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("score") is not None:
            out[r["user_id"]] = (r["gender"] == "male", r["score"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bpr", type=Path, default=Path("runs/EXP-003_bpr_d32_311/llm_aia"))
    p.add_argument("--lightgcn", type=Path, default=Path("runs/EXP-004_lightgcn_d32_311/llm_aia"))
    p.add_argument("--out", type=Path, default=Path("runs/EXP-003_bpr_d32_311/shot_sweep/bpr_vs_lightgcn_llm.md"))
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    rng = np.random.default_rng(2024)
    lines = ["LLM attacker AUC on BPR vs LightGCN top-10 lists, same 500 users (paired bootstrap)", "",
             "| Style | Attacker tag | n | AUC BPR | AUC LightGCN | Δ BPR − LightGCN [95% CI] |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for style, tag in PAIRS:
        fa, fb = args.bpr / f"preds_{style}_bpr_{tag}.jsonl", args.lightgcn / f"preds_{style}_lightgcn_{tag}.jsonl"
        if not (fa.exists() and fb.exists()):
            continue
        a, b = _load(fa), _load(fb)
        users = sorted(set(a) & set(b))
        y = np.array([a[u][0] for u in users])
        sa, sb = np.array([a[u][1] for u in users]), np.array([b[u][1] for u in users])
        d = []
        for _ in range(args.n_boot):
            i = rng.integers(0, len(users), len(users))
            if len(set(y[i])) == 2:
                d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
        lines.append(f"| {style} | {tag} | {len(users)} | {roc_auc_score(y, sa):.3f} | {roc_auc_score(y, sb):.3f} | "
                     f"{roc_auc_score(y, sa) - roc_auc_score(y, sb):+.3f} "
                     f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] |")
    md = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
