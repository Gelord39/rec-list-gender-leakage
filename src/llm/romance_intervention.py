"""Intervention test of "the zero-shot LLM attacker is largely a Romance detector".

Until now the evidence was correlational (2026-09-28 Table E: the LLM's P(male)
correlates with the list's Romance share, and Romance share alone gets most of
the LLM's AUC). This edits the lists and re-asks the same LLM.

Arms, built from one recommender's 500 fixed test lists (order and length kept):
  original        the lists as they are (re-run so every arm shares one Ollama build)
  romance_swap    every Romance-tagged film -> a non-Romance film whose genre set
                  equals the original's minus Romance, closest in popularity
  control_swap_s  the SAME NUMBER of films per user, chosen at random among the
                  list's non-Romance films (seed s), replaced by the same rule
                  (same genre set, closest popularity, never Romance)

Replacement rule (both arms): never an item already in the list or already
watched by the user (the RecBole lists' unseen rule), never Romance; prefer an
exact genre-set match, else the largest genre overlap; then the closest
log-popularity (train interactions); ties by item id. Deterministic given the
seed of the control arm's position choice.

If the LLM reads Romance, romance_swap should move AUC towards 0.5 and raise
P(male) mostly for women, while the control arm changes little.

    uv run python -m src.llm.romance_intervention build --run-dir runs/EXP-003_bpr_d32_311
    sbatch scripts/hendrix_romance_intervention.sh            # zero-shot GLM-4-9B per arm
    uv run python -m src.llm.romance_intervention report --run-dir runs/EXP-003_bpr_d32_311 --prefix bpr
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ITEMS = Path("data/ml-1m-311/splits/items.csv")
TRAIN = Path("data/ml-1m-311/splits/train.csv")
ROMANCE = "Romance"
OUT = "romance_intervention"
CONTROL_SEEDS = (2024, 2025, 2026)


def _catalogue() -> tuple[dict[str, str], dict[str, frozenset[str]], dict[str, int], dict[str, set[str]]]:
    with ITEMS.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    title = {r["item_id"]: r["movie_title"] for r in rows}
    genres = {r["item_id"]: frozenset(r["genres"].split()) for r in rows}
    pop: dict[str, int] = {i: 0 for i in title}
    seen: dict[str, set[str]] = {}
    with TRAIN.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pop[r["item_id"]] = pop.get(r["item_id"], 0) + 1
            seen.setdefault(r["user_id"], set()).add(r["item_id"])
    return title, genres, pop, seen


def _replacement(item: str, target_genres: frozenset[str], banned: set[str],
                 genres: dict[str, frozenset[str]], pop: dict[str, int]) -> str:
    lp = math.log1p(pop.get(item, 0))
    best, best_key = None, None
    for cand, g in genres.items():
        if ROMANCE in g or cand in banned or pop.get(cand, 0) == 0:
            continue
        key = (g != target_genres, -len(g & target_genres), abs(math.log1p(pop[cand]) - lp), cand)
        if best_key is None or key < best_key:
            best, best_key = cand, key
    assert best is not None
    return best


def build(args: argparse.Namespace) -> None:
    title, genres, pop, seen = _catalogue()
    rows = [json.loads(line) for line in (args.run_dir / "lists.jsonl").open(encoding="utf-8")]
    out = args.run_dir / OUT
    out.mkdir(parents=True, exist_ok=True)

    def write(arm: str, edited: list[dict[str, Any]], stats: dict[str, Any]) -> None:
        with (out / f"lists_{arm}.jsonl").open("w", encoding="utf-8") as f:
            for r in edited:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{arm:18s} " + "  ".join(f"{k} {v}" for k, v in stats.items()))
        manifest[arm] = stats

    def edit(r: dict[str, Any], positions: list[int]) -> dict[str, Any]:
        items = list(r["rec_item_ids"])
        banned = set(items) | seen.get(r["user_id"], set())
        for k in positions:
            new = _replacement(items[k], genres[items[k]] - {ROMANCE}, banned, genres, pop)
            banned.add(new)
            items[k] = new
        return {**r, "rec_item_ids": items, "rec_titles": [title[i] for i in items],
                "replaced_positions": positions}

    def stats(edited: list[dict[str, Any]]) -> dict[str, Any]:
        rom = [np.mean([ROMANCE in genres[i] for i in r["rec_item_ids"]]) for r in edited]
        lp = [np.mean([math.log1p(pop[i]) for i in r["rec_item_ids"]]) for r in edited]
        return {"romance_share": round(float(np.mean(rom)), 4), "mean_log_pop": round(float(np.mean(lp)), 3),
                "n_replaced": int(sum(len(r.get("replaced_positions", [])) for r in edited))}

    manifest: dict[str, Any] = {}
    write("original", rows, stats(rows))
    rom_pos = [[k for k, i in enumerate(r["rec_item_ids"]) if ROMANCE in genres[i]] for r in rows]
    write("romance_swap", [edit(r, p) for r, p in zip(rows, rom_pos)],
          stats([edit(r, p) for r, p in zip(rows, rom_pos)]))
    n_short = 0
    for seed in CONTROL_SEEDS:
        rng = np.random.default_rng(seed)
        edited = []
        for r, p in zip(rows, rom_pos):
            non = [k for k, i in enumerate(r["rec_item_ids"]) if ROMANCE not in genres[i]]
            n = min(len(p), len(non))
            n_short += len(p) - n
            edited.append(edit(r, sorted(rng.choice(non, size=n, replace=False).tolist()) if n else []))
        write(f"control_swap_s{seed}", edited, stats(edited))
    manifest["note"] = (f"control arms replace min(#Romance, #non-Romance) films per user; "
                        f"{n_short // len(CONTROL_SEEDS)} Romance slots per arm had no non-Romance partner")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _load(path: Path) -> dict[str, tuple[bool, float]]:
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("score") is not None:
            out[r["user_id"]] = (r["gender"] == "male", float(r["score"]))
    return out


def report(args: argparse.Namespace) -> None:
    from sklearn.metrics import roc_auc_score

    llm = args.run_dir / OUT / "llm_aia"
    arms = ["original", "romance_swap", *[f"control_swap_s{s}" for s in CONTROL_SEEDS]]
    preds = {a: _load(llm / f"preds_zero_shot_{args.prefix}_{args.model_slug}_500_{a}.jsonl") for a in arms
             if (llm / f"preds_zero_shot_{args.prefix}_{args.model_slug}_500_{a}.jsonl").exists()}
    manifest = json.loads((args.run_dir / OUT / "manifest.json").read_text())
    base = preds["original"]
    rng = np.random.default_rng(2024)
    lines = [f"Romance intervention, {args.run_dir.name}, {args.model_slug} zero-shot, same 500 users", "",
             "| Arm | Romance share | mean log-pop | AUC | Δ AUC vs original [95% CI] | mean P(male), women | mean P(male), men |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    res = {}
    for a, p in preds.items():
        users = sorted(set(p) & set(base))
        y = np.array([base[u][0] for u in users])
        s, s0 = np.array([p[u][1] for u in users]), np.array([base[u][1] for u in users])
        auc = roc_auc_score(y, s)
        d = []
        for _ in range(args.n_boot):
            i = rng.integers(0, len(users), len(users))
            if len(set(y[i])) == 2:
                d.append(roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], s0[i]))
        ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        res[a] = {"n": len(users), "auc": auc, "delta_ci95": ci,
                  "p_male_women": float(s[~y].mean()), "p_male_men": float(s[y].mean()), **manifest[a]}
        delta = "—" if a == "original" else f"{auc - roc_auc_score(y, s0):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        lines.append(f"| {a} | {manifest[a]['romance_share']:.3f} | {manifest[a]['mean_log_pop']:.2f} | {auc:.3f} | "
                     f"{delta} | {s[~y].mean():.3f} | {s[y].mean():.3f} |")
    lines += ["", manifest["note"]]
    md = "\n".join(lines) + "\n"
    (args.run_dir / OUT / f"report_{args.model_slug}.md").write_text(md)
    (args.run_dir / OUT / f"report_{args.model_slug}.json").write_text(json.dumps(res, indent=2))
    print(md)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run-dir", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--prefix", required=True, help="bpr or lightgcn")
    r.add_argument("--model-slug", default="glm4_9b")
    r.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    {"build": build, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
