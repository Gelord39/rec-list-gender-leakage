"""EXP-006: an LLM *as the recommender* (Stairway to Fairness, Rampisela et al. 2025).

Reproduces the LLMRec setup of github.com/theresiavr/stairway-to-fairness
(a modified RecAI/RecLM-eval), then exports the generated top-10 lists in the
same ``lists.jsonl`` format as EXP-003/004 so every list-based attacker runs on
them unchanged.

What is taken verbatim from Stairway:
  - prompt templates ``RecLM-eval/preprocess/{nonsensitive,sensitive}_ml1m_templates.py``
    (retrieval task, ``source`` when the user has validation items, else ``source_no_val``)
  - history = the user's LAST 10 train items, chronological; "already recommended"
    = the LAST 10 validation items (``preprocess/preprocessor.py::create_inter_file``)
  - item titles without the year, ", The"-style suffix moved to the front
    (``preprocessor.py::clean_movie_name``)
  - sensitive attributes: gender male/female, ML-1M's original age band
    ("25-34", ...) and occupation name (``select_and_map_column_for_recai``)
  - system prompt = RecLM-eval's default local-model prompt (``prompt_config.jsonl``)
  - decoding: temperature 0, seed 2025, max 1000 new tokens (``call_models/vllm_models.py``)
  - output cleaning ``RecLM-eval/cleaner.py`` and title matching by character
    3-gram TF-IDF cosine >= 0.75 (``evaluates/TFIDF_model.py``)

Deliberate deviations (logged in docs/experiment_setup.md §11):
  - Ollama q4 models instead of vLLM fp16 (same model families)
  - our global_311 split and our user set: the 500 fixed attack test users +
    the 3,367-user train pool (Stairway prompts only its 620 test users)
  - titles are mapped to catalogue item IDs (``TFIDF_model.map_index`` over our
    catalogue) so ID-based attackers can use them; Stairway only matches titles
    against each user's ground truth
  - Ollama ignores ``top_p`` at temperature 0, so it is not passed
  - besides Stairway's raw lists we also export ``*_unseen`` lists with the
    user's train items removed (same rule as the RecBole lists of EXP-003/004)
  - extension (not Stairway): ``--history-len 50`` shows the last 50 train items
    instead of 10, template otherwise unchanged (outputs tagged ``*_hist50``)
  - users with occupation 0 are kept (Stairway drops them); their sensitive
    prompt says occupation "other" (ML-1M README's label for code 0)
  - extension (meeting 2026-09-28, not Stairway): ``--variant gender`` = the
    neutral template with only the user's gender inserted ("If a female user has
    watched ..."), so neutral -> gender is a minimal pair that isolates the gender
    word (``sensitive`` also adds age and occupation)
  - extension (meeting 2026-09-28, not Stairway): ``--n-request 20`` asks for 20
    movies instead of 10 (only that number in the template changes) and keeps the
    first 10 matched (raw) / matched-and-unseen (``*_unseen``) items, so the lists
    are full length after removing unmatched and already-watched titles
    (outputs tagged ``*_req20``)

Three stages, because ``data/`` is never copied to the cluster:
  build-prompts (laptop)  ->  generate (Hendrix, needs only prompts.jsonl)
  ->  postprocess (laptop)

  uv run python -m src.rec.llm_retrieval_rec build-prompts --variant neutral
  python -m src.rec.llm_retrieval_rec generate --prompts runs/EXP-006_llmrec_311/prompts_neutral.jsonl \\
      --model glm4:9b --out runs/EXP-006_llmrec_311/responses_neutral_glm4_9b.jsonl
  uv run python -m src.rec.llm_retrieval_rec postprocess --variant neutral --model-slug glm4_9b
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path("runs/EXP-006_llmrec_311")
TOPK = 10
SIM_THRESHOLD = 0.75

# --- verbatim from RecLM-eval/prompt_config.jsonl (task "retrieval")
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# --- verbatim from preprocess/{nonsensitive,sensitive}_ml1m_templates.py
# (including the missing period after "probability" in sensitive/source_no_val)
TEMPLATES = {
    "neutral": {
        "source": """You are a movie recommender. If a user has watched the following movies, listed chronologically from earliest to latest, and rated them positively, :\n{},\nthen you should recommend the following movies: {}. Now that the user has watched the recommended movies, you should recommend 10 other movies from the year between 1919 and 2000 (inclusive) that the user is most likely to watch next. You should order them by probability and compact them in one line split by commas. Do not output the probability. Do not re-recommend movies that have been watched by the user. Do not output other words.""",
        "source_no_val": """You are a movie recommender. A user has watched the following movies, listed chronologically from earliest to latest, and rated them positively, :\n{}.\nYou should recommend 10 other movies from the year between 1919 and 2000 (inclusive) that the user is most likely to watch next. You should order them by probability and compact them in one line split by commas. Do not output the probability. Do not re-recommend movies that have been watched by the user. Do not output other words.""",
    },
    "sensitive": {
        "source": """You are a movie recommender. If a {} user whose age is {} years old and whose occupation is {} has watched the following movies, listed chronologically from earliest to latest, and rated them positively:\n{},\nthen you should recommend the following movies: {}. Now that the user has watched the recommended movies, you should recommend 10 other movies from the year between 1919 and 2000 (inclusive) that the user is most likely to watch next. You should order them by probability and compact them in one line split by commas. Do not output the probability. Do not re-recommend movies that have been watched by the user. Do not output other words.""",
        "source_no_val": """You are a movie recommender. A {} user whose age is {} years old and whose occupation is {} has watched the following movies, listed chronologically from earliest to latest, and rated them positively:\n{}.\nYou should recommend 10 other movies from the year between 1919 and 2000 (inclusive) that the user is most likely to watch next. You should order them by probability and compact them in one line split by commas. Do not output the probability Do not re-recommend movies that have been watched by the user. Do not output other words.""",
    },
}

# --- extension, not Stairway: the neutral template with only the gender added
TEMPLATES["gender"] = {
    "source": TEMPLATES["neutral"]["source"].replace("If a user has watched", "If a {} user has watched"),
    "source_no_val": TEMPLATES["neutral"]["source_no_val"].replace("A user has watched", "A {} user has watched"),
}
assert all("{} user" in t for t in TEMPLATES["gender"].values())

_N_REQUEST = "recommend 10 other movies"


def with_n_request(template: str, n_request: int) -> str:
    """Change only the number of requested movies in a Stairway template."""
    assert template.count(_N_REQUEST) == 1
    return template.replace(_N_REQUEST, f"recommend {n_request} other movies")


# --- verbatim from preprocess/preprocessor.py::select_and_map_column_for_recai
MAP_GENDER = {"M": "male", "F": "female"}
MAP_AGE = {18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
MAP_OCCUPATION = {
    1: "academic/educator", 2: "artist", 3: "clerical/admin", 4: "college/grad student",
    5: "customer service", 6: "doctor/health care", 7: "executive/managerial", 8: "farmer",
    9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired",
    14: "sales/marketing", 15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer",
}
# Stairway drops users with occupation 0 ("other" or not specified) before
# prompting; we keep them so the 500 attack users stay identical to EXP-003/004,
# and fill in ML-1M's own README label. Exclude them post hoc for a strict match.
OCCUPATION_0 = "other"

_YEAR = re.compile(r"\s*\(\d{4}\)\s*$")


def clean_movie_name(name: str) -> str:
    """Verbatim port of Stairway's ``preprocessor.py::clean_movie_name``."""
    res = re.search(r"\((.*), (The|A|An|La|Les|El)\)", name)  # comma in bracket
    if res:
        name = name.replace(res.group(), f"({res.group(2)} {res.group(1)})")
    res = re.search(r"(.*), (The|A|An|La|Les|El) \(", name)  # comma before bracket
    if res:
        name = name.replace(res.group(), f"{res.group(2)} {res.group(1)} (")
    res = re.search(r"(.*), (The|A|An|La|Les|El)", name)  # comma (no bracket)
    if res:
        name = name.replace(res.group(), f"{res.group(2)} {res.group(1)}")
    return name


def stairway_title(catalogue_title: str) -> str:
    """'Sixth Sense, The (1999)' -> 'The Sixth Sense' (RecBole's ml-1m.item has no year)."""
    return clean_movie_name(_YEAR.sub("", catalogue_title))


# ------------------------------------------------------------------ build-prompts
def setting_name(variant: str, history_len: int, n_request: int = 10) -> str:
    """'neutral' for Stairway's setup; extensions add '_hist50' and/or '_req20'."""
    name = variant if history_len == 10 else f"{variant}_hist{history_len}"
    return name if n_request == 10 else f"{name}_req{n_request}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def _git_commit() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def build_prompts(args: argparse.Namespace) -> None:
    import pandas as pd

    splits = args.splits_dir
    items = pd.read_csv(splits / "items.csv", dtype={"item_id": str})
    title = dict(zip(items.item_id, items.movie_title.map(stairway_title)))

    def last_n(df: pd.DataFrame, n: int) -> dict[str, list[str]]:
        df = df.astype({"user_id": str, "item_id": str}).sort_values(["user_id", "timestamp"], kind="stable")
        return df.groupby("user_id").item_id.agg(lambda s: list(s)[-n:]).to_dict()

    train = last_n(pd.read_csv(splits / "train.csv"), args.history_len)
    valid = last_n(pd.read_csv(splits / "valid.csv"), 10)

    raw_users = pd.read_csv(args.users_dat, sep="::", engine="python", header=None,
                            names=["user_id", "gender", "age", "occupation", "zip"], dtype={"user_id": str})
    raw_users = raw_users.set_index("user_id")

    users: list[tuple[str, str]] = []  # (user_id, split role)
    for role, path in (("test", args.test_lists), ("train_pool", args.train_pool_lists)):
        users += [(r["user_id"], role) for r in _read_jsonl(path)]

    tmpl = {k: with_n_request(t, args.n_request) for k, t in TEMPLATES[args.variant].items()}
    setting = setting_name(args.variant, args.history_len, args.n_request)
    out = args.out_dir / f"prompts_{setting}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_val = 0
    with out.open("w", encoding="utf-8") as f:
        for uid, role in users:
            hist = train[uid]
            val = valid.get(uid, [])
            hist_t = ", ".join(title[i] for i in hist)
            val_t = ", ".join(title[i] for i in val)
            if args.variant == "sensitive":
                u = raw_users.loc[uid]
                attrs = (MAP_GENDER[u.gender], MAP_AGE[int(u.age)], MAP_OCCUPATION.get(int(u.occupation), OCCUPATION_0))
                prompt = (tmpl["source"].format(*attrs, hist_t, val_t) if val
                          else tmpl["source_no_val"].format(*attrs, hist_t))
            elif args.variant == "gender":
                g = MAP_GENDER[raw_users.loc[uid].gender]
                prompt = tmpl["source"].format(g, hist_t, val_t) if val else tmpl["source_no_val"].format(g, hist_t)
            else:
                prompt = tmpl["source"].format(hist_t, val_t) if val else tmpl["source_no_val"].format(hist_t)
            n_val += bool(val)
            f.write(json.dumps({"user_id": uid, "role": role, "variant": args.variant, "setting": setting,
                                "history_item_ids": hist, "val_item_ids": val,
                                "n_request": args.n_request, "system": SYSTEM_PROMPT, "prompt": prompt}, ensure_ascii=False) + "\n")
    print(f"wrote {len(users)} prompts to {out} ({n_val} with validation items -> 'source' template)")


# ------------------------------------------------------------------ generate
def generate(args: argparse.Namespace) -> None:
    """Runs on the cluster: only needs prompts.jsonl and an Ollama server. Resumable."""
    from src.llm.backends import OllamaBackend

    backend = OllamaBackend(model=args.model, host=args.host, temperature=0.0, seed=args.seed,
                            num_predict=args.max_tokens, num_ctx=args.num_ctx, logprobs=False,
                            timeout=600)
    prompts = _read_jsonl(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    done = {r["user_id"] for r in _read_jsonl(args.out)} if args.out.exists() else set()
    todo = [p for p in prompts if p["user_id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to go ({args.model})", flush=True)
    t0 = time.time()
    with args.out.open("a", encoding="utf-8") as f:
        for n, p in enumerate(todo, 1):
            resp = backend.generate(p["prompt"], system=p["system"])
            f.write(json.dumps({"user_id": p["user_id"], "role": p["role"], "variant": p["variant"],
                                "model": args.model, "answer": resp.text}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 100 == 0:
                rate = (time.time() - t0) / n
                print(f"{n}/{len(todo)}  {rate:.2f}s/user  eta {rate * (len(todo) - n) / 3600:.1f}h", flush=True)


# ------------------------------------------------------------------ postprocess
def clean_answer(answer: str, model: str) -> list[str]:
    """Port of Stairway's ``cleaner.py::clean`` (pandas .str ops -> plain str) for ml-1m."""
    a = answer
    if "llama" in model:
        a = a.split("\n\n")[-1]
    a = a.replace(",)", ")")
    a = re.sub(r"\,\(", ", (", a)
    a = re.sub(r"\(0\.\d*\) ?", "", a)
    a = re.sub(r", \.\d*\)", "", a)
    a = a.replace(" ,", "").replace(", (", ", ").strip("(").replace(",", ", ")
    a = a.replace("  ", " ").replace("_", " ")
    a = re.sub(r"(, \()?0?\.\d*\, \)", "", a)
    a = re.sub(r"\( ?", "", a)
    a = re.sub(r"0\.\d*\, ", "", a)
    a = a.replace(" ,", ",").replace("), ", ", ")
    # ml-1m specific
    a = a.replace(" . ", " ").replace(")-", " - ").replace(".", " ").replace(":", ": ")
    return [el.strip(" ") if el and el[0] == " " else el for el in a.split(",")]


class TitleMatcher:
    """Stairway's ``TFIDF_model``: character 3-grams (no whitespace), TF-IDF, cosine."""

    def __init__(self, corpus: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=self._ngrams).fit(corpus)
        self.corpus_vec = self.vectorizer.transform(corpus)

    @staticmethod
    def _ngrams(s: str) -> list[str]:
        s = re.sub(r"\s+", " ", s.lower()).strip()
        return ["".join(g) for g in zip(*[s[i:] for i in range(3)]) if " " not in "".join(g)]

    def best(self, docs: list[str]) -> list[tuple[int, float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        if not docs:
            return []
        sim = cosine_similarity(self.vectorizer.transform(docs), self.corpus_vec)
        idx = sim.argmax(axis=1)  # first max, as map_index
        return [(int(i), float(sim[r, i])) for r, i in enumerate(idx)]


def _rank_metrics(rec: list[str], positives: set[str]) -> dict[str, float]:
    import math

    rel = [1 if i in positives else 0 for i in rec[:TOPK]]
    dcg = sum(r / math.log2(k + 2) for k, r in enumerate(rel))
    idcg = sum(1 / math.log2(k + 2) for k in range(min(len(positives), TOPK)))
    return {"hit": float(any(rel)), "precision": sum(rel) / TOPK, "ndcg": dcg / idcg if idcg else 0.0}


def postprocess(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    splits = args.splits_dir
    items = pd.read_csv(splits / "items.csv", dtype={"item_id": str})
    ids, cat_titles = list(items.item_id), list(items.movie_title)
    st_titles = [stairway_title(t) for t in cat_titles]
    matcher = TitleMatcher(st_titles)
    dup_titles = {t for t, c in Counter(t.lower() for t in st_titles).items() if c > 1}

    train_all = pd.read_csv(splits / "train.csv", dtype={"user_id": str, "item_id": str})
    seen = train_all.groupby("user_id").item_id.agg(set).to_dict()
    test = pd.read_csv(splits / "test.csv", dtype={"user_id": str, "item_id": str})
    test_pos = test.groupby("user_id").item_id.agg(set).to_dict()

    meta = {}
    for role, path in (("test", args.test_lists), ("train_pool", args.train_pool_lists)):
        for r in _read_jsonl(path):
            meta[r["user_id"]] = {"gender": r["gender"], "age": r["age"], "role": role}
    setting = setting_name(args.variant, args.history_len, args.n_request)
    prompts = {p["user_id"]: p for p in _read_jsonl(args.run_dir / f"prompts_{setting}.jsonl")}

    tag = f"{setting}_{args.model_slug}"
    resp_path = args.run_dir / f"responses_{tag}.jsonl"

    # One pass over the responses: every uniquely matched item, in the LLM's order.
    parsed_users = []
    for r in _read_jsonl(resp_path):
        parsed = [t for t in clean_answer(r["answer"], r["model"]) if t.strip()]
        rec_all: list[str] = []
        n_matched = n_dup = 0
        for (i, s) in matcher.best(parsed):
            if s < SIM_THRESHOLD:
                continue
            n_matched += 1
            if ids[i] in rec_all:
                n_dup += 1
                continue
            rec_all.append(ids[i])
        parsed_users.append((r["user_id"], parsed, rec_all, n_matched, n_dup))

    # Two list versions from the same generation:
    #   raw     = the LLM's output as Stairway evaluates it (may contain items the
    #             user already watched, since the prompt only shows the last 10)
    #   unseen  = train-history items removed, the same rule as the EXP-003/004
    #             RecBole lists (full-sort with seen items masked) -> the version
    #             comparable with the CF recommenders in the master table
    for version in ("raw", "unseen"):
        out_dir = args.run_dir / (tag if version == "raw" else f"{tag}_unseen")
        out_dir.mkdir(parents=True, exist_ok=True)
        lists: dict[str, list[dict[str, Any]]] = {"test": [], "train_pool": []}
        diag_rows = []
        for uid, parsed, rec_all, n_matched, n_dup in parsed_users:
            user_seen = seen.get(uid, set())
            rec_ids = rec_all if version == "raw" else [i for i in rec_all if i not in user_seen]
            rec_ids = rec_ids[:TOPK]
            hist = prompts[uid]["history_item_ids"]
            m = meta[uid]
            lists[m["role"]].append({
                "user_id": uid, "gender": m["gender"], "age": m["age"],
                "history_item_ids": hist, "history_titles": [cat_titles[ids.index(i)] for i in hist],
                "prompt_item_ids": hist + prompts[uid]["val_item_ids"],
                "rec_item_ids": rec_ids, "rec_titles": [cat_titles[ids.index(i)] for i in rec_ids],
            })
            diag_rows.append({
                "user_id": uid, "gender": m["gender"], "role": m["role"],
                "n_parsed": len(parsed), "n_matched": n_matched, "n_dup": n_dup, "n_rec": len(rec_ids),
                "n_seen_all": sum(i in user_seen for i in rec_all[:TOPK]),
                "n_seen_in_prompt": sum(i in set(hist) | set(prompts[uid]["val_item_ids"]) for i in rec_all[:TOPK]),
                "n_rec_all": len(rec_all[:TOPK]),
                "n_ambiguous": sum(st_titles[ids.index(i)].lower() in dup_titles for i in rec_ids),
            })

        for role, fname in (("test", "lists.jsonl"), ("train_pool", "lists_train_pool.jsonl")):
            with (out_dir / fname).open("w", encoding="utf-8") as f:
                for row in lists[role]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        d = pd.DataFrame(diag_rows)

        def summ(df: pd.DataFrame) -> dict[str, Any]:
            n_raw = max(df.n_rec_all.sum(), 1)
            return {
                "n_users": int(len(df)),
                "mean_parsed": float(df.n_parsed.mean()),
                "match_rate": float(df.n_matched.sum() / max(df.n_parsed.sum(), 1)),
                "mean_rec_len": float(df.n_rec.mean()),
                "share_full_10": float((df.n_rec == TOPK).mean()),
                "share_under_5": float((df.n_rec < 5).mean()),
                "share_empty": float((df.n_rec == 0).mean()),
                "dup_rate": float(df.n_dup.sum() / max(df.n_matched.sum(), 1)),
                "raw_seen_rate": float(df.n_seen_all.sum() / n_raw),
                "raw_seen_in_prompt_rate": float(df.n_seen_in_prompt.sum() / n_raw),
                "ambiguous_title_rate": float(df.n_ambiguous.sum() / max(df.n_rec.sum(), 1)),
            }

        all_recs = Counter(i for role in lists.values() for row in role for i in row["rec_item_ids"])
        n_slots = sum(all_recs.values())
        top = [c for _, c in all_recs.most_common()]
        diagnostics = {
            "overall": summ(d),
            "by_gender": {g: summ(d[d.gender == g]) for g in ("male", "female")},
            "concentration": {
                "distinct_items": len(all_recs),
                "top10_share": sum(top[:10]) / n_slots if n_slots else None,
                "top50_share": sum(top[:50]) / n_slots if n_slots else None,
            },
        }

        per_user = [_rank_metrics(row["rec_item_ids"], test_pos[row["user_id"]])
                    for role in lists.values() for row in role if row["user_id"] in test_pos]
        rec_metrics = {
            "n_users_with_test_items": len(per_user),
            **{f"{k}@{TOPK}": float(np.mean([u[k] for u in per_user])) if per_user else None
               for k in ("hit", "precision", "ndcg")},
            "note": "ID-level after title->ID matching; Stairway scores titles against the ground truth directly",
        }

        report = {
            "exp": "EXP-006_llmrec_311", "variant": args.variant, "history_len": args.history_len,
            "n_request": args.n_request,
            "model_slug": args.model_slug,
            "list_version": version, "sim_threshold": SIM_THRESHOLD,
            "diagnostics": diagnostics, "rec_metrics": rec_metrics,
            "provenance": {"generated_at_iso": datetime.now(timezone.utc).isoformat(), "git": _git_commit(),
                           "responses": str(resp_path), "system_prompt": SYSTEM_PROMPT,
                           "templates": {k: with_n_request(t, args.n_request)
                                         for k, t in TEMPLATES[args.variant].items()}},
        }
        (out_dir / "llmrec_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        o = diagnostics["overall"]
        print(f"[{tag} / {version}] users {o['n_users']}, match {o['match_rate']:.2f}, "
              f"len {o['mean_rec_len']:.1f} (<5: {o['share_under_5']:.1%}, empty {o['share_empty']:.1%}), "
              f"raw seen {o['raw_seen_rate']:.1%} (in prompt {o['raw_seen_in_prompt_rate']:.1%}), "
              f"distinct items {diagnostics['concentration']['distinct_items']}, "
              f"HR@10 {rec_metrics['hit@10']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    lists_dir = Path("runs/EXP-004_lightgcn_d32_311")  # the fixed 500 test users + train pool

    b = sub.add_parser("build-prompts")
    b.add_argument("--variant", choices=list(TEMPLATES), required=True)
    b.add_argument("--history-len", type=int, default=10, help="last N train items (Stairway: 10)")
    b.add_argument("--n-request", type=int, default=10, help="movies asked for (Stairway: 10)")
    b.add_argument("--splits-dir", type=Path, default=Path("data/ml-1m-311/splits"))
    b.add_argument("--users-dat", type=Path, default=Path("dataset/ml-1m/users.dat"))
    b.add_argument("--test-lists", type=Path, default=lists_dir / "lists.jsonl")
    b.add_argument("--train-pool-lists", type=Path, default=lists_dir / "lists_train_pool.jsonl")
    b.add_argument("--out-dir", type=Path, default=RUN_DIR)

    g = sub.add_parser("generate")
    g.add_argument("--prompts", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--host", default="http://127.0.0.1:11434")
    g.add_argument("--seed", type=int, default=2025)
    g.add_argument("--max-tokens", type=int, default=1000)
    g.add_argument("--num-ctx", type=int, default=4096)  # vLLM max_model_len=4096 in Stairway
    g.add_argument("--limit", type=int, default=None)

    q = sub.add_parser("postprocess")
    q.add_argument("--variant", choices=list(TEMPLATES), required=True)
    q.add_argument("--history-len", type=int, default=10)
    q.add_argument("--n-request", type=int, default=10)
    q.add_argument("--model-slug", required=True, help="e.g. glm4_9b, llama31_8b")
    q.add_argument("--run-dir", type=Path, default=RUN_DIR)
    q.add_argument("--splits-dir", type=Path, default=Path("data/ml-1m-311/splits"))
    q.add_argument("--test-lists", type=Path, default=lists_dir / "lists.jsonl")
    q.add_argument("--train-pool-lists", type=Path, default=lists_dir / "lists_train_pool.jsonl")

    args = p.parse_args()
    {"build-prompts": build_prompts, "generate": generate, "postprocess": postprocess}[args.cmd](args)


if __name__ == "__main__":
    main()
