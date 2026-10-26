"""LLM attribute inference attack on recommendation lists (list-only).

Pilot for EXP-003:
  1) export lists (history-size 0)
  2) run this module with --style zero_shot|few_shot|staab

Output files are tagged (``--tag``, required) so re-runs never overwrite a
previous run when the prompt/examples change, e.g. ``synthetic_v0`` for the
original hardcoded few-shot examples vs ``real_v1`` for the real-user ones.

Example:
  uv run python -m src.llm.list_aia \\
    --lists runs/EXP-003_bpr_d32_311/lists.jsonl \\
    --out-dir runs/EXP-003_bpr_d32_311/llm_aia \\
    --style zero_shot --tag real_v1 \\
    --backend ollama --model llama3.2 \\
    --limit 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from src.llm.backends import make_backend
from src.llm.prompts import (
    BASE_STYLES,
    HISTORY_STYLES,
    MANY_SHOT_STYLES,
    PROMPT_BUILDERS,
    build_prompt,
    get_few_shot_examples,
)


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


def parse_gender(text: str) -> str | None:
    """Extract the model's final male/female answer, or None if it did not give one.

    If the response contains a ``Gender:`` line, the LAST one is the answer and
    nothing else counts — matching where ``gender_score`` reads the logprobs.
    This matters for the reasoning styles: a model that reasons "often
    associated with a female audience" and then answers "Gender: Indeterminate"
    has abstained, and the old fallback (first male/female token anywhere in the
    text) mislabelled such answers from the reasoning. The free-text fallback is
    only used when there is no ``Gender:`` line at all (zero_shot / few_shot).
    """
    cleaned = text.strip().lower()
    labelled = re.findall(r"\bgender\s*[:\-]\s*([a-z]+)", cleaned)
    if labelled:
        return labelled[-1] if labelled[-1] in {"male", "female"} else None
    for token in re.split(r"[\s,.\n!?;:]+", cleaned):
        if token in {"male", "m", "man", "boy"}:
            return "male"
        if token in {"female", "f", "woman", "girl"}:
            return "female"
    has_male = bool(re.search(r"\bmale\b", cleaned))
    has_female = bool(re.search(r"\bfemale\b", cleaned))
    if has_male and not has_female:
        return "male"
    if has_female and not has_male:
        return "female"
    return None


def _answer_word(token: str) -> str | None:
    w = token.strip().strip("*").lower()
    return w if w in {"male", "female"} else None


def gender_score(
    logprobs: list[dict[str, Any]] | None,
) -> tuple[float | None, list[dict[str, Any]] | None, str | None]:
    """Continuous attack score from token logprobs: P(male) / (P(male) + P(female)).

    The *answer slot* is the first non-whitespace generated token after the LAST
    ``Gender:`` (staab styles reason first); with no ``Gender:`` it is the first
    male/female token, else the first token generated (zero_shot / few_shot).

    Scored at that slot even when the emitted token is NOT male/female — e.g.
    glm4:9b sometimes answers "Gender: Neutral" / "Indeterminate" after
    reasoning that the list is not conclusive. Those users get no hard label
    (``pred_gender`` stays None, so BAcc is unaffected) but still get a score,
    because the model's distribution over male/female at that slot is exactly
    the leakage the AUC is meant to measure. Protocol change approved
    2026-09-20; before that a score required the model to emit male/female.

    Score convention matches ``fixed_split_aia`` (``predict_proba[:, 1]`` with
    classes ``[female, male]``). Returns ``(None, None, None)`` when the answer
    slot has no male/female mass in the returned top_logprobs (e.g. a hard
    safety refusal that never reaches an answer). The third element is the
    token actually emitted at the slot, for abstention analysis.
    """
    if not logprobs:
        return None, None, None

    start, text = None, ""
    for i, t in enumerate(logprobs):
        text += t.get("token", "")
        if re.search(r"gender\s*:\s*$", text.lower()):
            start = i + 1
    if start is None:
        start = next(
            (i for i, t in enumerate(logprobs) if _answer_word(t.get("token", "")) is not None),
            0,
        )

    for t in logprobs[start:]:
        token = t.get("token", "")
        if not token.strip():
            continue  # skip whitespace between "Gender:" and the answer
        top = t.get("top_logprobs") or []
        mass = {"male": 0.0, "female": 0.0}
        for c in top:
            w = _answer_word(c.get("token", ""))
            if w:
                mass[w] += math.exp(c["logprob"])
        total = mass["male"] + mass["female"]
        if total <= 0:
            return None, top, token
        return mass["male"] / total, top, token
    return None, None, None


def _safe_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float | None]:
    if not y_true:
        return {"n": 0, "accuracy": None, "bacc": None, "f1_micro": None}
    acc = float(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true))
    try:
        bacc = float(balanced_accuracy_score(y_true, y_pred))
    except ValueError:
        bacc = None
    try:
        f1m = float(f1_score(y_true, y_pred, average="micro"))
    except ValueError:
        f1m = None
    return {
        "n": len(y_true),
        "accuracy": acc,
        "bacc": bacc,
        "f1_micro": f1m,
    }


def aggregate_by_gender(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Overall + per true-gender attack success (among parsed predictions)."""
    y_true: list[str] = []
    y_pred: list[str] = []
    by_group: dict[str, list[tuple[str, str]]] = defaultdict(list)
    n_unparsed = 0

    for r in records:
        gold = r.get("gender")
        pred = r.get("pred_gender")
        if gold is None:
            continue
        if pred is None:
            n_unparsed += 1
            continue
        y_true.append(gold)
        y_pred.append(pred)
        by_group[gold].append((gold, pred))

    overall = _safe_metrics(y_true, y_pred)
    per_gender = {}
    for g, pairs in by_group.items():
        gt = [a for a, _ in pairs]
        pr = [b for _, b in pairs]
        # Within a single true gender, BAcc is less meaningful; report accuracy
        # (= recall for that class) as "how often this group is correctly guessed".
        per_gender[g] = {
            "n": len(pairs),
            "accuracy": float(sum(a == b for a, b in pairs) / len(pairs)),
            "note": "accuracy = fraction of users with true gender=g correctly guessed",
        }

    scored = [
        r for r in records if r.get("gender") is not None and r.get("score") is not None
    ]
    auc = None
    if scored and len({r["gender"] for r in scored}) == 2:
        auc = float(
            roc_auc_score(
                [r["gender"] == "male" for r in scored], [r["score"] for r in scored]
            )
        )
    overall["auc"] = auc
    overall["n_scored"] = len(scored)
    overall["auc_score_definition"] = (
        "score = P(male)/(P(male)+P(female)) from answer-token logprobs; "
        "AUC over users with a score (refusals excluded)"
    )

    majority = max(set(y_true), key=y_true.count) if y_true else None
    majority_acc = (
        float(sum(t == majority for t in y_true) / len(y_true)) if y_true else None
    )

    return {
        "n_records": len(records),
        "n_parsed": len(y_true),
        "n_unparsed": n_unparsed,
        "overall": overall,
        "by_true_gender": per_gender,
        "baselines": {
            "majority_label": majority,
            "majority_accuracy": majority_acc,
            "chance_bacc": 0.5,
        },
    }


def sample_shot_examples(
    pool_path: Path, n_shots: int, *, seed: int = 2024
) -> list[dict[str, Any]]:
    """Draw ``n_shots`` labelled exemplars from a train-pool lists.jsonl.

    The pool (``lists_train_pool.jsonl``) is disjoint from the fixed test set,
    so exemplars never leak test users. Sampling is uniform at random with a
    fixed seed — i.e. the natural ~70/30 male/female mix of the population,
    not rebalanced, so the prompt reflects the real prior.
    """
    rows = [json.loads(line) for line in Path(pool_path).open(encoding="utf-8")]
    rows = [r for r in rows if r.get("rec_titles") and r.get("gender")]
    if n_shots > len(rows):
        raise ValueError(f"--n-shots {n_shots} exceeds pool size {len(rows)}")
    rng = random.Random(seed)
    return [
        {"user_id": r["user_id"], "rec_titles": r["rec_titles"], "gender": r["gender"]}
        for r in rng.sample(rows, n_shots)
    ]


def run_list_aia(
    lists_path: Path,
    out_dir: Path,
    *,
    style: str,
    tag: str,
    backend_kind: str = "ollama",
    model: str = "llama3.2",
    host: str = "http://127.0.0.1:11434",
    limit: int | None = None,
    dry_run: bool = False,
    examples: list[dict[str, Any]] | None = None,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Run one prompt style's list-only AIA and write tagged result files.

    Filenames are ``preds_{style}_{tag}.jsonl`` / ``metrics_{style}_{tag}.json``
    — ``tag`` is a required, human-meaningful label for this run (e.g.
    ``synthetic_v0`` for the original hardcoded few-shot examples,
    ``real_v1`` for the real-user ones) so re-runs never silently overwrite a
    previous run's results when the prompt/examples change. The metrics file
    additionally embeds a ``provenance`` block — the exact system prompt, an
    example fully-rendered user prompt, the few-shot exemplars actually used
    (if the style has any), the git commit, and a UTC timestamp — so the
    file is self-describing without needing to cross-reference code history.
    """
    root = Path(__file__).resolve().parents[2]
    lists_path = lists_path if lists_path.is_absolute() else root / lists_path
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with lists_path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]

    backend = None if dry_run else make_backend(backend_kind, model, host=host, num_ctx=num_ctx)

    pred_path = out_dir / f"preds_{style}_{tag}.jsonl"
    records: list[dict[str, Any]] = []

    with pred_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            if style in HISTORY_STYLES:
                titles = row.get("history_titles") or []
                if not titles:
                    raise ValueError(
                        f"style {style!r} needs history_titles; re-export the lists with "
                        "--history-size N (they are empty in list-only exports)"
                    )
            else:
                titles = row["rec_titles"]
            system, prompt = build_prompt(style, titles, examples)
            rec: dict[str, Any] = {
                "user_id": row["user_id"],
                "gender": row["gender"],
                "age": row.get("age"),
                "style": style,
                "system": system,
                "prompt": prompt,
                "rec_titles": row["rec_titles"],
                "history_titles": row.get("history_titles") if style in HISTORY_STYLES else None,
            }
            if dry_run:
                rec["raw_response"] = None
                rec["pred_gender"] = None
                rec["score"] = None
            else:
                assert backend is not None
                resp = backend.generate(prompt, system=system)
                pred = parse_gender(resp.text)
                rec["raw_response"] = resp.text
                rec["pred_gender"] = pred
                score, top, answer_token = gender_score(resp.logprobs)
                rec["score"] = score
                rec["answer_top_logprobs"] = top
                rec["answer_token"] = answer_token
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)

    if examples is not None:
        few_shot_examples_used = examples
    else:
        try:
            few_shot_examples_used = get_few_shot_examples() if "few_shot" in style else None
        except FileNotFoundError:
            few_shot_examples_used = None

    metrics = {
        "exp": "list_aia",
        "style": style,
        "tag": tag,
        "backend": backend_kind,
        "model": model,
        "dry_run": dry_run,
        "num_ctx": num_ctx,
        "lists_path": str(lists_path),
        "preds_path": str(pred_path),
        "input": "watched_history_only" if style in HISTORY_STYLES else "rec_list_only",
        "attribute": "gender",
        "provenance": {
            "generated_at_iso": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "system_prompt": records[0]["system"] if records else None,
            "example_rendered_prompt": records[0]["prompt"] if records else None,
            "n_shots": len(few_shot_examples_used) if few_shot_examples_used else 0,
            "few_shot_examples_used": few_shot_examples_used,
        },
        **aggregate_by_gender(records),
    }
    metrics_path = out_dir / f"metrics_{style}_{tag}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="LLM AIA on recommendation lists")
    p.add_argument(
        "--lists",
        type=Path,
        default=Path("runs/EXP-003_bpr_d32_311/lists.jsonl"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/EXP-003_bpr_d32_311/llm_aia"),
    )
    p.add_argument(
        "--style",
        choices=sorted(PROMPT_BUILDERS),
        default="zero_shot",
    )
    p.add_argument(
        "--tag",
        required=True,
        help="Human-meaningful run label, e.g. synthetic_v0, real_v1 — becomes part of the output filenames",
    )
    p.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompts only; do not call the LLM",
    )
    p.add_argument(
        "--all-styles",
        action="store_true",
        help="Run the four base styles sequentially (not the many-shot ones)",
    )
    p.add_argument(
        "--n-shots",
        type=int,
        default=None,
        help="number of in-context exemplars for --style many_shot/staab_many_shot",
    )
    p.add_argument(
        "--shot-pool",
        type=Path,
        default=None,
        help="lists.jsonl to draw many-shot exemplars from (must be disjoint from --lists)",
    )
    p.add_argument("--shot-seed", type=int, default=2024)
    p.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Ollama context window (default 4096 truncates many-shot prompts silently)",
    )
    args = p.parse_args()

    styles = list(BASE_STYLES) if args.all_styles else [args.style]
    examples = None
    if any(st in MANY_SHOT_STYLES for st in styles):
        if not args.n_shots or not args.shot_pool:
            raise SystemExit("many-shot styles need both --n-shots and --shot-pool")
        examples = sample_shot_examples(args.shot_pool, args.n_shots, seed=args.shot_seed)
        n_male = sum(e["gender"] == "male" for e in examples)
        print(f"[many-shot] {len(examples)} exemplars from {args.shot_pool} "
              f"({n_male} male / {len(examples) - n_male} female, seed {args.shot_seed})")
    for style in styles:
        run_list_aia(
            args.lists,
            args.out_dir,
            style=style,
            tag=args.tag,
            backend_kind=args.backend,
            model=args.model,
            host=args.host,
            limit=args.limit,
            dry_run=args.dry_run,
            examples=examples,
            num_ctx=args.num_ctx,
        )


if __name__ == "__main__":
    main()
