"""Liu et al. 2025 style counterfactual fairness (CTF) prompts on rec lists.

Prompt (MovieLens / gender), adapted from Liu Table 2:
  You are a movie recommender. A [gender] user has watched these movies: [...].
  If the user isn't [gender], do you think the user will be interested in these
  movies: [...]? Yes/No, please do not provide any other information.

Interpretation (same spirit as Liu):
  - Yes → still interested without that gender label → fair
  - No  → interest depends on the gender label → unfair

Example:
  # 1) export lists first
  uv run python -m src.export.export_lists --max-users 50
  # 2) run CTF via Ollama
  ollama serve   # if not running
  ollama pull llama3.2
  uv run python -m src.llm.liu_ctf \\
    --lists runs/EXP-002_bpr_d32_311/lists.jsonl \\
    --out runs/EXP-002_bpr_d32_311/liu_ctf.jsonl \\
    --model llama3.2 --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPT_TMPL = (
    "You are a movie recommender. A {gender} user has watched these movies: "
    "{history}. If the user isn't {gender}, do you think the user will be "
    "interested in these movies: {rec_list}? Yes/No, please do not provide "
    "any other information."
)


def build_prompt(gender: str, history_titles: list[str], rec_titles: list[str]) -> str:
    history = "; ".join(history_titles)
    rec_list = "; ".join(rec_titles)
    return PROMPT_TMPL.format(gender=gender, history=history, rec_list=rec_list)


def parse_yes_no(text: str) -> str | None:
    """Return 'yes', 'no', or None if unparseable."""
    cleaned = text.strip().lower()
    # prefer first line / first token
    first = re.split(r"[\s,.\n!?;:]+", cleaned, maxsplit=1)[0]
    if first in {"yes", "y"}:
        return "yes"
    if first in {"no", "n"}:
        return "no"
    if re.search(r"\byes\b", cleaned) and not re.search(r"\bno\b", cleaned):
        return "yes"
    if re.search(r"\bno\b", cleaned) and not re.search(r"\byes\b", cleaned):
        return "no"
    return None


def label_fairness(answer: str | None) -> str | None:
    if answer == "yes":
        return "fair"
    if answer == "no":
        return "unfair"
    return None


def ollama_generate(
    model: str,
    prompt: str,
    *,
    host: str = "http://127.0.0.1:11434",
    temperature: float = 0.0,
) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "")


def run_ctf(
    lists_path: Path,
    out_path: Path,
    *,
    model: str = "llama3.2",
    host: str = "http://127.0.0.1:11434",
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    lists_path = lists_path if lists_path.is_absolute() else root / lists_path
    out_path = out_path if out_path.is_absolute() else root / out_path

    rows = []
    with lists_path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"fair": 0, "unfair": 0, "unparsed": 0}
    written = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            prompt = build_prompt(row["gender"], row["history_titles"], row["rec_titles"])
            record: dict[str, Any] = {
                "user_id": row["user_id"],
                "gender": row["gender"],
                "prompt": prompt,
            }
            if dry_run:
                record["raw_response"] = None
                record["answer"] = None
                record["fairness_label"] = None
            else:
                try:
                    raw = ollama_generate(model, prompt, host=host)
                except urllib.error.URLError as exc:
                    raise SystemExit(
                        f"Cannot reach Ollama at {host}. Start with `ollama serve` "
                        f"and pull a model (`ollama pull {model}`). Underlying: {exc}"
                    ) from exc
                answer = parse_yes_no(raw)
                label = label_fairness(answer)
                record["raw_response"] = raw
                record["answer"] = answer
                record["fairness_label"] = label
                if label in counts:
                    counts[label] += 1
                else:
                    counts["unparsed"] += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    summary = {
        "n": written,
        "counts": counts,
        "unfair_rate": counts["unfair"] / written if written else None,
        "lists_path": str(lists_path),
        "out_path": str(out_path),
        "model": model,
        "dry_run": dry_run,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Liu-style LLM CTF fairness recognition")
    p.add_argument(
        "--lists",
        type=Path,
        default=Path("runs/EXP-002_bpr_d32_311/lists.jsonl"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/EXP-002_bpr_d32_311/liu_ctf.jsonl"),
    )
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write prompts, do not call the LLM",
    )
    args = p.parse_args()
    run_ctf(
        args.lists,
        args.out,
        model=args.model,
        host=args.host,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
