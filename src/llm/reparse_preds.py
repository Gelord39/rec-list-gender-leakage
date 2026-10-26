"""Re-derive pred_gender (and the metrics that depend on it) from stored raw responses.

``parse_gender`` was fixed on 2026-09-21: when a response contains a ``Gender:``
line, only the LAST one counts, and a non-male/female answer there ("Gender:
Indeterminate") is an abstention. The old free-text fallback could pick a
male/female word out of the *reasoning* and mislabel an abstention — which hit
glm4:9b's staab runs hard (74/500 labels) and llama3.2's barely at all.

Every preds file stores the full ``raw_response``, so labels and metrics can be
re-derived offline without calling any LLM again. Scores are untouched: they
come from the answer-token logprobs and were always read at the last
``Gender:``. Usage:

  uv run python -m src.llm.reparse_preds runs/EXP-004_*/llm_aia/preds_*.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.llm.list_aia import aggregate_by_gender, parse_gender


def reparse(preds_path: Path, *, dry_run: bool = False) -> dict[str, int]:
    rows = [json.loads(line) for line in preds_path.open(encoding="utf-8")]
    changed = 0
    for r in rows:
        if r.get("raw_response") is None:
            continue
        new = parse_gender(r["raw_response"])
        if new != r.get("pred_gender"):
            changed += 1
            r["pred_gender"] = new
    if not dry_run:
        with preds_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        metrics_path = preds_path.parent / preds_path.name.replace("preds_", "metrics_").replace(".jsonl", ".json")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics.update(aggregate_by_gender(rows))
            metrics["reparsed"] = {
                "at": datetime.now(timezone.utc).isoformat(),
                "reason": "parse_gender fix: last Gender: line only; non-male/female there = abstention",
                "labels_changed": changed,
            }
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"n": len(rows), "changed": changed}


def main() -> None:
    p = argparse.ArgumentParser(description="Re-derive pred_gender from stored raw responses")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for path in args.paths:
        if not path.name.startswith("preds_"):
            continue
        res = reparse(path, dry_run=args.dry_run)
        flag = "" if res["changed"] == 0 else "  <-- changed"
        print(f"{path.name:<58} {res['changed']:>4}/{res['n']} labels changed{flag}")


if __name__ == "__main__":
    main()
