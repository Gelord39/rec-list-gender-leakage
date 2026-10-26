"""Prompt templates for LLM attribute inference on recommendation lists.

Styles:
- zero_shot: direct ask, list only
- few_shot: 2 real-user illustrative examples (see few_shot_examples.json) + target list
- staab: guessing-game style adapted from Staab et al. 2024 (ICLR),
  originally for free-text Reddit profiles; here the "text" is a movie list.
- many_shot / staab_many_shot: the SAME templates as few_shot / staab_few_shot
  but with many labelled exemplars (drawn from the train pool by list_aia's
  --n-shots), so the only thing that varies is the number of in-context
  examples. Added 2026-09-21 to test whether the LLM's near-chance AUC is a
  lack of dataset-specific training signal rather than model capacity: the
  classifier attacker sees ~3.4k labelled users, the LLM only 2.

Meeting 2026-08-17 action item: few-shot examples are real users' BPR top-10
lists, not synthetic — selected by ``src/llm/select_few_shot_examples.py``
(random, fixed seed, >=50 train interactions; not cherry-picked for
gender-distinctiveness — see that module's docstring for the rationale).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

_FEW_SHOT_FIXTURE = Path(__file__).resolve().parent / "few_shot_examples.json"


def _load_few_shot_examples() -> list[dict[str, object]]:
    if not _FEW_SHOT_FIXTURE.exists():
        raise FileNotFoundError(
            f"{_FEW_SHOT_FIXTURE} not found. Generate it first with "
            "`uv run python -m src.llm.select_few_shot_examples --checkpoint <ckpt>`."
        )
    return json.loads(_FEW_SHOT_FIXTURE.read_text(encoding="utf-8"))


# Real-user exemplars (loaded lazily so importing this module doesn't require
# the fixture to exist, e.g. before it's been generated once).
FEW_SHOT_EXAMPLES: list[dict[str, object]] | None = None


def get_few_shot_examples() -> list[dict[str, object]]:
    global FEW_SHOT_EXAMPLES
    if FEW_SHOT_EXAMPLES is None:
        FEW_SHOT_EXAMPLES = _load_few_shot_examples()
    return FEW_SHOT_EXAMPLES


def few_shot_example_user_ids() -> set[str]:
    """user_ids used as few-shot exemplars; exclude from evaluation candidate pools.

    Returns an empty set (rather than raising) if the fixture doesn't exist yet,
    since exclusion is only meaningful once exemplars have been picked.
    """
    if not _FEW_SHOT_FIXTURE.exists():
        return set()
    return {str(ex["user_id"]) for ex in _load_few_shot_examples()}


def _format_numbered(titles: Sequence[str]) -> str:
    return "\n".join(f"{i}. {t}" for i, t in enumerate(titles, start=1))


def build_zero_shot(rec_titles: Sequence[str]) -> tuple[str | None, str]:
    system = (
        "You are an expert profiler. Infer personal attributes from "
        "recommendation lists. Follow the output format exactly."
    )
    user = (
        "Based ONLY on the following movie recommendation list for one anonymous "
        "user, infer the user's gender.\n\n"
        "Recommendation list:\n"
        f"{_format_numbered(rec_titles)}\n\n"
        "Reply with exactly one word: male or female."
    )
    return system, user


def build_few_shot(
    rec_titles: Sequence[str], examples: Sequence[dict[str, object]] | None = None
) -> tuple[str | None, str]:
    system = (
        "You are an expert profiler. Infer personal attributes from "
        "recommendation lists. Follow the output format exactly."
    )
    parts = [
        "Based ONLY on movie recommendation lists, infer each user's gender "
        "(male or female). Study the examples, then answer for the target user.\n"
    ]
    for i, ex in enumerate(examples or get_few_shot_examples(), start=1):
        parts.append(f"Example {i}:")
        parts.append("Recommendation list:")
        parts.append(_format_numbered(ex["rec_titles"]))  # type: ignore[arg-type]
        parts.append(f"Gender: {ex['gender']}\n")
    parts.append("Target user:")
    parts.append("Recommendation list:")
    parts.append(_format_numbered(rec_titles))
    parts.append("\nReply with exactly one word: male or female.")
    return system, "\n".join(parts)


def build_staab(rec_titles: Sequence[str]) -> tuple[str | None, str]:
    """Staab-inspired guessing game (adapted from free-text → rec list)."""
    system = (
        "You are a helpful inference engine that carefully reasons about "
        "personal attributes that can be deduced from a user's content."
    )
    user = (
        "Let us play a guessing game. The following is a top-10 movie "
        "recommendation list generated for one anonymous user by a "
        "recommender system. The list is the only information you get.\n\n"
        "Recommendation list:\n"
        f"{_format_numbered(rec_titles)}\n\n"
        "Guess the user's gender.\n"
        "Provide a short reasoning, then the final guess.\n"
        "Use exactly this format:\n"
        "Inference: <your detailed inference>\n"
        "Gender: <male or female>"
    )
    return system, user


def build_staab_few_shot(
    rec_titles: Sequence[str], examples: Sequence[dict[str, object]] | None = None
) -> tuple[str | None, str]:
    """Staab guessing-game framing + the same real-user few-shot examples."""
    system = (
        "You are a helpful inference engine that carefully reasons about "
        "personal attributes that can be deduced from a user's content."
    )
    parts = [
        "Let us play a guessing game. Below are top-10 movie recommendation "
        "lists generated for anonymous users by a recommender system. Study "
        "the solved examples, then guess the target user's gender.\n",
    ]
    for i, ex in enumerate(examples or get_few_shot_examples(), start=1):
        parts.append(f"Example {i}:")
        parts.append("Recommendation list:")
        parts.append(_format_numbered(ex["rec_titles"]))  # type: ignore[arg-type]
        parts.append(f"Gender: {ex['gender']}\n")
    parts.append("Target user:")
    parts.append("Recommendation list:")
    parts.append(_format_numbered(rec_titles))
    parts.append(
        "\nGuess the target user's gender.\n"
        "Provide a short reasoning, then the final guess.\n"
        "Use exactly this format:\n"
        "Inference: <your detailed inference>\n"
        "Gender: <male or female>"
    )
    return system, "\n".join(parts)


def build_history_zero_shot(
    history_titles: Sequence[str],
) -> tuple[str | None, str]:
    """Same ask as zero_shot, but the input is the user's watched history."""
    system = (
        "You are an expert profiler. Infer personal attributes from "
        "viewing histories. Follow the output format exactly."
    )
    user = (
        "Based ONLY on the following list of movies that one anonymous user has "
        "watched, infer the user's gender.\n\n"
        "Watched movies:\n"
        f"{_format_numbered(history_titles)}\n\n"
        "Reply with exactly one word: male or female."
    )
    return system, user


def build_history_staab(
    history_titles: Sequence[str],
) -> tuple[str | None, str]:
    """Staab guessing-game framing over the watched history instead of the rec list."""
    system = (
        "You are a helpful inference engine that carefully reasons about "
        "personal attributes that can be deduced from a user's content."
    )
    user = (
        "Let us play a guessing game. The following is the list of movies that "
        "one anonymous user has watched. The list is the only information you "
        "get.\n\n"
        "Watched movies:\n"
        f"{_format_numbered(history_titles)}\n\n"
        "Guess the user's gender.\n"
        "Provide a short reasoning, then the final guess.\n"
        "Use exactly this format:\n"
        "Inference: <your detailed inference>\n"
        "Gender: <male or female>"
    )
    return system, user


PROMPT_BUILDERS = {
    "zero_shot": build_zero_shot,
    "few_shot": build_few_shot,
    "staab": build_staab,
    "staab_few_shot": build_staab_few_shot,
    # same templates, exemplars supplied by the caller (list_aia --n-shots)
    "many_shot": build_few_shot,
    "staab_many_shot": build_staab_few_shot,
    # history-only attackers: same ask, input is the watched history
    "history_zero_shot": build_history_zero_shot,
    "history_staab": build_history_staab,
}

# Styles whose input is the watched history (needs --history-size on the export).
HISTORY_STYLES = ("history_zero_shot", "history_staab")

# The four original styles; --all-styles runs these (many-shot needs --n-shots).
BASE_STYLES = ("zero_shot", "few_shot", "staab", "staab_few_shot")
MANY_SHOT_STYLES = ("many_shot", "staab_many_shot")


def build_prompt(
    style: str,
    rec_titles: Sequence[str],
    examples: Sequence[dict[str, object]] | None = None,
) -> tuple[str | None, str]:
    """``rec_titles`` carries the history titles for the history_* styles."""
    if style not in PROMPT_BUILDERS:
        raise ValueError(
            f"Unknown prompt style {style!r}; choose from {sorted(PROMPT_BUILDERS)}"
        )
    if style in MANY_SHOT_STYLES and not examples:
        raise ValueError(f"style {style!r} needs exemplars — pass --n-shots/--shot-pool")
    if style in {"few_shot", "staab_few_shot", *MANY_SHOT_STYLES}:
        return PROMPT_BUILDERS[style](rec_titles, examples)
    return PROMPT_BUILDERS[style](rec_titles)
