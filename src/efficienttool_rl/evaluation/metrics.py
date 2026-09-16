"""HotpotQA-compatible answer and aggregate agent-behavior metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    without_punctuation = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(remove_articles(without_punctuation).split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_metrics(prediction: str, reference: str) -> dict[str, float]:
    return {"exact_match": exact_match(prediction, reference), "f1": token_f1(prediction, reference)}


def answer_metrics_any(prediction: str, references: Sequence[str]) -> dict[str, float]:
    """Score against the best canonical answer or accepted alias."""
    valid = [reference for reference in references if isinstance(reference, str) and reference]
    if not valid:
        raise ValueError("at least one non-empty reference answer is required")
    scores = [answer_metrics(prediction, reference) for reference in valid]
    return {
        "exact_match": max(item["exact_match"] for item in scores),
        "f1": max(item["f1"] for item in scores),
    }


def summarize_episodes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Aggregate behavior without treating non-completions as valid answers."""
    count = len(episodes)
    if count == 0:
        return {
            "episodes": 0,
            "completion_rate": 0.0,
            "avg_attempted_tool_calls": 0.0,
            "avg_valid_tool_calls": 0.0,
            "avg_executed_tool_calls": 0.0,
            "avg_executed_search_calls": 0.0,
            "avg_search_calls": 0.0,
            "avg_turns": 0.0,
            "invalid_action_rate": 0.0,
        }
    completions = sum(item.get("termination_reason") == "final_answer" for item in episodes)
    attempted_tool_calls = sum(
        int(item.get("attempted_tool_calls", item.get("tool_calls", 0))) for item in episodes
    )
    valid_tool_calls = sum(
        int(item.get("valid_tool_calls", item.get("tool_calls", 0))) for item in episodes
    )
    executed_tool_calls = sum(
        int(item.get("executed_tool_calls", item.get("tool_calls", 0))) for item in episodes
    )
    executed_search_calls = sum(
        int(
            item.get(
                "executed_search_calls",
                item.get("executed_tool_calls", item.get("tool_calls", 0)),
            )
        )
        for item in episodes
    )
    invalid_actions = sum(int(item.get("invalid_actions", 0)) for item in episodes)
    turns = sum(len(item.get("steps", [])) for item in episodes)
    return {
        "episodes": count,
        "completion_rate": completions / count,
        "avg_attempted_tool_calls": attempted_tool_calls / count,
        "avg_valid_tool_calls": valid_tool_calls / count,
        "avg_executed_tool_calls": executed_tool_calls / count,
        "avg_executed_search_calls": executed_search_calls / count,
        # Backward-compatible name; it now has an explicit executed-search meaning.
        "avg_search_calls": executed_search_calls / count,
        "avg_turns": turns / count,
        "invalid_action_rate": invalid_actions / max(turns, 1),
    }
