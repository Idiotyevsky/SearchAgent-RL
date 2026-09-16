"""Strict loader for official MuSiQue-Answerable JSON/JSONL records."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import Passage, SearchExample

_HOP_ID = re.compile(r"^(?P<hops>[2-4])hop__")


class MuSiQueExample(SearchExample):
    """Normalized MuSiQue example with explicit required-hop metadata."""


def _required_string(item: dict[str, Any], key: str, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"example {index}: {key!r} must be a non-empty string")
    return value.strip()


def _read_records(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            records: list[Any] = []
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"line {line_number}: invalid JSON: {exc.msg}"
                    ) from exc
            return records
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("MuSiQue JSON file must contain a list")
    return payload


def _hop_count(example_id: str, decomposition: list[Any], index: int) -> int:
    match = _HOP_ID.match(example_id)
    inferred = len(decomposition)
    hops = int(match.group("hops")) if match else inferred
    if hops not in {2, 3, 4}:
        raise ValueError(f"example {index}: hop count must be 2, 3, or 4")
    if inferred != hops:
        raise ValueError(
            f"example {index}: id declares {hops} hops but decomposition has {inferred}"
        )
    return hops


def load_musique(
    path: str | Path,
    split: str,
    *,
    answerable_only: bool = True,
) -> list[MuSiQueExample]:
    """Load official MuSiQue records without exposing decomposition metadata.

    Gold supporting-document labels and answer aliases are retained on the
    returned object for reward/evaluation only. Only passages are later passed
    to the search environment.
    """
    if not split.strip():
        raise ValueError("split must be explicit and non-empty")

    payload = _read_records(Path(path))
    examples: list[MuSiQueExample] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"example {index}: expected an object")

        answerable = item.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"example {index}: answerable must be boolean when present")
        if not answerable:
            if answerable_only:
                continue
            raise ValueError(
                f"example {index}: unanswerable MuSiQue-Full rows are unsupported"
            )

        example_id = _required_string(item, "id", index)
        if example_id in seen_ids:
            raise ValueError(f"example {index}: duplicate id {example_id!r}")
        seen_ids.add(example_id)

        raw_paragraphs = item.get("paragraphs")
        if not isinstance(raw_paragraphs, list) or not raw_paragraphs:
            raise ValueError(f"example {index}: paragraphs must be a non-empty list")

        passages_by_idx: dict[int, Passage] = {}
        declared_support: set[int] = set()
        ordered_passage_ids: list[int] = []
        for paragraph_index, raw in enumerate(raw_paragraphs):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"example {index}: paragraph {paragraph_index} must be an object"
                )
            paragraph_id = raw.get("idx")
            if not isinstance(paragraph_id, int) or isinstance(paragraph_id, bool):
                raise ValueError(
                    f"example {index}: paragraph {paragraph_index} has invalid idx"
                )
            if paragraph_id in passages_by_idx:
                raise ValueError(
                    f"example {index}: duplicate paragraph idx {paragraph_id}"
                )
            title = _required_string(raw, "title", index)
            text = _required_string(raw, "paragraph_text", index)
            is_supporting = raw.get("is_supporting")
            if not isinstance(is_supporting, bool):
                raise ValueError(
                    f"example {index}: paragraph {paragraph_index} "
                    "requires boolean is_supporting"
                )
            passage = Passage(title=title, text=text)
            passages_by_idx[paragraph_id] = passage
            ordered_passage_ids.append(paragraph_id)
            if is_supporting:
                declared_support.add(paragraph_id)

        title_counts = Counter(passage.title for passage in passages_by_idx.values())
        for paragraph_id, passage in tuple(passages_by_idx.items()):
            if title_counts[passage.title] > 1:
                passages_by_idx[paragraph_id] = Passage(
                    title=f"{passage.title} [MuSiQue paragraph {paragraph_id}]",
                    text=passage.text,
                )
        ordered_passages = [
            passages_by_idx[paragraph_id] for paragraph_id in ordered_passage_ids
        ]

        decomposition = item.get("question_decomposition")
        if not isinstance(decomposition, list):
            raise ValueError(f"example {index}: question_decomposition must be a list")
        hops = _hop_count(example_id, decomposition, index)

        support_indices: list[int] = []
        for step_index, step in enumerate(decomposition):
            if not isinstance(step, dict):
                raise ValueError(
                    f"example {index}: decomposition step {step_index} must be an object"
                )
            support_idx = step.get("paragraph_support_idx")
            if (
                not isinstance(support_idx, int)
                or isinstance(support_idx, bool)
                or support_idx not in passages_by_idx
            ):
                raise ValueError(
                    f"example {index}: decomposition step {step_index} "
                    "has invalid paragraph_support_idx"
                )
            if support_idx not in support_indices:
                support_indices.append(support_idx)

        if set(support_indices) != declared_support:
            raise ValueError(
                f"example {index}: is_supporting and decomposition support disagree"
            )

        aliases = item.get("answer_aliases", [])
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) and alias.strip() for alias in aliases
        ):
            raise ValueError(f"example {index}: answer_aliases must contain strings")
        unique_aliases = tuple(dict.fromkeys(alias.strip() for alias in aliases))

        supporting_titles = tuple(
            dict.fromkeys(passages_by_idx[support_idx].title for support_idx in support_indices)
        )
        examples.append(
            MuSiQueExample(
                example_id=example_id,
                question=_required_string(item, "question", index),
                answer=_required_string(item, "answer", index),
                passages=tuple(ordered_passages),
                supporting_titles=supporting_titles,
                split=split,
                question_type="composition",
                level=f"{hops}hop",
                dataset_name="musique",
                hop_count=hops,
                answer_aliases=unique_aliases,
            )
        )
    return examples
