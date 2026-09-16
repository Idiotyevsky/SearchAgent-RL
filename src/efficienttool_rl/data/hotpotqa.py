"""Strict loader for normalized HotpotQA distractor/fullwiki records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Passage, SearchExample


class HotpotExample(SearchExample):
    """Normalized HotpotQA example; retained for API compatibility."""



def _required_string(item: dict[str, Any], key: str, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"example {index}: {key!r} must be a non-empty string")
    return value.strip()


def _metadata_string(item: dict[str, Any], key: str, index: int) -> str:
    """Read optional official metadata while keeping legacy fixtures valid."""
    value = item.get(key, "unknown")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"example {index}: {key!r} must be a non-empty string when present")
    return value.strip().lower()


def _read_records(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            records: list[Any] = []
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            return records
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("HotpotQA JSON file must contain a list")
    return payload


def load_hotpotqa(path: str | Path, split: str) -> list[HotpotExample]:
    """Load normalized JSON/JSONL without inferring or mixing dataset splits."""
    if not split.strip():
        raise ValueError("split must be explicit and non-empty")
    payload = _read_records(Path(path))

    examples: list[HotpotExample] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"example {index}: expected an object")
        id_key = "_id" if "_id" in item else "id"
        example_id = _required_string(item, id_key, index)
        if example_id in seen_ids:
            raise ValueError(f"example {index}: duplicate id {example_id!r}")
        seen_ids.add(example_id)

        raw_context = item.get("context")
        if not isinstance(raw_context, list):
            raise ValueError(f"example {index}: context must be a list")
        passages: list[Passage] = []
        for context_index, context in enumerate(raw_context):
            if not isinstance(context, list) or len(context) != 2:
                raise ValueError(
                    f"example {index}: context {context_index} must be [title, sentences]"
                )
            title, sentences = context
            if not isinstance(title, str) or not isinstance(sentences, list) or not all(
                isinstance(sentence, str) for sentence in sentences
            ):
                raise ValueError(f"example {index}: malformed context {context_index}")
            passages.append(Passage(title=title.strip(), text="".join(sentences).strip()))

        raw_support = item.get("supporting_facts", [])
        if not isinstance(raw_support, list):
            raise ValueError(f"example {index}: supporting_facts must be a list")
        supporting_titles: list[str] = []
        for fact in raw_support:
            if not isinstance(fact, list) or len(fact) != 2 or not isinstance(fact[0], str):
                raise ValueError(f"example {index}: malformed supporting fact")
            if fact[0] not in supporting_titles:
                supporting_titles.append(fact[0])

        examples.append(
            HotpotExample(
                example_id=example_id,
                question=_required_string(item, "question", index),
                answer=_required_string(item, "answer", index),
                passages=tuple(passages),
                supporting_titles=tuple(supporting_titles),
                split=split,
                question_type=_metadata_string(item, "type", index),
                level=_metadata_string(item, "level", index),
                dataset_name="hotpotqa",
            )
        )
    return examples
