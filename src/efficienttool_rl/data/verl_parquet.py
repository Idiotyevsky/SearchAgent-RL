"""Load dataset-neutral search examples from verl parquet or source JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hotpotqa import HotpotExample, load_hotpotqa
from .musique import MuSiQueExample, load_musique
from .schema import Passage, SearchExample


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_strings(value: Any, *, field: str, row_index: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"row {row_index}: malformed {field}")
    return tuple(value)


def _parquet_examples(path: Path, split: str) -> list[SearchExample]:
    """Load the exact verl records used by native training."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("pyarrow is required to read parquet input") from exc

    table = pq.read_table(str(path), columns=["reward_model", "extra_info"])
    examples: list[SearchExample] = []
    for row_index, row in enumerate(table.to_pylist()):
        info = _as_dict(row.get("extra_info"))
        reward_model = _as_dict(row.get("reward_model"))
        tools_kwargs = _as_dict(info.get("tools_kwargs"))
        search_kwargs = _as_dict(_as_dict(tools_kwargs.get("search")).get("create_kwargs"))
        raw_passages = search_kwargs.get("passages")
        if not isinstance(raw_passages, list) or not raw_passages:
            raise ValueError(f"row {row_index}: missing search passages")
        passages: list[Passage] = []
        for passage_index, raw in enumerate(raw_passages):
            item = _as_dict(raw)
            if not isinstance(item.get("title"), str) or not isinstance(item.get("text"), str):
                raise ValueError(f"row {row_index}, passage {passage_index}: malformed passage")
            passages.append(Passage(title=item["title"], text=item["text"]))

        question = info.get("question")
        answer = reward_model.get("ground_truth")
        example_id = info.get("example_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"row {row_index}: missing question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"row {row_index}: missing ground-truth answer")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"row {row_index}: missing example_id")

        support = _optional_strings(
            info.get("supporting_titles", []),
            field="supporting_titles",
            row_index=row_index,
        )
        aliases = _optional_strings(
            info.get("answer_aliases", []),
            field="answer_aliases",
            row_index=row_index,
        )
        raw_hop_count = info.get("hop_count")
        if raw_hop_count is not None and (
            not isinstance(raw_hop_count, int) or isinstance(raw_hop_count, bool)
        ):
            raise ValueError(f"row {row_index}: malformed hop_count")

        example_class = (
            MuSiQueExample
            if str(info.get("dataset_name", "unknown")) == "musique"
            else HotpotExample
        )
        examples.append(
            example_class(
                example_id=example_id,
                question=question,
                answer=answer,
                passages=tuple(passages),
                supporting_titles=support,
                split=str(info.get("split", split)),
                question_type=str(info.get("question_type", "unknown")),
                level=str(info.get("level", "unknown")),
                dataset_name=str(info.get("dataset_name", "unknown")),
                hop_count=raw_hop_count,
                answer_aliases=aliases,
            )
        )
    return examples


def _looks_like_musique(path: Path) -> bool:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            first = next((json.loads(line) for line in handle if line.strip()), None)
        else:
            payload = json.load(handle)
            first = payload[0] if isinstance(payload, list) and payload else None
    return isinstance(first, dict) and {
        "paragraphs",
        "question_decomposition",
    }.issubset(first)


def load_verl_examples(path: str | Path, split: str = "validation") -> list[SearchExample]:
    """Read verl parquet, official MuSiQue, or normalized HotpotQA input."""
    path = Path(path)
    if path.suffix == ".parquet":
        return _parquet_examples(path, split)
    if _looks_like_musique(path):
        return list(load_musique(path, split=split))
    return list(load_hotpotqa(path, split=split))


__all__ = ["load_verl_examples"]
