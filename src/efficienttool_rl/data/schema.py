"""Dataset-neutral records for controlled multi-turn search tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    """One searchable document in a per-example corpus."""

    title: str
    text: str


@dataclass(frozen=True)
class SearchExample:
    """Common input contract shared by HotpotQA and MuSiQue adapters."""

    example_id: str
    question: str
    answer: str
    passages: tuple[Passage, ...]
    supporting_titles: tuple[str, ...]
    split: str
    question_type: str = "unknown"
    level: str = "unknown"
    dataset_name: str = "unknown"
    hop_count: int | None = None
    answer_aliases: tuple[str, ...] = ()

    @property
    def type(self) -> str:
        """Readable alias for datasets exposing question-type metadata."""
        return self.question_type
