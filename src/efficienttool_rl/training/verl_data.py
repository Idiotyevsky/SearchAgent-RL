"""Convert normalized search examples to verl RL records."""

from __future__ import annotations

from typing import Any

from ..data import SearchExample
from ..protocol import SYSTEM_PROMPT


def to_verl_record(
    example: SearchExample,
    *,
    index: int,
    max_observation_tokens: int = 512,
    max_top_k: int = 3,
    max_executed_search_calls: int = 3,
    data_source: str = "hotpotqa_distractor",
) -> dict[str, Any]:
    """Create a JSON-serializable record with reward metadata kept model-hidden."""
    if max_observation_tokens < 1 or max_top_k < 1 or max_executed_search_calls < 0:
        raise ValueError("observation/top-k limits must be positive and search budget non-negative")
    passages = [{"title": passage.title, "text": passage.text} for passage in example.passages]
    return {
        "data_source": data_source,
        "agent_name": "tool_agent",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example.question},
        ],
        "ability": "multi_hop_qa",
        "reward_model": {"style": "rule", "ground_truth": example.answer},
        "extra_info": {
            "split": example.split,
            "index": index,
            "example_id": example.example_id,
            "question": example.question,
            "question_type": example.question_type,
            "level": example.level,
            "dataset_name": example.dataset_name,
            "hop_count": example.hop_count,
            # Evaluation-only metadata. It is not included in prompt or tool kwargs.
            "supporting_titles": list(example.supporting_titles),
            "answer_aliases": list(example.answer_aliases),
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {
                        "passages": passages,
                        "max_observation_tokens": max_observation_tokens,
                        "max_top_k": max_top_k,
                        "max_executed_search_calls": max_executed_search_calls,
                    }
                }
            },
        },
    }
