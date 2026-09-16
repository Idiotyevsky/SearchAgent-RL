#!/usr/bin/env python3
"""Unified held-out evaluation for fixed policies (Transformers or vLLM).

The runner performs inference only: it reuses the production ``AgentRunner``,
strict local BM25 search environment, and the project action protocol, so any
checkpoint (base, GRPO, DAPO) is evaluated under identical conditions. Rows
follow the stored fixed-policy trajectory schema consumed by
``scripts/analyze_cost_reward.py`` and ``scripts/audit_cost_signal.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from efficienttool_rl.agent import AgentConfig, AgentRunner
from efficienttool_rl.data import load_verl_examples
from efficienttool_rl.evaluation import (
    answer_metrics_any,
    episode_search_usage,
    summarize_episodes,
)
from efficienttool_rl.policies import TransformersToolPolicy
from efficienttool_rl.protocol import SYSTEM_PROMPT
from efficienttool_rl.tools import BM25Search


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _build_policy(args: argparse.Namespace):
    if args.backend == "transformers":
        return TransformersToolPolicy(
            args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + args.shard_id,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    from efficienttool_rl.policies import VLLMToolPolicy

    return VLLMToolPolicy(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument(
        "--backend",
        choices=("auto", "transformers", "vllm"),
        default="auto",
        help="vLLM is chosen automatically when tensor-parallel-size > 1.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--rollouts-per-prompt", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-search-calls", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-top-k", type=int, default=1)
    parser.add_argument("--max-observation-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    args = parser.parse_args()
    if args.backend == "auto":
        args.backend = "vllm" if args.tensor_parallel_size > 1 else "transformers"
    return args


def _validate(args: argparse.Namespace) -> None:
    if args.start_index < 0 or args.limit < 1:
        raise ValueError("start-index must be non-negative and limit must be positive")
    if args.rollouts_per_prompt < 1:
        raise ValueError("rollouts-per-prompt must be positive")
    if args.backend == "transformers" and (
        args.shard_id < 0 or args.num_shards < 1 or args.shard_id >= args.num_shards
    ):
        raise ValueError("shard-id must be in [0, num-shards)")
    if args.tensor_parallel_size < 1:
        raise ValueError("tensor-parallel-size must be positive")
    if args.max_turns < 1 or args.max_search_calls < 0:
        raise ValueError("invalid turn or search budget")
    if args.top_k < 1 or args.max_top_k < args.top_k:
        raise ValueError("max-top-k must be at least top-k and both must be positive")
    if args.max_observation_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("token limits must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be positive and top-p must be in (0, 1]")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")
    if args.output.exists() and args.output.is_dir():
        raise IsADirectoryError(args.output)


def _flat_metrics_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [record["trajectory"] for record in records]
    behavior = summarize_episodes(episodes)
    count = len(records)
    if count == 0:
        return {**behavior, "em": 0.0, "f1": 0.0}
    multi_search = sum(
        int(record["executed_search_calls"] >= 2) for record in records
    )
    return {
        **behavior,
        "em": sum(record["exact_match"] for record in records) / count,
        "f1": sum(record["f1"] for record in records) / count,
        "avg_task_reward": sum(record["task_reward"] for record in records) / count,
        "multi_search_rate": multi_search / count,
    }


def _metrics_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _flat_metrics_summary(records)
    hop_counts = sorted(
        {
            record["hop_count"]
            for record in records
            if isinstance(record.get("hop_count"), int)
        }
    )
    if hop_counts:
        summary["by_hop"] = {
            str(hops): _flat_metrics_summary(
                [record for record in records if record.get("hop_count") == hops]
            )
            for hops in hop_counts
        }
    return summary


def main() -> None:
    args = _parse_args()
    _validate(args)

    examples = load_verl_examples(args.data, args.split)
    selected = examples[args.start_index : args.start_index + args.limit]
    if len(selected) != args.limit:
        raise ValueError("requested range exceeds the dataset")
    if args.backend == "transformers":
        selected = selected[args.shard_id :: args.num_shards]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    existing[str(row["trajectory_id"])] = row
    expected_ids = {
        f"{example.example_id}__rollout_{rollout_index}"
        for example in selected
        for rollout_index in range(args.rollouts_per_prompt)
    }
    if expected_ids.issubset(existing):
        print(json.dumps({"status": "already_complete", "rows": len(expected_ids)}))
        return

    policy = _build_policy(args)

    completed = len(existing)
    with args.output.open("a", encoding="utf-8") as handle:
        for example_offset, example in enumerate(selected):
            search = BM25Search(
                example.passages,
                max_observation_tokens=args.max_observation_tokens,
            )

            def bounded_search(arguments: dict[str, object]) -> list[dict[str, object]]:
                supplied = dict(arguments)
                top_k = supplied.get("top_k", args.top_k)
                if isinstance(top_k, int) and not isinstance(top_k, bool):
                    supplied["top_k"] = min(top_k, args.max_top_k)
                return search.tool(supplied)

            runner = AgentRunner(
                policy,
                tools={"search": bounded_search},
                config=AgentConfig(
                    max_turns=args.max_turns,
                    max_tool_calls=args.max_search_calls,
                ),
                system_prompt=SYSTEM_PROMPT,
            )
            for rollout_index in range(args.rollouts_per_prompt):
                trajectory_id = f"{example.example_id}__rollout_{rollout_index}"
                if trajectory_id in existing:
                    continue
                episode = runner.run(example.question, episode_id=trajectory_id)
                scores = answer_metrics_any(
                    episode.final_answer or "",
                    (example.answer, *example.answer_aliases),
                )
                usage = episode_search_usage(episode, example.supporting_titles)
                record: dict[str, Any] = {
                    "trajectory_id": trajectory_id,
                    "input": example.question,
                    "episode_id": episode.episode_id,
                    "reference_answer": example.answer,
                    "reference_aliases": list(example.answer_aliases),
                    "dataset_name": example.dataset_name,
                    "hop_count": example.hop_count,
                    "task_reward": 0.5 * scores["exact_match"] + 0.5 * scores["f1"],
                    "exact_match": scores["exact_match"],
                    "f1": scores["f1"],
                    "valid_answer": float(bool(episode.final_answer)),
                    "wasted_search_calls": usage["wasted_search_calls"],
                    "useful_search_calls": usage["useful_search_calls"],
                    "executed_search_calls": usage["executed_search_calls"],
                    "attempted_tool_calls": episode.attempted_tool_calls,
                    "valid_tool_calls": episode.valid_tool_calls,
                    "invalid_actions": episode.invalid_actions,
                    "generated_tokens": sum(
                        policy.count_tokens(step.model_output) for step in episode.steps
                    ),
                    "turns": len(episode.steps),
                    "termination_reason": episode.termination_reason,
                    "trajectory": episode.to_dict(),
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                existing[trajectory_id] = record
                completed += 1
                print(
                    json.dumps(
                        {
                            "example_id": example.example_id,
                            "rollout": rollout_index,
                            "completed": completed,
                            "em": scores["exact_match"],
                            "f1": scores["f1"],
                            **usage,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    records = list(existing.values())
    _write_json(args.output.parent / "metrics.json", _metrics_summary(records))
    failures = [record for record in records if record["exact_match"] == 0]
    failures_path = args.output.parent / "failures.jsonl"
    if failures:
        with failures_path.open("w", encoding="utf-8") as handle:
            for record in failures:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    elif failures_path.exists():
        failures_path.unlink()

    if args.config_output:
        _write_json(
            args.config_output,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data": str(args.data.resolve()),
                "data_sha256": _sha256(args.data),
                "model": str(args.model.resolve()),
                "backend": args.backend,
                "device": args.device if args.backend == "transformers" else None,
                "split": args.split,
                "start_index": args.start_index,
                "limit": args.limit,
                "rollouts_per_prompt": args.rollouts_per_prompt,
                "shard_id": args.shard_id if args.backend == "transformers" else None,
                "num_shards": args.num_shards if args.backend == "transformers" else None,
                "tensor_parallel_size": (
                    args.tensor_parallel_size if args.backend == "vllm" else None
                ),
                "max_turns": args.max_turns,
                "max_search_calls": args.max_search_calls,
                "top_k": args.top_k,
                "max_top_k": args.max_top_k,
                "max_observation_tokens": args.max_observation_tokens,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed + args.shard_id
                if args.backend == "transformers"
                else args.seed,
                "gpu_memory_utilization": (
                    args.gpu_memory_utilization if args.backend == "vllm" else None
                ),
                "max_model_len": args.max_model_len if args.backend == "vllm" else None,
                "max_num_batched_tokens": (
                    args.max_num_batched_tokens if args.backend == "vllm" else None
                ),
                "max_num_seqs": args.max_num_seqs if args.backend == "vllm" else None,
                "inference_only": True,
            },
        )


if __name__ == "__main__":
    main()
