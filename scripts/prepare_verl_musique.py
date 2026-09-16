#!/usr/bin/env python3
"""Convert official MuSiQue-Answerable JSONL into local-corpus verl parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset

from efficienttool_rl.data import load_musique
from efficienttool_rl.training import to_verl_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int)
    group.add_argument(
        "--per-hop-limit",
        type=int,
        help="Select this many examples for each requested hop count.",
    )
    parser.add_argument("--hops", type=int, nargs="+", choices=(2, 3, 4), default=(2, 3, 4))
    parser.add_argument(
        "--answerable-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-observation-tokens", type=int, default=384)
    parser.add_argument("--max-top-k", type=int, default=1)
    parser.add_argument("--max-executed-search-calls", type=int, default=4)
    parser.add_argument("--data-source", default="musique_ans_local")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_examples(examples, args: argparse.Namespace):
    allowed = set(args.hops)
    filtered = [example for example in examples if example.hop_count in allowed]
    if args.per_hop_limit is not None:
        selected = []
        for hops in args.hops:
            bucket = [example for example in filtered if example.hop_count == hops]
            chunk = bucket[args.start_index : args.start_index + args.per_hop_limit]
            if len(chunk) != args.per_hop_limit:
                raise ValueError(f"requested range exceeds available {hops}-hop rows")
            selected.extend(chunk)
        return selected
    selected = filtered[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
        if len(selected) != args.limit:
            raise ValueError("requested range exceeds filtered input dataset")
    return selected


def main() -> None:
    args = parse_args()
    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.per_hop_limit is not None and args.per_hop_limit < 1:
        raise ValueError("per-hop-limit must be positive")
    if (
        args.max_observation_tokens < 1
        or args.max_top_k < 1
        or args.max_executed_search_calls < 0
    ):
        raise ValueError(
            "observation/top-k limits must be positive and search budget non-negative"
        )
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")

    examples = load_musique(
        args.input,
        split=args.split,
        answerable_only=args.answerable_only,
    )
    selected = select_examples(examples, args)
    if not selected:
        raise ValueError("selection produced no examples")

    records = [
        to_verl_record(
            example,
            index=index,
            max_observation_tokens=args.max_observation_tokens,
            max_top_k=args.max_top_k,
            max_executed_search_calls=args.max_executed_search_calls,
            data_source=args.data_source,
        )
        for index, example in enumerate(selected, start=args.start_index)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    Dataset.from_list(records).to_parquet(str(temporary))
    temporary.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "StonyBrookNLP/MuSiQue",
        "source_format": "official MuSiQue-Answerable JSONL",
        "source_license": "CC BY 4.0",
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "split": args.split,
        "start_index": args.start_index,
        "rows": len(records),
        "hop_counts": dict(sorted(Counter(example.hop_count for example in selected).items())),
        "requested_hops": args.hops,
        "answerable_only": args.answerable_only,
        "max_observation_tokens": args.max_observation_tokens,
        "max_top_k": args.max_top_k,
        "max_executed_search_calls": args.max_executed_search_calls,
        "data_source": args.data_source,
        "output": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "sha256": digest,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
