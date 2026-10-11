#!/usr/bin/env python3
"""Create a stratified JSONL sample (30 examples with the default label counts)."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/processed/synthetic_messages.jsonl")
DEFAULT_OUTPUT = Path("data/processed/evaluation_sample.jsonl")

TARGET_COUNTS = {
    "ACTIONABLE_DDI": 20,
    "NO_SOURCE_SUPPORTED_DDI": 5,
    "MULTI_DDI_SYNTHESIS": 5,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            records.append(record)
    return records


def label_for(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    label = metadata.get("label")
    return label if isinstance(label, str) else None


def sample_records(
    records: list[dict[str, Any]],
    target_counts: dict[str, int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    records_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        label = label_for(record)
        if label in target_counts:
            records_by_label[label].append(record)

    sampled: list[dict[str, Any]] = []
    for label, needed_count in target_counts.items():
        available_count = len(records_by_label[label])
        if available_count < needed_count:
            raise ValueError(
                f"Need {needed_count} records with label {label}, but only found {available_count}."
            )
        sampled.extend(rng.sample(records_by_label[label], needed_count))

    rng.shuffle(sampled)
    return sampled


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument("--actionable", type=int, default=20)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--multi", type=int, default=5)
    args = parser.parse_args()

    target_counts = {
        "ACTIONABLE_DDI": args.actionable,
        "NO_SOURCE_SUPPORTED_DDI": args.negative,
        "MULTI_DDI_SYNTHESIS": args.multi,
    }
    if any(n < 0 for n in target_counts.values()):
        parser.error("label counts must be non-negative")
    rng = random.Random(args.seed)
    records = load_jsonl(args.input)
    sampled = sample_records(records, target_counts, rng)
    write_jsonl(sampled, args.output)

    counts = Counter(label_for(record) for record in sampled)
    print(f"Wrote {len(sampled)} records to {args.output}")
    for label, expected_count in target_counts.items():
        print(f"{label}: {counts[label]} / {expected_count}")


if __name__ == "__main__":
    main()
