"""Create source-negative pairs; missing records do not establish drug safety."""

from __future__ import annotations

import argparse
import random
from itertools import combinations
from pathlib import Path

from patient_aware_ddi.sample import load_jsonl, write_jsonl


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a.strip().lower(), b.strip().lower())))


def generate_negatives(rows: list[dict], count: int, seed: int = 42) -> list[dict]:
    if count < 0:
        raise ValueError("count must be non-negative")
    names = {}
    existing = set()
    for row in rows:
        a, b = row.get("drug_a"), row.get("drug_b")
        if not a or not b:
            continue
        names.setdefault(a.strip().lower(), a.strip())
        names.setdefault(b.strip().lower(), b.strip())
        existing.add(pair_key(a, b))
    # Reservoir sampling uses O(count) memory even for a large drug vocabulary.
    rng = random.Random(seed)
    selected = []
    available = 0
    for key in combinations(sorted(names), 2):
        if key in existing:
            continue
        available += 1
        if len(selected) < count:
            selected.append(key)
        else:
            position = rng.randrange(available)
            if position < count:
                selected[position] = key
    if available < count:
        raise ValueError(f"Requested {count} negative pairs, but only {available} are available")
    return [
        {
            "interaction_id": "N/A",
            "url": "N/A",
            "drug_a": names[a],
            "drug_b": names[b],
            "source_status": "no_record_in_project_source",
            "label": "NO_SOURCE_SUPPORTED_DDI",
            "severity": "No source-supported actionable interaction",
            "mechanism_type": "None identified in project source",
            "interaction": "No interaction record found in the available project dataset.",
            "management": "Absence from this subset does not establish safety. Review independently.",
            "references": "N/A",
            "is_synthetic_tn": True,
        }
        for a, b in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/combined.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/source_negatives.jsonl")
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-source", action="store_true", help="Write source rows before negatives"
    )
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    negatives = generate_negatives(rows, args.count, args.seed)
    write_jsonl((rows if args.include_source else []) + negatives, args.output)
    print(f"Wrote {len(negatives)} source-negative pairs to {args.output}")


if __name__ == "__main__":
    main()
