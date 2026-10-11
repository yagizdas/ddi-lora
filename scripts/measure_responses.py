#!/usr/bin/env python3
"""Measure the recovered paired responses without assigning clinical scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"\b[\w]+(?:[’'-][\w]+)*\b")
SECTIONS = [
    "Priority Interactions",
    "Mechanistic Explanation",
    "Patient-Specific Risk Assessment",
    "Recommendations",
]


def read_cases(path: Path) -> dict[int, dict]:
    parts = re.split(r"^## (\d+)\)?\s*$", path.read_text(encoding="utf-8"), flags=re.M)
    cases = {}
    for i in range(1, len(parts), 2):
        prompt, answer = parts[i + 1].split("### User Request", 1)[1].split("### Model Response", 1)
        answer = re.sub(r"\n---\s*$", "", answer.strip())
        counts = [
            len(re.findall(rf"^\s*\[?{n}[.)]\s*(?:\*\*)?{label}", answer, re.M | re.I))
            for n, label in enumerate(SECTIONS, 1)
        ]
        cases[int(parts[i])] = {
            "user": re.sub(r"\s+", " ", prompt).strip(),
            "words": len(WORD.findall(answer)),
            "section_occurrences": counts,
        }
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="Local baseline transcript")
    parser.add_argument("--lora", required=True, type=Path, help="Local LoRA transcript")
    args = parser.parse_args()
    paths = {"baseline": args.baseline, "lora": args.lora}
    models = {name: read_cases(path) for name, path in paths.items()}
    if set(models["baseline"]) != set(models["lora"]) or len(models["baseline"]) != 15:
        raise ValueError("Expected the same 15 case IDs in each file")
    cases = []
    for case in sorted(models["baseline"]):
        baseline, lora = models["baseline"][case], models["lora"][case]
        if baseline["user"] != lora["user"]:
            raise ValueError(f"Case {case} prompts differ")
        cases.append(
            {
                "case": case,
                "baseline_words": baseline["words"],
                "lora_words": lora["words"],
                "baseline_section_occurrences": baseline["section_occurrences"],
                "lora_section_occurrences": lora["section_occurrences"],
            }
        )
    totals = {name: sum(c[f"{name}_words"] for c in cases) for name in models}
    metrics = {
        "scope": "Presentation measurements only; original aggregate clinical scores are not recalculated.",
        "word_definition": "Unicode word-like tokens, including headings and numbers; internal hyphens/apostrophes remain in one token. Excludes user prompts and Markdown separators.",
        "word_pattern": WORD.pattern,
        "transcript_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
        "paired_cases": len(cases),
        "totals": totals,
        "mean_words": {name: mean(c[f"{name}_words"] for c in cases) for name in models},
        "median_words": {name: median(c[f"{name}_words"] for c in cases) for name in models},
        "lora_word_reduction_percent": (1 - totals["lora"] / totals["baseline"]) * 100,
        "lora_shorter_cases": sum(c["lora_words"] < c["baseline_words"] for c in cases),
        "all_four_sections_present": {
            name: sum(all(n > 0 for n in c[f"{name}_section_occurrences"]) for c in cases)
            for name in models
        },
        "all_four_sections_once": {
            name: sum(c[f"{name}_section_occurrences"] == [1, 1, 1, 1] for c in cases)
            for name in models
        },
        "cases": cases,
    }
    output = ROOT / "docs/response_presentation_metrics.json"
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in metrics.items()
                if k not in {"cases", "transcript_sha256", "word_pattern"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
