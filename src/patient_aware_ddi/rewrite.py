#!/usr/bin/env python3
"""Rewrite sampled user prompts while preserving source DDI semantics."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


INPUT_JSONL = Path("data/processed/evaluation_sample.jsonl")
OUTPUT_JSONL = Path("data/processed/evaluation_dataset.jsonl")

REWRITER_MODEL = "gemini-3.1-flash-lite"
EVALUATOR_MODEL = "gemini-3.1-flash-lite"
REWRITE_TEMPERATURE = 0.55
EVALUATOR_TEMPERATURE = 0.1
MAX_REWRITE_ROUNDS = 2

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewrites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_index": {"type": "integer"},
                    "rewritten_user": {"type": "string"},
                    "quality_notes": {"type": "string"},
                },
                "required": ["source_index", "rewritten_user", "quality_notes"],
            },
        }
    },
    "required": ["rewrites"],
}

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["accepted", "needs_rewrite"],
        },
        "flagged": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_index": {"type": "integer"},
                    "reason": {"type": "string"},
                    "rewrite_instruction": {"type": "string"},
                },
                "required": ["source_index", "reason", "rewrite_instruction"],
            },
        },
        "batch_notes": {"type": "string"},
    },
    "required": ["verdict", "flagged", "batch_notes"],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json_response(response: Any) -> Any:
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Empty Gemini response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)


def source_index_for(row: dict[str, Any], fallback: int) -> int:
    source_index = row.get("metadata", {}).get("source_index", fallback)
    try:
        return int(source_index)
    except (TypeError, ValueError):
        return fallback


DRUG_STOP_TOKENS = {
    "acid",
    "acids",
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "hydrochloride",
    "hcl",
    "succinate",
    "tartrate",
    "phosphate",
    "sulfate",
    "sulphate",
    "chloride",
    "bromide",
    "nasal",
    "oral",
    "topical",
    "injection",
    "injectable",
    "tablet",
    "capsule",
    "solution",
    "suspension",
    "extended",
    "release",
    "er",
    "xr",
    "sr",
    "fatty",
    "mg",
    "mcg",
    "g",
    "ml",
    "type",
    "types",
}


def normalize_text_for_match(text: str) -> str:
    text = str(text).lower().replace("ω", "omega")
    text = re.sub(r"[^a-z0-9\- ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def meaningful_drug_tokens(drug_name: str) -> list[str]:
    tokens = []
    for token in normalize_text_for_match(drug_name).split():
        if token in DRUG_STOP_TOKENS:
            continue
        if len(token) < 4 and not token.startswith("omega"):
            continue
        tokens.append(token)
        if token.startswith("omega"):
            tokens.append("omega")
    return list(dict.fromkeys(tokens))


def drug_mentioned_in_text(drug_name: str, text: str) -> bool:
    normalized = normalize_text_for_match(text)
    words = normalized.split()
    for token in meaningful_drug_tokens(drug_name):
        if token in normalized:
            return True
        if any(SequenceMatcher(None, token, word).ratio() >= 0.80 for word in words):
            return True
    return False


def count_numeric_tokens(text: str) -> int:
    patterns = [
        r"\b\d{2,3}\s*[- ]?year[- ]?old\b",
        r"\b\d+(\.\d+)?\s*(mg|mcg|g|units|iu|mmol/l|meq/l|mg/dl|ml/min|bpm|kg|mmhg|%)\b",
        r"\b\d{2,3}/\d{2,3}\b",
        r"\bqtc\s*(of|is|was|:)?\s*\d{3}\b",
        r"\begfr\s*(is|of|was|:)?\s*\d+",
        r"\bcreatinine\s*(is|of|was|:)?\s*\d+(\.\d+)?",
        r"\bpotassium\s*(is|of|was|:)?\s*\d+(\.\d+)?",
        r"\bmagnesium\s*(is|of|was|:)?\s*\d+(\.\d+)?",
        r"\binr\s*(is|of|was|:)?\s*\d+(\.\d+)?",
        r"\bheart rate\s*(is|of|was|:)?\s*\d+",
        r"\bhr\s*(is|of|was|:)?\s*\d+",
        r"\bbp\s*(is|of|was|:)?\s*\d{2,3}/\d{2,3}",
    ]
    lowered = text.lower()
    return sum(len(re.findall(pattern, lowered)) for pattern in patterns)


def user_message(row: dict[str, Any]) -> str:
    for message in row["messages"]:
        if message.get("role") == "user":
            return message.get("content", "")
    raise ValueError("messages does not contain a user message")


def assistant_message(row: dict[str, Any]) -> str:
    for message in row["messages"]:
        if message.get("role") == "assistant":
            return message.get("content", "")
    raise ValueError("messages does not contain an assistant message")


def set_rewrited_messages(row: dict[str, Any], rewritten_user: str) -> dict[str, Any]:
    new_row = deepcopy(row)
    new_row["rewrited_messages"] = deepcopy(row["messages"])
    for message in new_row["rewrited_messages"]:
        if message.get("role") == "user":
            message["content"] = rewritten_user.strip()
            break
    return new_row


def compact_for_prompt(row: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    meta = row.get("metadata", {})
    rewritten_user = ""
    for message in row.get("rewrited_messages", []):
        if message.get("role") == "user":
            rewritten_user = message.get("content", "")

    return {
        "source_index": source_index_for(row, fallback_index),
        "drug_a": meta.get("drug_a"),
        "drug_b": meta.get("drug_b"),
        "expected_label": meta.get("label"),
        "mechanism_category": meta.get("source_mechanism_type"),
        "source_severity": meta.get("source_severity"),
        "original_user": user_message(row),
        "current_rewritten_user": rewritten_user,
        "assistant_management_logic": assistant_message(row),
    }


def build_rewrite_prompt(
    rows: list[dict[str, Any]],
    instructions_by_index: dict[int, str] | None = None,
) -> str:
    payload = []
    instructions_by_index = instructions_by_index or {}

    for fallback_index, row in enumerate(rows):
        item = compact_for_prompt(row, fallback_index)
        item["rewrite_instruction"] = instructions_by_index.get(
            item["source_index"],
            "Create a different clinician-style version of the user prompt.",
        )
        payload.append(item)

    return f"""
You are rewriting synthetic clinical DDI evaluation prompts.

Rewrite ONLY the clinician/user prompt. Do not rewrite the assistant answer.

For each item:
- Keep the exact same source drug pair. Both drugs must appear by name.
- Keep the same expected interaction label.
- Keep the same mechanism category.
- Keep the same source-supported management logic implied by the assistant answer.
- Do not make the case wildly different or change the clinical question's purpose.
- Change age and sex when clinically plausible.
- Change comorbidities, labs, wording, clinical setting, and medication-list style.
- Include exact age, sex, at least two clinical conditions or comorbidities, and at least two numeric clinical details.
- Preserve plausibility of dose, lab, vital-sign, and setting details.
- For NO_SOURCE_SUPPORTED_DDI examples, do not imply there is a real interaction; keep the question neutral.
- Do not mention that this is a rewrite, source row, dataset, label, or evaluation item.

Return valid JSON only, with one rewritten user prompt per source_index.

Items:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def build_evaluation_prompt(rows: list[dict[str, Any]]) -> str:
    payload = []
    for fallback_index, row in enumerate(rows):
        meta = row.get("metadata", {})
        rewritten_user = ""
        for message in row.get("rewrited_messages", []):
            if message.get("role") == "user":
                rewritten_user = message.get("content", "")

        payload.append(
            {
                "source_index": source_index_for(row, fallback_index),
                "drug_a": meta.get("drug_a"),
                "drug_b": meta.get("drug_b"),
                "expected_label": meta.get("label"),
                "mechanism_category": meta.get("source_mechanism_type"),
                "source_severity": meta.get("source_severity"),
                "original_user": user_message(row),
                "rewritten_user": rewritten_user,
                "unchanged_assistant": assistant_message(row),
                "local_validation_errors": local_validate_rewrite(row),
            }
        )

    return f"""
You are evaluating rewritten synthetic clinical DDI prompts.

Evaluate only the rewritten_user field. The assistant answer intentionally remains unchanged.

Accept a rewritten prompt only if:
- It keeps the same source drug pair and mentions both source drugs by name.
- It keeps the same expected label, mechanism category, and management logic.
- It changes wording and scenario details enough to be useful as an evaluation variant.
- It changes several of age/sex, comorbidities, labs, clinical setting, and medication-list style.
- It still includes exact age, sex, at least two clinical conditions or comorbidities, and at least two numeric clinical details.
- It does not introduce unsupported interaction claims or contradict the unchanged assistant answer.
- It does not say it is synthetic, rewritten, source-supported metadata, or an evaluation case.

Return verdict="accepted" only if every rewritten prompt is acceptable.
For any item that fails, add it to flagged with a concrete rewrite_instruction.

Items:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def local_validate_rewrite(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    meta = row.get("metadata", {})
    rewritten_messages = row.get("rewrited_messages", [])

    if not rewritten_messages:
        return ["missing rewrited_messages"]

    original_roles = [message.get("role") for message in row.get("messages", [])]
    rewritten_roles = [message.get("role") for message in rewritten_messages]
    if original_roles != rewritten_roles:
        errors.append("rewrited_messages must keep the original role order")

    rewritten_user = ""
    for original, rewritten in zip(row.get("messages", []), rewritten_messages):
        role = original.get("role")
        if role == "user":
            rewritten_user = rewritten.get("content", "")
            if rewritten_user.strip() == original.get("content", "").strip():
                errors.append("rewritten user content is unchanged")
        elif rewritten.get("content") != original.get("content"):
            errors.append(f"{role} content changed, but only user content may change")

    drug_a = str(meta.get("drug_a", "") or "")
    drug_b = str(meta.get("drug_b", "") or "")
    if drug_a and not drug_mentioned_in_text(drug_a, rewritten_user):
        errors.append(f"source drug_a appears missing from rewritten prompt: {drug_a}")
    if drug_b and not drug_mentioned_in_text(drug_b, rewritten_user):
        errors.append(f"source drug_b appears missing from rewritten prompt: {drug_b}")
    if count_numeric_tokens(rewritten_user) < 2:
        errors.append("rewritten prompt has fewer than two numeric clinical details")

    return errors


def call_rewriter(
    client: genai.Client,
    rows: list[dict[str, Any]],
    instructions_by_index: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    response = client.models.generate_content(
        model=REWRITER_MODEL,
        contents=build_rewrite_prompt(rows, instructions_by_index),
        config=types.GenerateContentConfig(
            temperature=REWRITE_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=REWRITE_SCHEMA,
        ),
    )
    return parse_json_response(response)["rewrites"]


def call_evaluator(client: genai.Client, rows: list[dict[str, Any]]) -> dict[str, Any]:
    response = client.models.generate_content(
        model=EVALUATOR_MODEL,
        contents=build_evaluation_prompt(rows),
        config=types.GenerateContentConfig(
            temperature=EVALUATOR_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=EVALUATION_SCHEMA,
        ),
    )
    return parse_json_response(response)


def apply_rewrites(
    rows: list[dict[str, Any]],
    rewrites: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rewrites_by_index = {int(item["source_index"]): item for item in rewrites}
    rewritten_rows = []

    for fallback_index, row in enumerate(rows):
        source_index = source_index_for(row, fallback_index)
        if source_index not in rewrites_by_index:
            raise ValueError(f"Missing rewrite for source_index={source_index}")

        rewrite = rewrites_by_index[source_index]
        new_row = set_rewrited_messages(row, rewrite["rewritten_user"])
        new_row["rewrite_metadata"] = {
            "rewriter_model": REWRITER_MODEL,
            "quality_notes": rewrite.get("quality_notes", ""),
        }
        rewritten_rows.append(new_row)

    return rewritten_rows


def select_rows(
    rows: list[dict[str, Any]],
    limit: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if len(rows) < limit:
        raise ValueError(f"Need {limit} rows, but {len(rows)} rows were found")

    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[:limit]


def main() -> None:
    global REWRITER_MODEL, EVALUATOR_MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_JSONL)
    parser.add_argument("--output", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rewrite-rounds", type=int, default=MAX_REWRITE_ROUNDS)
    parser.add_argument("--rewriter-model", default=REWRITER_MODEL)
    parser.add_argument("--evaluator-model", default=EVALUATOR_MODEL)
    args = parser.parse_args()
    if args.limit < 1 or args.max_rewrite_rounds < 0:
        parser.error("--limit must be positive and rewrite rounds non-negative")
    REWRITER_MODEL, EVALUATOR_MODEL = args.rewriter_model, args.evaluator_model
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    rows = select_rows(load_jsonl(args.input), args.limit, args.seed)
    started_at = time.monotonic()

    print(f"Selected {len(rows)} rows from {args.input}")
    print(
        f"Input label counts: {dict(Counter(row.get('metadata', {}).get('label') for row in rows))}"
    )

    rewrites = call_rewriter(client, rows)
    rewritten_rows = apply_rewrites(rows, rewrites)

    for round_number in range(1, args.max_rewrite_rounds + 1):
        local_flagged = [
            {
                "source_index": source_index_for(row, fallback_index),
                "reason": "; ".join(local_validate_rewrite(row)),
                "rewrite_instruction": "Fix the local validation errors while preserving the same source pair and management logic.",
            }
            for fallback_index, row in enumerate(rewritten_rows)
            if local_validate_rewrite(row)
        ]

        evaluation = call_evaluator(client, rewritten_rows)
        flagged_by_index = {
            int(item["source_index"]): item for item in evaluation.get("flagged", [])
        }
        for item in local_flagged:
            flagged_by_index[int(item["source_index"])] = item
        flagged = list(flagged_by_index.values())

        print(
            f"Review round {round_number}: {evaluation.get('verdict')} with {len(flagged)} flagged"
        )

        if not flagged:
            for row in rewritten_rows:
                row.setdefault("rewrite_metadata", {})["evaluation_verdict"] = "accepted"
                row["rewrite_metadata"]["evaluator_model"] = EVALUATOR_MODEL
            break

        if round_number == args.max_rewrite_rounds:
            for row in rewritten_rows:
                source_index = source_index_for(row, 0)
                row.setdefault("rewrite_metadata", {})["evaluation_verdict"] = "needs_rewrite"
                row["rewrite_metadata"]["evaluator_model"] = EVALUATOR_MODEL
                row["rewrite_metadata"]["remaining_issues"] = [
                    item for item in flagged if int(item["source_index"]) == source_index
                ]
            break

        flagged_indices = {int(item["source_index"]) for item in flagged}
        rows_by_index = {
            source_index_for(row, fallback_index): row
            for fallback_index, row in enumerate(rewritten_rows)
        }
        repair_rows = [rows_by_index[idx] for idx in flagged_indices]
        instructions = {int(item["source_index"]): item["rewrite_instruction"] for item in flagged}
        repaired = apply_rewrites(
            repair_rows,
            call_rewriter(client, repair_rows, instructions),
        )
        repaired_by_index = {
            source_index_for(row, fallback_index): row
            for fallback_index, row in enumerate(repaired)
        }
        rewritten_rows = [
            repaired_by_index.get(source_index_for(row, fallback_index), row)
            for fallback_index, row in enumerate(rewritten_rows)
        ]

    save_jsonl(args.output, rewritten_rows)
    elapsed = time.monotonic() - started_at
    print(f"Wrote {len(rewritten_rows)} rows to {args.output} in {elapsed:.1f}s")
    print(
        f"Output label counts: {dict(Counter(row.get('metadata', {}).get('label') for row in rewritten_rows))}"
    )


if __name__ == "__main__":
    main()
