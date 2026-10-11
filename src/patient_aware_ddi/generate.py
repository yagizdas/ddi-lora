import argparse
import os
import json
import time
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

from google import genai
from google.genai import types
from dotenv import load_dotenv


INPUT_JSONL = "data/raw/combined.jsonl"
OUTPUT_JSONL = "data/processed/synthetic_messages.jsonl"
CHECKPOINT_JSONL = "data/processed/synthetic_checkpoint.jsonl"

BATCH_SIZE = 50
MAX_REWRITE_ROUNDS = 2

GENERATOR_MODEL = "gemini-2.5-flash-lite"
EVALUATOR_MODEL = "gemini-3.1-flash-lite"
REWRITER_MODEL = EVALUATOR_MODEL

GENERATOR_TEMPERATURE = 0.35
REWRITE_TEMPERATURE = 0.55
EVALUATOR_TEMPERATURE = 0.1

SYSTEM_MESSAGE = (
    "You are a cautious educational clinical decision-support assistant. "
    "Use only source-supported reasoning, avoid unsupported claims, and clearly "
    "state uncertainty when evidence is insufficient. This is synthetic educational "
    "data and not medical advice."
)


client = None


def get_client() -> genai.Client:
    global client
    if client is None:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY in the environment or .env before generation")
        client = genai.Client(api_key=api_key)
    return client


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunked(items: List[Dict[str, Any]], size: int):
    for i in range(0, len(items), size):
        yield i, items[i : i + size]


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def print_progress(
    completed: int,
    total: int,
    accepted_this_run: int,
    started_at: float,
    status: str = "",
    bar_width: int = 30,
) -> None:
    if total <= 0:
        print("[PROGRESS] No rows to process.", flush=True)
        return

    completed = min(completed, total)
    percent = completed / total
    filled = int(bar_width * percent)
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = time.monotonic() - started_at
    eta = "?"

    if completed:
        seconds_per_item = elapsed / completed
        remaining = (total - completed) * seconds_per_item
        eta = format_duration(remaining)

    status_text = f" | {status}" if status else ""
    print(
        f"[PROGRESS] |{bar}| {completed}/{total} "
        f"({percent * 100:5.1f}%) | accepted: {accepted_this_run} | "
        f"elapsed: {format_duration(elapsed)} | eta: {eta}{status_text}",
        flush=True,
    )


def get_pair_key(row: Dict[str, Any]) -> str:
    a = str(row.get("drug_a", "")).strip().lower()
    b = str(row.get("drug_b", "")).strip().lower()
    return " + ".join(sorted([a, b]))


def infer_label(row: Dict[str, Any]) -> str:
    if row.get("is_synthetic_tn") is True:
        return "NO_SOURCE_SUPPORTED_DDI"

    if row.get("label"):
        return row["label"]

    severity = str(row.get("severity", "")).lower()
    management = str(row.get("management", "")).strip()

    if "no source-supported" in severity:
        return "NO_SOURCE_SUPPORTED_DDI"

    if severity == "unknown" or management in {"", "-", "N/A"}:
        return "WEAK_OR_UNKNOWN_EVIDENCE"

    return "ACTIONABLE_DDI"


def minimal_source_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_index": row["_source_index"],
        "drug_a": row.get("drug_a"),
        "drug_b": row.get("drug_b"),
        "severity": row.get("severity"),
        "mechanism_type": row.get("mechanism_type"),
        "interaction": row.get("interaction"),
        "management": row.get("management"),
        "label": infer_label(row),
        "is_synthetic_tn": bool(row.get("is_synthetic_tn", False)),
    }


def metadata_from_source(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_index": row["_source_index"],
        "source_interaction_id": row.get("interaction_id"),
        "source_url": row.get("url"),
        "drug_a": row.get("drug_a"),
        "drug_b": row.get("drug_b"),
        "drug_pair_key": get_pair_key(row),
        "source_severity": row.get("severity"),
        "source_mechanism_type": row.get("mechanism_type"),
        "label": infer_label(row),
        "is_synthetic_tn": bool(row.get("is_synthetic_tn", False)),
        "raw_id_field": row.get("raw_id_field"),
        "references": row.get("references"),
    }


def extract_json_text(response) -> str:
    text = response.text
    if not text:
        raise ValueError("Empty Gemini response")
    return text.strip()


def parse_json_response(response) -> Any:
    text = extract_json_text(response)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)


GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_index": {"type": "integer"},
                    "user": {"type": "string"},
                    "assistant": {"type": "string"},
                    "quality_notes": {"type": "string"},
                },
                "required": ["source_index", "user", "assistant", "quality_notes"],
            },
        }
    },
    "required": ["examples"],
}

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["good_enough", "needs_rewrite"],
        },
        "flagged": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_index": {"type": "integer"},
                    "similar_to_source_index": {"type": "integer"},
                    "reason": {"type": "string"},
                    "rewrite_instruction": {"type": "string"},
                },
                "required": [
                    "source_index",
                    "reason",
                    "rewrite_instruction",
                ],
            },
        },
        "batch_notes": {"type": "string"},
    },
    "required": ["verdict", "flagged", "batch_notes"],
}


def build_generation_prompt(records: List[Dict[str, Any]]) -> str:
    return f"""
You are generating synthetic training examples for a computational medicine class project.

You will receive source drug-pair records. For each source record, create exactly one natural clinician-style user request and one assistant answer.

The final training user request must NOT look like raw JSON or a source database row.
It should look like a clinician or pharmacist asking about a patient.

Critical source-pair constraint:
- The user request must explicitly mention BOTH source drugs by name.
- Do not replace either source drug with another medication, another drug from the same class, or a more common alternative.
- Do not generalize the source drug into only a drug class in the user request.
- The assistant answer must assess the same two source drugs.
- If either source drug would be unusual in the scenario, create a plausible scenario where that exact drug is present.
- For NO_SOURCE_SUPPORTED_DDI rows, the source pair must still appear explicitly in the user request. Do not create a negative example around a different pair from the patient's medication list.
- If the source drug includes a route or formulation in parentheses, mention the base drug name naturally.

Rules:
- Use the source record as the factual grounding.
- Do not invent interaction mechanisms not supported by the source record.
- Do not add unsupported drug-drug interaction claims.
- Patient details may modify concern only when relevant to the source mechanism or management.
- Every user request must include a realistic patient context with:
  - exact age and sex,
  - at least two relevant comorbidities or clinical conditions,
  - at least three current medications including the two source drugs or one source drug plus the proposed added drug,
  - at least two numeric clinical details such as eGFR, serum creatinine, potassium, magnesium, AST/ALT, INR, QTc, blood pressure, heart rate, weight, dose, duration of therapy, or relevant lab values.
- The user request should be 2-5 sentences long, not a one-sentence database query.
- Use numeric values only when clinically plausible for the scenario.
- Prefer mechanism-relevant numbers when possible:
  - QT-risk examples: QTc, potassium, magnesium, heart rate.
  - renal-risk examples: eGFR, creatinine, potassium.
  - bleeding-risk examples: INR, platelet count, hemoglobin, age.
  - hepatic-risk examples: AST, ALT, bilirubin.
  - CNS/sedation-risk examples: age, opioid/benzodiazepine dose, respiratory rate, oxygen saturation if relevant.
  - blood-pressure examples: systolic/diastolic BP, heart rate, fall/syncope history.
- Do NOT repeat the same common values too often, such as age 65, eGFR 60, potassium 4.0, QTc 450, BP 130/80.
- For NO_SOURCE_SUPPORTED_DDI examples, still include age, sex, comorbidities, medication list, and at least two numeric clinical details. The assistant must still avoid inventing a pair-specific interaction.
- For ACTIONABLE_DDI rows, the assistant should explain the source-supported interaction and recommendations.
- For WEAK_OR_UNKNOWN_EVIDENCE rows, the assistant should explicitly state that evidence is insufficient and should not invent a mechanism.
- For NO_SOURCE_SUPPORTED_DDI rows, the assistant must not warn about a drug-drug interaction. It should say no source-supported clinically actionable interaction was found in the available project dataset.
- Never say "no interaction exists." Use "no source-supported clinically actionable interaction was found."
- Avoid phrases like "assuming the question implies", "the source interaction is with", or "not listed in the patient's medications."
- The assistant answer must use this structure:
  [1. Priority Interactions]
  [2. Mechanistic Explanation]
  [3. Patient-Specific Risk Assessment]
  [4. Recommendations]
- This is synthetic educational data, not medical advice.
- Return valid JSON only.

Source records:
{json.dumps([minimal_source_record(r) for r in records], ensure_ascii=False, indent=2)}
""".strip()


def build_evaluation_prompt(
    examples: List[Dict[str, Any]],
    review_scope: str = "full batch",
) -> str:
    compact_examples = [
        {
            "source_index": ex["metadata"]["source_index"],
            "drug_a": ex["metadata"]["drug_a"],
            "drug_b": ex["metadata"]["drug_b"],
            "label": ex["metadata"]["label"],
            "local_validation_errors": ex["metadata"].get("validation_errors", []),
            "user": ex["messages"][1]["content"],
            "assistant": ex["messages"][2]["content"],
        }
        for ex in examples
    ]

    return f"""
You are evaluating diversity and consistency in a synthetic clinical training batch.

You will receive {review_scope}.
Your job is NOT to fully judge medical correctness. Your main jobs are:
1. find near-duplicate scenarios,
2. find source-drug mismatch problems,
3. find overly generic or underspecified examples.

Flag examples that are very similar in scenario construction, such as:
- same or nearly same patient age/sex pattern,
- same comorbidity bundle,
- same lab values or nearly identical numeric values,
- same clinical setting and wording,
- same medication-list style,
- assistant wording that is almost templated beyond the required section headings.

Also flag examples where:
- local_validation_errors is not empty,
- the user request does not explicitly include both source drugs from drug_a and drug_b,
- one source drug is replaced by another drug from the same class,
- the assistant discusses a drug pair that is not actually present in the user request,
- the answer contains phrases like "assuming the question implies", "not listed in the patient's current medications", "the source interaction is with", or "while the source specifically mentions",
- the user prompt has no exact age,
- the user prompt has fewer than two numeric clinical values,
- the user prompt is a one-sentence database query,
- TN examples are much shorter or less clinically detailed than actionable examples.

Do not be too strict about ordinary overlap. Some similarity is acceptable because all examples are DDI reasoning examples.
Flag only cases that reduce dataset quality or create source-pair inconsistency.

Return:
- verdict = "good_enough" if no significant problems exist.
- verdict = "needs_rewrite" if some examples should be rewritten.
- flagged = list of examples that should be rewritten. Prefer flagging the later/less distinctive item, not both.
- rewrite_instruction = specific advice for making that example different or fixing the source-pair issue.
- On follow-up review, only evaluate and return items from the selected examples you received. Do not ask to re-review or return the whole original batch.

Training pairs:
{json.dumps(compact_examples, ensure_ascii=False, indent=2)}
""".strip()


def build_rewrite_prompt(
    source_records_by_index: Dict[int, Dict[str, Any]],
    current_examples_by_index: Dict[int, Dict[str, Any]],
    flagged_items: List[Dict[str, Any]],
) -> str:
    rewrite_payload = []

    for item in flagged_items:
        idx = item["source_index"]
        rewrite_payload.append(
            {
                "source_record": minimal_source_record(source_records_by_index[idx]),
                "current_user": current_examples_by_index[idx]["messages"][1]["content"],
                "current_assistant": current_examples_by_index[idx]["messages"][2]["content"],
                "reason_flagged": item["reason"],
                "rewrite_instruction": item["rewrite_instruction"],
            }
        )

    return f"""
You are rewriting flagged synthetic clinical training examples to improve diversity and fix validation problems.

For each flagged item:
- Rewrite only the flagged/selected items provided below.
- Return exactly one rewritten example for each flagged source_index, and do not return unflagged examples from the original batch.
- Keep the same source drug pair and same source-supported clinical meaning.
- The rewritten user request must explicitly include BOTH source drugs by name.
- Do not substitute a different drug from the same class.
- Rewrite the patient scenario so it is clearly different from the original.
- Change age, sex if clinically plausible, care setting, comorbidities, lab values, clinical reason, and wording style.
- The rewritten user request must include exact age, sex, at least two comorbidities or clinical conditions, a medication list, and at least two numeric clinical details.
- If the original was vague, make it concrete with plausible values such as eGFR, creatinine, potassium, magnesium, QTc, INR, BP, HR, weight, dose, or duration.
- Do not reuse the same numeric values from the current user prompt unless directly required by the source.
- Do not invent unsupported interaction mechanisms.
- Keep the same four-section assistant structure.
- For NO_SOURCE_SUPPORTED_DDI rows, do not invent an interaction.
- Avoid phrases like "assuming the question implies", "the source interaction is with", or "not listed in the patient's medications."
- Return valid JSON only.

Flagged items:
{json.dumps(rewrite_payload, ensure_ascii=False, indent=2)}
""".strip()


def call_generator(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt = build_generation_prompt(records)

    response = get_client().models.generate_content(
        model=GENERATOR_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=GENERATOR_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=GENERATION_SCHEMA,
        ),
    )

    parsed = parse_json_response(response)
    return parsed["examples"]


def call_evaluator(
    examples: List[Dict[str, Any]],
    review_scope: str = "a full batch of 50 or fewer user/assistant training pairs",
) -> Dict[str, Any]:
    prompt = build_evaluation_prompt(examples, review_scope=review_scope)

    response = get_client().models.generate_content(
        model=EVALUATOR_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=EVALUATOR_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=EVALUATION_SCHEMA,
        ),
    )

    return parse_json_response(response)


def call_rewriter(
    source_records_by_index: Dict[int, Dict[str, Any]],
    current_examples_by_index: Dict[int, Dict[str, Any]],
    flagged_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    prompt = build_rewrite_prompt(
        source_records_by_index=source_records_by_index,
        current_examples_by_index=current_examples_by_index,
        flagged_items=flagged_items,
    )

    response = get_client().models.generate_content(
        model=REWRITER_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=REWRITE_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=GENERATION_SCHEMA,
        ),
    )

    parsed = parse_json_response(response)
    return parsed["examples"]


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
    text = str(text).lower()
    text = re.sub(r"$begin:math:text$\[\^\)\]\*$end:math:text$", " ", text)
    text = text.replace("ω", "omega")
    text = re.sub(r"[^a-z0-9\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def meaningful_drug_tokens(drug_name: str) -> List[str]:
    normalized = normalize_text_for_match(drug_name)
    raw_tokens = normalized.split()

    tokens = []
    for token in raw_tokens:
        if token in DRUG_STOP_TOKENS:
            continue
        if len(token) < 4 and not token.startswith("omega"):
            continue
        tokens.append(token)

    expanded = []
    for token in tokens:
        expanded.append(token)
        if token.startswith("omega"):
            expanded.append("omega")

    return list(dict.fromkeys(expanded))


def fuzzy_contains_token(text: str, token: str, threshold: float = 0.80) -> bool:
    normalized_text = normalize_text_for_match(text)
    text_tokens = normalized_text.split()

    if token in normalized_text:
        return True

    for word in text_tokens:
        if len(word) < 4:
            continue

        ratio = SequenceMatcher(None, token, word).ratio()
        if ratio >= threshold:
            return True

    return False


def drug_mentioned_in_text(drug_name: str, text: str, threshold: float = 0.80) -> bool:
    tokens = meaningful_drug_tokens(drug_name)

    if not tokens:
        return False

    return any(fuzzy_contains_token(text, token, threshold=threshold) for token in tokens)


def count_numeric_tokens(text: str) -> int:
    patterns = [
        r"\b\d{2,3}\s*[- ]?year[- ]?old\b",
        r"\b\d+(\.\d+)?\s*(mg|mcg|g|units|iu|mmol/l|meq/l|mg/dl|ml/min|ml/min/1\.73m²|bpm|kg|mmhg|%)\b",
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
        r"\bblood pressure\s*(is|of|was|:)?\s*\d{2,3}/\d{2,3}",
        r"\bfev1\s*(is|of|was|:)?\s*\d+",
        r"\bpeak flow\s*(is|of|was|:)?\s*\d+",
    ]

    lowered = text.lower()
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, lowered))
    return count


GENERIC_USER_PATTERNS = [
    r"^what is the interaction potential between\b",
    r"^is there an interaction between\b",
    r"^are there interactions between\b",
    r"^any interaction between\b",
]


def looks_too_generic_user_prompt(text: str) -> bool:
    stripped = text.strip().lower()
    return any(re.search(pattern, stripped) for pattern in GENERIC_USER_PATTERNS)


def to_training_example(
    generated: Dict[str, Any],
    source_record: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": generated["user"].strip(),
            },
            {
                "role": "assistant",
                "content": generated["assistant"].strip(),
            },
        ],
        "metadata": metadata_from_source(source_record)
        | {
            "quality_notes": generated.get("quality_notes", ""),
            "generator_model": GENERATOR_MODEL,
        },
    }


def basic_validate_example(example: Dict[str, Any]) -> List[str]:
    errors = []

    messages = example.get("messages", [])
    if len(messages) != 3:
        errors.append("messages must contain system, user, assistant")
        return errors

    user = messages[1]["content"]
    assistant = messages[2]["content"]
    meta = example["metadata"]

    lower_assistant = assistant.lower()

    required_sections = [
        "[1. Priority Interactions]",
        "[2. Mechanistic Explanation]",
        "[3. Patient-Specific Risk Assessment]",
        "[4. Recommendations]",
    ]

    for section in required_sections:
        if section not in assistant:
            errors.append(f"missing assistant section: {section}")

    drug_a = str(meta.get("drug_a", "") or "")
    drug_b = str(meta.get("drug_b", "") or "")

    if drug_a and not drug_mentioned_in_text(drug_a, user, threshold=0.80):
        errors.append(f"source drug_a appears missing from user prompt: {drug_a}")

    if drug_b and not drug_mentioned_in_text(drug_b, user, threshold=0.80):
        errors.append(f"source drug_b appears missing from user prompt: {drug_b}")

    if drug_a and not drug_mentioned_in_text(drug_a, assistant, threshold=0.80):
        errors.append(f"source drug_a appears missing from assistant answer: {drug_a}")

    if drug_b and not drug_mentioned_in_text(drug_b, assistant, threshold=0.80):
        errors.append(f"source drug_b appears missing from assistant answer: {drug_b}")

    numeric_count = count_numeric_tokens(user)
    if numeric_count < 2:
        errors.append(
            f"user prompt should include at least two numeric clinical details; found {numeric_count}"
        )

    if looks_too_generic_user_prompt(user):
        errors.append("user prompt is too generic/database-query-like")

    forbidden_phrases = [
        "no interaction exists",
        "definitely safe",
        "guaranteed safe",
        "safe to use together",
        "can be used safely",
        "assuming the question implies",
        "not listed in the patient's current medications",
        "not listed in the patient’s current medications",
        "the source interaction is with",
        "while the source specifically mentions",
        "the principle applies to",
        "not the exact source pair",
    ]

    for phrase in forbidden_phrases:
        if phrase in lower_assistant:
            errors.append(f"forbidden phrase: {phrase}")

    if meta.get("label") == "NO_SOURCE_SUPPORTED_DDI":
        if "no source-supported" not in lower_assistant:
            errors.append("TN example should contain 'no source-supported'")

        tn_forbidden = [
            "major interaction",
            "moderate interaction",
            "minor interaction",
            "no identified interaction risk",
            "no interaction risk",
            "no specific management is required",
            "no specific interaction management is required",
        ]

        for phrase in tn_forbidden:
            if phrase in lower_assistant:
                errors.append(f"TN unsafe/overstrong phrase: {phrase}")

    return errors


def convert_generated_batch(
    generated_items: List[Dict[str, Any]],
    source_records_by_index: Dict[int, Dict[str, Any]],
    raise_on_validation: bool = True,
) -> List[Dict[str, Any]]:
    examples = []

    for item in generated_items:
        idx = item["source_index"]
        if idx not in source_records_by_index:
            raise ValueError(f"Generated unknown source_index: {idx}")

        example = to_training_example(item, source_records_by_index[idx])
        errors = basic_validate_example(example)

        if errors:
            example["metadata"]["validation_errors"] = errors
            if raise_on_validation:
                raise ValueError(
                    f"Validation failed for source_index={idx}: {errors}\n"
                    f"User: {example['messages'][1]['content']}\n"
                    f"Assistant: {example['messages'][2]['content'][:700]}"
                )

        examples.append(example)

    return examples


def run_pipeline() -> None:
    raw_rows = load_jsonl(INPUT_JSONL)

    for i, row in enumerate(raw_rows):
        row["_source_index"] = i

    random.seed(42)
    random.shuffle(raw_rows)

    already_done_indices = set()
    if Path(OUTPUT_JSONL).exists():
        existing = load_jsonl(OUTPUT_JSONL)
        already_done_indices = {
            ex["metadata"]["source_index"]
            for ex in existing
            if "metadata" in ex and "source_index" in ex["metadata"]
        }
        print(f"Already completed: {len(already_done_indices)} examples")

    rows_to_process = [row for row in raw_rows if row["_source_index"] not in already_done_indices]

    total_to_process = len(rows_to_process)
    total_accepted = 0
    started_at = time.monotonic()

    print(f"Rows to process this run: {total_to_process}")
    print_progress(0, total_to_process, total_accepted, started_at, "starting")

    for batch_start, batch in chunked(rows_to_process, BATCH_SIZE):
        print(f"\n=== Processing batch starting at {batch_start}, size={len(batch)} ===")
        completed_before_batch = total_accepted

        source_records_by_index = {row["_source_index"]: row for row in batch}

        print_progress(
            completed_before_batch,
            total_to_process,
            total_accepted,
            started_at,
            f"generating batch {batch_start // BATCH_SIZE + 1}",
        )

        generated = call_generator(batch)
        examples = convert_generated_batch(
            generated,
            source_records_by_index,
            raise_on_validation=False,
        )

        print_progress(
            completed_before_batch,
            total_to_process,
            total_accepted,
            started_at,
            "evaluating batch diversity",
        )

        selected_source_indices = None
        for rewrite_round in range(MAX_REWRITE_ROUNDS + 1):
            if rewrite_round == 0:
                review_examples = examples
                review_scope = "a full batch of 50 or fewer user/assistant training pairs"
            else:
                review_examples = [
                    ex
                    for ex in examples
                    if ex["metadata"]["source_index"] in selected_source_indices
                ]
                review_scope = (
                    "only the examples selected in the first review, after rewrite. "
                    "Evaluate only these selected examples and return only selected "
                    "examples that still need another rewrite."
                )

            if not review_examples:
                break

            evaluation = call_evaluator(review_examples, review_scope=review_scope)

            print(f"Evaluator verdict: {evaluation['verdict']}")
            print(f"Evaluator notes: {evaluation.get('batch_notes', '')}")

            if evaluation["verdict"] == "good_enough" or not evaluation["flagged"]:
                break

            flagged = evaluation["flagged"]
            if rewrite_round == 0:
                selected_source_indices = {item["source_index"] for item in flagged}
            else:
                flagged = [
                    item for item in flagged if item["source_index"] in selected_source_indices
                ]

            if not flagged:
                break

            if rewrite_round >= MAX_REWRITE_ROUNDS:
                print("Max rewrite rounds reached. Accepting current batch with notes.")
                break

            print(f"Flagged for rewrite: {[x['source_index'] for x in flagged]}")

            print_progress(
                completed_before_batch,
                total_to_process,
                total_accepted,
                started_at,
                f"rewriting {len(flagged)} flagged examples",
            )

            current_examples_by_index = {ex["metadata"]["source_index"]: ex for ex in examples}

            rewritten_generated = call_rewriter(
                source_records_by_index=source_records_by_index,
                current_examples_by_index=current_examples_by_index,
                flagged_items=flagged,
            )

            rewritten_examples = convert_generated_batch(
                rewritten_generated,
                source_records_by_index,
                raise_on_validation=False,
            )

            flagged_indices = {item["source_index"] for item in flagged}
            rewritten_by_index = {
                ex["metadata"]["source_index"]: ex
                for ex in rewritten_examples
                if ex["metadata"]["source_index"] in flagged_indices
            }

            new_examples = []
            for ex in examples:
                idx = ex["metadata"]["source_index"]
                if idx in rewritten_by_index:
                    replacement = rewritten_by_index[idx]
                    replacement["metadata"]["rewritten_due_to_diversity"] = True
                    replacement["metadata"]["rewrite_round"] = rewrite_round + 1
                    replacement["metadata"]["rewriter_model"] = REWRITER_MODEL
                    new_examples.append(replacement)
                else:
                    new_examples.append(ex)

            examples = new_examples

            time.sleep(1)

        append_jsonl(OUTPUT_JSONL, examples)
        append_jsonl(CHECKPOINT_JSONL, examples)
        total_accepted += len(examples)

        print(f"Accepted batch size: {len(examples)}")
        print(f"Total accepted this run: {total_accepted}")

        print_progress(
            total_accepted,
            total_to_process,
            total_accepted,
            started_at,
            "batch saved",
        )

        time.sleep(2)

    print_progress(
        total_accepted,
        total_to_process,
        total_accepted,
        started_at,
        "complete",
    )
    print("Pipeline complete.")


def main() -> None:
    global INPUT_JSONL, OUTPUT_JSONL, CHECKPOINT_JSONL, BATCH_SIZE
    global GENERATOR_MODEL, EVALUATOR_MODEL, REWRITER_MODEL
    parser = argparse.ArgumentParser(
        description="Generate source-grounded synthetic conversations via Gemini."
    )
    parser.add_argument("--input", default=INPUT_JSONL)
    parser.add_argument("--output", default=OUTPUT_JSONL)
    parser.add_argument("--checkpoint", default=CHECKPOINT_JSONL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--generator-model", default=GENERATOR_MODEL)
    parser.add_argument("--evaluator-model", default=EVALUATOR_MODEL)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    INPUT_JSONL, OUTPUT_JSONL, CHECKPOINT_JSONL = args.input, args.output, args.checkpoint
    if len({Path(p).resolve() for p in [INPUT_JSONL, OUTPUT_JSONL, CHECKPOINT_JSONL]}) != 3:
        parser.error("input, output, and checkpoint must use different paths")
    BATCH_SIZE = args.batch_size
    GENERATOR_MODEL, EVALUATOR_MODEL = args.generator_model, args.evaluator_model
    REWRITER_MODEL = EVALUATOR_MODEL
    get_client()
    run_pipeline()


if __name__ == "__main__":
    main()
