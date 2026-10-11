#!/usr/bin/env python3
"""Build three local review pages from fictional fixtures without API calls."""

from pathlib import Path

from patient_aware_ddi.evaluate import build_review_html, load_jsonl, make_dataset
from patient_aware_ddi.merge import combine_jsonl_by_url
from patient_aware_ddi.review_sources import render_html
from patient_aware_ddi.review_synthetic import build_review


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    examples = root / "examples"
    output = root / "outputs" / "demo"
    output.mkdir(parents=True, exist_ok=True)
    combine_jsonl_by_url([str(examples / "interactions.jsonl")], str(output / "combined.jsonl"))
    rows = load_jsonl(output / "combined.jsonl")
    for i, row in enumerate(rows, 1):
        row.update({"_index": i, "_source_file": "examples/interactions.jsonl"})
    from patient_aware_ddi.review_sources import counts

    payload = {
        "generated_from": {
            "input_files": ["examples/interactions.jsonl"],
            "combined_file": "outputs/demo/combined.jsonl",
            "dedupe_key": "url",
            "duplicate_urls_skipped": 0,
        },
        "summary": {
            "records": len(rows),
            "unique_urls": len(rows),
            "source_counts": {"examples/interactions.jsonl": len(rows)},
            "severity_counts": counts(rows, "severity"),
            "mechanism_counts": counts(rows, "mechanism_type"),
        },
        "rows": rows,
    }
    (output / "source_review.html").write_text(render_html(payload), encoding="utf-8")
    build_review(examples / "conversations.jsonl", output / "synthetic_review.html")
    datasets = [
        make_dataset(
            "fictional_fixture",
            "Fictional demo — no model run",
            load_jsonl(examples / "evaluated_outputs.jsonl"),
        )
    ]
    build_review_html(datasets, output / "model_review.html")
    print("Built offline review pages from fictional fixtures:")
    for page in sorted(output.glob("*.html")):
        print(f"  {page}")


if __name__ == "__main__":
    main()
