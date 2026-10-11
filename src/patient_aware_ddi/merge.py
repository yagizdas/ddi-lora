"""Merge interaction JSONL files, retaining the first record for each URL."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def combine_jsonl_by_url(input_files: list[str], output_file: str) -> dict[str, int]:
    output = Path(output_file)
    if any(Path(p).resolve() == output.resolve() for p in input_files):
        raise ValueError("Output must differ from every input path")
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_urls = set()
    skipped = 0
    # Preserve an existing output if any input is invalid or unreadable.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            for input_file in input_files:
                with Path(input_file).open(encoding="utf-8") as source:
                    for number, line in enumerate(source, 1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"{input_file}:{number}: invalid JSON") from exc
                        if (
                            not isinstance(record, dict)
                            or not isinstance(record.get("url"), str)
                            or not record["url"].strip()
                        ):
                            raise ValueError(f"{input_file}:{number}: missing or invalid url")
                        if record["url"] in seen_urls:
                            skipped += 1
                            continue
                        seen_urls.add(record["url"])
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"written": len(seen_urls), "skipped_duplicates": skipped, "unique_urls": len(seen_urls)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Source JSONL files, in precedence order")
    parser.add_argument("-o", "--output", default="data/raw/combined.jsonl")
    args = parser.parse_args()
    stats = combine_jsonl_by_url(args.inputs, args.output)
    print(
        f"Wrote {stats['written']} unique records; skipped {stats['skipped_duplicates']} duplicates."
    )


if __name__ == "__main__":
    main()
