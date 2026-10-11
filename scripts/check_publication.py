#!/usr/bin/env python3
"""Check Git's publishable file set without printing matched secret values."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERNS = [
    r"AIza[0-9A-Za-z_-]{30,}",
    r"hf_[A-Za-z0-9]{20,}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"sk-[A-Za-z0-9_-]{20,}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
]


def inspect_file(path: Path) -> list[str]:
    relative = path.relative_to(ROOT)
    problems = []
    if relative.parts[0] in {"local_archive", ".venv", "outputs", "responses"} and path.name != ".gitkeep":
        problems.append("local-only artifact included")
    if relative.parts[:2] in {("data", "raw"), ("data", "processed")} and path.name != ".gitkeep":
        problems.append("local-only dataset included")
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        problems.append("environment credential file included")
    if path.suffix in {".zip", ".safetensors", ".gguf", ".pt", ".pth", ".bin", ".pem", ".key"}:
        problems.append("archive, weights, or key file included")
    if path.suffix in {".jsonl", ".csv", ".parquet", ".html"} and relative.parts[0] != "examples":
        problems.append("uncurated dataset or export included")
    if path.suffix == ".pdf":
        return problems
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return problems + ["unexpected binary file: review before publishing"]
    if any(re.search(pattern, text) for pattern in TOKEN_PATTERNS):
        problems.append("potential secret detected (value withheld)")
    if path.suffix == ".ipynb":
        notebook = json.loads(text)
        if any(
            c.get("outputs") or c.get("execution_count") is not None
            for c in notebook["cells"]
            if c["cell_type"] == "code"
        ):
            problems.append("notebook contains execution output")
    return problems


def main() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    files = sorted(set(p for p in result.stdout.decode().split("\0") if p))
    failures = []
    for name in files:
        path = ROOT / name
        if path.is_file():
            failures.extend(f"{name}: {problem}" for problem in inspect_file(path))
    for local_path in [
        ".env",
        "local_archive/private.jsonl",
        "data/raw/private.jsonl",
        "outputs/private.html",
        "responses/private.md",
        ".venv/private.txt",
    ]:
        ignored = subprocess.run(["git", "check-ignore", "--no-index", "-q", local_path], cwd=ROOT)
        if ignored.returncode != 0:
            failures.append(f"{local_path}: expected ignore rule is missing")
    if failures:
        print("Publication check failed:")
        for failure in failures:
            print(f"  {failure}")
        sys.exit(1)
    print(f"Publication checks passed for {len(files)} Git-visible files.")


if __name__ == "__main__":
    main()
