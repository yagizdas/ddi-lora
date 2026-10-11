import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

from patient_aware_ddi.crawl import parse_interaction_page
from patient_aware_ddi.evaluate import build_review_html, make_dataset
from patient_aware_ddi.review_sources import render_html
from patient_aware_ddi.review_synthetic import build_review


ROOT = Path(__file__).resolve().parents[1]
CONTENT = '</script><script>alert("fixture")</script> x < y & z > 0'


def embedded_json(page, element_id):
    match = re.search(
        rf'<script id="{element_id}" type="application/json">(.*?)</script>', page, re.S
    )
    assert match
    assert "</script>" not in match.group(1)
    return json.loads(match.group(1))


def test_synthetic_review_roundtrips_special_characters(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "nested/review.html"
    source.write_text(
        json.dumps(
            {"messages": [{"role": "user", "content": CONTENT}], "metadata": {"label": "fixture"}}
        )
        + "\n"
    )
    build_review(source, output)
    rows = embedded_json(output.read_text(), "dataset")
    assert rows[0]["messages"][0]["content"] == CONTENT


def test_evaluation_review_roundtrips_special_characters(tmp_path):
    output = tmp_path / "nested/review.html"
    dataset = make_dataset(
        "fixture", "Fixture", [{"index": 0, "user": CONTENT, "generated_text": CONTENT}]
    )
    build_review_html([dataset], output)
    rows = embedded_json(output.read_text(), "review-data")
    assert rows[0]["rows"][0]["generated_text"] == CONTENT


def test_source_review_roundtrips_special_characters():
    assert embedded_json(render_html({"fixture": CONTENT}), "payload")["fixture"] == CONTENT


def test_crawler_parses_local_html_without_network():
    html = """<div class="title">Interaction between Alpha and Beta</div>
    <span class="badge">Moderate</span><span class="badge">Example</span>
    <table><tr><td>Interaction</td><td>Source text <div id="effect-block">Remove duplicate</div></td></tr>
    <tr><td>Management</td><td> Example management </td></tr></table>"""
    row = parse_interaction_page(html, 123)
    assert (row["drug_a"], row["drug_b"]) == ("Alpha", "Beta")
    assert row["severity"] == "Moderate"
    assert row["interaction"] == "Source text"
    assert row["management"] == "Example management"


@pytest.mark.parametrize(
    "module",
    [
        "crawl",
        "merge",
        "negatives",
        "generate",
        "sample",
        "rewrite",
        "evaluate",
        "review_synthetic",
        "review_sources",
    ],
)
def test_cli_help_works_without_credentials(module, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = subprocess.run(
        [sys.executable, "-m", f"patient_aware_ddi.{module}", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_generator_import_does_not_initialize_api_client(monkeypatch):
    from google import genai

    def unexpected_client(*args, **kwargs):
        pytest.fail("Import must not construct an API client")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(genai, "Client", unexpected_client)
    module = importlib.import_module("patient_aware_ddi.generate")
    importlib.reload(module)
    assert module.client is None


def test_notebook_schema_clean_outputs_and_correct_export():
    notebook = nbformat.read(ROOT / "notebooks/finetune_qwen3_14b.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "for line in model_outputs:" in code
    assert "for line in evaluation_dataset:" not in code
    assert all(
        not c.outputs and c.execution_count is None for c in notebook.cells if c.cell_type == "code"
    )
