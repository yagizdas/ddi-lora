#!/usr/bin/env python3
"""Evaluate non-finetuned model outputs and build a comparison review page."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_INPUT = Path("outputs/finetuned_2.jsonl")
DEFAULT_OUTPUT = Path("outputs/finetuned_2_evaluated.jsonl")
DEFAULT_NON_FINE_TUNED_OUTPUT = Path("outputs/baseline_evaluated.jsonl")
DEFAULT_FINE_TUNED_OUTPUT = Path("outputs/finetuned_1_evaluated.jsonl")
DEFAULT_HTML = Path("outputs/model_review.html")

EVALUATOR_MODEL = "gemini-3.1-flash-lite"
EVALUATOR_TEMPERATURE = 0.1
BATCH_SIZE = 5

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {
                        "type": "string",
                        "enum": ["pass", "needs_review", "fail"],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "summary": {"type": "string"},
                    "expected_behavior": {"type": "string"},
                    "model_behavior": {"type": "string"},
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "missing_or_wrong_points": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "human_review_priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": [
                    "index",
                    "verdict",
                    "confidence",
                    "summary",
                    "expected_behavior",
                    "model_behavior",
                    "issues",
                    "missing_or_wrong_points",
                    "human_review_priority",
                ],
            },
        }
    },
    "required": ["reviews"],
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


def chunked(items: list[dict[str, Any]], size: int):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def parse_json_response(response: Any) -> Any:
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Empty evaluator response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)


def compact_case(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    return {
        "index": row.get("index"),
        "user_request": row.get("user", ""),
        "model_response": row.get("generated_text", ""),
        "metadata": {
            "drug_a": metadata.get("drug_a"),
            "drug_b": metadata.get("drug_b"),
            "drug_pair_key": metadata.get("drug_pair_key"),
            "expected_label": metadata.get("label"),
            "source_severity": metadata.get("source_severity"),
            "source_mechanism_type": metadata.get("source_mechanism_type"),
            "source_url": metadata.get("source_url"),
            "is_synthetic_tn": metadata.get("is_synthetic_tn"),
            "quality_notes": metadata.get("quality_notes"),
        },
    }


def build_evaluation_prompt(rows: list[dict[str, Any]]) -> str:
    cases = [compact_case(row) for row in rows]
    return f"""
You are evaluating a clinical drug-drug interaction model answer.

Use the metadata as the ground-truth source expectation. Evaluate whether the model_response correctly answers the user_request.

Verdicts:
- pass: The answer identifies the expected interaction status, severity/actionability, mechanism category, and management logic well enough for this case.
- needs_review: The answer is partly useful but has a notable omission, weak wording, unsupported extra claim, formatting issue, or minor mismatch that a human should inspect.
- fail: The answer gives the wrong interaction status, misses the source drug pair, invents unsupported interaction logic, says no interaction for an actionable DDI, warns about an interaction for a NO_SOURCE_SUPPORTED_DDI case, or gives unsafe/conflicting management.

Evaluation rules:
- The source drug pair must stay centered. Penalize answers that answer about other drugs or omit one source drug.
- For ACTIONABLE_DDI, the answer should recognize an actionable interaction and preserve the source severity/mechanism/management direction.
- For NO_SOURCE_SUPPORTED_DDI, the answer should avoid inventing a clinically actionable interaction and should communicate that no source-supported actionable interaction was found.
- For MULTI_DDI_SYNTHESIS, the answer should address the relevant multiple interaction risks without collapsing the case into one unrelated pair.
- Do not require identical wording or identical patient demographics.
- Be strict about clinically meaningful contradictions.
- Keep descriptions concise and useful for manual review.

Return valid JSON only.

Cases:
{json.dumps(cases, ensure_ascii=False, indent=2)}
""".strip()


def call_evaluator(client: genai.Client, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response = client.models.generate_content(
        model=EVALUATOR_MODEL,
        contents=build_evaluation_prompt(rows),
        config=types.GenerateContentConfig(
            temperature=EVALUATOR_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=EVALUATION_SCHEMA,
        ),
    )
    parsed = parse_json_response(response)
    return parsed["reviews"]


def evaluate_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
    batch_size: int,
) -> list[dict[str, Any]]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    evaluated: list[dict[str, Any]] = []
    started_at = time.monotonic()

    for batch_number, (start, batch) in enumerate(chunked(rows, batch_size), start=1):
        reviews = call_evaluator(client, batch)
        reviews_by_index = {review["index"]: review for review in reviews}

        for row in batch:
            idx = row.get("index")
            if idx not in reviews_by_index:
                raise ValueError(f"Evaluator did not return a review for index={idx}")
            evaluated_row = dict(row)
            evaluated_row["evaluation"] = reviews_by_index[idx] | {
                "evaluator_model": EVALUATOR_MODEL,
            }
            evaluated.append(evaluated_row)

        save_jsonl(output_path, evaluated)
        elapsed = time.monotonic() - started_at
        print(
            f"Evaluated batch {batch_number}: {len(evaluated)}/{len(rows)} rows in {elapsed:.1f}s",
            flush=True,
        )

    return evaluated


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Result Review</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #fff;
      --ink: #20242a;
      --muted: #667085;
      --line: #d8dde6;
      --accent: #1457d9;
      --pass: #167647;
      --review: #9a5b00;
      --fail: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(245, 246, 248, .96);
      border-bottom: 1px solid var(--line);
      padding: 12px 18px;
    }}
    h1 {{ margin: 0 0 10px; font-size: 18px; }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 10px;
    }}
    .tabs button {{
      min-height: 32px;
      padding: 6px 10px;
    }}
    .tabs button.active {{
      background: var(--ink);
      border-color: var(--ink);
      color: #fff;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 150px 150px 150px auto;
      gap: 8px;
      align-items: center;
    }}
    .performance {{
      margin: 0 0 10px;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
    }}
    .performance table {{ width: 100%; border-collapse: collapse; }}
    .performance th, .performance td {{
      padding: 7px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    .performance tr:last-child td {{ border-bottom: 0; }}
    .performance th {{ color: var(--muted); font-size: 12px; }}
    .metric-pass {{ color: var(--pass); font-weight: 750; }}
    .metric-review {{ color: var(--review); font-weight: 750; }}
    .metric-fail {{ color: var(--fail); font-weight: 750; }}
    input, select, button, textarea {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    input, select, button {{ min-height: 36px; padding: 7px 10px; }}
    button {{ cursor: pointer; font-weight: 650; }}
    button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
    main {{
      display: grid;
      grid-template-columns: minmax(300px, 390px) minmax(0, 1fr);
      gap: 14px;
      padding: 14px 18px 24px;
    }}
    .list, .detail {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: calc(100vh - 108px);
    }}
    .list {{ overflow: auto; max-height: calc(100vh - 108px); }}
    .row {{
      display: block;
      width: 100%;
      padding: 10px 12px;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      background: #fff;
      text-align: left;
    }}
    .row.active {{ background: #eef4ff; }}
    .row-title {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
      font-weight: 750;
    }}
    .row-sub {{
      margin-top: 3px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .detail {{ padding: 16px; overflow: auto; max-height: calc(100vh - 108px); }}
    .case-title {{ margin: 0 0 8px; font-size: 20px; }}
    .summary, .actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #f9fafb;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }}
    .badge.pass {{ color: var(--pass); border-color: #9ed8b8; background: #effaf4; }}
    .badge.needs_review {{ color: var(--review); border-color: #f3c781; background: #fff8ea; }}
    .badge.fail {{ color: var(--fail); border-color: #f5aaa4; background: #fff1f0; }}
    .actions button.selected {{ outline: 2px solid currentColor; }}
    .block {{ border-top: 1px solid var(--line); padding-top: 14px; margin-top: 14px; }}
    .block h2 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); text-transform: uppercase; }}
    pre {{
      margin: 0;
      padding: 12px;
      border-radius: 6px;
      background: #f4f6f8;
      white-space: pre-wrap;
      word-break: break-word;
      overflow: auto;
      font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    textarea {{ width: 100%; min-height: 90px; padding: 10px; resize: vertical; }}
    .empty {{ padding: 24px; color: var(--muted); }}
    @media (max-width: 900px) {{
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
      main {{ grid-template-columns: 1fr; }}
      .list, .detail {{ max-height: none; min-height: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Model Result Review</h1>
    <section class="performance" id="performance"></section>
    <div class="tabs" id="model-tabs"></div>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search drug, text, issue">
      <select id="verdict"></select>
      <select id="label"></select>
      <select id="human"></select>
      <button id="export" class="primary">Export Reviews</button>
    </div>
  </header>
  <main>
    <section class="list" id="list"></section>
    <section class="detail" id="detail"></section>
  </main>
  <script id="review-data" type="application/json">{data_json}</script>
  <script>
    const datasets = JSON.parse(document.getElementById('review-data').textContent);
    const rows = datasets.flatMap(dataset => dataset.rows.map(row => ({{
      ...row,
      model_id: dataset.id,
      model_label: dataset.label,
    }})));
    const state = {{ query: '', verdict: '', label: '', human: '', model: 'non_finetuned', active: '' }};
    const saved = JSON.parse(localStorage.getItem('model_result_human_reviews_v2') || '{{}}');
    const els = {{
      performance: document.getElementById('performance'),
      tabs: document.getElementById('model-tabs'),
      search: document.getElementById('search'),
      verdict: document.getElementById('verdict'),
      label: document.getElementById('label'),
      human: document.getElementById('human'),
      list: document.getElementById('list'),
      detail: document.getElementById('detail'),
      export: document.getElementById('export'),
    }};
    const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({{
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }}[ch]));
    const keyFor = row => `${{row.model_id}}:${{row.index}}`;
    const humanFor = row => saved[keyFor(row)]?.verdict || 'unreviewed';
    const activeFor = row => keyFor(row);
    const rate = (count, total) => total ? `${{(count / total * 100).toFixed(1)}}%` : '0.0%';
    function optionize(select, values, first) {{
      select.innerHTML = `<option value="">${{first}}</option>` + values.map(v => `<option value="${{esc(v)}}">${{esc(v)}}</option>`).join('');
    }}
    function filtered() {{
      const q = state.query.toLowerCase();
      return rows.filter(row => {{
        const meta = row.metadata || {{}};
        const ev = row.evaluation || {{}};
        const hay = JSON.stringify([row.model_label, row.user, row.generated_text, meta, ev]).toLowerCase();
        if (state.model && row.model_id !== state.model) return false;
        if (q && !hay.includes(q)) return false;
        if (state.verdict && ev.verdict !== state.verdict) return false;
        if (state.label && meta.label !== state.label) return false;
        if (state.human && humanFor(row) !== state.human) return false;
        return true;
      }});
    }}
    function badge(text, cls='') {{ return `<span class="badge ${{cls}}">${{esc(text)}}</span>`; }}
    function renderPerformance() {{
      const body = datasets.map(dataset => {{
        const modelRows = rows.filter(row => row.model_id === dataset.id);
        const counts = {{ pass: 0, needs_review: 0, fail: 0 }};
        modelRows.forEach(row => {{
          const verdict = row.evaluation?.verdict;
          if (verdict in counts) counts[verdict] += 1;
        }});
        const total = modelRows.length;
        return `<tr>
          <td><strong>${{esc(dataset.label)}}</strong></td>
          <td>${{total}}</td>
          <td class="metric-pass">${{counts.pass}} (${{rate(counts.pass, total)}})</td>
          <td class="metric-review">${{counts.needs_review}} (${{rate(counts.needs_review, total)}})</td>
          <td class="metric-fail">${{counts.fail}} (${{rate(counts.fail, total)}})</td>
        </tr>`;
      }}).join('');
      els.performance.innerHTML = `<table>
        <thead><tr><th>Model</th><th>Total</th><th>Pass</th><th>Needs review</th><th>Fail</th></tr></thead>
        <tbody>${{body}}</tbody>
      </table>`;
    }}
    function renderTabs() {{
      const counts = Object.fromEntries(datasets.map(dataset => [
        dataset.id,
        rows.filter(row => row.model_id === dataset.id).length,
      ]));
      els.tabs.innerHTML = datasets.map(dataset => `
        <button class="${{state.model === dataset.id ? 'active' : ''}}" data-model="${{dataset.id}}">
          ${{esc(dataset.label)}} (${{counts[dataset.id] || 0}})
        </button>
      `).join('');
      els.tabs.querySelectorAll('[data-model]').forEach(btn => btn.addEventListener('click', () => {{
        state.model = btn.dataset.model;
        state.active = '';
        render();
      }}));
    }}
    function renderList() {{
      const shown = filtered();
      if (!shown.length) {{
        els.list.innerHTML = '<div class="empty">No matching cases.</div>';
        els.detail.innerHTML = '<div class="empty">No case selected.</div>';
        return;
      }}
      if (!shown.some(row => activeFor(row) === state.active)) state.active = activeFor(shown[0]);
      els.list.innerHTML = shown.map(row => {{
        const meta = row.metadata || {{}};
        const ev = row.evaluation || {{}};
        return `<button class="row ${{activeFor(row) === state.active ? 'active' : ''}}" data-key="${{activeFor(row)}}">
          <div class="row-title">
            <span>${{esc(row.model_label)}} #${{row.index}} · ${{esc(meta.drug_pair_key || '')}}</span>
            ${{badge(ev.verdict || 'missing', ev.verdict || '')}}
          </div>
          <div class="row-sub">${{esc(meta.label || '')}} · ${{esc(ev.summary || row.user || '')}}</div>
        </button>`;
      }}).join('');
      els.list.querySelectorAll('.row').forEach(btn => btn.addEventListener('click', () => {{
        state.active = btn.dataset.key;
        render();
      }}));
    }}
    function setHuman(row, verdict) {{
      const key = keyFor(row);
      saved[key] = saved[key] || {{}};
      saved[key].verdict = verdict;
      saved[key].notes = document.getElementById('notes')?.value || saved[key].notes || '';
      saved[key].updated_at = new Date().toISOString();
      localStorage.setItem('model_result_human_reviews_v2', JSON.stringify(saved));
      render();
    }}
    function comparisonBlocks(row) {{
      return rows
        .filter(item => item.model_id !== row.model_id && item.index === row.index)
        .map(peer => {{
          const peerEv = peer.evaluation || {{}};
          return `
            <div class="block"><h2>${{esc(peer.model_label)}} Result</h2>
              <div class="summary">
                ${{badge(peerEv.verdict || 'missing', peerEv.verdict || '')}}
                ${{badge('priority: ' + (peerEv.human_review_priority || ''))}}
              </div>
              <pre>${{esc(peer.generated_text)}}</pre>
            </div>
            <div class="block"><h2>${{esc(peer.model_label)}} Evaluation</h2><pre>${{esc(peerEv.summary || '')}}</pre></div>
          `;
        }}).join('');
    }}
    function renderDetail() {{
      const row = rows.find(item => activeFor(item) === state.active) || filtered()[0];
      if (!row) return;
      const meta = row.metadata || {{}};
      const ev = row.evaluation || {{}};
      const review = saved[keyFor(row)] || {{}};
      els.detail.innerHTML = `
        <h2 class="case-title">${{esc(row.model_label)}} Case #${{row.index}}</h2>
        <div class="summary">
          ${{badge(row.model_label)}}
          ${{badge(meta.drug_pair_key || '')}}
          ${{badge(meta.label || '')}}
          ${{badge(meta.source_severity || '')}}
          ${{badge(meta.source_mechanism_type || '')}}
          ${{badge(ev.verdict || 'missing', ev.verdict || '')}}
          ${{badge('priority: ' + (ev.human_review_priority || ''))}}
          ${{badge('human: ' + humanFor(row))}}
        </div>
        <div class="actions">
          <button data-human="pass" class="${{review.verdict === 'pass' ? 'selected' : ''}}">Human Pass</button>
          <button data-human="needs_review" class="${{review.verdict === 'needs_review' ? 'selected' : ''}}">Human Review</button>
          <button data-human="fail" class="${{review.verdict === 'fail' ? 'selected' : ''}}">Human Fail</button>
          <button data-human="unreviewed">Clear</button>
        </div>
        <div class="block"><h2>User Request</h2><pre>${{esc(row.user)}}</pre></div>
        <div class="block"><h2>Model Answer</h2><pre>${{esc(row.generated_text)}}</pre></div>
        ${{comparisonBlocks(row)}}
        <div class="block"><h2>Evaluator Summary</h2><pre>${{esc(ev.summary)}}</pre></div>
        <div class="block"><h2>Expected Behavior</h2><pre>${{esc(ev.expected_behavior)}}</pre></div>
        <div class="block"><h2>Model Behavior</h2><pre>${{esc(ev.model_behavior)}}</pre></div>
        <div class="block"><h2>Issues</h2><pre>${{esc((ev.issues || []).join('\\n') || 'None listed')}}</pre></div>
        <div class="block"><h2>Missing Or Wrong Points</h2><pre>${{esc((ev.missing_or_wrong_points || []).join('\\n') || 'None listed')}}</pre></div>
        <div class="block"><h2>Metadata</h2><pre>${{esc(JSON.stringify(meta, null, 2))}}</pre></div>
        <div class="block"><h2>Human Notes</h2><textarea id="notes" placeholder="Add manual review notes">${{esc(review.notes || '')}}</textarea></div>
      `;
      els.detail.querySelectorAll('[data-human]').forEach(btn => btn.addEventListener('click', () => setHuman(row, btn.dataset.human)));
      document.getElementById('notes').addEventListener('input', event => {{
        const key = keyFor(row);
        saved[key] = saved[key] || {{}};
        saved[key].notes = event.target.value;
        saved[key].updated_at = new Date().toISOString();
        localStorage.setItem('model_result_human_reviews_v2', JSON.stringify(saved));
      }});
    }}
    function render() {{ renderPerformance(); renderTabs(); renderList(); renderDetail(); }}
    optionize(els.verdict, ['pass', 'needs_review', 'fail'], 'All evaluator verdicts');
    optionize(els.label, [...new Set(rows.map(row => row.metadata?.label).filter(Boolean))].sort(), 'All labels');
    optionize(els.human, ['unreviewed', 'pass', 'needs_review', 'fail'], 'All human verdicts');
    els.search.addEventListener('input', event => {{ state.query = event.target.value; render(); }});
    els.verdict.addEventListener('change', event => {{ state.verdict = event.target.value; render(); }});
    els.label.addEventListener('change', event => {{ state.label = event.target.value; render(); }});
    els.human.addEventListener('change', event => {{ state.human = event.target.value; render(); }});
    els.export.addEventListener('click', () => {{
      const payload = rows.map(row => ({{
        model_id: row.model_id,
        model_label: row.model_label,
        index: row.index,
        metadata: row.metadata,
        evaluation: row.evaluation,
        human_review: saved[keyFor(row)] || {{ verdict: 'unreviewed', notes: '' }},
      }}));
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'model_result_human_reviews_combined.json';
      link.click();
      URL.revokeObjectURL(link.href);
    }});
    render();
  </script>
</body>
</html>
"""


def make_dataset(dataset_id: str, label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": dataset_id,
        "label": label,
        "rows": rows,
    }


def build_review_html(datasets: list[dict[str, Any]], output_path: Path) -> None:
    data_json = json.dumps(datasets, ensure_ascii=False).replace("<", "\\u003c")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(HTML_TEMPLATE.format(data_json=data_json), encoding="utf-8")


def main() -> None:
    global EVALUATOR_MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--non-finetuned-output",
        type=Path,
        default=DEFAULT_NON_FINE_TUNED_OUTPUT,
    )
    parser.add_argument("--fine-tuned-output", type=Path, default=DEFAULT_FINE_TUNED_OUTPUT)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Build the review HTML from already evaluated JSONL files.",
    )
    parser.add_argument("--model", default=EVALUATOR_MODEL)
    parser.add_argument(
        "--comparison", action="store_true", help="Also load the baseline and first fine-tuned run"
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    EVALUATOR_MODEL = args.model
    load_dotenv()

    if args.html_only:
        fine_tuned_2_evaluated = load_jsonl(args.output)
    else:
        rows = load_jsonl(args.input)
        fine_tuned_2_evaluated = evaluate_rows(rows, args.output, args.batch_size)

    datasets = [make_dataset("fine_tuned_2", "Evaluated run", fine_tuned_2_evaluated)]
    if args.comparison:
        datasets = [
            make_dataset("non_finetuned", "Non-finetuned", load_jsonl(args.non_finetuned_output)),
            make_dataset("fine_tuned", "Fine-tuned 1", load_jsonl(args.fine_tuned_output)),
        ] + datasets
    build_review_html(datasets, args.html)

    print(f"Wrote Fine-tuned 2 evaluated JSONL: {args.output}")
    print(f"Wrote review page: {args.html}")
    for dataset in datasets:
        counts = Counter(
            row.get("evaluation", {}).get("verdict", "missing") for row in dataset["rows"]
        )
        print(f"{dataset['label']} evaluator verdict counts: {dict(counts)}")


if __name__ == "__main__":
    main()
