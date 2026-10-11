#!/usr/bin/env python3
"""Build a static manual review page for synthetic_messages.jsonl."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = Path("data/processed/synthetic_messages.jsonl")
DEFAULT_OUTPUT = Path("outputs/synthetic_review.html")


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic DDI Review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #20242a;
      --muted: #667085;
      --line: #d8dde6;
      --accent: #1463ff;
      --ok: #127c45;
      --bad: #b42318;
      --warn: #a15c07;
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
      z-index: 10;
      background: rgba(246, 247, 249, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 12px 18px;
      backdrop-filter: blur(10px);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 180px 180px 180px auto auto;
      gap: 8px;
      align-items: center;
    }}
    input, select, button, textarea {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    input, select, button {{
      min-height: 36px;
      padding: 7px 10px;
    }}
    button {{
      cursor: pointer;
      font-weight: 650;
    }}
    button.primary {{
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(280px, 380px) minmax(0, 1fr);
      gap: 14px;
      padding: 14px 18px 24px;
    }}
    .list, .detail {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: calc(100vh - 108px);
    }}
    .list {{
      overflow: auto;
      max-height: calc(100vh - 108px);
    }}
    .row {{
      width: 100%;
      display: block;
      padding: 10px 12px;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      text-align: left;
      background: #fff;
    }}
    .row.active {{ background: #eef4ff; }}
    .row-title {{
      display: flex;
      gap: 8px;
      justify-content: space-between;
      align-items: baseline;
      font-weight: 700;
    }}
    .row-sub {{
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      margin-top: 3px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      background: #f9fafb;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .badge.pass {{ color: var(--ok); border-color: #98d8b7; background: #edfff5; }}
    .badge.fail {{ color: var(--bad); border-color: #ffb3ad; background: #fff1f0; }}
    .badge.review {{ color: var(--warn); border-color: #ffd49b; background: #fff8eb; }}
    .detail {{
      padding: 16px;
      overflow: auto;
      max-height: calc(100vh - 108px);
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .case-title {{
      margin: 0 0 8px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 14px 0;
    }}
    .actions button.pass {{ border-color: #98d8b7; color: var(--ok); }}
    .actions button.fail {{ border-color: #ffb3ad; color: var(--bad); }}
    .actions button.review {{ border-color: #ffd49b; color: var(--warn); }}
    .actions button.selected {{ outline: 2px solid currentColor; }}
    .block {{
      border-top: 1px solid var(--line);
      padding-top: 14px;
      margin-top: 14px;
    }}
    .block h2 {{
      margin: 0 0 8px;
      font-size: 14px;
      letter-spacing: 0;
      color: var(--muted);
      text-transform: uppercase;
    }}
    pre {{
      margin: 0;
      padding: 12px;
      border-radius: 6px;
      background: #f4f6f8;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .message {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      margin: 8px 0;
      background: #fff;
    }}
    .role {{
      color: var(--muted);
      font-weight: 800;
      margin-bottom: 6px;
      text-transform: uppercase;
      font-size: 12px;
    }}
    textarea {{
      width: 100%;
      min-height: 90px;
      padding: 10px;
      resize: vertical;
    }}
    .empty {{
      padding: 24px;
      color: var(--muted);
    }}
    @media (max-width: 860px) {{
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
      main {{ grid-template-columns: 1fr; }}
      .list, .detail {{ min-height: auto; max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Synthetic DDI Review</h1>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search drug, text, URL, line...">
      <select id="label"></select>
      <select id="severity"></select>
      <select id="verdict">
        <option value="">All verdicts</option>
        <option value="unreviewed">Unreviewed</option>
        <option value="pass">Pass</option>
        <option value="review">Needs review</option>
        <option value="fail">Fail</option>
      </select>
      <button id="export" class="primary">Export notes</button>
      <button id="clear">Clear filters</button>
    </div>
  </header>
  <main>
    <section class="list" id="list"></section>
    <section class="detail" id="detail"></section>
  </main>
  <script id="dataset" type="application/json">{dataset_json}</script>
  <script>
    const cases = JSON.parse(document.getElementById('dataset').textContent);
    const storageKey = 'synthetic_messages_review_v1';
    const state = {{
      active: 0,
      query: '',
      label: '',
      severity: '',
      verdict: '',
      reviews: JSON.parse(localStorage.getItem(storageKey) || '{{}}')
    }};

    const els = {{
      search: document.getElementById('search'),
      label: document.getElementById('label'),
      severity: document.getElementById('severity'),
      verdict: document.getElementById('verdict'),
      list: document.getElementById('list'),
      detail: document.getElementById('detail'),
      export: document.getElementById('export'),
      clear: document.getElementById('clear')
    }};

    function esc(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[ch]));
    }}

    function optionize(select, values, allText) {{
      select.innerHTML = '<option value="">' + esc(allText) + '</option>' +
        values.map(value => '<option value="' + esc(value) + '">' + esc(value) + '</option>').join('');
    }}

    function reviewFor(item) {{
      return state.reviews[item.id] || {{ verdict: 'unreviewed', notes: '' }};
    }}

    function saveReview(item, patch) {{
      const current = reviewFor(item);
      state.reviews[item.id] = {{ ...current, ...patch, updated_at: new Date().toISOString() }};
      localStorage.setItem(storageKey, JSON.stringify(state.reviews));
      render();
    }}

    function haystack(item) {{
      const meta = item.metadata || {{}};
      return [
        item.line_number, meta.drug_a, meta.drug_b, meta.drug_pair_key, meta.label,
        meta.source_severity, meta.source_url,
        ...item.messages.map(m => m.content)
      ].join(' ').toLowerCase();
    }}

    function filteredCases() {{
      const q = state.query.trim().toLowerCase();
      return cases.filter(item => {{
        const meta = item.metadata || {{}};
        const review = reviewFor(item);
        if (state.label && meta.label !== state.label) return false;
        if (state.severity && meta.source_severity !== state.severity) return false;
        if (state.verdict && review.verdict !== state.verdict) return false;
        return !q || haystack(item).includes(q);
      }});
    }}

    function renderList(items) {{
      if (!items.length) {{
        els.list.innerHTML = '<div class="empty">No cases match the current filters.</div>';
        return;
      }}
      els.list.innerHTML = items.map((item, idx) => {{
        const meta = item.metadata || {{}};
        const review = reviewFor(item);
        const active = item.id === cases[state.active]?.id ? ' active' : '';
        const pair = meta.drug_pair_key || [meta.drug_a, meta.drug_b].filter(Boolean).join(' + ');
        return `<button class="row${{active}}" data-id="${{esc(item.id)}}">
          <div class="row-title">
            <span>#${{item.line_number}} ${{esc(pair || 'Untitled pair')}}</span>
            <span class="badge ${{esc(review.verdict)}}">${{esc(review.verdict)}}</span>
          </div>
          <div class="row-sub">${{esc(meta.label || '')}} · ${{esc(meta.source_severity || '')}}</div>
        </button>`;
      }}).join('');
      els.list.querySelectorAll('.row').forEach(row => {{
        row.addEventListener('click', () => {{
          const id = row.dataset.id;
          state.active = cases.findIndex(item => item.id === id);
          render();
        }});
      }});
    }}

    function renderDetail(item) {{
      if (!item) {{
        els.detail.innerHTML = '<div class="empty">Select a case to review.</div>';
        return;
      }}
      const meta = item.metadata || {{}};
      const review = reviewFor(item);
      const pair = meta.drug_pair_key || [meta.drug_a, meta.drug_b].filter(Boolean).join(' + ');
      const messages = item.messages.map(m => `
        <div class="message">
          <div class="role">${{esc(m.role)}}</div>
          <pre>${{esc(m.content)}}</pre>
        </div>
      `).join('');
      const metaText = JSON.stringify(meta, null, 2);
      els.detail.innerHTML = `
        <h2 class="case-title">#${{item.line_number}} ${{esc(pair || 'Untitled pair')}}</h2>
        <div class="summary">
          <span class="badge">${{esc(meta.label || 'missing label')}}</span>
          <span class="badge">${{esc(meta.source_severity || 'missing severity')}}</span>
          <span class="badge">${{esc(meta.source_mechanism_type || 'missing mechanism')}}</span>
          <span class="badge">source_index: ${{esc(meta.source_index ?? 'N/A')}}</span>
        </div>
        <div class="actions">
          <button class="pass ${{review.verdict === 'pass' ? 'selected' : ''}}" data-verdict="pass">Pass</button>
          <button class="review ${{review.verdict === 'review' ? 'selected' : ''}}" data-verdict="review">Needs review</button>
          <button class="fail ${{review.verdict === 'fail' ? 'selected' : ''}}" data-verdict="fail">Fail</button>
          <button data-verdict="unreviewed">Reset</button>
        </div>
        <textarea id="notes" placeholder="Reviewer notes...">${{esc(review.notes || '')}}</textarea>
        <div class="block">
          <h2>Messages</h2>
          ${{messages}}
        </div>
        <div class="block">
          <h2>Metadata</h2>
          <pre>${{esc(metaText)}}</pre>
        </div>
      `;
      els.detail.querySelectorAll('button[data-verdict]').forEach(button => {{
        button.addEventListener('click', () => saveReview(item, {{ verdict: button.dataset.verdict }}));
      }});
      els.detail.querySelector('#notes').addEventListener('input', event => {{
        const current = reviewFor(item);
        state.reviews[item.id] = {{ ...current, notes: event.target.value, updated_at: new Date().toISOString() }};
        localStorage.setItem(storageKey, JSON.stringify(state.reviews));
        renderList(filteredCases());
      }});
    }}

    function render() {{
      const items = filteredCases();
      if (!items.some(item => item.id === cases[state.active]?.id) && items.length) {{
        state.active = cases.findIndex(item => item.id === items[0].id);
      }}
      renderList(items);
      renderDetail(cases[state.active]);
    }}

    function exportNotes() {{
      const payload = cases.map(item => {{
        const meta = item.metadata || {{}};
        const review = reviewFor(item);
        return {{
          id: item.id,
          line_number: item.line_number,
          drug_pair_key: meta.drug_pair_key,
          label: meta.label,
          source_severity: meta.source_severity,
          verdict: review.verdict,
          notes: review.notes || '',
          updated_at: review.updated_at || null
        }};
      }});
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'synthetic_messages_review_notes.json';
      link.click();
      URL.revokeObjectURL(url);
    }}

    optionize(els.label, [...new Set(cases.map(c => c.metadata?.label).filter(Boolean))].sort(), 'All labels');
    optionize(els.severity, [...new Set(cases.map(c => c.metadata?.source_severity).filter(Boolean))].sort(), 'All severities');

    els.search.addEventListener('input', event => {{ state.query = event.target.value; render(); }});
    els.label.addEventListener('change', event => {{ state.label = event.target.value; render(); }});
    els.severity.addEventListener('change', event => {{ state.severity = event.target.value; render(); }});
    els.verdict.addEventListener('change', event => {{ state.verdict = event.target.value; render(); }});
    els.export.addEventListener('click', exportNotes);
    els.clear.addEventListener('click', () => {{
      state.query = state.label = state.severity = state.verdict = '';
      els.search.value = els.label.value = els.severity.value = els.verdict.value = '';
      render();
    }});
    document.addEventListener('keydown', event => {{
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {{
        const items = filteredCases();
        const currentId = cases[state.active]?.id;
        const pos = items.findIndex(item => item.id === currentId);
        const next = event.key === 'ArrowDown' ? Math.min(items.length - 1, pos + 1) : Math.max(0, pos - 1);
        if (items[next]) {{
          state.active = cases.findIndex(item => item.id === items[next].id);
          render();
          event.preventDefault();
        }}
      }}
    }});
    render();
  </script>
</body>
</html>
"""


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["line_number"] = line_number
            record["id"] = f"{path.name}:{line_number}"
            records.append(record)
    return records


def build_review(input_path: Path, output_path: Path) -> Counter:
    records = load_jsonl(input_path)
    counts = Counter(record.get("metadata", {}).get("label", "missing") for record in records)
    dataset_json = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(HTML_TEMPLATE.format(dataset_json=dataset_json), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    counts = build_review(args.input, args.output)
    total = sum(counts.values())
    print(f"Wrote {args.output} for {total} records")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
