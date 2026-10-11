import argparse
import json
from collections import Counter
from pathlib import Path


INPUT_FILES = [
    "data/raw/ddinter_interactions.jsonl",
    "data/raw/additional_interactions.jsonl",
]
COMBINED_FILE = "data/raw/combined.jsonl"
OUTPUT_HTML = "outputs/source_review.html"


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
    return rows


def build_source_lookup(input_files: list[str]) -> tuple[dict[str, str], int]:
    seen = set()
    source_by_url = {}
    duplicate_count = 0

    for input_file in input_files:
        for row in read_jsonl(input_file):
            url = row.get("url")
            if not url:
                continue
            if url in seen:
                duplicate_count += 1
                continue
            seen.add(url)
            source_by_url[url] = input_file

    return source_by_url, duplicate_count


def counts(rows: list[dict], key: str) -> dict[str, int]:
    return dict(Counter(row.get(key) or "Unknown" for row in rows))


def build_payload() -> dict:
    combined_rows = read_jsonl(COMBINED_FILE)
    source_by_url, duplicate_count = build_source_lookup(INPUT_FILES)

    rows = []
    for index, row in enumerate(combined_rows, start=1):
        item = dict(row)
        item["_index"] = index
        item["_source_file"] = source_by_url.get(row.get("url"), "Unknown")
        rows.append(item)

    source_counts = Counter(row["_source_file"] for row in rows)
    urls = [row.get("url") for row in rows if row.get("url")]

    return {
        "generated_from": {
            "input_files": INPUT_FILES,
            "combined_file": COMBINED_FILE,
            "dedupe_key": "url",
            "duplicate_urls_skipped": duplicate_count,
        },
        "summary": {
            "records": len(rows),
            "unique_urls": len(set(urls)),
            "source_counts": dict(source_counts),
            "severity_counts": counts(rows, "severity"),
            "mechanism_counts": counts(rows, "mechanism_type"),
        },
        "rows": rows,
    }


def render_html(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DDInter Merge Review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f4;
      --panel: #ffffff;
      --panel-soft: #f9faf7;
      --ink: #182026;
      --muted: #66737d;
      --line: #dce2df;
      --blue: #2563eb;
      --teal: #0f766e;
      --amber: #b45309;
      --red: #b42318;
      --green: #167647;
      --shadow: 0 14px 34px rgba(24, 32, 38, .08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(180deg, rgba(15, 118, 110, .08), transparent 280px),
        var(--bg);
      color: var(--ink);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    button, input, select, textarea {{
      font: inherit;
      color: inherit;
    }}
    button {{
      cursor: pointer;
      border: 1px solid var(--line);
      background: var(--panel);
      min-height: 36px;
      border-radius: 7px;
      padding: 7px 11px;
      font-weight: 700;
    }}
    button:hover {{ border-color: #b6c2bf; }}
    a {{ color: var(--blue); text-decoration: none; }}
    header {{
      padding: 22px 24px 14px;
      border-bottom: 1px solid rgba(220, 226, 223, .88);
      background: rgba(246, 247, 244, .92);
      backdrop-filter: blur(14px);
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    .topline {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      margin: 0 auto 14px;
      max-width: 1440px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(24px, 3vw, 38px);
      line-height: 1.05;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin-top: 8px;
      color: var(--muted);
      max-width: 760px;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .primary {{
      background: var(--ink);
      border-color: var(--ink);
      color: #fff;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, minmax(128px, 1fr));
      gap: 10px;
      max-width: 1440px;
      margin: 0 auto;
    }}
    .metric {{
      background: rgba(255, 255, 255, .82);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      box-shadow: 0 8px 24px rgba(24, 32, 38, .04);
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .metric strong {{
      display: block;
      margin-top: 4px;
      font-size: 24px;
      line-height: 1;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(240px, 1fr) 150px 170px 180px 110px;
      gap: 9px;
      max-width: 1440px;
      margin: 12px auto 0;
    }}
    input, select {{
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      padding: 8px 10px;
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px 24px 28px;
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 14px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-height: calc(100vh - 246px);
    }}
    .list {{
      overflow: auto;
      max-height: calc(100vh - 246px);
    }}
    .row {{
      width: 100%;
      display: block;
      text-align: left;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      padding: 12px 13px;
      background: #fff;
    }}
    .row:hover, .row.active {{ background: #eef7f5; }}
    .row-title {{
      display: flex;
      gap: 9px;
      justify-content: space-between;
      align-items: baseline;
      font-weight: 800;
    }}
    .row-sub {{
      margin-top: 4px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .chips {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-top: 8px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      color: var(--muted);
      background: var(--panel-soft);
      font-size: 12px;
      font-weight: 750;
    }}
    .chip.major {{ color: var(--red); background: #fff3f1; border-color: #ffd4ce; }}
    .chip.moderate {{ color: var(--amber); background: #fff7ed; border-color: #fed7aa; }}
    .detail {{
      padding: 18px;
      overflow: auto;
      max-height: calc(100vh - 246px);
    }}
    .detail-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }}
    .detail h2 {{
      margin: 0;
      font-size: 25px;
      line-height: 1.16;
      letter-spacing: 0;
    }}
    .pair {{
      margin-top: 8px;
      color: var(--muted);
      font-weight: 700;
    }}
    .section {{
      padding: 15px 0;
      border-bottom: 1px solid var(--line);
    }}
    .section:last-child {{ border-bottom: 0; }}
    .section h3 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .text {{
      margin: 0;
      white-space: pre-wrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .field {{
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-width: 0;
    }}
    .field span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      text-transform: uppercase;
    }}
    .field strong {{
      display: block;
      margin-top: 3px;
      overflow-wrap: anywhere;
    }}
    .bars {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 110px 1fr 42px;
      gap: 8px;
      align-items: center;
      margin: 8px 0;
      color: var(--muted);
      font-weight: 700;
    }}
    .bar {{
      height: 8px;
      border-radius: 999px;
      background: #edf0ee;
      overflow: hidden;
    }}
    .bar i {{
      display: block;
      height: 100%;
      border-radius: inherit;
      background: var(--teal);
    }}
    .bar.alt i {{ background: var(--blue); }}
    .empty {{
      padding: 28px;
      color: var(--muted);
      text-align: center;
    }}
    @media (max-width: 980px) {{
      header {{ position: static; }}
      .topline {{ display: block; }}
      .actions {{ justify-content: flex-start; margin-top: 12px; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
      main {{ grid-template-columns: 1fr; }}
      .panel, .list, .detail {{ min-height: 0; max-height: none; }}
    }}
    @media (max-width: 640px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .metrics, .toolbar, .grid, .bars {{ grid-template-columns: 1fr; }}
      .detail-head {{ display: block; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topline">
      <div>
        <h1>DDInter Merge Review</h1>
        <div class="subtitle">URL-deduplicated interaction records from ddinter_interactions.jsonl and a_ddinter_interactions_2.jsonl.</div>
      </div>
      <div class="actions">
        <button id="copyUrlBtn">Copy URL</button>
        <button id="exportBtn" class="primary">Export Visible JSON</button>
      </div>
    </div>
    <section class="metrics" id="metrics"></section>
    <section class="toolbar">
      <input id="searchInput" type="search" placeholder="Search drugs, title, interaction, management">
      <select id="severityFilter"></select>
      <select id="mechanismFilter"></select>
      <select id="sourceFilter"></select>
      <select id="sortSelect">
        <option value="index">Original order</option>
        <option value="severity">Severity</option>
        <option value="drug">Drug A-Z</option>
      </select>
    </section>
  </header>
  <main>
    <section class="panel list" id="list"></section>
    <section class="panel detail" id="detail"></section>
  </main>

  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    const rows = payload.rows;
    const state = {{ query: '', severity: 'all', mechanism: 'all', source: 'all', sort: 'index', selected: 0 }};

    const els = {{
      metrics: document.getElementById('metrics'),
      list: document.getElementById('list'),
      detail: document.getElementById('detail'),
      search: document.getElementById('searchInput'),
      severity: document.getElementById('severityFilter'),
      mechanism: document.getElementById('mechanismFilter'),
      source: document.getElementById('sourceFilter'),
      sort: document.getElementById('sortSelect'),
      export: document.getElementById('exportBtn'),
      copyUrl: document.getElementById('copyUrlBtn')
    }};

    function byCount(values) {{
      return Object.entries(values).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    }}

    function optionList(select, label, values) {{
      select.innerHTML = `<option value="all">${{label}}</option>` +
        byCount(values).map(([name, count]) => `<option value="${{escapeAttr(name)}}">${{escapeHtml(name)}} (${{count}})</option>`).join('');
    }}

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, char => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[char]));
    }}

    function escapeAttr(value) {{
      return escapeHtml(value).replace(/`/g, '&#96;');
    }}

    function severityClass(value) {{
      return String(value || '').toLowerCase() === 'major' ? 'major' : 'moderate';
    }}

    function pct(count, total) {{
      return total ? Math.round((count / total) * 100) : 0;
    }}

    function filteredRows() {{
      const q = state.query.trim().toLowerCase();
      let out = rows.filter(row => {{
        if (state.severity !== 'all' && row.severity !== state.severity) return false;
        if (state.mechanism !== 'all' && row.mechanism_type !== state.mechanism) return false;
        if (state.source !== 'all' && row._source_file !== state.source) return false;
        if (!q) return true;
        const haystack = [
          row.title, row.drug_a, row.drug_b, row.severity, row.mechanism_type,
          row.interaction, row.management, row.raw_id_field, row.url
        ].join(' ').toLowerCase();
        return haystack.includes(q);
      }});

      if (state.sort === 'severity') {{
        out = out.slice().sort((a, b) => String(a.severity).localeCompare(String(b.severity)) || a._index - b._index);
      }} else if (state.sort === 'drug') {{
        out = out.slice().sort((a, b) => String(a.drug_a).localeCompare(String(b.drug_a)) || String(a.drug_b).localeCompare(String(b.drug_b)));
      }}
      return out;
    }}

    function renderMetrics(visible) {{
      const total = rows.length;
      const major = payload.summary.severity_counts.Major || 0;
      const moderate = payload.summary.severity_counts.Moderate || 0;
      els.metrics.innerHTML = [
        ['Combined records', total],
        ['Unique URLs', payload.summary.unique_urls],
        ['Major', `${{major}} / ${{pct(major, total)}}%`],
        ['Moderate', `${{moderate}} / ${{pct(moderate, total)}}%`],
        ['Visible', visible.length]
      ].map(([label, value]) => `<div class="metric"><span>${{label}}</span><strong>${{value}}</strong></div>`).join('');
    }}

    function renderList(visible) {{
      if (!visible.length) {{
        els.list.innerHTML = '<div class="empty">No matching records.</div>';
        els.detail.innerHTML = '<div class="empty">Adjust the filters to select a record.</div>';
        return;
      }}
      if (!visible.some(row => row._index === state.selected)) state.selected = visible[0]._index;
      els.list.innerHTML = visible.map(row => `
        <button class="row ${{row._index === state.selected ? 'active' : ''}}" data-index="${{row._index}}">
          <div class="row-title">
            <span>${{escapeHtml(row.drug_a)}} + ${{escapeHtml(row.drug_b)}}</span>
            <span>#${{row._index}}</span>
          </div>
          <div class="row-sub">${{escapeHtml(row.title)}}</div>
          <div class="chips">
            <span class="chip ${{severityClass(row.severity)}}">${{escapeHtml(row.severity)}}</span>
            <span class="chip">${{escapeHtml(row.mechanism_type || 'Unknown')}}</span>
            <span class="chip">${{escapeHtml(row._source_file)}}</span>
          </div>
        </button>
      `).join('');
      els.list.querySelectorAll('.row').forEach(button => {{
        button.addEventListener('click', () => {{
          state.selected = Number(button.dataset.index);
          render();
        }});
      }});
    }}

    function renderBars() {{
      const severityEntries = byCount(payload.summary.severity_counts);
      const mechanismEntries = byCount(payload.summary.mechanism_counts);
      const severityMax = Math.max(...severityEntries.map(([, count]) => count), 1);
      const mechanismMax = Math.max(...mechanismEntries.map(([, count]) => count), 1);
      const bar = ([name, count], max, alt=false) => `
        <div class="bar-row">
          <span>${{escapeHtml(name)}}</span>
          <div class="bar ${{alt ? 'alt' : ''}}"><i style="width:${{pct(count, max)}}%"></i></div>
          <strong>${{count}}</strong>
        </div>`;
      return `
        <div class="bars">
          <div>${{severityEntries.map(entry => bar(entry, severityMax)).join('')}}</div>
          <div>${{mechanismEntries.map(entry => bar(entry, mechanismMax, true)).join('')}}</div>
        </div>`;
    }}

    function renderDetail() {{
      const row = rows.find(item => item._index === state.selected);
      if (!row) return;
      els.detail.innerHTML = `
        <div class="detail-head">
          <div>
            <h2>${{escapeHtml(row.title)}}</h2>
            <div class="pair">${{escapeHtml(row.drug_a)}} + ${{escapeHtml(row.drug_b)}}</div>
            <div class="chips">
              <span class="chip ${{severityClass(row.severity)}}">${{escapeHtml(row.severity)}}</span>
              <span class="chip">${{escapeHtml(row.mechanism_type || 'Unknown')}}</span>
              <span class="chip">${{escapeHtml(row._source_file)}}</span>
            </div>
          </div>
          <a href="${{escapeAttr(row.url)}}" target="_blank" rel="noreferrer">Open DDInter</a>
        </div>
        <div class="section">
          <div class="grid">
            <div class="field"><span>Interaction ID</span><strong>${{escapeHtml(row.interaction_id)}}</strong></div>
            <div class="field"><span>Raw IDs</span><strong>${{escapeHtml(row.raw_id_field || 'Empty')}}</strong></div>
            <div class="field"><span>Source file</span><strong>${{escapeHtml(row._source_file)}}</strong></div>
            <div class="field"><span>URL dedupe key</span><strong>${{escapeHtml(row.url)}}</strong></div>
          </div>
        </div>
        <div class="section">
          <h3>Interaction</h3>
          <p class="text">${{escapeHtml(row.interaction || 'Empty')}}</p>
        </div>
        <div class="section">
          <h3>Management</h3>
          <p class="text">${{escapeHtml(row.management || 'Empty')}}</p>
        </div>
        <div class="section">
          <h3>Dataset Shape</h3>
          ${{renderBars()}}
        </div>
      `;
    }}

    function render() {{
      const visible = filteredRows();
      renderMetrics(visible);
      renderList(visible);
      renderDetail();
    }}

    function download(filename, text) {{
      const blob = new Blob([text], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    }}

    optionList(els.severity, 'All severities', payload.summary.severity_counts);
    optionList(els.mechanism, 'All mechanisms', payload.summary.mechanism_counts);
    optionList(els.source, 'All sources', payload.summary.source_counts);

    els.search.addEventListener('input', event => {{ state.query = event.target.value; render(); }});
    els.severity.addEventListener('change', event => {{ state.severity = event.target.value; render(); }});
    els.mechanism.addEventListener('change', event => {{ state.mechanism = event.target.value; render(); }});
    els.source.addEventListener('change', event => {{ state.source = event.target.value; render(); }});
    els.sort.addEventListener('change', event => {{ state.sort = event.target.value; render(); }});
    els.export.addEventListener('click', () => download('visible_ddinter_records.json', JSON.stringify(filteredRows(), null, 2)));
    els.copyUrl.addEventListener('click', async () => {{
      const row = rows.find(item => item._index === state.selected);
      if (row?.url && navigator.clipboard) await navigator.clipboard.writeText(row.url);
    }});

    state.selected = rows[0]?._index || 0;
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    global INPUT_FILES, COMBINED_FILE
    parser = argparse.ArgumentParser(description="Build a source-provenance review page.")
    parser.add_argument("--sources", nargs="+", default=INPUT_FILES)
    parser.add_argument("--combined", default=COMBINED_FILE)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_HTML))
    args = parser.parse_args()
    INPUT_FILES, COMBINED_FILE = args.sources, args.combined
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote {args.output} with {payload['summary']['records']} records.")


if __name__ == "__main__":
    main()
