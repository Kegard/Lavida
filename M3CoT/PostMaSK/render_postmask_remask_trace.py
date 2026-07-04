import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(
    "M3CoT/PostMaSK/outputs/postmask_sr0p5_d16_p16_proposalconf_fixed32_refill2_seed42_n400"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a static HTML report for PostMaSK remask/refill traces."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def load_records(path, max_records=None):
    records = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_records is not None and len(records) >= max_records:
                break
    return records


def transition(record):
    if record.get("final_correct") and not record.get("draft_correct"):
        return "improved"
    if record.get("draft_correct") and not record.get("final_correct"):
        return "worsened"
    if record.get("final_correct"):
        return "kept_correct"
    return "kept_wrong"


def short_text(text, limit=220):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def first_postmask(record):
    postmask = record.get("postmask_records") or []
    return postmask[0] if postmask else {}


def build_position_rows(record):
    first = first_postmask(record)
    positions = first.get("remasked_answer_positions") or []
    scores = first.get("selection_scores") or []
    proposal = record.get("proposal_confidence") or []
    rows = []
    refilled_by = {}
    for step in record.get("postmask_records") or []:
        for pos in step.get("refilled_answer_positions") or []:
            refilled_by.setdefault(int(pos), int(step.get("step", 0)))
    for rank, pos in enumerate(positions, start=1):
        pos = int(pos)
        score = scores[rank - 1] if rank - 1 < len(scores) else None
        proposal_score = proposal[pos] if 0 <= pos < len(proposal) else None
        rows.append(
            {
                "rank": rank,
                "position": pos,
                "selection_score": score,
                "proposal_confidence": proposal_score,
                "refilled_at_step": refilled_by.get(pos),
            }
        )
    return rows


def compact_record(record):
    postmask_records = []
    for step in record.get("postmask_records") or []:
        postmask_records.append(
            {
                "step": step.get("step"),
                "remasked_answer_positions": step.get("remasked_answer_positions") or [],
                "refilled_answer_positions": step.get("refilled_answer_positions") or [],
                "selection_scores": step.get("selection_scores"),
                "state_text": step.get("state_text", ""),
            }
        )
    return {
        "dataset_index": record.get("dataset_index"),
        "id": record.get("id"),
        "domain": record.get("domain"),
        "topic": record.get("topic"),
        "answer": record.get("answer"),
        "choices": record.get("choices") or [],
        "question": record.get("question", ""),
        "draft_correct": bool(record.get("draft_correct")),
        "final_correct": bool(record.get("final_correct")),
        "transition": transition(record),
        "draft_text": record.get("draft_text", ""),
        "final_text": record.get("final_text", ""),
        "draft_preview": short_text(record.get("draft_text", "")),
        "final_preview": short_text(record.get("final_text", "")),
        "position_rows": build_position_rows(record),
        "postmask_records": postmask_records,
        "meta": record.get("meta") or {},
    }


def build_data(output_dir, max_records=None):
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    records = load_records(records_path, max_records=max_records)
    summary = load_json(summary_path) if summary_path.exists() else {}
    transitions = Counter(transition(record) for record in records)
    topics = Counter(record.get("topic") or "unknown" for record in records)
    fixed_set_sizes = Counter(
        len(first_postmask(record).get("remasked_answer_positions") or [])
        for record in records
    )
    refill_sizes = Counter(
        len(step.get("refilled_answer_positions") or [])
        for record in records
        for step in (record.get("postmask_records") or [])
    )
    return {
        "records_path": str(records_path),
        "summary": summary,
        "stats": {
            "num_records": len(records),
            "transitions": dict(transitions),
            "topics": dict(topics),
            "fixed_set_sizes": dict(fixed_set_sizes),
            "refill_sizes": dict(refill_sizes),
        },
        "records": [compact_record(record) for record in records],
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PostMaSK Remask Trace</title>
  <style>
    :root {
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --green: #1f8a63;
      --red: #c64b3d;
      --blue: #356ac3;
      --amber: #b7791f;
      --code: #f1f4f8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }
    .shell { width: min(1480px, calc(100vw - 32px)); margin: 24px auto 48px; }
    header { border-bottom: 1px solid var(--line); padding-bottom: 18px; margin-bottom: 18px; }
    h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }
    h2 { margin: 0 0 10px; font-size: 17px; letter-spacing: 0; }
    h3 { margin: 0 0 6px; font-size: 15px; letter-spacing: 0; }
    p { margin: 0; color: var(--muted); }
    code { background: var(--code); border: 1px solid var(--line); border-radius: 5px; padding: 1px 5px; }
    .metrics { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin: 16px 0; }
    .metric, .panel, .sample { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .metric { padding: 12px; }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { font-size: 26px; font-weight: 760; margin-top: 4px; font-variant-numeric: tabular-nums; }
    .layout { display: grid; grid-template-columns: 390px 1fr; gap: 14px; align-items: start; }
    .panel { padding: 12px; }
    .controls { display: grid; gap: 8px; margin-bottom: 10px; }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    .list { display: grid; gap: 8px; max-height: calc(100vh - 235px); overflow: auto; padding-right: 4px; }
    .sample { padding: 10px; cursor: pointer; }
    .sample.active { outline: 2px solid var(--blue); }
    .sample-title { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin-bottom: 6px; }
    .tag { display: inline-flex; align-items: center; min-height: 22px; padding: 2px 7px; border: 1px solid var(--line); border-radius: 5px; background: #fafbfc; color: var(--muted); font-size: 12px; font-weight: 650; }
    .tag.improved { color: white; background: var(--green); border-color: var(--green); }
    .tag.worsened { color: white; background: var(--red); border-color: var(--red); }
    .tag.kept_correct { color: white; background: var(--blue); border-color: var(--blue); }
    .tag.kept_wrong { color: white; background: var(--amber); border-color: var(--amber); }
    .preview { color: var(--muted); font-size: 13px; }
    .detail { display: grid; gap: 14px; }
    .twocol { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .textblock {
      white-space: pre-wrap;
      word-break: break-word;
      background: #fbfcfe;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      max-height: 260px;
      overflow: auto;
      font-size: 13px;
    }
    .state { max-height: 190px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 6px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 700; background: #fbfcfe; position: sticky; top: 0; }
    .tablewrap { max-height: 320px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; }
    .steps { display: grid; gap: 10px; }
    .step { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: white; }
    .step-head { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .poslist { color: var(--muted); font-size: 12px; word-break: break-word; }
    mark { background: #ffe8a3; border-radius: 3px; padding: 0 2px; }
    .empty { color: var(--muted); padding: 18px; text-align: center; }
    @media (max-width: 1050px) {
      .layout, .twocol, .metrics { grid-template-columns: 1fr; }
      .list { max-height: 420px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>PostMaSK Remask Trace</h1>
      <p>实验目录：<code id="recordsPath"></code>。本页聚焦 draft 后被选中的 remask answer positions，以及 PostMask 每一步 refill 后的文本状态。</p>
    </header>
    <section class="metrics" id="metrics"></section>
    <main class="layout">
      <aside class="panel">
        <h2>样本</h2>
        <div class="controls">
          <input id="query" placeholder="搜索 id / index / question / text">
          <select id="transitionFilter">
            <option value="all">all transitions</option>
            <option value="improved">improved: draft wrong -> final right</option>
            <option value="worsened">worsened: draft right -> final wrong</option>
            <option value="kept_correct">kept correct</option>
            <option value="kept_wrong">kept wrong</option>
          </select>
          <select id="topicFilter"></select>
        </div>
        <div class="list" id="sampleList"></div>
      </aside>
      <section class="detail" id="detail"></section>
    </main>
  </div>
  <script>
    const DATA = __DATA__;
    let filtered = [];
    let selectedIndex = 0;
    const esc = (s) => String(s ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
    const fmt = (x) => Number.isFinite(Number(x)) ? Number(x).toFixed(6) : 'NA';
    const transitionText = {
      improved: 'improved',
      worsened: 'worsened',
      kept_correct: 'kept correct',
      kept_wrong: 'kept wrong'
    };
    function maskText(s) {
      return esc(s).replaceAll('&lt;|mdm_mask|&gt;', '<mark>[MASK]</mark>');
    }
    function renderMetrics() {
      const s = DATA.stats;
      const sum = DATA.summary || {};
      const items = [
        ['records', s.num_records],
        ['draft acc', sum.draft_accuracy == null ? 'NA' : (sum.draft_accuracy * 100).toFixed(2) + '%'],
        ['final acc', sum.final_accuracy == null ? 'NA' : (sum.final_accuracy * 100).toFixed(2) + '%'],
        ['improved', s.transitions.improved || 0],
        ['worsened', s.transitions.worsened || 0],
        ['fixed set', Object.keys(s.fixed_set_sizes).map(k => `${k} x ${s.fixed_set_sizes[k]}`).join(', ')]
      ];
      document.getElementById('metrics').innerHTML = items.map(([label, value]) => `
        <div class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>
      `).join('');
      document.getElementById('recordsPath').textContent = DATA.records_path;
    }
    function initTopics() {
      const topics = ['all', ...Object.keys(DATA.stats.topics).sort()];
      document.getElementById('topicFilter').innerHTML = topics.map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join('');
    }
    function applyFilters() {
      const q = document.getElementById('query').value.trim().toLowerCase();
      const tr = document.getElementById('transitionFilter').value;
      const topic = document.getElementById('topicFilter').value;
      filtered = DATA.records.filter(r => {
        if (tr !== 'all' && r.transition !== tr) return false;
        if (topic !== 'all' && r.topic !== topic) return false;
        if (!q) return true;
        const hay = `${r.dataset_index} ${r.id} ${r.question} ${r.draft_text} ${r.final_text}`.toLowerCase();
        return hay.includes(q);
      });
      selectedIndex = Math.min(selectedIndex, Math.max(filtered.length - 1, 0));
      renderList();
      renderDetail();
    }
    function renderList() {
      const root = document.getElementById('sampleList');
      if (!filtered.length) {
        root.innerHTML = '<div class="empty">没有匹配样本</div>';
        return;
      }
      root.innerHTML = filtered.map((r, i) => `
        <article class="sample ${i === selectedIndex ? 'active' : ''}" data-i="${i}">
          <div class="sample-title">
            <span class="tag ${r.transition}">${transitionText[r.transition]}</span>
            <span class="tag">idx ${r.dataset_index}</span>
            <span class="tag">${esc(r.topic)}</span>
            <span class="tag">gold ${esc(r.answer)}</span>
          </div>
          <div class="preview">${esc(r.id)} · ${esc(r.draft_preview)}</div>
        </article>
      `).join('');
    }
    function renderPositionTable(r) {
      if (!r.position_rows.length) return '<p>没有 postmask remask positions。</p>';
      return `<div class="tablewrap"><table>
        <thead><tr><th>rank</th><th>answer position</th><th>selection score</th><th>proposal confidence</th><th>refilled at step</th></tr></thead>
        <tbody>${r.position_rows.map(row => `
          <tr>
            <td>${row.rank}</td>
            <td>${row.position}</td>
            <td>${fmt(row.selection_score)}</td>
            <td>${fmt(row.proposal_confidence)}</td>
            <td>${row.refilled_at_step ?? 'not refilled'}</td>
          </tr>
        `).join('')}</tbody>
      </table></div>`;
    }
    function renderSteps(r) {
      return `<div class="steps">${r.postmask_records.map(step => `
        <article class="step">
          <div class="step-head">
            <strong>Step ${step.step}</strong>
            <span class="tag">refilled: ${(step.refilled_answer_positions || []).join(', ') || 'none'}</span>
          </div>
          <div class="poslist">remask set: ${(step.remasked_answer_positions || []).join(', ')}</div>
          <div class="textblock state">${maskText(step.state_text)}</div>
        </article>
      `).join('')}</div>`;
    }
    function renderDetail() {
      const root = document.getElementById('detail');
      const r = filtered[selectedIndex];
      if (!r) {
        root.innerHTML = '<div class="panel empty">请选择样本</div>';
        return;
      }
      root.innerHTML = `
        <section class="panel">
          <div class="sample-title">
            <span class="tag ${r.transition}">${transitionText[r.transition]}</span>
            <span class="tag">idx ${r.dataset_index}</span>
            <span class="tag">${esc(r.id)}</span>
            <span class="tag">${esc(r.domain)} / ${esc(r.topic)}</span>
            <span class="tag">draft ${r.draft_correct ? 'right' : 'wrong'}</span>
            <span class="tag">final ${r.final_correct ? 'right' : 'wrong'}</span>
            <span class="tag">gold ${esc(r.answer)}</span>
          </div>
          <h3>Question</h3>
          <div class="textblock">${esc(r.question)}</div>
          <h3>Choices</h3>
          <div class="textblock">${r.choices.map((c, i) => `${String.fromCharCode(65 + i)}. ${esc(c)}`).join('\\n')}</div>
        </section>
        <section class="twocol">
          <div class="panel"><h2>Draft Text</h2><div class="textblock">${esc(r.draft_text)}</div></div>
          <div class="panel"><h2>Final Text</h2><div class="textblock">${esc(r.final_text)}</div></div>
        </section>
        <section class="panel">
          <h2>Draft 后选中的 remask token positions</h2>
          <p>当前记录没有保存 tokenizer token 字符串；这里显示精确 answer position、selection score、proposal confidence，以及它在哪个 PostMask step 被 refill。</p>
          ${renderPositionTable(r)}
        </section>
        <section class="panel">
          <h2>每一步 remask/refill 结果</h2>
          ${renderSteps(r)}
        </section>
      `;
    }
    document.getElementById('sampleList').addEventListener('click', event => {
      const item = event.target.closest('.sample');
      if (!item) return;
      selectedIndex = Number(item.dataset.i);
      renderList();
      renderDetail();
    });
    document.getElementById('query').addEventListener('input', applyFilters);
    document.getElementById('transitionFilter').addEventListener('change', applyFilters);
    document.getElementById('topicFilter').addEventListener('change', applyFilters);
    renderMetrics();
    initTopics();
    applyFilters();
  </script>
</body>
</html>
"""


def main():
    args = parse_args()
    output_dir = args.output_dir
    html_path = args.html or output_dir / "postmask_remask_trace.html"
    data = build_data(output_dir, max_records=args.max_records)
    html = HTML_TEMPLATE.replace(
        "__DATA__",
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
