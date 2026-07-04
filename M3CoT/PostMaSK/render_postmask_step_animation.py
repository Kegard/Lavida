import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(
    "M3CoT/PostMaSK/outputs/postmask_sr0p5_d16_p16_conf_r4_seed42_n400"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render an animated HTML report for dynamic PostMaSK remask steps."
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


def jaccard(left, right):
    left = set(left)
    right = set(right)
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def short_text(text, limit=220):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def step_frames(record):
    frames = []
    previous = []
    for item in record.get("postmask_records") or []:
        selected = [int(pos) for pos in item.get("remasked_answer_positions") or []]
        refilled = [int(pos) for pos in item.get("refilled_answer_positions") or []]
        frames.append(
            {
                "step": int(item.get("step", 0)),
                "selected": selected,
                "refilled": refilled,
                "same_as_previous": bool(frames and set(selected) == set(previous)),
                "overlap_previous": len(set(selected) & set(previous)) if frames else None,
                "jaccard_previous": jaccard(selected, previous) if frames else None,
                "state_text": item.get("state_text", ""),
            }
        )
        previous = selected
    return frames


def record_step_stats(frames, max_new_tokens):
    selected_sets = [frozenset(frame["selected"]) for frame in frames]
    counts = Counter(pos for frame in frames for pos in frame["selected"])
    repeated_steps = sum(
        1
        for idx in range(1, len(selected_sets))
        if selected_sets[idx] == selected_sets[idx - 1]
    )
    consecutive_jaccards = [
        jaccard(frames[idx]["selected"], frames[idx - 1]["selected"])
        for idx in range(1, len(frames))
    ]
    most_selected = [
        {"position": int(pos), "count": int(count)}
        for pos, count in counts.most_common()
    ]
    ever_selected = len(counts)
    never_selected = max(int(max_new_tokens) - ever_selected, 0)
    return {
        "unique_step_sets": len(set(selected_sets)),
        "repeated_consecutive_steps": repeated_steps,
        "mean_consecutive_jaccard": (
            sum(consecutive_jaccards) / len(consecutive_jaccards)
            if consecutive_jaccards
            else None
        ),
        "ever_selected": ever_selected,
        "never_selected": never_selected,
        "most_selected": most_selected[:10],
        "position_counts": {str(int(pos)): int(count) for pos, count in counts.items()},
    }


def compact_record(record):
    meta = record.get("meta") or {}
    max_new_tokens = int(meta.get("max_new_tokens") or 64)
    frames = step_frames(record)
    stats = record_step_stats(frames, max_new_tokens)
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
        "frames": frames,
        "step_stats": stats,
        "max_new_tokens": max_new_tokens,
        "meta": meta,
    }


def build_global_stats(records):
    transitions = Counter(transition(record) for record in records)
    topics = Counter(record.get("topic") or "unknown" for record in records)
    remask_set_sizes = Counter()
    repeat_counter = Counter()
    all_jaccards = []
    position_counts = Counter()
    for record in records:
        frames = step_frames(record)
        for frame in frames:
            remask_set_sizes[len(frame["selected"])] += 1
            position_counts.update(frame["selected"])
        for idx in range(1, len(frames)):
            current = set(frames[idx]["selected"])
            previous = set(frames[idx - 1]["selected"])
            repeat_counter["same"] += int(current == previous)
            repeat_counter["different"] += int(current != previous)
            all_jaccards.append(jaccard(current, previous))
    return {
        "num_records": len(records),
        "transitions": dict(transitions),
        "topics": dict(topics),
        "remask_set_sizes": dict(remask_set_sizes),
        "consecutive_same_sets": int(repeat_counter["same"]),
        "consecutive_different_sets": int(repeat_counter["different"]),
        "mean_consecutive_jaccard": (
            sum(all_jaccards) / len(all_jaccards) if all_jaccards else None
        ),
        "top_positions": [
            {"position": int(pos), "count": int(count)}
            for pos, count in position_counts.most_common(16)
        ],
    }


def build_data(output_dir, max_records=None):
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    records = load_records(records_path, max_records=max_records)
    summary = load_json(summary_path) if summary_path.exists() else {}
    return {
        "records_path": str(records_path),
        "summary": summary,
        "stats": build_global_stats(records),
        "records": [compact_record(record) for record in records],
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PostMaSK Step Animation</title>
  <style>
    :root {
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #18212c;
      --muted: #657184;
      --line: #d9dee7;
      --green: #1f8a63;
      --red: #c84d3f;
      --blue: #3368c4;
      --amber: #b6781e;
      --purple: #7a58b5;
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
    .shell { width: min(1500px, calc(100vw - 32px)); margin: 24px auto 48px; }
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
    .metric .value { font-size: 25px; font-weight: 760; margin-top: 4px; font-variant-numeric: tabular-nums; }
    .layout { display: grid; grid-template-columns: 390px 1fr; gap: 14px; align-items: start; }
    .panel { padding: 12px; }
    .controls { display: grid; gap: 8px; margin-bottom: 10px; }
    input, select, button {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    button { cursor: pointer; font-weight: 700; }
    button.primary { background: var(--ink); color: white; border-color: var(--ink); }
    input, select { width: 100%; }
    input[type="range"] { padding: 0; }
    .list { display: grid; gap: 8px; max-height: calc(100vh - 235px); overflow: auto; padding-right: 4px; }
    .sample { padding: 10px; cursor: pointer; }
    .sample.active { outline: 2px solid var(--blue); }
    .sample-title, .row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
    .sample-title { margin-bottom: 6px; }
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
      max-height: 230px;
      overflow: auto;
      font-size: 13px;
    }
    .animation { display: grid; grid-template-columns: 430px 1fr; gap: 14px; align-items: start; }
    .board {
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      gap: 6px;
      width: 100%;
    }
    .cell {
      aspect-ratio: 1;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f3f5f8;
      display: grid;
      place-items: center;
      color: #566275;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      position: relative;
    }
    .cell.ever { background: #eaf0fb; border-color: #c7d6f4; }
    .cell.current { background: var(--blue); color: white; border-color: var(--blue); transform: scale(1.04); }
    .cell.previous { box-shadow: inset 0 0 0 2px var(--amber); }
    .cell.refilled::after {
      content: "";
      position: absolute;
      width: 7px;
      height: 7px;
      right: 4px;
      top: 4px;
      border-radius: 50%;
      background: var(--green);
    }
    .timeline {
      display: grid;
      grid-template-columns: repeat(16, minmax(0, 1fr));
      gap: 5px;
      margin-top: 10px;
    }
    .tick {
      height: 28px;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: #f3f5f8;
      color: var(--muted);
      font-size: 12px;
      display: grid;
      place-items: center;
      cursor: pointer;
    }
    .tick.active { background: var(--ink); color: white; border-color: var(--ink); }
    .tick.same { border-color: var(--purple); }
    .playbar { display: grid; grid-template-columns: auto 1fr auto; gap: 8px; align-items: center; margin: 10px 0; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 6px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 700; background: #fbfcfe; position: sticky; top: 0; }
    .tablewrap { max-height: 290px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; }
    mark { background: #ffe8a3; border-radius: 3px; padding: 0 2px; }
    .empty { color: var(--muted); padding: 18px; text-align: center; }
    @media (max-width: 1120px) {
      .layout, .twocol, .animation, .metrics { grid-template-columns: 1fr; }
      .list { max-height: 420px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>PostMaSK Step Animation</h1>
      <p>实验目录：<code id="recordsPath"></code>。动画逐步高亮每个 PostMask step 选择的 answer token positions，用来观察不同 step 是否选中了同一批 token。</p>
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
    let frameIndex = 0;
    let timer = null;
    const esc = (s) => String(s ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
    const pct = (x) => x == null ? 'NA' : `${(x * 100).toFixed(1)}%`;
    const transitionText = { improved: 'improved', worsened: 'worsened', kept_correct: 'kept correct', kept_wrong: 'kept wrong' };
    function maskText(s) {
      return esc(s).replaceAll('&lt;|mdm_mask|&gt;', '<mark>[MASK]</mark>');
    }
    function stop() {
      if (timer) clearInterval(timer);
      timer = null;
      const btn = document.getElementById('playButton');
      if (btn) btn.textContent = 'Play';
    }
    function currentRecord() {
      return filtered[selectedIndex];
    }
    function renderMetrics() {
      const s = DATA.stats;
      const sum = DATA.summary || {};
      const same = s.consecutive_same_sets || 0;
      const diff = s.consecutive_different_sets || 0;
      const items = [
        ['records', s.num_records],
        ['draft acc', sum.draft_accuracy == null ? 'NA' : (sum.draft_accuracy * 100).toFixed(2) + '%'],
        ['final acc', sum.final_accuracy == null ? 'NA' : (sum.final_accuracy * 100).toFixed(2) + '%'],
        ['improved / worsened', `${s.transitions.improved || 0} / ${s.transitions.worsened || 0}`],
        ['same adjacent sets', `${same} / ${same + diff}`],
        ['mean adjacent Jaccard', pct(s.mean_consecutive_jaccard)]
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
      frameIndex = 0;
      stop();
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
            <span class="tag">unique ${r.step_stats.unique_step_sets}/${r.frames.length}</span>
            <span class="tag">J ${pct(r.step_stats.mean_consecutive_jaccard)}</span>
          </div>
          <div class="preview">${esc(r.id)} · ${esc(r.draft_preview)}</div>
        </article>
      `).join('');
    }
    function renderBoard(r, frame) {
      const current = new Set(frame.selected);
      const previous = new Set(frameIndex > 0 ? r.frames[frameIndex - 1].selected : []);
      const refilled = new Set(frame.refilled);
      const ever = new Set(r.frames.flatMap(f => f.selected));
      let html = '<div class="board">';
      for (let pos = 0; pos < r.max_new_tokens; pos++) {
        const classes = ['cell'];
        if (ever.has(pos)) classes.push('ever');
        if (previous.has(pos)) classes.push('previous');
        if (current.has(pos)) classes.push('current');
        if (refilled.has(pos)) classes.push('refilled');
        html += `<div class="${classes.join(' ')}" title="position ${pos}">${pos}</div>`;
      }
      return html + '</div>';
    }
    function renderTimeline(r) {
      return `<div class="timeline">${r.frames.map((f, i) => `
        <button class="tick ${i === frameIndex ? 'active' : ''} ${f.same_as_previous ? 'same' : ''}" data-frame="${i}">${f.step}</button>
      `).join('')}</div>`;
    }
    function renderFrequencyTable(r) {
      const rows = Object.entries(r.step_stats.position_counts)
        .map(([pos, count]) => [Number(pos), count])
        .sort((a, b) => b[1] - a[1] || a[0] - b[0]);
      return `<div class="tablewrap"><table>
        <thead><tr><th>position</th><th>selected count</th><th>steps</th></tr></thead>
        <tbody>${rows.map(([pos, count]) => {
          const steps = r.frames.filter(f => f.selected.includes(pos)).map(f => f.step).join(', ');
          return `<tr><td>${pos}</td><td>${count}</td><td>${steps}</td></tr>`;
        }).join('')}</tbody>
      </table></div>`;
    }
    function renderFrameOnly() {
      const r = currentRecord();
      if (!r) return;
      const frame = r.frames[frameIndex] || r.frames[0];
      document.getElementById('frameTitle').innerHTML = `
        <span class="tag">step ${frame.step}</span>
        <span class="tag">selected: ${frame.selected.join(', ')}</span>
        <span class="tag">refilled: ${frame.refilled.join(', ')}</span>
        <span class="tag">overlap prev: ${frame.overlap_previous ?? 'NA'}</span>
        <span class="tag">Jaccard prev: ${pct(frame.jaccard_previous)}</span>
      `;
      document.getElementById('boardWrap').innerHTML = renderBoard(r, frame);
      document.getElementById('timelineWrap').innerHTML = renderTimeline(r);
      document.getElementById('stepRange').value = frameIndex;
      document.getElementById('stateText').innerHTML = maskText(frame.state_text);
    }
    function renderDetail() {
      const root = document.getElementById('detail');
      const r = currentRecord();
      if (!r) {
        root.innerHTML = '<div class="panel empty">请选择样本</div>';
        return;
      }
      const frame = r.frames[frameIndex] || r.frames[0];
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
          <div class="row">
            <span class="tag">unique sets ${r.step_stats.unique_step_sets}/${r.frames.length}</span>
            <span class="tag">same adjacent ${r.step_stats.repeated_consecutive_steps}</span>
            <span class="tag">mean adjacent Jaccard ${pct(r.step_stats.mean_consecutive_jaccard)}</span>
            <span class="tag">ever selected ${r.step_stats.ever_selected}/${r.max_new_tokens}</span>
          </div>
        </section>
        <section class="panel animation">
          <div>
            <h2>Step 动画</h2>
            <div class="playbar">
              <button class="primary" id="playButton">Play</button>
              <input id="stepRange" type="range" min="0" max="${Math.max(r.frames.length - 1, 0)}" value="${frameIndex}">
              <button id="nextButton">Next</button>
            </div>
            <div class="row" id="frameTitle"></div>
            <div id="boardWrap"></div>
            <div id="timelineWrap"></div>
            <p style="margin-top:8px">蓝色是当前 step 选择的位置，黄色描边是上一 step 选择的位置，绿色点表示本步 refill 的位置。</p>
          </div>
          <div>
            <h2>当前 step state_text</h2>
            <div class="textblock" id="stateText">${maskText(frame.state_text)}</div>
          </div>
        </section>
        <section class="twocol">
          <div class="panel"><h2>Draft Text</h2><div class="textblock">${esc(r.draft_text)}</div></div>
          <div class="panel"><h2>Final Text</h2><div class="textblock">${esc(r.final_text)}</div></div>
        </section>
        <section class="twocol">
          <div class="panel"><h2>Question</h2><div class="textblock">${esc(r.question)}</div></div>
          <div class="panel"><h2>Position 被选频率</h2>${renderFrequencyTable(r)}</div>
        </section>
      `;
      bindAnimation();
      renderFrameOnly();
    }
    function bindAnimation() {
      document.getElementById('playButton').addEventListener('click', () => {
        const r = currentRecord();
        if (!r) return;
        if (timer) {
          stop();
          return;
        }
        document.getElementById('playButton').textContent = 'Pause';
        timer = setInterval(() => {
          frameIndex = (frameIndex + 1) % r.frames.length;
          renderFrameOnly();
        }, 850);
      });
      document.getElementById('nextButton').addEventListener('click', () => {
        const r = currentRecord();
        if (!r) return;
        stop();
        frameIndex = (frameIndex + 1) % r.frames.length;
        renderFrameOnly();
      });
      document.getElementById('stepRange').addEventListener('input', event => {
        stop();
        frameIndex = Number(event.target.value);
        renderFrameOnly();
      });
      document.getElementById('timelineWrap').addEventListener('click', event => {
        const tick = event.target.closest('.tick');
        if (!tick) return;
        stop();
        frameIndex = Number(tick.dataset.frame);
        renderFrameOnly();
      });
    }
    document.getElementById('sampleList').addEventListener('click', event => {
      const item = event.target.closest('.sample');
      if (!item) return;
      selectedIndex = Number(item.dataset.i);
      frameIndex = 0;
      stop();
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
    html_path = args.html or output_dir / "postmask_step_animation.html"
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
