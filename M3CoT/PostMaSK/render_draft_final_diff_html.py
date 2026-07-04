#!/usr/bin/env python
import argparse
import html
import json
import re
from pathlib import Path


DEFAULT_RUNS = [
    (
        "visual_select",
        "Visual Select",
        "M3CoT/PostMaSK/outputs/postmask_visualgain_vcdnoise500_fixed32_refill2_seed42_n999999/records.jsonl",
    ),
    (
        "proposal_select",
        "Proposal Select",
        "M3CoT/PostVRG/outputs/main_proposal_postmask_seed42_n999999/records.jsonl",
    ),
    (
        "visual_vrg",
        "Visual Select + VRG/VCD Refill",
        "M3CoT/PostMaSK/outputs/postmask_visualgain_vcdrefill_k4_alpha0p5_noise500_fixed32_refill2_seed42_n999999/records.jsonl",
    ),
    (
        "proposal_vrg",
        "Proposal Select + VRG/VCD Refill",
        "M3CoT/PostVRG/outputs/main_postvrg_alpha0p5_noise500_fixed32_refill2_seed42_n999999/records.jsonl",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Render a standalone HTML for draft-vs-final text changes.")
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Run spec: key,label,records.jsonl. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("M3CoT/PostMaSK/outputs/draft_final_diff_full.html"),
    )
    parser.add_argument(
        "--only-improved",
        action="store_true",
        help="Keep only samples where draft is wrong and final is correct.",
    )
    return parser.parse_args()


def parse_runs(raw_runs):
    if not raw_runs:
        return DEFAULT_RUNS
    runs = []
    for raw in raw_runs:
        parts = raw.split(",", 2)
        if len(parts) != 3:
            raise ValueError(f"--run must be key,label,path, got: {raw}")
        runs.append((parts[0], parts[1], parts[2]))
    return runs


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def change_type(row):
    draft = bool(row.get("draft_correct"))
    final = bool(row.get("final_correct"))
    if not draft and final:
        return "improved"
    if draft and not final:
        return "worsened"
    if draft and final:
        return "both_correct"
    return "both_wrong"


def summarize_diff(draft_text, final_text):
    import difflib

    draft_tokens = normalize_text(draft_text).split()
    final_tokens = normalize_text(final_text).split()
    matcher = difflib.SequenceMatcher(a=draft_tokens, b=final_tokens, autojunk=False)
    deleted = 0
    inserted = 0
    replaced = 0
    equal = 0
    spans = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            equal += i2 - i1
        elif tag == "delete":
            deleted += i2 - i1
            spans.append({"type": "delete", "draft": draft_tokens[i1:i2], "final": []})
        elif tag == "insert":
            inserted += j2 - j1
            spans.append({"type": "insert", "draft": [], "final": final_tokens[j1:j2]})
        elif tag == "replace":
            replaced += max(i2 - i1, j2 - j1)
            spans.append({"type": "replace", "draft": draft_tokens[i1:i2], "final": final_tokens[j1:j2]})
    total = max(len(draft_tokens), len(final_tokens), 1)
    return {
        "draft_len": len(draft_tokens),
        "final_len": len(final_tokens),
        "equal": equal,
        "deleted": deleted,
        "inserted": inserted,
        "replaced": replaced,
        "changed_tokens": deleted + inserted + replaced,
        "change_ratio": (deleted + inserted + replaced) / total,
        "num_spans": len(spans),
        "spans_preview": spans[:30],
    }


def build_payload(runs, only_improved=False):
    run_payloads = []
    for key, label, path in runs:
        rows = load_jsonl(path)
        cases = []
        counts = {"improved": 0, "worsened": 0, "both_correct": 0, "both_wrong": 0, "text_changed": 0}
        changed_ratio_sum = 0.0
        for row in rows:
            draft_text = row.get("draft_text")
            final_text = row.get("final_text")
            if draft_text is None or final_text is None:
                continue
            diff = summarize_diff(draft_text, final_text)
            ctype = change_type(row)
            if only_improved and ctype != "improved":
                continue
            text_changed = normalize_text(draft_text) != normalize_text(final_text)
            counts[ctype] += 1
            counts["text_changed"] += int(text_changed)
            changed_ratio_sum += diff["change_ratio"]
            cases.append(
                {
                    "id": row.get("id"),
                    "dataset_index": row.get("dataset_index"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "topic": row.get("topic"),
                    "domain": row.get("domain"),
                    "draft_correct": bool(row.get("draft_correct")),
                    "final_correct": bool(row.get("final_correct")),
                    "change_type": ctype,
                    "text_changed": text_changed,
                    "draft_text": normalize_text(draft_text),
                    "final_text": normalize_text(final_text),
                    "diff": diff,
                }
            )
        count = len(cases)
        run_payloads.append(
            {
                "key": key,
                "label": label,
                "records_path": path,
                "num_cases": count,
                "counts": counts,
                "text_changed_rate": counts["text_changed"] / count if count else None,
                "mean_change_ratio": changed_ratio_sum / count if count else None,
                "cases": cases,
            }
        )
    return {"runs": run_payloads, "only_improved": bool(only_improved)}


def render_html(payload):
    data_json = json.dumps(payload, ensure_ascii=False)
    data_json = data_json.replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Draft vs Final Text Diff</title>
<style>
:root{{--bg:#f7f8fa;--panel:#fff;--ink:#17191f;--muted:#667085;--line:#d8dde6;--green:#13795b;--red:#bd3f36;--blue:#245fc7;--amber:#a76b16;--soft:#f1f4f8}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}}
.shell{{width:min(1500px,calc(100vw - 32px));margin:24px auto 48px}}
header{{border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:14px}}
h1{{margin:0 0 8px;font-size:28px;letter-spacing:0}} h2{{margin:0 0 10px;font-size:18px}} h3{{font-size:15px;margin:0 0 8px}}
p{{margin:0;color:var(--muted)}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}} th{{background:var(--soft);color:#344054}}
.panel,.case{{background:var(--panel);border:1px solid var(--line);border-radius:8px}} .panel{{padding:12px;margin-bottom:12px}}
.metrics{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:14px 0}} .metric{{background:white;border:1px solid var(--line);border-radius:8px;padding:12px}} .metric .label{{color:var(--muted);font-size:12px}} .metric .value{{font-size:23px;font-weight:760;margin-top:4px}}
.grid{{display:grid;grid-template-columns:420px 1fr;gap:14px;align-items:start}} .twocol{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.controls{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}} select{{width:100%;border:1px solid var(--line);border-radius:6px;padding:8px 9px;background:white;color:var(--ink);font:inherit}}
.list{{display:grid;gap:8px;max-height:calc(100vh - 280px);overflow:auto;padding-right:4px}} .case{{padding:10px;cursor:pointer}} .case.active{{outline:2px solid var(--blue)}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:7px}} .tag{{display:inline-flex;align-items:center;min-height:22px;padding:2px 7px;border:1px solid var(--line);border-radius:5px;background:#fafbfc;color:var(--muted);font-size:12px;font-weight:650}}
.improved{{color:white;background:var(--green);border-color:var(--green)}} .worsened{{color:white;background:var(--red);border-color:var(--red)}} .both_correct{{color:white;background:var(--blue);border-color:var(--blue)}} .both_wrong{{color:white;background:var(--amber);border-color:var(--amber)}}
.textblock{{white-space:pre-wrap;word-break:break-word;background:#fbfcfe;border:1px solid var(--line);border-radius:6px;padding:10px;max-height:260px;overflow:auto;font-size:13px}}
.diff{{white-space:normal;word-break:break-word;line-height:1.9}} .tok{{display:inline;padding:2px 3px;border-radius:4px;margin-right:2px}} .eq{{color:#344054}} .del{{background:#ffe4e0;color:#8f241d;text-decoration:line-through}} .ins{{background:#dbf7e8;color:#0b684b;font-weight:650}} .rep-old{{background:#fff1cc;color:#7a4b00;text-decoration:line-through}} .rep-new{{background:#dbeafe;color:#184aa5;font-weight:650}}
.muted{{color:var(--muted)}} code{{background:var(--soft);border:1px solid var(--line);border-radius:5px;padding:1px 5px}}
@media(max-width:1000px){{.grid,.twocol,.metrics{{grid-template-columns:1fr}}.list{{max-height:420px}}}}
</style>
</head>
<body><div class="shell">
<header><h1>Draft vs Final Text Diff</h1><p>对比每个实验中 draft_text 和 final_text 的变化。红色表示 draft 中被删除，绿色/蓝色表示 final 中新增或替换后的文本。</p></header>
<div id="app"></div>
</div>
<script id="payload" type="application/json">{data_json}</script>
<script>
const payload = JSON.parse(document.getElementById('payload').textContent);
let state = {{run: payload.runs[0]?.key || '', outcome: payload.only_improved ? 'improved' : 'all', index: 0}};
function esc(x){{return String(x??'').replace(/[&<>"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]));}}
function pct(x){{return x===null||x===undefined?'n/a':(100*Number(x)).toFixed(1)+'%';}}
function run(){{return payload.runs.find(r=>r.key===state.run)||payload.runs[0];}}
function filtered(){{
 const r=run(); if(!r) return [];
 return r.cases.filter(c=>state.outcome==='all'||c.change_type===state.outcome);
}}
function tokenized(text){{return String(text||'').split(/\\s+/).filter(Boolean);}}
function diffHtml(draft, finalText){{
 const a=tokenized(draft), b=tokenized(finalText);
 const n=a.length, m=b.length;
 const dp=Array.from({{length:n+1}},()=>Array(m+1).fill(0));
 for(let i=n-1;i>=0;i--) for(let j=m-1;j>=0;j--) dp[i][j]=a[i]===b[j]?dp[i+1][j+1]+1:Math.max(dp[i+1][j],dp[i][j+1]);
 let i=0,j=0,out=[];
 while(i<n&&j<m){{
   if(a[i]===b[j]){{out.push(`<span class="tok eq">${{esc(a[i])}}</span>`);i++;j++;}}
   else if(dp[i+1][j]>=dp[i][j+1]){{out.push(`<span class="tok del">${{esc(a[i])}}</span>`);i++;}}
   else{{out.push(`<span class="tok ins">${{esc(b[j])}}</span>`);j++;}}
 }}
 while(i<n){{out.push(`<span class="tok del">${{esc(a[i++])}}</span>`);}}
 while(j<m){{out.push(`<span class="tok ins">${{esc(b[j++])}}</span>`);}}
 return out.join(' ');
}}
function metrics(r){{
 const c=r.counts||{{}};
 return `<div class="metrics">
 <div class="metric"><div class="label">samples</div><div class="value">${{r.num_cases}}</div></div>
 <div class="metric"><div class="label">text changed</div><div class="value">${{pct(r.text_changed_rate)}}</div></div>
 <div class="metric"><div class="label">mean change ratio</div><div class="value">${{pct(r.mean_change_ratio)}}</div></div>
 <div class="metric"><div class="label">improved / worsened</div><div class="value">${{c.improved||0}} / ${{c.worsened||0}}</div></div>
 <div class="metric"><div class="label">both correct / wrong</div><div class="value">${{c.both_correct||0}} / ${{c.both_wrong||0}}</div></div>
 </div>`;
}}
function list(items){{
 return `<div class="list">${{items.map((c,i)=>`<div class="case ${{i===state.index?'active':''}}" onclick="state.index=${{i}};render()"><div class="tags"><span class="tag ${{c.change_type}}">${{esc(c.change_type)}}</span><span class="tag">#${{esc(c.dataset_index)}}</span><span class="tag">chg ${{pct(c.diff.change_ratio)}}</span></div><div>${{esc((c.question||'').slice(0,180))}}</div></div>`).join('')}}</div>`;
}}
function detail(c){{
 if(!c)return'<section class="panel">No samples.</section>';
 return `<section class="panel"><h2>#${{esc(c.dataset_index)}} · ${{esc(c.id)}} · ${{esc(c.change_type)}}</h2><p>${{esc(c.question)}}</p><p class="muted">answer=${{esc(c.answer)}} · topic=${{esc(c.topic)}} · domain=${{esc(c.domain)}} · draft=${{c.draft_correct?'correct':'wrong'}} · final=${{c.final_correct?'correct':'wrong'}}</p></section>
 <div class="twocol"><section class="panel"><h3>Draft</h3><div class="textblock">${{esc(c.draft_text)}}</div></section><section class="panel"><h3>Final</h3><div class="textblock">${{esc(c.final_text)}}</div></section></div>
 <section class="panel"><h3>Diff</h3><div class="diff">${{diffHtml(c.draft_text,c.final_text)}}</div></section>`;
}}
function render(){{
 const r=run(); const items=filtered(); if(state.index>=items.length)state.index=0;
 document.getElementById('app').innerHTML=`<section class="panel"><div class="controls"><select onchange="state.run=this.value;state.index=0;render()">${{payload.runs.map(r=>`<option value="${{esc(r.key)}}" ${{r.key===state.run?'selected':''}}>${{esc(r.label)}}</option>`).join('')}}</select>${{payload.only_improved ? '<select disabled><option>improved only</option></select>' : `<select onchange="state.outcome=this.value;state.index=0;render()"><option value="all">all outcomes</option><option value="improved" ${{state.outcome==='improved'?'selected':''}}>improved</option><option value="worsened" ${{state.outcome==='worsened'?'selected':''}}>worsened</option><option value="both_correct" ${{state.outcome==='both_correct'?'selected':''}}>both correct</option><option value="both_wrong" ${{state.outcome==='both_wrong'?'selected':''}}>both wrong</option></select>`}}</div><p class="muted">records: <code>${{esc(r.records_path)}}</code></p></section>${{metrics(r)}}<div class="grid"><section class="panel"><h2>Samples</h2>${{list(items)}}</section><section>${{detail(items[state.index])}}</section></div>`;
}}
render();
</script>
</body></html>"""


def main():
    args = parse_args()
    payload = build_payload(parse_runs(args.run), only_improved=args.only_improved)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote HTML: {args.output_html}")
    for run in payload["runs"]:
        print(
            f"{run['key']}: n={run['num_cases']} text_changed={run['counts']['text_changed']} "
            f"improved={run['counts']['improved']} worsened={run['counts']['worsened']}"
        )


if __name__ == "__main__":
    main()
