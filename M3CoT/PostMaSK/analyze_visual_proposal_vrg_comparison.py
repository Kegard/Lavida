#!/usr/bin/env python
import argparse
import html
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean


STOPWORDS = set(
    "the a an is are was were be been being of to in on at for from with by and or but if then than as that this these those it its we need determine correct answer therefore because can see based given option options question image diagram figure shown shows using use compare each which what when where how there more less between into left right country year hourly weekly average per capita wage wages hours".split()
)

CAT_ORDER = ["content", "number", "option", "mixednum", "template", "punct", "white", "special", "other"]
CAT_LABEL = {
    "content": "content",
    "number": "number",
    "option": "option",
    "mixednum": "mixed number/unit",
    "template": "template",
    "punct": "punct",
    "white": "white",
    "special": "special",
    "other": "other",
}
CAT_COLORS = {
    "content": "#2563eb",
    "number": "#16a34a",
    "option": "#f97316",
    "mixednum": "#0f766e",
    "template": "#94a3b8",
    "punct": "#64748b",
    "white": "#cbd5e1",
    "special": "#a855f7",
    "other": "#eab308",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare visual select, proposal select, visual+VRG/VCD, and proposal+VRG/VCD runs."
    )
    parser.add_argument(
        "--visual-select-records",
        type=Path,
        default=Path("M3CoT/PostMaSK/outputs/postmask_sr0p5_d16_p16_visualgain_fixed32_refill2_seed42_n400/records.jsonl"),
    )
    parser.add_argument(
        "--proposal-select-records",
        type=Path,
        default=Path("M3CoT/PostMaSK/outputs/postmask_sr0p5_d16_p16_proposalconf_fixed32_refill2_seed42_n400/records.jsonl"),
    )
    parser.add_argument(
        "--visual-vrg-records",
        type=Path,
        default=Path("M3CoT/PostMaSK/outputs/postmask_visualgain_vcdrefill_k4_alpha0p5_noise500_fixed32_refill2_seed42_n999999/records.jsonl"),
    )
    parser.add_argument(
        "--proposal-vrg-records",
        type=Path,
        default=Path("M3CoT/PostVRG/outputs/postvrg_proposalconf_vcdrefill_alpha0p5_noise500_fixed32_refill2_seed42_n400/records.jsonl"),
    )
    parser.add_argument(
        "--baseline-records",
        type=Path,
        default=Path("M3CoT/benchmark/StepRemask/baseline_no_remask_seed42_n400/records.jsonl"),
        help="Optional no-remask records used only to recover draft token text by position.",
    )
    parser.add_argument("--tokenizer", default="weight/lavida-reason")
    parser.add_argument("--max-html-samples", type=int, default=999999)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("M3CoT/PostMaSK/outputs/visual_proposal_vrg_comparison.json"),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("M3CoT/PostMaSK/outputs/visual_proposal_vrg_comparison.html"),
    )
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_summary(records_path):
    path = records_path.with_name("summary.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(values):
    finite = [
        float(value)
        for value in values
        if value is not None and isinstance(value, (int, float, bool)) and math.isfinite(float(value))
    ]
    return mean(finite) if finite else None


def jaccard(left, right):
    return len(left & right) / len(left | right) if left or right else 1.0


def overlap(left, right):
    denom = min(len(left), len(right))
    return len(left & right) / denom if denom else (1.0 if not left and not right else 0.0)


def clean_token(text):
    return str(text or "").replace("\n", "\\n")


def display_token(text):
    text = clean_token(text)
    return text if text.strip() else "·"


def token_category(text):
    stripped = str(text or "").strip()
    lowered = stripped.lower()
    if stripped == "":
        return "white"
    if stripped in {"<|eot_id|>", "<|endoftext|>", "<|mdm_mask|>"}:
        return "special"
    if re.fullmatch(r"[\.,:;!?\-()\[\]{}\\/\"']+", stripped):
        return "punct"
    if re.fullmatch(r"\{?[A-E]\}?", stripped):
        return "option"
    if re.fullmatch(r"[+-]?\d+(\.\d+)?%?", stripped):
        return "number"
    if any(ch.isdigit() for ch in stripped):
        return "mixednum"
    if lowered in STOPWORDS:
        return "template"
    if re.fullmatch(r"[A-Za-z]+", stripped):
        return "content"
    return "other"


def load_tokenizer(tokenizer_path):
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as exc:
        print(f"Warning: tokenizer unavailable at {tokenizer_path}: {exc}")
        return None


def decode_token(tokenizer, token_id):
    if token_id is None:
        return "<missing>"
    if tokenizer is None:
        return f"<id:{int(token_id)}>"
    return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def first_selection_record(row):
    for record in row.get("postmask_records") or []:
        if record.get("remasked_answer_positions"):
            return record
    return None


def build_baseline_tokens(path, tokenizer):
    by_id = {}
    for row in load_jsonl(path):
        token_ids = row.get("final_answer_ids") or row.get("draft_answer_ids") or []
        tokens = {}
        for pos, token_id in enumerate(token_ids):
            text = decode_token(tokenizer, token_id)
            tokens[int(pos)] = {
                "pos": int(pos),
                "id": int(token_id),
                "text": clean_token(text),
                "display": display_token(text),
                "cat": token_category(text),
            }
        by_id[row.get("id")] = tokens
    return by_id


def selected_items(row, baseline_tokens, tokenizer):
    record = first_selection_record(row)
    if record is None:
        return []
    positions = [int(pos) for pos in record.get("remasked_answer_positions") or []]
    token_ids = record.get("remasked_token_ids") or []
    token_texts = record.get("remasked_token_texts") or []
    scores = record.get("selection_scores") or []
    visual_gains = record.get("selected_visual_gains") or []
    items = []
    for idx, pos in enumerate(positions):
        base = baseline_tokens.get(pos)
        token_id = token_ids[idx] if idx < len(token_ids) else (base or {}).get("id")
        if idx < len(token_texts):
            text = token_texts[idx]
        elif base is not None:
            text = base["text"]
        else:
            text = decode_token(tokenizer, token_id)
        items.append(
            {
                "pos": pos,
                "token_id": int(token_id) if token_id is not None else None,
                "token": clean_token(text),
                "display": display_token(text),
                "cat": token_category(text),
                "score": scores[idx] if idx < len(scores) else None,
                "visual_gain": visual_gains[idx] if idx < len(visual_gains) else None,
            }
        )
    return items


def summarize_run(key, label, records_path, tokenizer, baseline_by_id):
    rows = load_jsonl(records_path)
    summary = load_summary(records_path)
    by_id = {}
    all_selected = []
    for row in rows:
        sample_id = row.get("id")
        selected = selected_items(row, baseline_by_id.get(sample_id, {}), tokenizer)
        cats = Counter(item["cat"] for item in selected)
        semantic = cats["content"] + cats["number"] + cats["mixednum"] + cats["option"]
        data = {
            "key": key,
            "id": sample_id,
            "dataset_index": row.get("dataset_index"),
            "question": row.get("question"),
            "choices": row.get("choices") or [],
            "answer": row.get("answer"),
            "domain": row.get("domain"),
            "topic": row.get("topic"),
            "draft_text": row.get("draft_text"),
            "final_text": row.get("final_text"),
            "draft_correct": bool(row.get("draft_correct")),
            "final_correct": bool(row.get("final_correct")),
            "selected_positions": [item["pos"] for item in selected],
            "selected": selected,
            "category_counts": dict(cats),
            "semantic_ratio": semantic / len(selected) if selected else None,
        }
        by_id[sample_id] = data
        all_selected.extend(selected)

    cats = Counter(item["cat"] for item in all_selected)
    semantic_total = cats["content"] + cats["number"] + cats["mixednum"] + cats["option"]
    return {
        "key": key,
        "label": label,
        "records_path": str(records_path),
        "summary_path": str(records_path.with_name("summary.json")),
        "summary": summary,
        "num_records": len(rows),
        "draft_accuracy": safe_mean([row.get("draft_correct", False) for row in rows]),
        "final_accuracy": safe_mean([row.get("final_correct", False) for row in rows]),
        "accuracy_delta": (
            safe_mean([row.get("final_correct", False) for row in rows])
            - safe_mean([row.get("draft_correct", False) for row in rows])
            if rows
            else None
        ),
        "selected_token_count": len(all_selected),
        "category_counts": dict(cats),
        "semantic_ratio": semantic_total / len(all_selected) if all_selected else None,
        "content_ratio": cats["content"] / len(all_selected) if all_selected else None,
        "template_ratio": cats["template"] / len(all_selected) if all_selected else None,
        "top_tokens": Counter(item["token"] for item in all_selected).most_common(20),
        "by_id": by_id,
    }


def compare_pair(left, right, label):
    shared_ids = sorted(set(left["by_id"]) & set(right["by_id"]))
    outcome = Counter()
    jaccards = []
    overlaps = []
    left_only_cats = Counter()
    right_only_cats = Counter()
    shared_cats = Counter()
    cases = []
    for sample_id in shared_ids:
        left_row = left["by_id"][sample_id]
        right_row = right["by_id"][sample_id]
        left_set = set(left_row["selected_positions"])
        right_set = set(right_row["selected_positions"])
        inter = left_set & right_set
        jaccards.append(jaccard(left_set, right_set))
        overlaps.append(overlap(left_set, right_set))

        left_items = {item["pos"]: item for item in left_row["selected"]}
        right_items = {item["pos"]: item for item in right_row["selected"]}
        for pos in left_set - right_set:
            left_only_cats[left_items[pos]["cat"]] += 1
        for pos in right_set - left_set:
            right_only_cats[right_items[pos]["cat"]] += 1
        for pos in inter:
            shared_cats[left_items[pos]["cat"]] += 1

        left_ok = bool(left_row["final_correct"])
        right_ok = bool(right_row["final_correct"])
        if left_ok and right_ok:
            outcome["both_correct"] += 1
        elif left_ok and not right_ok:
            outcome["left_only_correct"] += 1
        elif (not left_ok) and right_ok:
            outcome["right_only_correct"] += 1
        else:
            outcome["both_wrong"] += 1

        cases.append(
            {
                "id": sample_id,
                "dataset_index": left_row.get("dataset_index"),
                "question": left_row.get("question"),
                "answer": left_row.get("answer"),
                "domain": left_row.get("domain"),
                "topic": left_row.get("topic"),
                "left_correct": left_ok,
                "right_correct": right_ok,
                "left_final_text": left_row.get("final_text"),
                "right_final_text": right_row.get("final_text"),
                "left_selected": left_row["selected"],
                "right_selected": right_row["selected"],
                "shared_positions": sorted(inter),
                "left_only_positions": sorted(left_set - right_set),
                "right_only_positions": sorted(right_set - left_set),
                "jaccard": jaccards[-1],
                "overlap": overlaps[-1],
            }
        )

    return {
        "label": label,
        "left_key": left["key"],
        "right_key": right["key"],
        "left_label": left["label"],
        "right_label": right["label"],
        "num_shared_samples": len(shared_ids),
        "mean_jaccard": safe_mean(jaccards),
        "mean_overlap": safe_mean(overlaps),
        "outcome": dict(outcome),
        "net_right_minus_left": outcome["right_only_correct"] - outcome["left_only_correct"],
        "left_only_category_counts": dict(left_only_cats),
        "right_only_category_counts": dict(right_only_cats),
        "shared_category_counts": dict(shared_cats),
        "cases": cases,
    }


def compact_report(report, max_samples):
    compact = json.loads(json.dumps(report, ensure_ascii=False))
    for run in compact["runs"].values():
        run.pop("by_id", None)
    for pair in compact["pairwise"].values():
        pair["cases"] = pair["cases"][:max_samples]
    compact.pop("samples", None)
    return compact


def pct(value):
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def fmt(value, digits=3):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def cat_bar(counts):
    total = sum(counts.values())
    if not total:
        return ""
    parts = []
    for cat in CAT_ORDER:
        count = counts.get(cat, 0)
        if not count:
            continue
        width = 100.0 * count / total
        parts.append(
            f'<span class="seg" title="{html.escape(CAT_LABEL[cat])}: {count}" '
            f'style="width:{width:.3f}%;background:{CAT_COLORS[cat]}"></span>'
        )
    return "".join(parts)


def render_html(report):
    data = json.dumps(compact_report(report, report["html_max_samples"]), ensure_ascii=False)
    run_rows = []
    for run in report["runs"].values():
        run_rows.append(
            "<tr>"
            f"<td>{html.escape(run['label'])}</td>"
            f"<td>{run['num_records']}</td>"
            f"<td>{pct(run['draft_accuracy'])}</td>"
            f"<td>{pct(run['final_accuracy'])}</td>"
            f"<td>{fmt((run['accuracy_delta'] or 0) * 100, 2)} pts</td>"
            f"<td>{run['selected_token_count']}</td>"
            f"<td>{pct(run['semantic_ratio'])}</td>"
            f"<td><div class='bar'>{cat_bar(Counter(run['category_counts']))}</div></td>"
            "</tr>"
        )
    pair_rows = []
    for key, pair in report["pairwise"].items():
        out = pair["outcome"]
        pair_rows.append(
            "<tr>"
            f"<td>{html.escape(pair['label'])}</td>"
            f"<td>{pair['num_shared_samples']}</td>"
            f"<td>{fmt(pair['mean_jaccard'])}</td>"
            f"<td>{fmt(pair['mean_overlap'])}</td>"
            f"<td>{out.get('left_only_correct', 0)}</td>"
            f"<td>{out.get('right_only_correct', 0)}</td>"
            f"<td>{pair['net_right_minus_left']:+d}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visual vs Proposal Select + VRG Comparison</title>
<style>
:root{{--bg:#f7f8fa;--panel:#fff;--ink:#17191f;--muted:#667085;--line:#d8dde6;--blue:#2563eb;--orange:#f97316;--green:#16a34a;--red:#c2410c;--soft:#f1f4f8}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}}
.shell{{width:min(1500px,calc(100vw - 32px));margin:24px auto 48px}}
header{{border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:14px}}
h1{{margin:0 0 8px;font-size:28px;letter-spacing:0}} h2{{font-size:18px;margin:0 0 10px}} h3{{font-size:15px;margin:0 0 8px}}
p{{margin:0;color:var(--muted)}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}} th{{background:var(--soft);color:#344054}}
.panel,.case{{background:var(--panel);border:1px solid var(--line);border-radius:8px}} .panel{{padding:12px;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:420px 1fr;gap:14px;align-items:start}} .twocol{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.bar{{display:flex;height:18px;overflow:hidden;border-radius:5px;background:#edf0f4;border:1px solid var(--line)}} .seg{{display:block;height:100%}}
.controls{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}}
input,select{{width:100%;border:1px solid var(--line);border-radius:6px;padding:8px 9px;background:white;color:var(--ink);font:inherit}}
.list{{display:grid;gap:8px;max-height:calc(100vh - 280px);overflow:auto;padding-right:4px}} .case{{padding:10px;cursor:pointer}} .case.active{{outline:2px solid var(--blue)}}
.tags{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:7px}} .tag{{display:inline-flex;align-items:center;min-height:22px;padding:2px 7px;border:1px solid var(--line);border-radius:5px;background:#fafbfc;color:var(--muted);font-size:12px;font-weight:650}}
.left{{color:white;background:var(--blue);border-color:var(--blue)}} .right{{color:white;background:var(--orange);border-color:var(--orange)}} .both{{color:white;background:var(--green);border-color:var(--green)}} .bad{{color:white;background:var(--red);border-color:var(--red)}}
.textblock{{white-space:pre-wrap;word-break:break-word;background:#fbfcfe;border:1px solid var(--line);border-radius:6px;padding:10px;max-height:220px;overflow:auto;font-size:13px}}
.tokens{{display:flex;gap:5px;flex-wrap:wrap}} code{{background:#f8fafc;border:1px solid var(--line);border-radius:5px;padding:2px 5px}} .muted{{color:var(--muted)}} .mono{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
@media(max-width:1000px){{.grid,.twocol{{grid-template-columns:1fr}}.list{{max-height:420px}}}}
</style>
</head>
<body><div class="shell">
<header><h1>Visual Select vs Proposal Select, With and Without VRG</h1>
<p>Compares four aligned runs by selected token positions, token categories, final correctness, and per-sample outputs.</p></header>
<section class="panel"><h2>Run Summary</h2><table><thead><tr><th>run</th><th>samples</th><th>draft acc</th><th>final acc</th><th>delta</th><th>selected tokens</th><th>semantic ratio</th><th>categories</th></tr></thead><tbody>{''.join(run_rows)}</tbody></table></section>
<section class="panel"><h2>Paired Comparisons</h2><table><thead><tr><th>comparison</th><th>shared samples</th><th>position Jaccard</th><th>overlap</th><th>left only correct</th><th>right only correct</th><th>net right-left</th></tr></thead><tbody>{''.join(pair_rows)}</tbody></table></section>
<div class="grid"><section class="panel"><h2>Samples</h2><div class="controls"><select id="pair"><option value="visual_vs_proposal">visual vs proposal</option><option value="visual_vrg_vs_proposal_vrg">visual+VRG vs proposal+VRG</option><option value="visual_vrg_gain">visual + VRG effect</option><option value="proposal_vrg_gain">proposal + VRG effect</option></select><select id="outcome"><option value="all">all outcomes</option><option value="left">left correct, right wrong</option><option value="right">right correct, left wrong</option><option value="both">both correct</option><option value="wrong">both wrong</option></select></div><p class="muted">Default shows all samples for the selected comparison. Use the filters above to narrow by comparison type and correctness outcome.</p><div id="list" class="list"></div></section><section id="detail"></section></div>
</div>
<script id="report" type="application/json">{html.escape(data)}</script>
<script>
const report = JSON.parse(document.getElementById('report').textContent);
let state = {{pair:'visual_vs_proposal', outcome:'all', index:0}};
const listEl = document.getElementById('list'), detailEl = document.getElementById('detail');
function esc(x){{return String(x??'').replace(/[&<>"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]));}}
function num(x){{return x===null||x===undefined?'n/a':Number(x).toFixed(3);}}
function filtered(){{
 const pair = report.pairwise[state.pair]; if(!pair) return [];
 return (pair.cases||[]).filter(c=>{{
   const ok = state.outcome==='all' || (state.outcome==='left'&&c.left_correct&&!c.right_correct) || (state.outcome==='right'&&!c.left_correct&&c.right_correct) || (state.outcome==='both'&&c.left_correct&&c.right_correct) || (state.outcome==='wrong'&&!c.left_correct&&!c.right_correct);
   return ok;
 }});
}}
function tokenList(items, klass){{return `<div class="tokens">${{(items||[]).map(t=>`<code class="${{klass}} mono" title="pos ${{esc(t.pos)}} score ${{num(t.score)}}">${{esc(t.display)}}<span class="muted">:${{esc(t.pos)}}</span></code>`).join('')}}</div>`;}}
function renderList(){{
 const items = filtered(); if(state.index>=items.length) state.index=0;
 listEl.innerHTML = items.map((c,i)=>`<div class="case ${{i===state.index?'active':''}}" onclick="state.index=${{i}}; render()"><div class="tags"><span class="tag">#${{esc(c.dataset_index)}}</span><span class="tag ${{c.left_correct?'both':'bad'}}">L ${{c.left_correct?'ok':'bad'}}</span><span class="tag ${{c.right_correct?'both':'bad'}}">R ${{c.right_correct?'ok':'bad'}}</span><span class="tag">J ${{num(c.jaccard)}}</span></div><div>${{esc((c.question||'').slice(0,180))}}</div></div>`).join('');
 renderDetail(items[state.index]);
}}
function renderDetail(c){{
 const pair = report.pairwise[state.pair];
 if(!c){{detailEl.innerHTML='<section class="panel">No matching samples.</section>';return;}}
 detailEl.innerHTML = `<section class="panel"><h2>${{esc(pair.label)}} · #${{esc(c.dataset_index)}} · ${{esc(c.id)}}</h2><p>${{esc(c.question)}}</p><p class="muted">answer=${{esc(c.answer)}} · topic=${{esc(c.topic)}} · domain=${{esc(c.domain)}} · Jaccard=${{num(c.jaccard)}} · overlap=${{num(c.overlap)}}</p></section>
 <div class="twocol"><section class="panel"><h3>${{esc(pair.left_label)}} ${{c.left_correct?'✓':'✗'}}</h3><div class="textblock">${{esc(c.left_final_text)}}</div></section><section class="panel"><h3>${{esc(pair.right_label)}} ${{c.right_correct?'✓':'✗'}}</h3><div class="textblock">${{esc(c.right_final_text)}}</div></section></div>
 <div class="twocol"><section class="panel"><h3>Left Selected Tokens</h3>${{tokenList(c.left_selected,'left')}}</section><section class="panel"><h3>Right Selected Tokens</h3>${{tokenList(c.right_selected,'right')}}</section></div>
 <section class="panel"><h3>Position Difference</h3><p class="muted">shared: ${{esc((c.shared_positions||[]).join(', '))}}</p><p class="muted">left-only: ${{esc((c.left_only_positions||[]).join(', '))}}</p><p class="muted">right-only: ${{esc((c.right_only_positions||[]).join(', '))}}</p></section>`;
}}
function render(){{renderList();}}
document.getElementById('pair').onchange=e=>{{state.pair=e.target.value;state.index=0;render();}};
document.getElementById('outcome').onchange=e=>{{state.outcome=e.target.value;state.index=0;render();}};
render();
</script>
</body></html>"""


def build_samples(runs):
    shared = sorted(set.intersection(*(set(run["by_id"]) for run in runs.values())))
    samples = []
    for sample_id in shared:
        first = next(run["by_id"][sample_id] for run in runs.values() if sample_id in run["by_id"])
        samples.append(
            {
                "id": sample_id,
                "dataset_index": first.get("dataset_index"),
                "question": first.get("question"),
                "answer": first.get("answer"),
                "domain": first.get("domain"),
                "topic": first.get("topic"),
                "runs": {
                    key: {
                        "final_correct": run["by_id"][sample_id]["final_correct"],
                        "draft_correct": run["by_id"][sample_id]["draft_correct"],
                        "final_text": run["by_id"][sample_id]["final_text"],
                        "selected": run["by_id"][sample_id]["selected"],
                    }
                    for key, run in runs.items()
                    if sample_id in run["by_id"]
                },
            }
        )
    return samples


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    baseline_by_id = build_baseline_tokens(args.baseline_records, tokenizer)
    runs = {
        "visual_select": summarize_run(
            "visual_select", "Visual Select", args.visual_select_records, tokenizer, baseline_by_id
        ),
        "proposal_select": summarize_run(
            "proposal_select", "Proposal Select", args.proposal_select_records, tokenizer, baseline_by_id
        ),
        "visual_vrg": summarize_run(
            "visual_vrg", "Visual Select + VRG/VCD Refill", args.visual_vrg_records, tokenizer, baseline_by_id
        ),
        "proposal_vrg": summarize_run(
            "proposal_vrg", "Proposal Select + VRG/VCD Refill", args.proposal_vrg_records, tokenizer, baseline_by_id
        ),
    }
    pairwise = {
        "visual_vs_proposal": compare_pair(runs["visual_select"], runs["proposal_select"], "Visual Select vs Proposal Select"),
        "visual_vrg_vs_proposal_vrg": compare_pair(
            runs["visual_vrg"], runs["proposal_vrg"], "Visual Select + VRG vs Proposal Select + VRG"
        ),
        "visual_vrg_gain": compare_pair(runs["visual_select"], runs["visual_vrg"], "Visual Select vs Visual Select + VRG"),
        "proposal_vrg_gain": compare_pair(
            runs["proposal_select"], runs["proposal_vrg"], "Proposal Select vs Proposal Select + VRG"
        ),
    }
    report = {
        "inputs": {
            "visual_select_records": str(args.visual_select_records),
            "proposal_select_records": str(args.proposal_select_records),
            "visual_vrg_records": str(args.visual_vrg_records),
            "proposal_vrg_records": str(args.proposal_vrg_records),
            "baseline_records": str(args.baseline_records),
            "tokenizer": args.tokenizer,
        },
        "notes": [
            "Visual select means the visual_gain remask selector, not visual warmup.",
            "The default +VRG runs use saved VCD/weak-visual refill guidance; exact config is preserved in each run summary.",
            "Token comparison is position-based on the generated answer token stream.",
        ],
        "runs": runs,
        "pairwise": pairwise,
        "samples": build_samples(runs),
        "html_max_samples": args.max_html_samples,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(compact_report(report, len(report["samples"])), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output_html.write_text(render_html(report), encoding="utf-8")

    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote HTML: {args.output_html}")
    for key, run in runs.items():
        print(f"{key}: n={run['num_records']} final_acc={run['final_accuracy']:.4f}")
    for key, pair in pairwise.items():
        print(
            f"{key}: shared={pair['num_shared_samples']} jaccard={pair['mean_jaccard']:.4f} "
            f"net_right_minus_left={pair['net_right_minus_left']:+d}"
        )


if __name__ == "__main__":
    main()
