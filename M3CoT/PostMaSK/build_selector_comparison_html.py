import html
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer


ROOT = Path("M3CoT/PostMaSK/outputs")
BASE_RECORDS = Path("M3CoT/benchmark/StepRemask/baseline_no_remask_seed42_n400/records.jsonl")
OUT_HTML = ROOT / "postmask_selector_strategy_comparison.html"

STRATEGIES = [
    ("proposal_confidence", "Proposal confidence", ROOT / "postmask_sr0p5_d16_p16_proposalconf_fixed32_refill2_seed42_n400", "#2563eb"),
    ("topk_margin", "Top-k margin", ROOT / "postmask_sr0p5_d16_p16_topkmargin_fixed32_refill2_seed42_n400", "#f97316"),
    ("visual_gain_null", "Null visual gain", ROOT / "postmask_sr0p5_d16_p16_visualgain_fixed32_refill2_seed42_n400", "#84cc16"),
    ("visual_gain_vcd", "VCD visual gain", ROOT / "postmask_sr0p5_d16_p16_visualgain_vcdnoise500_fixed32_refill2_seed42_n400", "#0f766e"),
    ("conf_low_visual_null", "Conf-low null visual", ROOT / "postmask_sr0p5_d16_p16_conflowvisualgain_fixed32_refill2_seed42_n400", "#14b8a6"),
    ("proposal_then_vcd_k48", "Proposal then VCD K48", ROOT / "postmask_sr0p5_d16_p16_proposalthenvcdvisual_k48_fixed32_refill2_seed42_n400", "#0891b2"),
    ("proposal_then_vcd_k64", "Proposal then VCD K64", ROOT / "postmask_sr0p5_d16_p16_proposalthenvcdvisual_k64_fixed32_refill2_seed42_n400", "#0369a1"),
    ("rank_conf_vcd_l1", "Rank conf+VCD lambda1", ROOT / "postmask_sr0p5_d16_p16_rankconfvcdvisual_l1_fixed32_refill2_seed42_n400", "#be123c"),
    ("norm_vcd_a1_b1", "Norm VCD a1 b1", ROOT / "postmask_sr0p5_d16_p16_normconfvcdvisual_a1_b1_fixed32_refill2_seed42_n400", "#9333ea"),
    ("norm_vcd_a1_b2", "Norm VCD a1 b2", ROOT / "postmask_sr0p5_d16_p16_normconfvcdvisual_a1_b2_fixed32_refill2_seed42_n400", "#7e22ce"),
    ("norm_vcd_a2_b1", "Norm VCD a2 b1", ROOT / "postmask_sr0p5_d16_p16_normconfvcdvisual_a2_b1_fixed32_refill2_seed42_n400", "#a21caf"),
]

VISUAL_SERIES = [
    "visual_gain_null",
    "visual_gain_vcd",
    "conf_low_visual_null",
    "proposal_then_vcd_k48",
    "proposal_then_vcd_k64",
    "rank_conf_vcd_l1",
    "norm_vcd_a1_b1",
    "norm_vcd_a1_b2",
    "norm_vcd_a2_b1",
]

STOPWORDS = set(
    "the a an is are was were be been being of to in on at for from with by and or but if then than as that this these those it its we need determine correct answer therefore because can see based given option options question image diagram figure shown shows using use compare each which what when where how there more less between into left right".split()
)

CAT_ORDER = [
    "content",
    "template",
    "number",
    "option",
    "punct",
    "white",
    "special",
    "other",
    "mixednum",
]

CAT_LABEL = {
    "content": "content word",
    "template": "reasoning/template",
    "number": "number",
    "option": "answer option",
    "punct": "punct/format",
    "white": "whitespace",
    "special": "special/eos",
    "other": "other/subword",
    "mixednum": "mixed number/unit",
}

CAT_COLORS = {
    "content": "#2563eb",
    "template": "#94a3b8",
    "number": "#16a34a",
    "option": "#f97316",
    "punct": "#64748b",
    "white": "#cbd5e1",
    "special": "#a855f7",
    "other": "#eab308",
    "mixednum": "#0f766e",
}


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def first_remask_record(row):
    for record in row.get("postmask_records") or []:
        if record.get("remasked_answer_positions"):
            return record
    return None


def decode_token(tokenizer, token_id):
    if token_id is None:
        return "<missing>"
    return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def clean_token(text):
    return text.replace("\n", "\\n").strip()


def display_token(text):
    return text.replace("\n", "↵") if text else "·"


def token_category(text):
    stripped = text.strip()
    lowered = stripped.lower()
    if text == "\n" or stripped == "":
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


def pct(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def safe_mean(values):
    finite = [
        float(value)
        for value in values
        if value is not None and isinstance(value, (int, float, bool)) and math.isfinite(float(value))
    ]
    return mean(finite) if finite else None


def jaccard(left, right):
    return len(left & right) / len(left | right) if left or right else 0.0


def category_bar(counter, total):
    parts = []
    for cat in CAT_ORDER:
        count = counter.get(cat, 0)
        if not count or not total:
            continue
        width = count / total * 100
        label = CAT_LABEL[cat]
        parts.append(
            f'<div class="seg" style="width:{width:.3f}%;background:{CAT_COLORS[cat]}" '
            f'title="{html.escape(label)}: {count} ({width:.1f}%)"></div>'
        )
    return "".join(parts)


def category_legend(counter, total):
    parts = []
    for cat in CAT_ORDER:
        count = counter.get(cat, 0)
        if not count or not total:
            continue
        parts.append(
            f'<span><i style="background:{CAT_COLORS[cat]}"></i>{html.escape(CAT_LABEL[cat])} {count / total * 100:.1f}%</span>'
        )
    return "".join(parts)


def selected_items_from_row(row, base_tokens):
    record = first_remask_record(row)
    if record is None:
        return []
    positions = [int(pos) for pos in record.get("remasked_answer_positions") or []]
    scores = record.get("selection_scores") or []
    visual_gains = record.get("selected_visual_gains") or []
    items = []
    for idx, pos in enumerate(positions):
        token = base_tokens.get(
            pos,
            {"pos": pos, "id": None, "text": "<missing>", "display": "<missing>", "clean": "<missing>", "cat": "other"},
        )
        item = {
            "pos": pos,
            "token": token["clean"],
            "display": token["display"],
            "cat": token["cat"],
            "score": scores[idx] if idx < len(scores) else None,
            "visual_gain": visual_gains[idx] if idx < len(visual_gains) else None,
        }
        items.append(item)
    return items


def main():
    tokenizer = AutoTokenizer.from_pretrained("weight/lavida-reason", trust_remote_code=True)

    base_rows = load_jsonl(BASE_RECORDS)
    base_by_id = {}
    for row in base_rows:
        tokens = []
        for pos, token_id in enumerate((row.get("final_answer_ids") or [])[:64]):
            text = decode_token(tokenizer, token_id)
            tokens.append(
                {
                    "pos": pos,
                    "id": int(token_id),
                    "text": text,
                    "display": display_token(text),
                    "clean": clean_token(text),
                    "cat": token_category(text),
                }
            )
        base_by_id[row["id"]] = {"row": row, "tokens": tokens}

    strategies = []
    for key, label, directory, color in STRATEGIES:
        records_path = directory / "records.jsonl"
        summary_path = directory / "summary.json"
        if not records_path.exists():
            continue
        rows = load_jsonl(records_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        by_id = {}
        selected_all = []
        for row in rows:
            base_tokens = {token["pos"]: token for token in base_by_id.get(row["id"], {}).get("tokens", [])}
            selected = selected_items_from_row(row, base_tokens)
            by_id[row["id"]] = {
                "draft_correct": bool(row.get("draft_correct")),
                "final_correct": bool(row.get("final_correct")),
                "positions": [item["pos"] for item in selected],
                "selected": selected,
            }
            selected_all.extend(selected)
        strategies.append(
            {
                "key": key,
                "label": label,
                "color": color,
                "rows": rows,
                "summary": summary,
                "by_id": by_id,
                "selected": selected_all,
            }
        )

    strategy_by_key = {strategy["key"]: strategy for strategy in strategies}
    proposal = strategy_by_key.get("proposal_confidence")

    summary_rows = []
    for strategy in strategies:
        selected = strategy["selected"]
        cats = Counter(item["cat"] for item in selected)
        total = len(selected)
        semantic = cats["content"] + cats["number"] + cats["mixednum"] + cats["option"]
        rows = strategy["rows"]
        draft_acc = safe_mean([row.get("draft_correct", False) for row in rows])
        final_acc = safe_mean([row.get("final_correct", False) for row in rows])
        summary_rows.append(
            {
                "key": strategy["key"],
                "label": strategy["label"],
                "color": strategy["color"],
                "samples": len(rows),
                "selected": total,
                "draft_acc": draft_acc,
                "final_acc": final_acc,
                "content_ratio": cats["content"] / total if total else None,
                "semantic_ratio": semantic / total if total else None,
                "template_ratio": cats["template"] / total if total else None,
                "mean_pos": safe_mean([item["pos"] for item in selected]),
                "cats": cats,
                "top_tokens": Counter(item["token"] for item in selected).most_common(10),
            }
        )

    proposal_vs_visual = []
    if proposal is not None:
        for key in VISUAL_SERIES:
            visual = strategy_by_key.get(key)
            if visual is None:
                continue
            shared_ids = sorted(set(proposal["by_id"]) & set(visual["by_id"]))
            overlap_values = []
            proposal_only_cats = Counter()
            visual_only_cats = Counter()
            both_cats = Counter()
            outcome = Counter()
            for sample_id in shared_ids:
                proposal_data = proposal["by_id"][sample_id]
                visual_data = visual["by_id"][sample_id]
                proposal_positions = set(proposal_data["positions"])
                visual_positions = set(visual_data["positions"])
                overlap_values.append(jaccard(proposal_positions, visual_positions))
                proposal_items = {item["pos"]: item for item in proposal_data["selected"]}
                visual_items = {item["pos"]: item for item in visual_data["selected"]}
                for pos in proposal_positions - visual_positions:
                    proposal_only_cats[proposal_items[pos]["cat"]] += 1
                for pos in visual_positions - proposal_positions:
                    visual_only_cats[visual_items[pos]["cat"]] += 1
                for pos in proposal_positions & visual_positions:
                    both_cats[proposal_items[pos]["cat"]] += 1
                proposal_correct = bool(proposal_data["final_correct"])
                visual_correct = bool(visual_data["final_correct"])
                if proposal_correct and visual_correct:
                    outcome["both_correct"] += 1
                elif proposal_correct and not visual_correct:
                    outcome["proposal_only_correct"] += 1
                elif (not proposal_correct) and visual_correct:
                    outcome["visual_only_correct"] += 1
                else:
                    outcome["both_wrong"] += 1
            proposal_vs_visual.append(
                {
                    "key": key,
                    "label": visual["label"],
                    "color": visual["color"],
                    "samples": len(shared_ids),
                    "jaccard": safe_mean(overlap_values),
                    "proposal_only_cats": proposal_only_cats,
                    "visual_only_cats": visual_only_cats,
                    "both_cats": both_cats,
                    "outcome": outcome,
                }
            )

    sample_payload = []
    for base in sorted(base_rows, key=lambda row: row.get("dataset_index", 0)):
        sample_id = base["id"]
        sample_strategies = {}
        for strategy in strategies:
            data = strategy["by_id"].get(sample_id)
            if data:
                sample_strategies[strategy["key"]] = data
        if not sample_strategies:
            continue
        sample_payload.append(
            {
                "id": sample_id,
                "dataset_index": base.get("dataset_index"),
                "topic": base.get("topic"),
                "domain": base.get("domain"),
                "answer": base.get("answer"),
                "question": base.get("question", ""),
                "baseline_correct": bool(base.get("final_correct")),
                "tokens": base_by_id[sample_id]["tokens"],
                "strategies": sample_strategies,
            }
        )

    summary_html = []
    for row in summary_rows:
        delta = row["final_acc"] - row["draft_acc"] if row["final_acc"] is not None and row["draft_acc"] is not None else None
        summary_html.append(
            "<tr>"
            f'<td><span class="swatch" style="background:{row["color"]}"></span>{html.escape(row["label"])}</td>'
            f'<td>{row["samples"]}</td><td>{row["selected"]}</td><td>{pct(row["draft_acc"])}</td>'
            f'<td>{pct(row["final_acc"])}</td><td>{fmt(delta * 100 if delta is not None else None, 2)} pts</td>'
            f'<td>{pct(row["content_ratio"])}</td><td>{pct(row["semantic_ratio"])}</td>'
            f'<td>{pct(row["template_ratio"])}</td><td>{fmt(row["mean_pos"], 1)}</td>'
            "</tr>"
        )

    cat_cards = []
    for row in summary_rows:
        top_tokens = ", ".join(f"<code>{html.escape(token)}</code> {count}" for token, count in row["top_tokens"][:8])
        cat_cards.append(
            '<section class="card compact">'
            f'<h3><span class="swatch" style="background:{row["color"]}"></span>{html.escape(row["label"])}</h3>'
            f'<div class="bar">{category_bar(row["cats"], row["selected"])}</div>'
            f'<div class="legend">{category_legend(row["cats"], row["selected"])}</div>'
            f'<p class="small"><b>Top selected draft tokens:</b> {top_tokens}</p>'
            "</section>"
        )

    focus_rows = []
    for row in proposal_vs_visual:
        prop_only_total = sum(row["proposal_only_cats"].values())
        visual_only_total = sum(row["visual_only_cats"].values())
        both_total = sum(row["both_cats"].values())
        outcome = row["outcome"]
        net_vs_proposal = outcome["visual_only_correct"] - outcome["proposal_only_correct"]
        focus_rows.append(
            '<section class="card focus-card">'
            f'<h3><span class="swatch" style="background:{row["color"]}"></span>{html.escape(row["label"])} vs Proposal</h3>'
            '<div class="focus-grid">'
            f'<div class="metric"><b>{fmt(row["jaccard"], 3)}</b><span>selected-position Jaccard</span></div>'
            f'<div class="metric"><b>{outcome["visual_only_correct"]}</b><span>visual correct, proposal wrong</span></div>'
            f'<div class="metric"><b>{outcome["proposal_only_correct"]}</b><span>proposal correct, visual wrong</span></div>'
            f'<div class="metric"><b>{net_vs_proposal:+d}</b><span>net vs proposal</span></div>'
            "</div>"
            '<div class="diff-bars">'
            '<div><b>Proposal-only tokens</b>'
            f'<div class="bar">{category_bar(row["proposal_only_cats"], prop_only_total)}</div>'
            f'<div class="legend">{category_legend(row["proposal_only_cats"], prop_only_total)}</div></div>'
            '<div><b>Visual-only tokens</b>'
            f'<div class="bar">{category_bar(row["visual_only_cats"], visual_only_total)}</div>'
            f'<div class="legend">{category_legend(row["visual_only_cats"], visual_only_total)}</div></div>'
            '<div><b>Shared tokens</b>'
            f'<div class="bar">{category_bar(row["both_cats"], both_total)}</div>'
            f'<div class="legend">{category_legend(row["both_cats"], both_total)}</div></div>'
            "</div></section>"
        )

    strategy_js = [
        {"key": strategy["key"], "label": strategy["label"], "color": strategy["color"]}
        for strategy in strategies
    ]
    visual_js = [key for key in VISUAL_SERIES if key in strategy_by_key]

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PostMask Selector Strategy Comparison</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#decfb3;--card:#fffaf0}}
*{{box-sizing:border-box}}
body{{margin:0;background:radial-gradient(circle at 10% 0%,#fff0ba 0,transparent 28%),linear-gradient(135deg,#f5f1e8,#edf5ef);color:var(--ink);font-family:ui-serif,Georgia,Cambria,"Times New Roman",serif}}
main{{max-width:1380px;margin:0 auto;padding:34px 24px 80px}}
h1{{margin:0 0 8px;font-size:36px;letter-spacing:-.03em}}
h2{{margin-top:34px}} p,li{{color:var(--muted);line-height:1.55}}
.card{{background:rgba(255,250,240,.95);border:1px solid var(--line);border-radius:18px;box-shadow:0 18px 48px rgba(59,45,20,.08);padding:18px;margin:16px 0}}
.compact h3,.focus-card h3{{margin:0 0 12px}}
.grid2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:rgba(255,250,240,.8)}}
th,td{{border-bottom:1px solid #e7dcc6;padding:8px 9px;text-align:left;vertical-align:middle}}
th{{color:#475569;font-weight:700}}
.swatch{{display:inline-block;width:11px;height:11px;border-radius:999px;margin-right:7px;vertical-align:-1px}}
.bar{{display:flex;height:28px;overflow:hidden;border-radius:999px;border:1px solid #d6c7aa;background:#fff;margin-top:8px}}
.seg{{height:100%}}
.legend{{display:flex;flex-wrap:wrap;gap:7px 13px;margin-top:9px;font-size:12px;color:#475569}}
.legend i{{display:inline-block;width:9px;height:9px;border-radius:999px;margin-right:4px}}
code{{background:#fff2cc;border:1px solid #ead79e;padding:1px 4px;border-radius:5px}}
.small{{font-size:13px}}
.note{{background:#fff8e6;border:1px dashed #d8b45f;border-radius:14px;padding:12px 14px}}
.focus-grid,.statline{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:10px 0 14px}}
.metric,.stat{{background:#fffdf7;border:1px solid #eadfc9;border-radius:12px;padding:10px}}
.metric b,.stat b{{display:block;font-size:20px;color:#172033}}
.metric span,.stat span{{font-size:12px;color:#64748b}}
.diff-bars{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
.controls{{display:grid;grid-template-columns:1.2fr 1fr 1fr;gap:12px;align-items:end;margin:14px 0}}
label{{display:block;font-size:12px;color:#64748b;font-weight:700;margin-bottom:5px}}
select,input{{width:100%;border:1px solid #d6c7aa;border-radius:10px;background:#fffdf7;padding:9px 10px;color:#1f2937}}
.strategy-pills{{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}}
.pill{{border:1px solid #d6c7aa;background:#fffdf7;border-radius:999px;padding:7px 10px;cursor:pointer;color:#334155;font-size:13px}}
.pill.active{{color:white;border-color:transparent}}
.sample-meta{{display:flex;flex-wrap:wrap;gap:8px 16px;color:#475569;font-size:13px;margin:10px 0}}
.question{{white-space:pre-wrap;background:#fffdf7;border-left:3px solid #d8c8aa;border-radius:8px;padding:10px 12px;font-size:13px;max-height:150px;overflow:auto}}
.draft{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#172033;color:#e5e7eb;border-radius:16px;padding:16px;line-height:2.25;font-size:14px;white-space:normal}}
.tok{{display:inline-block;padding:2px 4px;margin:1px;border-radius:6px;border:1px solid transparent}}
.tok.primary{{color:#fff;font-weight:800;box-shadow:0 0 0 2px rgba(250,204,21,.95),0 0 18px rgba(250,204,21,.28)}}
.tok.compare{{outline:2px dashed rgba(255,255,255,.7)}}
.tok.both{{background:#16a34a!important;color:white;font-weight:900}}
.tok.proposalOnly{{background:#2563eb!important;color:white;font-weight:900}}
.tok.visualOnly{{background:#f97316!important;color:white;font-weight:900}}
.selected-list{{display:flex;flex-wrap:wrap;gap:5px;margin-top:12px}}
.selected-list code{{background:#fff7dc}}
@media(max-width:900px){{.grid2,.controls,.focus-grid,.diff-bars,.statline{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<h1>PostMask Selector Strategy Comparison</h1>
<p>Focus: how <b>proposal confidence</b> differs from the visual selector family when choosing draft tokens to remask.</p>
<section class="note"><ul>
<li><b>Proposal vs Visual Focus</b> shows proposal-only, visual-only, and shared selected token types.</li>
<li>In the per-sample browser, choose <b>Proposal vs Visual overlay</b> to mark shared tokens green, proposal-only blue, and visual-only orange.</li>
<li>Token text/type is recovered from the shared no-remask draft answer ids, so every selector is compared on the same draft token stream.</li>
</ul></section>
<h2>Overall Strategy Summary</h2>
<section class="card"><table><thead><tr><th>Strategy</th><th>samples</th><th>selected tokens</th><th>draft acc</th><th>final acc</th><th>delta</th><th>content</th><th>semantic</th><th>template</th><th>mean pos</th></tr></thead><tbody>{''.join(summary_html)}</tbody></table></section>
<h2>Proposal vs Visual-Series Differences</h2>
{''.join(focus_rows)}
<h2>Selected Token Type Distribution</h2>
<div class="grid2">{''.join(cat_cards)}</div>
<h2>Per-Sample Draft Browser</h2>
<section class="card">
<div class="controls">
<div><label for="sampleSelect">Sample</label><select id="sampleSelect"></select></div>
<div><label for="strategySelect">Highlight strategy</label><select id="strategySelect"></select></div>
<div><label for="compareSelect">Optional compare overlay</label><select id="compareSelect"><option value="">None</option><option value="proposalVisual">Proposal vs selected visual</option></select></div>
</div>
<div class="strategy-pills" id="strategyPills"></div>
<div id="sampleMeta" class="sample-meta"></div>
<div id="question" class="question"></div>
<div class="statline" id="statline"></div>
<div id="draft" class="draft"></div>
<div id="selectedList" class="selected-list"></div>
</section>
</main>
<script>
const SAMPLES = {json.dumps(sample_payload, ensure_ascii=False)};
const STRATEGIES = {json.dumps(strategy_js, ensure_ascii=False)};
const VISUAL_KEYS = {json.dumps(visual_js, ensure_ascii=False)};
const sampleSelect = document.getElementById('sampleSelect');
const strategySelect = document.getElementById('strategySelect');
const compareSelect = document.getElementById('compareSelect');
const draftEl = document.getElementById('draft');
const metaEl = document.getElementById('sampleMeta');
const questionEl = document.getElementById('question');
const listEl = document.getElementById('selectedList');
const statlineEl = document.getElementById('statline');
const pillsEl = document.getElementById('strategyPills');
function esc(s) {{ return String(s ?? '').replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m])); }}
function strategyByKey(key) {{ return STRATEGIES.find(s => s.key === key); }}
function sampleLabel(s) {{ return `${{s.dataset_index}} · ${{s.id}} · ${{s.topic}}`; }}
SAMPLES.forEach((s, idx) => {{ const opt = document.createElement('option'); opt.value = idx; opt.textContent = sampleLabel(s); sampleSelect.appendChild(opt); }});
STRATEGIES.forEach(s => {{
  const opt = document.createElement('option'); opt.value = s.key; opt.textContent = s.label; strategySelect.appendChild(opt);
  if (s.key !== 'proposal_confidence') {{
    const opt2 = document.createElement('option'); opt2.value = s.key; opt2.textContent = s.label; compareSelect.appendChild(opt2);
  }}
  const pill = document.createElement('button'); pill.className = 'pill'; pill.textContent = s.label; pill.style.borderColor = s.color;
  pill.onclick = () => {{ strategySelect.value = s.key; render(); }};
  pillsEl.appendChild(pill);
}});
strategySelect.value = STRATEGIES.some(s => s.key === 'visual_gain_vcd') ? 'visual_gain_vcd' : STRATEGIES[0].key;
function render() {{
  const sample = SAMPLES[Number(sampleSelect.value || 0)];
  const key = strategySelect.value;
  const compareKey = compareSelect.value;
  const strat = strategyByKey(key) || STRATEGIES[0];
  const selected = sample.strategies[key]?.selected || [];
  const selectedSet = new Set(selected.map(x => x.pos));
  const proposalSet = new Set(sample.strategies['proposal_confidence']?.positions || []);
  const compareSet = compareKey && compareKey !== 'proposalVisual' ? new Set(sample.strategies[compareKey]?.positions || []) : new Set();
  document.querySelectorAll('.pill').forEach((p, i) => {{
    const active = STRATEGIES[i].key === key;
    p.classList.toggle('active', active);
    p.style.background = active ? STRATEGIES[i].color : '#fffdf7';
  }});
  metaEl.innerHTML = `<span><b>${{esc(sample.id)}}</b></span><span>idx=${{sample.dataset_index}}</span><span>topic=${{esc(sample.topic)}}</span><span>domain=${{esc(sample.domain)}}</span><span>gold=${{esc(sample.answer)}}</span><span>baseline_correct=${{sample.baseline_correct}}</span>`;
  questionEl.textContent = sample.question || '';
  const cats = {{}};
  selected.forEach(x => cats[x.cat] = (cats[x.cat] || 0) + 1);
  const semantic = (cats.content||0) + (cats.number||0) + (cats.mixednum||0) + (cats.option||0);
  const both = selected.filter(x => proposalSet.has(x.pos)).length;
  const proposalOnly = [...proposalSet].filter(pos => !selectedSet.has(pos)).length;
  const visualOnly = [...selectedSet].filter(pos => !proposalSet.has(pos)).length;
  statlineEl.innerHTML = `<div class="stat"><b>${{selected.length}}</b><span>selected tokens</span></div><div class="stat"><b>${{cats.content||0}}</b><span>content words</span></div><div class="stat"><b>${{both}}</b><span>shared with proposal</span></div><div class="stat"><b>${{proposalOnly}}</b><span>proposal-only</span></div><div class="stat"><b>${{visualOnly}}</b><span>selected-only</span></div>`;
  draftEl.innerHTML = sample.tokens.map(t => {{
    const inSelected = selectedSet.has(t.pos);
    const inProposal = proposalSet.has(t.pos);
    let cls = 'tok';
    let bg = 'transparent';
    if (compareKey === 'proposalVisual') {{
      if (inSelected && inProposal) cls += ' both';
      else if (inProposal) cls += ' proposalOnly';
      else if (inSelected) cls += ' visualOnly';
    }} else {{
      if (inSelected) {{ cls += ' primary'; bg = strat.color; }}
      if (compareSet.has(t.pos)) cls += ' compare';
    }}
    const title = `pos=${{t.pos}} token=${{t.clean}} cat=${{t.cat}} selected=${{inSelected}} proposal=${{inProposal}}`;
    return `<span class="${{cls}}" style="background:${{bg}}" title="${{esc(title)}}">${{esc(t.display || '·')}}</span>`;
  }}).join('');
  listEl.innerHTML = selected.slice().sort((a,b)=>a.pos-b.pos).map(x => `<code title="score=${{esc(x.score)}} visual_gain=${{esc(x.visual_gain)}}">${{x.pos}}:${{esc(x.token || '_')}}</code>`).join('');
}}
sampleSelect.onchange = render; strategySelect.onchange = render; compareSelect.onchange = render; render();
</script>
</body></html>"""
    OUT_HTML.write_text(html_text, encoding="utf-8")
    print(OUT_HTML)
    print(f"samples={len(sample_payload)} strategies={len(strategies)}")


if __name__ == "__main__":
    main()
