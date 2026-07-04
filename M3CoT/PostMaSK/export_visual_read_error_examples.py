import argparse
import json
from html import escape
from pathlib import Path


SELECTIONS = [
    {
        "dataset_index": 8,
        "id": "mathematics-1649",
        "category": "schedule_timetable",
        "reason": "把时刻表里的电影开始时间读错了，属于典型的表格时间读取错误。",
    },
    {
        "dataset_index": 9,
        "id": "cognitive-science-264",
        "category": "object_recognition",
        "reason": "根据图形外观把物体识别错了，属于整体轮廓感知错误。",
    },
    {
        "dataset_index": 22,
        "id": "biology-1745",
        "category": "species_recognition",
        "reason": "把主图里的青蛙物种认错了，属于生物视觉识别错误。",
    },
    {
        "dataset_index": 24,
        "id": "physical-commonsense-889",
        "category": "physical_state_reading",
        "reason": "把不同容器/液体的图中对应关系看错了，导致判断哪一杯先凝固时出错。",
    },
    {
        "dataset_index": 26,
        "id": "physical-commonsense-578",
        "category": "physical_state_reading",
        "reason": "对图中容器温度状态的读取有偏差，进而错判哪一个最先结露。",
    },
    {
        "dataset_index": 27,
        "id": "biology-2108",
        "category": "species_recognition",
        "reason": "把主图里的鸟类外观看错，映射到了错误物种。",
    },
    {
        "dataset_index": 30,
        "id": "social-commonsense-1148",
        "category": "scene_inference",
        "reason": "对场景中人物活动的视觉理解偏了，把场景类型判断错了。",
    },
    {
        "dataset_index": 36,
        "id": "social-commonsense-1648",
        "category": "scene_inference",
        "reason": "根据人物姿态和手部动作做了错误视觉判断，把行为理解成自拍。",
    },
    {
        "dataset_index": 46,
        "id": "geography-3579",
        "category": "map_chart_reading",
        "reason": "地形图颜色带和最南端位置的对应关系读错了。",
    },
    {
        "dataset_index": 48,
        "id": "geography-3721",
        "category": "map_chart_reading",
        "reason": "地形图上高程颜色和点位的对应关系看错了。",
    },
    {
        "dataset_index": 54,
        "id": "mathematics-1552",
        "category": "schedule_timetable",
        "reason": "把时刻表里下一班的等待时长读错了。",
    },
    {
        "dataset_index": 55,
        "id": "synthesis-problem-202",
        "category": "scene_inference",
        "reason": "根据人物姿态和表情做了错误视觉推断，把场景理解成在摆拍。",
    },
    {
        "dataset_index": 57,
        "id": "physical-commonsense-187",
        "category": "counting_or_mapping",
        "reason": "图中物体与密度条件的对应关系没有看准，导致浮起数量判断错。",
    },
    {
        "dataset_index": 58,
        "id": "social-commonsense-747",
        "category": "scene_inference",
        "reason": "对人物所处环境和动作线索的视觉解读偏了。",
    },
    {
        "dataset_index": 65,
        "id": "social-commonsense-270",
        "category": "scene_inference",
        "reason": "对酒吧场景细节的视觉理解偏差，导致错判人物意图。",
    },
    {
        "dataset_index": 71,
        "id": "social-commonsense-395",
        "category": "scene_inference",
        "reason": "对雪地防护栏用途的视觉推断偏了，没抓住场景中真正的风险对象。",
    },
    {
        "dataset_index": 81,
        "id": "geography-3067",
        "category": "map_chart_reading",
        "reason": "海洋深度图中最东点与颜色条的对应关系读错了。",
    },
    {
        "dataset_index": 89,
        "id": "cognitive-science-70",
        "category": "tangram_shape",
        "reason": "把 tangram 的整体形状看成了错误物体。",
    },
    {
        "dataset_index": 95,
        "id": "cognitive-science-380",
        "category": "object_part_recognition",
        "reason": "对鸟形图中被强调的部位识别错了。",
    },
    {
        "dataset_index": 134,
        "id": "geography-3548",
        "category": "map_chart_reading",
        "reason": "地形图最北端点位和色带对应的高度读错了。",
    },
    {
        "dataset_index": 142,
        "id": "geography-3038",
        "category": "map_chart_reading",
        "reason": "海洋深度图中最南端位置的读图结果有误。",
    },
    {
        "dataset_index": 148,
        "id": "cognitive-science-385",
        "category": "tangram_shape",
        "reason": "对 tangram 组成的整体轮廓理解偏了。",
    },
    {
        "dataset_index": 156,
        "id": "cognitive-science-534",
        "category": "tangram_shape",
        "reason": "把 tangram 形成的动物类别识别错了。",
    },
    {
        "dataset_index": 190,
        "id": "geography-3120",
        "category": "map_chart_reading",
        "reason": "海洋深度图最北点对应的颜色层级读取错误。",
    },
    {
        "dataset_index": 210,
        "id": "geography-3725",
        "category": "map_chart_reading",
        "reason": "地形图最低点对应的颜色区域判断错了。",
    },
    {
        "dataset_index": 246,
        "id": "geography-3680",
        "category": "map_chart_reading",
        "reason": "地形图上最高点所在区域读错，属于典型点位-颜色映射错误。",
    },
    {
        "dataset_index": 264,
        "id": "social-commonsense-624",
        "category": "scene_inference",
        "reason": "对环境类型的视觉判断偏差，把场景类别理解错了。",
    },
    {
        "dataset_index": 298,
        "id": "social-commonsense-1287",
        "category": "scene_inference",
        "reason": "对场景所属活动类型的视觉理解偏差，把环境误判成烹饪课。",
    },
    {
        "dataset_index": 301,
        "id": "cognitive-science-208",
        "category": "tangram_shape",
        "reason": "把 tangram 形状整体看成了错误物体。",
    },
    {
        "dataset_index": 305,
        "id": "cognitive-science-167",
        "category": "tangram_shape",
        "reason": "对图形组成的对象类别识别偏差较明显。",
    },
]


CATEGORY_LABELS = {
    "schedule_timetable": "Schedule / Timetable",
    "object_recognition": "Object Recognition",
    "species_recognition": "Species Recognition",
    "physical_state_reading": "Physical State Reading",
    "scene_inference": "Scene Inference",
    "map_chart_reading": "Map / Chart Reading",
    "tangram_shape": "Tangram Shape",
    "object_part_recognition": "Object Part Recognition",
    "counting_or_mapping": "Counting / Mapping",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export 30 curated draft errors that look like visual-read failures."
    )
    parser.add_argument(
        "--records",
        default="M3CoT/PostMaSK/outputs/postmask_sr0p5_d16_p16_conf_r4_seed42_n400/records.jsonl",
    )
    parser.add_argument(
        "--labels",
        default="M3CoT/PostMaSK/draft_semantic_error_labels.jsonl",
    )
    parser.add_argument(
        "--output-json",
        default="M3CoT/PostMaSK/visual_read_error_examples.json",
    )
    parser.add_argument(
        "--output-html",
        default="M3CoT/PostMaSK/visual_read_error_examples.html",
    )
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            rows.append(json.loads(line))
    return rows


def build_records_map(records):
    return {(row["dataset_index"], row["id"]): row for row in records}


def normalize_text(text):
    return " ".join(text.split())


def assemble_examples(records_map, labels_map):
    examples = []
    missing = []
    for item in SELECTIONS:
        key = (item["dataset_index"], item["id"])
        record = records_map.get(key)
        label = labels_map.get(key)
        if record is None or label is None:
            missing.append(key)
            continue
        examples.append(
            {
                "dataset_index": item["dataset_index"],
                "id": item["id"],
                "category": item["category"],
                "category_label": CATEGORY_LABELS[item["category"]],
                "reason": item["reason"],
                "topic": label.get("topic"),
                "domain": label.get("domain"),
                "gold": label.get("gold"),
                "pred": label.get("pred"),
                "extract_source": label.get("extract_source"),
                "question": record.get("question", ""),
                "choices": record.get("choices", []),
                "draft_text": record.get("draft_text", ""),
                "final_text": record.get("final_text", ""),
                "local_span": label.get("local_span", ""),
            }
        )
    if missing:
        raise RuntimeError(f"Missing records for selections: {missing}")
    return examples


def build_html(examples):
    category_counts = {}
    for item in examples:
        category_counts[item["category_label"]] = category_counts.get(item["category_label"], 0) + 1

    filter_buttons = "\n".join(
        f'<button class="filter-btn" data-filter="{escape(key)}">{escape(key)} ({count})</button>'
        for key, count in sorted(category_counts.items())
    )

    cards = []
    for item in examples:
        choices_html = "".join(
            f"<li><strong>{chr(65 + idx)}.</strong> {escape(choice)}</li>"
            for idx, choice in enumerate(item["choices"])
        )
        cards.append(
            f"""
<article class="card" data-category="{escape(item['category_label'])}">
  <div class="card-head">
    <div>
      <div class="eyebrow">{escape(item['category_label'])}</div>
      <h2>{escape(item['id'])}</h2>
    </div>
    <div class="meta">
      <span>idx {item['dataset_index']}</span>
      <span>{escape(item['topic'] or 'UNKNOWN')}</span>
      <span>pred {escape(item['pred'])} / gold {escape(item['gold'])}</span>
    </div>
  </div>
  <p class="reason">{escape(item['reason'])}</p>
  <div class="block">
    <h3>Question</h3>
    <p>{escape(item['question'])}</p>
  </div>
  <div class="grid">
    <section class="block">
      <h3>Choices</h3>
      <ol>{choices_html}</ol>
    </section>
    <section class="block">
      <h3>Why Counted As Visual Read Error</h3>
      <p>{escape(item['reason'])}</p>
      <p><strong>Local span:</strong> {escape(normalize_text(item['local_span']))}</p>
      <p><strong>Extract source:</strong> <code>{escape(item['extract_source'])}</code></p>
    </section>
  </div>
  <div class="block">
    <h3>Draft Text</h3>
    <pre>{escape(item['draft_text'])}</pre>
  </div>
</article>
""".strip()
        )

    cards_html = "\n\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Visual Read Error Examples</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf2;
      --panel-2: #f7efe1;
      --ink: #1f1a17;
      --muted: #6f645b;
      --accent: #9f3d22;
      --accent-2: #2f6d62;
      --line: #d8c6b0;
      --shadow: 0 18px 40px rgba(55, 34, 19, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Georgia", "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(159, 61, 34, 0.10), transparent 32%),
        radial-gradient(circle at top right, rgba(47, 109, 98, 0.10), transparent 28%),
        linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
    }}
    .wrap {{
      width: min(1200px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,250,242,0.95), rgba(247,239,225,0.95));
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 28px;
      box-shadow: var(--shadow);
      margin-bottom: 20px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: clamp(30px, 4vw, 48px);
      line-height: 1.02;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 8px 0;
      color: var(--muted);
      max-width: 900px;
      line-height: 1.6;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .chip {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.7);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 14px;
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 20px 0 24px;
    }}
    .filter-btn {{
      appearance: none;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      padding: 10px 14px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
      transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
    }}
    .filter-btn:hover {{
      transform: translateY(-1px);
      border-color: var(--accent);
    }}
    .filter-btn.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff8f0;
    }}
    .cards {{
      display: grid;
      gap: 18px;
    }}
    .card {{
      background: rgba(255, 250, 242, 0.96);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 22px;
      box-shadow: var(--shadow);
    }}
    .card[hidden] {{ display: none; }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 8px;
    }}
    .card-head h2 {{
      margin: 2px 0 0;
      font-size: 24px;
      line-height: 1.15;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--accent-2);
      font-size: 12px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }}
    .meta span {{
      font-size: 13px;
      color: var(--muted);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      white-space: nowrap;
    }}
    .reason {{
      margin: 10px 0 18px;
      padding-left: 14px;
      border-left: 3px solid var(--accent);
      line-height: 1.6;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .block {{
      background: rgba(255,255,255,0.6);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 14px;
    }}
    .block h3 {{
      margin: 0 0 10px;
      font-size: 15px;
      letter-spacing: 0.02em;
    }}
    .block p, .block li {{
      line-height: 1.6;
    }}
    .block ol {{
      margin: 0;
      padding-left: 20px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 13px;
      line-height: 1.6;
      color: #332720;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 12px;
    }}
    @media (max-width: 820px) {{
      .wrap {{ width: min(100vw - 20px, 1200px); padding-top: 20px; }}
      .hero {{ padding: 22px; border-radius: 18px; }}
      .card {{ padding: 18px; border-radius: 18px; }}
      .card-head {{ flex-direction: column; }}
      .meta {{ justify-content: flex-start; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>30 Curated Visual-Read Draft Errors</h1>
      <p>These examples are manually curated from <code>postmask_sr0p5_d16_p16_conf_r4_seed42_n400</code>. The goal here is narrower than generic draft errors: we only keep cases that look like “没看清楚图 / 读图读错” rather than truncation, extraction artifacts, or purely symbolic reasoning mistakes.</p>
      <p>Selection policy: start from draft errors already labeled as semantic <code>local</code>, then keep cases where the wrong step is best explained by reading the image, chart, map, scene, schedule, tangram, or visual attribute incorrectly.</p>
      <div class="summary">
        <span class="chip">Total examples: {len(examples)}</span>
        <span class="chip">Source records: postmask_sr0p5_d16_p16_conf_r4_seed42_n400</span>
        <span class="chip">Only visual-read style cases</span>
      </div>
    </section>

    <div class="filters">
      <button class="filter-btn active" data-filter="ALL">ALL ({len(examples)})</button>
      {filter_buttons}
    </div>

    <section class="cards">
      {cards_html}
    </section>
  </div>

  <script>
    const buttons = [...document.querySelectorAll('.filter-btn')];
    const cards = [...document.querySelectorAll('.card')];
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        buttons.forEach((b) => b.classList.remove('active'));
        button.classList.add('active');
        const filter = button.dataset.filter;
        cards.forEach((card) => {{
          card.hidden = filter !== 'ALL' && card.dataset.category !== filter;
        }});
      }});
    }});
  </script>
</body>
</html>
"""


def main():
    args = parse_args()
    records = load_jsonl(Path(args.records))
    labels = load_jsonl(Path(args.labels))
    records_map = build_records_map(records)
    labels_map = {(row["dataset_index"], row["id"]): row for row in labels}
    examples = assemble_examples(records_map, labels_map)

    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    output_json.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    output_html.write_text(build_html(examples), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
