import argparse
import html
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path


QUADRANT_KEYS = [
    "high_conf_high_vis",
    "high_conf_low_vis",
    "low_conf_high_vis",
    "low_conf_low_vis",
]


SVG_COLORS = {
    "red": "#d94841",
    "gold": "#f2a900",
    "green": "#2f7d32",
    "ink": "#30404f",
    "muted": "#6b7785",
    "line": "#d7cfc2",
    "panel": "#fffdf8",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a standalone HTML report for M3CoT token visual sensitivity results."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--title", default="M3CoT Token Visual Sensitivity Report")
    parser.add_argument("--max-scatter-points", type=int, default=8000)
    parser.add_argument("--top-k-tokens", type=int, default=20)
    return parser.parse_args()


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def sample_points(points, limit, seed=42):
    if len(points) <= limit:
        return list(points)
    rng = random.Random(seed)
    return rng.sample(points, limit)


def escape(text):
    return html.escape(str(text))


def format_float(value, digits=4):
    return f"{value:.{digits}f}"


def token_is_content(item):
    return not item["is_whitespace"] and not item["is_punctuation"]


def build_summary(rows):
    token_records = [item for row in rows for item in row["token_records"]]
    conf = [item["confidence"] for item in token_records]
    vis = [item["visual_sensitivity"] for item in token_records]
    status = Counter(str(row.get("is_correct_by_letter")) for row in rows)
    return {
        "num_rows": len(rows),
        "num_tokens": len(token_records),
        "correct": status["True"],
        "incorrect": status["False"],
        "unparsed": status["None"],
        "median_conf": st.median(conf),
        "median_vis": st.median(vis),
        "high_conf_ratio": sum(value >= 0.9 for value in conf) / len(conf),
        "high_vis_ratio": sum(value > 0.2 for value in vis) / len(vis),
        "mean_hc_lv": sum(
            row["summary"]["high_conf_low_sensitivity_ratio"] for row in rows
        )
        / len(rows),
    }


def build_group_stats(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("is_correct_by_letter"))].append(row)

    result = []
    for key, label in [("True", "Correct"), ("False", "Incorrect"), ("None", "Unparsed")]:
        subset = grouped.get(key, [])
        if not subset:
            continue
        result.append(
            {
                "label": label,
                "count": len(subset),
                "mean_conf": sum(row["summary"]["mean_confidence"] for row in subset) / len(subset),
                "mean_vis": sum(row["summary"]["mean_visual_sensitivity"] for row in subset) / len(subset),
                "mean_hc_lv": sum(
                    row["summary"]["high_conf_low_sensitivity_ratio"] for row in subset
                )
                / len(subset),
            }
        )
    return result


def build_top_token_tables(rows, top_k):
    token_records = [item for row in rows for item in row["token_records"]]
    return {
        "hc_lv": Counter(
            item["token_text"] for item in token_records if item["is_high_conf_low_sensitivity"]
        ).most_common(top_k),
        "high_vis": Counter(
            item["token_text"] for item in token_records if item["visual_sensitivity"] > 0.5
        ).most_common(top_k),
        "content_high_vis": Counter(
            item["token_text"]
            for item in token_records
            if item["visual_sensitivity"] > 0.5 and token_is_content(item)
        ).most_common(top_k),
    }


def build_case_tables(rows):
    top_hc_lv = sorted(
        rows,
        key=lambda row: row["summary"]["high_conf_low_sensitivity_ratio"],
        reverse=True,
    )[:8]
    top_vis = sorted(
        rows,
        key=lambda row: row["summary"]["mean_visual_sensitivity"],
        reverse=True,
    )[:8]
    return top_hc_lv, top_vis


def svg_scatter(token_records, limit):
    points = sample_points(token_records, limit)
    width = 820
    height = 520
    left = 70
    right = 20
    top = 20
    bottom = 55
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = 0.0, 1.0
    y_min = min(item["visual_sensitivity"] for item in points)
    y_max = max(item["visual_sensitivity"] for item in points)
    if y_max - y_min < 1e-6:
        y_max = y_min + 1.0

    def x_map(value):
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y_map(value):
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    circles = []
    for item in points:
        if item["confidence"] >= 0.9 and item["visual_sensitivity"] > 0.2:
            color = SVG_COLORS["red"]
        elif item["confidence"] >= 0.9:
            color = SVG_COLORS["gold"]
        elif item["visual_sensitivity"] > 0.2:
            color = SVG_COLORS["green"]
        else:
            color = SVG_COLORS["ink"]
        circles.append(
            f"<circle cx='{x_map(item['confidence']):.2f}' cy='{y_map(item['visual_sensitivity']):.2f}' "
            f"r='2.2' fill='{color}' fill-opacity='0.28' />"
        )

    x_thr = x_map(0.9)
    y_thr = y_map(0.2)
    y_ticks = [y_min, (y_min + y_max) / 2.0, y_max]
    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='scatter plot'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='{SVG_COLORS['panel']}' rx='18' />",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{width - right}' y2='{top + plot_h}' stroke='{SVG_COLORS['muted']}' stroke-width='1.2' />",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='{SVG_COLORS['muted']}' stroke-width='1.2' />",
    ]
    for y in y_ticks:
        yy = y_map(y)
        parts.append(
            f"<line x1='{left}' y1='{yy:.2f}' x2='{width-right}' y2='{yy:.2f}' stroke='{SVG_COLORS['line']}' stroke-dasharray='3 4' />"
        )
        parts.append(
            f"<text x='{left-10}' y='{yy+4:.2f}' text-anchor='end' font-size='12' fill='{SVG_COLORS['muted']}'>{format_float(y,2)}</text>"
        )
    for x, label in [(0.0, "0.0"), (0.5, "0.5"), (0.9, "0.9"), (1.0, "1.0")]:
        xx = x_map(x)
        parts.append(
            f"<line x1='{xx:.2f}' y1='{top}' x2='{xx:.2f}' y2='{top+plot_h}' stroke='{SVG_COLORS['line']}' stroke-dasharray='3 4' />"
        )
        parts.append(
            f"<text x='{xx:.2f}' y='{height-18}' text-anchor='middle' font-size='12' fill='{SVG_COLORS['muted']}'>{label}</text>"
        )
    parts.extend(circles)
    parts.append(
        f"<line x1='{x_thr:.2f}' y1='{top}' x2='{x_thr:.2f}' y2='{top+plot_h}' stroke='{SVG_COLORS['red']}' stroke-width='1.5' stroke-dasharray='6 6' />"
    )
    parts.append(
        f"<line x1='{left}' y1='{y_thr:.2f}' x2='{width-right}' y2='{y_thr:.2f}' stroke='{SVG_COLORS['red']}' stroke-width='1.5' stroke-dasharray='6 6' />"
    )
    parts.append(
        f"<text x='{width/2:.2f}' y='{height-2}' text-anchor='middle' font-size='13' fill='{SVG_COLORS['ink']}'>Confidence at transfer time</text>"
    )
    parts.append(
        f"<text x='18' y='{height/2:.2f}' text-anchor='middle' font-size='13' fill='{SVG_COLORS['ink']}' transform='rotate(-90 18 {height/2:.2f})'>Visual sensitivity</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def svg_bar(labels, values, colors, title, width=760, height=360, horizontal=False):
    left = 70
    right = 20
    top = 25
    bottom = 55 if not horizontal else 35
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1.0)
    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='{SVG_COLORS['panel']}' rx='18' />",
        f"<text x='{width/2:.2f}' y='18' text-anchor='middle' font-size='15' fill='{SVG_COLORS['ink']}'>{escape(title)}</text>",
    ]

    if not horizontal:
        n = max(len(labels), 1)
        bar_w = plot_w / n * 0.72
        gap = plot_w / n
        for i, tick in enumerate([0, max_value / 2.0, max_value]):
            y = top + plot_h - (tick / max_value) * plot_h
            parts.append(
                f"<line x1='{left}' y1='{y:.2f}' x2='{width-right}' y2='{y:.2f}' stroke='{SVG_COLORS['line']}' stroke-dasharray='3 4' />"
            )
            parts.append(
                f"<text x='{left-10}' y='{y+4:.2f}' text-anchor='end' font-size='12' fill='{SVG_COLORS['muted']}'>{format_float(tick,1)}</text>"
            )
        for i, (label, value, color) in enumerate(zip(labels, values, colors)):
            x = left + i * gap + (gap - bar_w) / 2.0
            h = (value / max_value) * plot_h
            y = top + plot_h - h
            parts.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w:.2f}' height='{h:.2f}' fill='{color}' rx='8' />"
            )
            parts.append(
                f"<text x='{x + bar_w/2:.2f}' y='{top+plot_h+18}' text-anchor='middle' font-size='12' fill='{SVG_COLORS['ink']}'>{escape(label)}</text>"
            )
            parts.append(
                f"<text x='{x + bar_w/2:.2f}' y='{y-6:.2f}' text-anchor='middle' font-size='12' fill='{SVG_COLORS['ink']}'>{format_float(value,1)}</text>"
            )
    else:
        n = max(len(labels), 1)
        bar_h = plot_h / n * 0.62
        gap = plot_h / n
        for i, tick in enumerate([0, max_value / 2.0, max_value]):
            x = left + (tick / max_value) * plot_w
            parts.append(
                f"<line x1='{x:.2f}' y1='{top}' x2='{x:.2f}' y2='{top+plot_h}' stroke='{SVG_COLORS['line']}' stroke-dasharray='3 4' />"
            )
            parts.append(
                f"<text x='{x:.2f}' y='{height-10}' text-anchor='middle' font-size='12' fill='{SVG_COLORS['muted']}'>{format_float(tick,1)}</text>"
            )
        for i, (label, value, color) in enumerate(zip(labels, values, colors)):
            y = top + i * gap + (gap - bar_h) / 2.0
            w = (value / max_value) * plot_w
            parts.append(
                f"<text x='{left-8}' y='{y+bar_h/2+4:.2f}' text-anchor='end' font-size='12' fill='{SVG_COLORS['ink']}'>{escape(label)}</text>"
            )
            parts.append(
                f"<rect x='{left}' y='{y:.2f}' width='{w:.2f}' height='{bar_h:.2f}' fill='{color}' rx='8' />"
            )
            parts.append(
                f"<text x='{left+w+6:.2f}' y='{y+bar_h/2+4:.2f}' font-size='12' fill='{SVG_COLORS['ink']}'>{format_float(value,1)}</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


def svg_grouped_bar(group_stats, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get("is_correct_by_letter"))].append(row)
    labels = ["Correct", "Incorrect", "Unparsed"]
    status_keys = ["True", "False", "None"]
    series = []
    for quadrant in QUADRANT_KEYS:
        values = []
        for key in status_keys:
            subset = grouped.get(key, [])
            if not subset:
                values.append(0.0)
            else:
                values.append(
                    100.0
                    * sum(row["summary"]["quadrant_ratios"][quadrant] for row in subset)
                    / len(subset)
                )
        series.append((quadrant, values))

    width = 820
    height = 380
    left = 70
    right = 20
    top = 30
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max(max(values) for _, values in series)
    max_value = max(max_value, 1.0)
    colors = [SVG_COLORS["red"], SVG_COLORS["gold"], SVG_COLORS["green"], SVG_COLORS["muted"]]
    group_w = plot_w / len(labels)
    bar_w = group_w / 5
    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='correctness grouped bar chart'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='{SVG_COLORS['panel']}' rx='18' />",
        f"<text x='{width/2:.2f}' y='18' text-anchor='middle' font-size='15' fill='{SVG_COLORS['ink']}'>Quadrant Ratios by Final Answer Status</text>",
    ]
    for tick in [0, max_value / 2.0, max_value]:
        y = top + plot_h - (tick / max_value) * plot_h
        parts.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{width-right}' y2='{y:.2f}' stroke='{SVG_COLORS['line']}' stroke-dasharray='3 4' />"
        )
        parts.append(
            f"<text x='{left-10}' y='{y+4:.2f}' text-anchor='end' font-size='12' fill='{SVG_COLORS['muted']}'>{format_float(tick,1)}</text>"
        )
    for i, label in enumerate(labels):
        group_x = left + i * group_w
        parts.append(
            f"<text x='{group_x + group_w/2:.2f}' y='{height-18}' text-anchor='middle' font-size='12' fill='{SVG_COLORS['ink']}'>{escape(label)}</text>"
        )
        for j, (_, values) in enumerate(series):
            x = group_x + (j + 0.5) * bar_w
            h = (values[i] / max_value) * plot_h
            y = top + plot_h - h
            parts.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w*0.82:.2f}' height='{h:.2f}' fill='{colors[j]}' rx='6' />"
            )
    legend_x = width - 220
    legend_y = 32
    for idx, name in enumerate(["HC+HV", "HC+LV", "LC+HV", "LC+LV"]):
        yy = legend_y + idx * 18
        parts.append(
            f"<rect x='{legend_x}' y='{yy-10}' width='12' height='12' fill='{colors[idx]}' rx='3' />"
        )
        parts.append(
            f"<text x='{legend_x+18}' y='{yy}' font-size='12' fill='{SVG_COLORS['ink']}'>{name}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def render_token_rows(items):
    return "\n".join(
        "<tr>"
        f"<td><code>{escape(repr(token))}</code></td>"
        f"<td>{count}</td>"
        "</tr>"
        for token, count in items
    )


def render_case_rows(items, mode):
    rows = []
    for row in items:
        summary = row["summary"]
        metric = (
            summary["high_conf_low_sensitivity_ratio"]
            if mode == "hc_lv"
            else summary["mean_visual_sensitivity"]
        )
        rows.append(
            "<tr>"
            f"<td>{row['dataset_index']}</td>"
            f"<td>{escape(row.get('topic'))}</td>"
            f"<td>{escape(row.get('gold_answer'))}</td>"
            f"<td>{escape(row.get('parsed_prediction'))}</td>"
            f"<td>{escape(row.get('is_correct_by_letter'))}</td>"
            f"<td>{format_float(metric)}</td>"
            f"<td>{escape((row.get('question') or '')[:120])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_html(args, summary, group_stats, token_tables, top_hc_lv, top_vis, plots):
    group_cards = "\n".join(
        (
            "<div class='mini-card'>"
            f"<h4>{escape(item['label'])}</h4>"
            f"<p>n = {item['count']}</p>"
            f"<p>mean conf = {format_float(item['mean_conf'])}</p>"
            f"<p>mean vis = {format_float(item['mean_vis'])}</p>"
            f"<p>mean HC+LV = {format_float(item['mean_hc_lv'])}</p>"
            "</div>"
        )
        for item in group_stats
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(args.title)}</title>
  <style>
    :root {{
      --bg: #f7f3ea;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #5b6c7d;
      --line: #d7cfc2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(242,169,0,0.12), transparent 28%),
        radial-gradient(circle at top left, rgba(217,72,65,0.08), transparent 25%),
        var(--bg);
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 22px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,253,248,0.96), rgba(247,243,234,0.96));
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 18px 50px rgba(57, 49, 36, 0.08);
    }}
    h1, h2, h3, h4 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 34px; }}
    h2 {{ font-size: 22px; margin-top: 28px; }}
    h3 {{ font-size: 18px; }}
    p {{ margin: 0; line-height: 1.5; }}
    .lede {{ color: var(--muted); max-width: 960px; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .card, .mini-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px;
    }}
    .card .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .card .value {{
      font-size: 30px;
      font-weight: 700;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}
    .plot-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      overflow: hidden;
    }}
    .plot-card svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .mini-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid var(--line);
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid #ece4d6;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      background: #f3ecdf;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
    }}
    .section {{
      margin-top: 26px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{escape(args.title)}</h1>
      <p class="lede">
        Report generated from <code>{escape(args.input_jsonl)}</code>. This summarizes token-level transfer-time
        confidence and synchronized conditional-vs-null-visual sensitivity on M3CoT.
      </p>
      <div class="stats">
        <div class="card"><div class="label">Samples</div><div class="value">{summary['num_rows']}</div></div>
        <div class="card"><div class="label">Tokens</div><div class="value">{summary['num_tokens']}</div></div>
        <div class="card"><div class="label">Median Confidence</div><div class="value">{format_float(summary['median_conf'], 3)}</div></div>
        <div class="card"><div class="label">Median Vis</div><div class="value">{format_float(summary['median_vis'], 3)}</div></div>
        <div class="card"><div class="label">High-Conf Ratio</div><div class="value">{format_float(summary['high_conf_ratio'] * 100, 1)}%</div></div>
        <div class="card"><div class="label">High-Vis Ratio</div><div class="value">{format_float(summary['high_vis_ratio'] * 100, 1)}%</div></div>
        <div class="card"><div class="label">Avg HC+LV</div><div class="value">{format_float(summary['mean_hc_lv'] * 100, 1)}%</div></div>
        <div class="card"><div class="label">Answer Status</div><div class="value">{summary['correct']} / {summary['incorrect']} / {summary['unparsed']}</div></div>
      </div>
    </section>

    <section class="section">
      <h2>Plots</h2>
      <div class="grid-2">
        <div class="plot-card">
          <h3>Confidence vs Visual Sensitivity</h3>
          {plots['scatter']}
        </div>
        <div class="plot-card">
          <h3>Average Quadrant Mix</h3>
          {plots['quadrant']}
        </div>
        <div class="plot-card">
          <h3>Correct vs Incorrect</h3>
          {plots['correctness']}
        </div>
        <div class="plot-card">
          <h3>Topic-Level HC+LV</h3>
          {plots['topic']}
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Status Breakdown</h2>
      <div class="mini-grid">
        {group_cards}
      </div>
    </section>

    <section class="section grid-2">
      <div>
        <h2>Top High-Conf Low-Vis Tokens</h2>
        <table>
          <thead><tr><th>Token</th><th>Count</th></tr></thead>
          <tbody>{render_token_rows(token_tables['hc_lv'])}</tbody>
        </table>
      </div>
      <div>
        <h2>Top High-Vis Tokens</h2>
        <table>
          <thead><tr><th>Token</th><th>Count</th></tr></thead>
          <tbody>{render_token_rows(token_tables['high_vis'])}</tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>Top High-Vis Content Tokens</h2>
      <table>
        <thead><tr><th>Token</th><th>Count</th></tr></thead>
        <tbody>{render_token_rows(token_tables['content_high_vis'])}</tbody>
      </table>
    </section>

    <section class="section grid-2">
      <div>
        <h2>Highest HC+LV Samples</h2>
        <table>
          <thead><tr><th>Idx</th><th>Topic</th><th>Gold</th><th>Pred</th><th>Status</th><th>HC+LV</th><th>Question</th></tr></thead>
          <tbody>{render_case_rows(top_hc_lv, 'hc_lv')}</tbody>
        </table>
      </div>
      <div>
        <h2>Highest Mean-Vis Samples</h2>
        <table>
          <thead><tr><th>Idx</th><th>Topic</th><th>Gold</th><th>Pred</th><th>Status</th><th>Mean Vis</th><th>Question</th></tr></thead>
          <tbody>{render_case_rows(top_vis, 'vis')}</tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>
"""


def main():
    args = parse_args()
    rows = load_rows(args.input_jsonl)
    summary = build_summary(rows)
    group_stats = build_group_stats(rows)
    token_tables = build_top_token_tables(rows, args.top_k_tokens)
    top_hc_lv, top_vis = build_case_tables(rows)

    quadrant_means = {
        key: 100.0
        * sum(row["summary"]["quadrant_ratios"][key] for row in rows)
        / len(rows)
        for key in QUADRANT_KEYS
    }
    topic_groups = defaultdict(list)
    for row in rows:
        topic_groups[row.get("topic") or "unknown"].append(row)
    top_topics = sorted(topic_groups.items(), key=lambda item: len(item[1]), reverse=True)[:8]
    topic_labels = [topic for topic, _ in top_topics]
    topic_values = [
        100.0
        * sum(r["summary"]["high_conf_low_sensitivity_ratio"] for r in subset)
        / len(subset)
        for _, subset in top_topics
    ]

    plots = {
        "scatter": svg_scatter([item for row in rows for item in row["token_records"]], args.max_scatter_points),
        "quadrant": svg_bar(
            ["HC+HV", "HC+LV", "LC+HV", "LC+LV"],
            [quadrant_means[key] for key in QUADRANT_KEYS],
            [SVG_COLORS["red"], SVG_COLORS["gold"], SVG_COLORS["green"], SVG_COLORS["muted"]],
            "Average Token Quadrant Ratios per Sample",
        ),
        "correctness": svg_grouped_bar(group_stats, rows),
        "topic": svg_bar(
            topic_labels,
            topic_values,
            [SVG_COLORS["gold"]] * len(topic_labels),
            "High-Confidence Low-Visual Ratio by Topic",
            width=860,
            height=420,
            horizontal=True,
        ),
    }

    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_html(args, summary, group_stats, token_tables, top_hc_lv, top_vis, plots),
        encoding="utf-8",
    )
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
