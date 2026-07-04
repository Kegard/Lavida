#!/usr/bin/env python
import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize token-switch events emitted by analyze_fullstage_vrg_token_switches.py."
    )
    parser.add_argument("--events-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=200)
    return parser.parse_args()


def load_json(path):
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path):
    events = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                events.append(json.loads(line))
    return events


def transition_key(event):
    return (
        int(event["cond_token_id"]),
        event.get("cond_token_text", ""),
        int(event["vrg_token_id"]),
        event.get("vrg_token_text", ""),
    )


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_stat(stats, key, value):
    value = safe_float(value)
    if value is None:
        return
    item = stats[key]
    item["sum"] += value
    item["count"] += 1
    item["min"] = value if item["min"] is None else min(item["min"], value)
    item["max"] = value if item["max"] is None else max(item["max"], value)


def stat_row(stats, key, prefix):
    item = stats.get(key)
    if not item or item["count"] == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_mean": item["sum"] / item["count"],
        f"{prefix}_min": item["min"],
        f"{prefix}_max": item["max"],
    }


def fmt_float(value, digits=3):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def summarize_events(events, top_k):
    by_scope = {}
    for scope in sorted({event.get("scope", "unknown") for event in events}):
        scope_events = [event for event in events if event.get("scope", "unknown") == scope]
        step_counts = Counter()
        position_counts = Counter()
        transition_counts = Counter()
        step_transition_counts = defaultdict(Counter)
        step_position_counts = defaultdict(Counter)
        position_transition_counts = defaultdict(Counter)
        transition_boost_stats = defaultdict(lambda: {"sum": 0.0, "count": 0, "min": None, "max": None})
        step_boost_stats = defaultdict(lambda: {"sum": 0.0, "count": 0, "min": None, "max": None})
        position_boost_stats = defaultdict(lambda: {"sum": 0.0, "count": 0, "min": None, "max": None})
        step_position_examples = defaultdict(list)

        for event in scope_events:
            step = int(event["step"])
            pos = int(event["answer_position"])
            key = transition_key(event)
            boost = event.get("guided_minus_cond_on_vrg_token")
            step_counts[step] += 1
            position_counts[pos] += 1
            transition_counts[key] += 1
            step_transition_counts[step][key] += 1
            step_position_counts[step][pos] += 1
            position_transition_counts[pos][key] += 1
            add_stat(transition_boost_stats, key, boost)
            add_stat(step_boost_stats, step, boost)
            add_stat(position_boost_stats, pos, boost)
            pair_key = (step, pos)
            if len(step_position_examples[pair_key]) < 10:
                step_position_examples[pair_key].append(event)

        step_rows = []
        for step in sorted(step_counts):
            top_transitions = [
                {
                    "cond_token_id": key[0],
                    "cond_token_text": key[1],
                    "vrg_token_id": key[2],
                    "vrg_token_text": key[3],
                    "count": count,
                    **stat_row(transition_boost_stats, key, "target_boost"),
                }
                for key, count in step_transition_counts[step].most_common(top_k)
            ]
            top_positions = [
                {"answer_position": pos, "count": count}
                for pos, count in step_position_counts[step].most_common(top_k)
            ]
            step_rows.append(
                {
                    "step": step,
                    "switch_count": step_counts[step],
                    **stat_row(step_boost_stats, step, "target_boost"),
                    "top_positions": top_positions,
                    "top_transitions": top_transitions,
                }
            )

        position_rows = []
        for pos, count in position_counts.most_common(top_k):
            position_rows.append(
                {
                    "answer_position": pos,
                    "switch_count": count,
                    "top_transitions": [
                        {
                            "cond_token_id": key[0],
                            "cond_token_text": key[1],
                            "vrg_token_id": key[2],
                            "vrg_token_text": key[3],
                            "count": trans_count,
                            **stat_row(transition_boost_stats, key, "target_boost"),
                        }
                        for key, trans_count in position_transition_counts[pos].most_common(top_k)
                    ],
                    **stat_row(position_boost_stats, pos, "target_boost"),
                }
            )

        transition_rows = [
            {
                "cond_token_id": key[0],
                "cond_token_text": key[1],
                "vrg_token_id": key[2],
                "vrg_token_text": key[3],
                "count": count,
                **stat_row(transition_boost_stats, key, "target_boost"),
            }
            for key, count in transition_counts.most_common(top_k)
        ]

        step_position_rows = []
        for (step, pos), examples in sorted(
            step_position_examples.items(),
            key=lambda item: (len(item[1]), -item[0][0], -item[0][1]),
            reverse=True,
        )[:top_k]:
            transition_counter = Counter(transition_key(event) for event in examples)
            step_position_rows.append(
                {
                    "step": step,
                    "answer_position": pos,
                    "example_count_saved": len(examples),
                    "sample_examples": [
                        {
                            "sample_id": event.get("sample_id"),
                            "dataset_index": event.get("dataset_index"),
                            "cond_token_text": event.get("cond_token_text"),
                            "vrg_token_text": event.get("vrg_token_text"),
                            "final_correct": event.get("final_correct"),
                        }
                        for event in examples
                    ],
                    "saved_transition_counts": [
                        {
                            "cond_token_id": key[0],
                            "cond_token_text": key[1],
                            "vrg_token_id": key[2],
                            "vrg_token_text": key[3],
                            "count": count,
                            **stat_row(transition_boost_stats, key, "target_boost"),
                        }
                        for key, count in transition_counter.most_common(top_k)
                    ],
                }
            )

        by_scope[scope] = {
            "scope": scope,
            "num_events": len(scope_events),
            "step_summary": step_rows,
            "position_summary": position_rows,
            "transition_summary": transition_rows,
            "step_position_examples": step_position_rows,
        }
    return by_scope


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def flatten_step_rows(scope_summary):
    rows = []
    for item in scope_summary["step_summary"]:
        rows.append(
            {
                "step": item["step"],
                "switch_count": item["switch_count"],
                "target_boost_mean": item.get("target_boost_mean"),
                "target_boost_min": item.get("target_boost_min"),
                "target_boost_max": item.get("target_boost_max"),
                "top_positions": json.dumps(item["top_positions"][:20], ensure_ascii=False),
                "top_transitions": json.dumps(item["top_transitions"][:20], ensure_ascii=False),
            }
        )
    return rows


def flatten_position_rows(scope_summary):
    rows = []
    for item in scope_summary["position_summary"]:
        rows.append(
            {
                "answer_position": item["answer_position"],
                "switch_count": item["switch_count"],
                "target_boost_mean": item.get("target_boost_mean"),
                "target_boost_min": item.get("target_boost_min"),
                "target_boost_max": item.get("target_boost_max"),
                "top_transitions": json.dumps(item["top_transitions"][:20], ensure_ascii=False),
            }
        )
    return rows


def render_html(report, output_path):
    def esc(value):
        return html.escape(str(value))

    sections = []
    for scope, scope_summary in report["scopes"].items():
        step_rows = []
        for row in scope_summary["step_summary"]:
            top_pos = ", ".join(
                f"{item['answer_position']}:{item['count']}" for item in row["top_positions"][:10]
            )
            top_trans = ", ".join(
                f"{item['cond_token_text']} -> {item['vrg_token_text']} "
                f"({item['count']}, boost {fmt_float(item.get('target_boost_mean'))})"
                for item in row["top_transitions"][:8]
            )
            step_rows.append(
                "<tr>"
                f"<td>{row['step']}</td>"
                f"<td>{row['switch_count']}</td>"
                f"<td>{fmt_float(row.get('target_boost_mean'))}</td>"
                f"<td>{esc(top_pos)}</td>"
                f"<td>{esc(top_trans)}</td>"
                "</tr>"
            )

        trans_rows = []
        for idx, row in enumerate(scope_summary["transition_summary"][:100], start=1):
            trans_rows.append(
                "<tr>"
                f"<td>{idx}</td>"
                f"<td><code>{row['cond_token_id']}</code></td>"
                f"<td>{esc(row['cond_token_text'])}</td>"
                f"<td><code>{row['vrg_token_id']}</code></td>"
                f"<td>{esc(row['vrg_token_text'])}</td>"
                f"<td>{row['count']}</td>"
                f"<td>{fmt_float(row.get('target_boost_mean'))}</td>"
                f"<td>{fmt_float(row.get('target_boost_min'))}</td>"
                f"<td>{fmt_float(row.get('target_boost_max'))}</td>"
                "</tr>"
            )

        sections.append(
            f"""
            <section class="panel">
              <h2>{esc(scope)} scope</h2>
              <p>Total events: <b>{scope_summary['num_events']}</b></p>
              <p><code>target boost</code> = <code>guided_logit(vrg_token) - cond_logit(vrg_token)</code>，只覆盖已经发生 top1 switch 的 events；不包含 <code>cond_top1 == guided_top1</code> 的 non-switch boost。</p>
              <h3>Switches by step</h3>
              <table><thead><tr><th>step</th><th>switches</th><th>mean target boost</th><th>top positions</th><th>top transitions</th></tr></thead><tbody>{''.join(step_rows)}</tbody></table>
              <h3>Top token transitions</h3>
              <table><thead><tr><th>#</th><th>from id</th><th>from token</th><th>to id</th><th>to token</th><th>count</th><th>mean target boost</th><th>min</th><th>max</th></tr></thead><tbody>{''.join(trans_rows)}</tbody></table>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VRG Token Switch Step Summary</title>
<style>
body{{margin:0;background:#f7f8fa;color:#17191f;font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}}
.shell{{width:min(1400px,calc(100vw - 32px));margin:24px auto 48px}}
.panel{{background:white;border:1px solid #d8dde6;border-radius:8px;padding:14px;margin-bottom:14px;overflow:auto}}
h1{{margin:0 0 8px;font-size:28px}} h2{{font-size:19px}} h3{{font-size:15px;margin-top:18px}}
p{{color:#667085}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{border-bottom:1px solid #e4e8f0;padding:7px 8px;text-align:left;vertical-align:top}} th{{background:#f1f4f8;color:#344054;position:sticky;top:0}}
code{{background:#f1f4f8;border:1px solid #d8dde6;border-radius:5px;padding:1px 5px}}
</style>
</head>
<body><div class="shell">
<header><h1>VRG Token Switch Step Summary</h1><p>Source events: <code>{esc(report['events_jsonl'])}</code></p></header>
{''.join(sections)}
</div></body></html>"""
    output_path.write_text(html_text, encoding="utf-8")


def main():
    args = parse_args()
    events = load_events(args.events_jsonl)
    trace_summary = load_json(args.summary_json)
    scopes = summarize_events(events, args.top_k)
    report = {
        "events_jsonl": str(args.events_jsonl),
        "trace_summary_json": str(args.summary_json) if args.summary_json else None,
        "trace_summary": trace_summary,
        "num_events": len(events),
        "scopes": scopes,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "step_position_transition_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for scope, scope_summary in scopes.items():
        write_csv(
            args.output_dir / f"{scope}_step_summary.csv",
            flatten_step_rows(scope_summary),
            [
                "step",
                "switch_count",
                "target_boost_mean",
                "target_boost_min",
                "target_boost_max",
                "top_positions",
                "top_transitions",
            ],
        )
        write_csv(
            args.output_dir / f"{scope}_position_summary.csv",
            flatten_position_rows(scope_summary),
            [
                "answer_position",
                "switch_count",
                "target_boost_mean",
                "target_boost_min",
                "target_boost_max",
                "top_transitions",
            ],
        )
        write_csv(
            args.output_dir / f"{scope}_transition_summary.csv",
            scope_summary["transition_summary"],
            [
                "cond_token_id",
                "cond_token_text",
                "vrg_token_id",
                "vrg_token_text",
                "count",
                "target_boost_mean",
                "target_boost_min",
                "target_boost_max",
            ],
        )
    render_html(report, args.output_dir / "step_position_transition_summary.html")
    print(f"Wrote summarized switch reports to {args.output_dir}")


if __name__ == "__main__":
    main()
