import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze step-wise x0 TextVQA trajectories and visualize case proportions."
    )
    parser.add_argument(
        "--input-jsonl",
        default="Draft/textvqa_stepwise_eval_samples.jsonl",
        help="Sample-level step-wise evaluation output.",
    )
    parser.add_argument(
        "--output-summary-json",
        default="Draft/textvqa_stepwise_case_summary.json",
        help="Summary JSON with counts, ratios, and step metrics.",
    )
    parser.add_argument(
        "--output-plot",
        default="Draft/textvqa_stepwise_case_summary.png",
        help="Visualization figure path.",
    )
    parser.add_argument(
        "--output-labeled-jsonl",
        default="Draft/textvqa_stepwise_eval_samples_labeled.jsonl",
        help="Optional labeled sample output.",
    )
    parser.add_argument(
        "--correct-threshold",
        type=float,
        default=0.0,
        help="Treat exact_match > threshold as correct. Default follows TextVQA soft score > 0.",
    )
    parser.add_argument(
        "--max-examples-per-case",
        type=int,
        default=5,
        help="How many example sample ids to keep per case in the summary.",
    )
    return parser.parse_args()


def is_correct(score, threshold):
    return float(score) > threshold


def first_true_index(flags):
    for idx, flag in enumerate(flags, start=1):
        if flag:
            return idx
    return None


def first_change_index(values):
    if not values:
        return None
    first_value = values[0]
    for idx, value in enumerate(values[1:], start=2):
        if value != first_value:
            return idx
    return None


def count_transitions(flags):
    if not flags:
        return 0
    return sum(1 for prev, curr in zip(flags[:-1], flags[1:]) if prev != curr)


def classify_case(step_results, threshold):
    scores = [float(item["exact_match"]) for item in step_results]
    predictions = [item.get("normalized_prediction", item.get("candidate_text", "")) for item in step_results]
    correct_flags = [is_correct(score, threshold) for score in scores]

    first_correct = correct_flags[0]
    final_correct = correct_flags[-1]
    prediction_changed = any(pred != predictions[0] for pred in predictions[1:])
    transition_count = count_transitions(correct_flags)

    if first_correct and final_correct:
        if prediction_changed:
            case_name = "correct_to_correct_changing"
        else:
            case_name = "correct_to_correct_stable"
    elif (not first_correct) and (not final_correct):
        if prediction_changed:
            case_name = "wrong_to_wrong_changing"
        else:
            case_name = "wrong_to_wrong_stable"
    elif (not first_correct) and final_correct:
        case_name = "wrong_to_correct"
    else:
        case_name = "correct_to_wrong"

    return {
        "case": case_name,
        "scores": scores,
        "predictions": predictions,
        "correct_flags": correct_flags,
        "first_correct_step": first_true_index(correct_flags),
        "first_prediction_change_step": first_change_index(predictions),
        "correctness_transition_count": transition_count,
        "prediction_changed": prediction_changed,
        "first_score": scores[0],
        "final_score": scores[-1],
    }


def short_case_label(case_name):
    mapping = {
        "correct_to_correct_stable": "correct->correct stable",
        "correct_to_correct_changing": "correct->correct changing",
        "wrong_to_wrong_stable": "wrong->wrong stable",
        "wrong_to_wrong_changing": "wrong->wrong changing",
        "wrong_to_correct": "wrong->correct",
        "correct_to_wrong": "correct->wrong",
    }
    return mapping.get(case_name, case_name)


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def draw_svg_bar_chart(case_counts, total_samples, width, height):
    ordered_cases = [
        "correct_to_correct_stable",
        "correct_to_correct_changing",
        "wrong_to_correct",
        "correct_to_wrong",
        "wrong_to_wrong_changing",
        "wrong_to_wrong_stable",
    ]
    case_names = [case for case in ordered_cases if case in case_counts]
    case_ratios = [
        (case_counts[case] / total_samples * 100.0) if total_samples else 0.0
        for case in case_names
    ]
    labels = [short_case_label(case) for case in case_names]
    colors = ["#0b7285", "#74c0fc", "#2f9e44", "#d9480f", "#f59f00", "#868e96"]

    left = 70
    right = 20
    top = 40
    bottom = 110
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_ratio = max(case_ratios + [5.0])
    bar_width = chart_width / max(len(case_names), 1) * 0.65
    gap = chart_width / max(len(case_names), 1) * 0.35

    parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-family="Arial">Case Proportions</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#333" stroke-width="1.5"/>',
    ]

    for tick_ratio in [0, max_ratio / 2.0, max_ratio]:
        y = top + chart_height - (tick_ratio / max_ratio * chart_height if max_ratio else 0)
        parts.append(
            f'<line x1="{left-5}" y1="{y:.2f}" x2="{left + chart_width}" y2="{y:.2f}" stroke="#e9ecef" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{tick_ratio:.1f}%</text>'
        )

    for idx, (label, ratio) in enumerate(zip(labels, case_ratios)):
        x = left + idx * (bar_width + gap) + gap / 2.0
        bar_height = ratio / max_ratio * chart_height if max_ratio else 0
        y = top + chart_height - bar_height
        color = colors[idx % len(colors)]
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-size="11" font-family="Arial">{ratio:.1f}%</text>'
        )
        label_x = x + bar_width / 2.0
        label_y = top + chart_height + 18
        parts.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="end" transform="rotate(-25 {label_x:.2f} {label_y:.2f})" font-size="10" font-family="Arial">{svg_escape(label)}</text>'
        )

    return "\n".join(parts)


def draw_svg_line_chart(step_means, width, height):
    step_ids = sorted(step_means)
    step_scores = [step_means[step] for step in step_ids]

    left = 60
    right = 20
    top = 40
    bottom = 60
    chart_width = width - left - right
    chart_height = height - top - bottom

    min_step = min(step_ids) if step_ids else 0
    max_step = max(step_ids) if step_ids else 1
    max_score = max(step_scores + [0.1])

    parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-family="Arial">Mean TextVQA Score by Step</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#333" stroke-width="1.5"/>',
    ]

    for tick_score in [0, max_score / 2.0, max_score]:
        y = top + chart_height - (tick_score / max_score * chart_height if max_score else 0)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_width}" y2="{y:.2f}" stroke="#e9ecef" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{tick_score:.2f}</text>'
        )

    points = []
    for step, score in zip(step_ids, step_scores):
        if max_step == min_step:
            x = left + chart_width / 2.0
        else:
            x = left + (step - min_step) / (max_step - min_step) * chart_width
        y = top + chart_height - (score / max_score * chart_height if max_score else 0)
        points.append((x, y, step, score))

    if points:
        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y, _, _ in points)
        parts.append(
            f'<polyline fill="none" stroke="#c2255c" stroke-width="2.5" points="{polyline}"/>'
        )
        for x, y, step, score in points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="#c2255c"/>')
        for step in [step_ids[0], step_ids[len(step_ids) // 2], step_ids[-1]]:
            if max_step == min_step:
                x = left + chart_width / 2.0
            else:
                x = left + (step - min_step) / (max_step - min_step) * chart_width
            parts.append(
                f'<text x="{x:.2f}" y="{top + chart_height + 18:.2f}" text-anchor="middle" font-size="11" font-family="Arial">{step}</text>'
            )

    return "\n".join(parts)


def plot_summary(case_counts, total_samples, step_means, output_path):
    width = 1400
    height = 520
    gap = 24
    half_width = (width - gap) // 2

    left_chart = draw_svg_bar_chart(case_counts, total_samples, half_width, height)
    right_chart = draw_svg_line_chart(step_means, half_width, height)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#f8f9fa"/>
<g transform="translate(0,0)">
{left_chart}
</g>
<g transform="translate({half_width + gap},0)">
{right_chart}
</g>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def main():
    args = parse_args()
    input_path = Path(args.input_jsonl)
    summary_path = Path(args.output_summary_json)
    plot_path = Path(args.output_plot)
    labeled_path = Path(args.output_labeled_jsonl)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_path.parent.mkdir(parents=True, exist_ok=True)

    case_counts = Counter()
    case_examples = defaultdict(list)
    step_score_totals = defaultdict(float)
    step_score_counts = defaultdict(int)
    labeled_records = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            step_results = record.get("step_results", [])
            if not step_results:
                continue

            case_info = classify_case(step_results, args.correct_threshold)
            case_name = case_info["case"]
            case_counts[case_name] += 1

            if len(case_examples[case_name]) < args.max_examples_per_case:
                case_examples[case_name].append(
                    {
                        "doc_id": record.get("doc_id"),
                        "question": record.get("question"),
                        "first_score": case_info["first_score"],
                        "final_score": case_info["final_score"],
                        "first_correct_step": case_info["first_correct_step"],
                        "final_text": record.get("final_text"),
                    }
                )

            for step_result in step_results:
                step = int(step_result["step"])
                step_score_totals[step] += float(step_result["exact_match"])
                step_score_counts[step] += 1

            labeled_record = dict(record)
            labeled_record["case_analysis"] = {
                "case": case_name,
                "case_label": short_case_label(case_name),
                "first_correct_step": case_info["first_correct_step"],
                "first_prediction_change_step": case_info["first_prediction_change_step"],
                "correctness_transition_count": case_info["correctness_transition_count"],
                "prediction_changed": case_info["prediction_changed"],
                "first_score": case_info["first_score"],
                "final_score": case_info["final_score"],
            }
            labeled_records.append(labeled_record)

    total_samples = sum(case_counts.values())
    step_means = {
        step: (step_score_totals[step] / step_score_counts[step])
        for step in sorted(step_score_totals)
    }

    case_summary = []
    ordered_cases = [
        "correct_to_correct_stable",
        "correct_to_correct_changing",
        "wrong_to_correct",
        "correct_to_wrong",
        "wrong_to_wrong_changing",
        "wrong_to_wrong_stable",
    ]
    for case_name in ordered_cases:
        count = case_counts.get(case_name, 0)
        ratio = (count / total_samples) if total_samples else 0.0
        case_summary.append(
            {
                "case": case_name,
                "label": short_case_label(case_name),
                "count": count,
                "ratio": ratio,
                "ratio_percent": ratio * 100.0,
                "examples": case_examples.get(case_name, []),
            }
        )

    summary = {
        "input_jsonl": str(input_path),
        "correct_threshold": args.correct_threshold,
        "num_samples": total_samples,
        "case_summary": case_summary,
        "step_mean_exact_match": [
            {"step": step, "mean_exact_match": step_means[step], "count": step_score_counts[step]}
            for step in sorted(step_means)
        ],
    }

    with summary_path.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)

    with labeled_path.open("w", encoding="utf-8") as fout:
        for record in labeled_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    plot_summary(case_counts, total_samples, step_means, plot_path)

    print(f"Wrote summary to {summary_path}")
    print(f"Wrote labeled samples to {labeled_path}")
    print(f"Wrote plot to {plot_path}")

    for item in case_summary:
        print(f"{item['label']}: {item['count']}/{total_samples} ({item['ratio_percent']:.1f}%)")


if __name__ == "__main__":
    main()
