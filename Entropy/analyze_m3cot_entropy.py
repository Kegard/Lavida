import argparse
import csv
import json
from pathlib import Path


DEFAULT_ENTROPY_SUMMARY = Path("Entropy/outputs/m3cot_entropy_reason_cot/summary.json")
DEFAULT_ACCURACY_SUMMARY = Path("M3CoT/outputs/64_stepwise_x0_reason_cot/summary.json")

SVG_WIDTH = 980
SVG_HEIGHT = 680
PLOT_LEFT = 88
PLOT_RIGHT = 44
PLOT_TOP = 58
PLOT_BOTTOM = 64
PANEL_GAP = 34
BG = "#fbfaf7"
AXIS = "#1f2937"
GRID = "#d6d3d1"
TEXT = "#374151"
TITLE = "#111827"
ENTROPY_LINE = "#0f766e"
SELECTED_LINE = "#b45309"
PRIORITY_LINE = "#2563eb"
ACCURACY_LINE = "#b91c1c"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize and visualize M3CoT entropy results.")
    parser.add_argument("--entropy-summary", default=str(DEFAULT_ENTROPY_SUMMARY))
    parser.add_argument("--accuracy-summary", default=str(DEFAULT_ACCURACY_SUMMARY))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def mean(values):
    return sum(values) / len(values) if values else 0.0


def pct_change(first, last):
    return ((last - first) / first * 100.0) if first else 0.0


def scale_x(step, max_step):
    width = SVG_WIDTH - PLOT_LEFT - PLOT_RIGHT
    if max_step <= 1:
        return PLOT_LEFT + width / 2
    return PLOT_LEFT + (step - 1) * width / (max_step - 1)


def scale_y(value, y_min, y_max, panel_top, panel_height):
    if y_max == y_min:
        return panel_top + panel_height / 2
    ratio = (value - y_min) / (y_max - y_min)
    return panel_top + panel_height - ratio * panel_height


def padded_range(values, pad_ratio=0.08):
    y_min = min(values)
    y_max = max(values)
    pad = max((y_max - y_min) * pad_ratio, 1e-6)
    return y_min - pad, y_max + pad


def polyline(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def draw_panel(elements, title, series, panel_top, panel_height, max_step, y_label):
    all_values = [value for _, values, _ in series for value in values]
    y_min, y_max = padded_range(all_values)
    plot_right = SVG_WIDTH - PLOT_RIGHT
    plot_bottom = panel_top + panel_height

    elements.append(
        f'<text x="{PLOT_LEFT}" y="{panel_top - 12}" font-size="17" font-family="Arial, Helvetica, sans-serif" fill="{TITLE}" font-weight="700">{escape_xml(title)}</text>'
    )
    for i in range(5):
        y_value = y_min + (y_max - y_min) * i / 4
        y = scale_y(y_value, y_min, y_max, panel_top, panel_height)
        elements.append(f'<line x1="{PLOT_LEFT}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>')
        elements.append(
            f'<text x="{PLOT_LEFT - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial, Helvetica, sans-serif" fill="{TEXT}">{y_value:.2f}</text>'
        )

    for x_tick in sorted(set([1, max_step // 4 or 1, max_step // 2 or 1, (3 * max_step) // 4 or 1, max_step])):
        x = scale_x(x_tick, max_step)
        elements.append(f'<line x1="{x:.2f}" y1="{panel_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="{GRID}" stroke-width="1"/>')
        elements.append(
            f'<text x="{x:.2f}" y="{plot_bottom + 18:.2f}" text-anchor="middle" font-size="11" font-family="Arial, Helvetica, sans-serif" fill="{TEXT}">{x_tick}</text>'
        )

    elements.append(f'<line x1="{PLOT_LEFT}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="{AXIS}" stroke-width="1.5"/>')
    elements.append(f'<line x1="{PLOT_LEFT}" y1="{panel_top}" x2="{PLOT_LEFT}" y2="{plot_bottom}" stroke="{AXIS}" stroke-width="1.5"/>')
    elements.append(
        f'<text x="24" y="{panel_top + panel_height / 2:.2f}" transform="rotate(-90 24 {panel_top + panel_height / 2:.2f})" text-anchor="middle" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="{TEXT}">{escape_xml(y_label)}</text>'
    )

    legend_x = PLOT_LEFT
    for label, values, color in series:
        steps = list(range(1, len(values) + 1))
        points = [
            (scale_x(step, max_step), scale_y(value, y_min, y_max, panel_top, panel_height))
            for step, value in zip(steps, values)
        ]
        elements.append(
            f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        elements.append(f'<line x1="{legend_x}" y1="{panel_top + 14}" x2="{legend_x + 24}" y2="{panel_top + 14}" stroke="{color}" stroke-width="3"/>')
        elements.append(
            f'<text x="{legend_x + 30}" y="{panel_top + 18}" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="{TEXT}">{escape_xml(label)}</text>'
        )
        legend_x += 220


def write_svg(path, entropy_summary, accuracy_summary, rows):
    max_step = len(rows)
    panel_height = 230
    panel1_top = PLOT_TOP + 42
    panel2_top = panel1_top + panel_height + PANEL_GAP + 38

    active_entropy = [row["mean_entropy_active_block_before"] for row in rows]
    selected_entropy = [row["mean_entropy_selected"] for row in rows]
    priority = [row["mean_priority_selected"] for row in rows]
    accuracy = [row["mean_acc"] for row in rows if row["mean_acc"] is not None]

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
        f'<text x="{PLOT_LEFT}" y="32" font-size="24" font-family="Arial, Helvetica, sans-serif" fill="{TITLE}" font-weight="700">M3CoT LaViDa-Reason Entropy Trajectory</text>',
        f'<text x="{PLOT_LEFT}" y="56" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="{TEXT}">samples={entropy_summary["num_samples"]}  max_new_tokens={entropy_summary["generation"]["max_new_tokens"]}  step_ratio={entropy_summary["generation"]["step_ratio"]}  final_acc={accuracy[-1]:.3f}</text>',
    ]

    draw_panel(
        elements,
        "Entropy over denoising steps",
        [
            ("active masked entropy", active_entropy, ENTROPY_LINE),
            ("selected token entropy", selected_entropy, SELECTED_LINE),
        ],
        panel1_top,
        panel_height,
        max_step,
        "entropy",
    )
    draw_panel(
        elements,
        "Selection confidence and accuracy",
        [
            ("selection priority", priority, PRIORITY_LINE),
            ("stepwise accuracy", accuracy, ACCURACY_LINE),
        ],
        panel2_top,
        panel_height,
        max_step,
        "score",
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_csv(path, rows):
    fieldnames = [
        "step",
        "mean_entropy_active_block_before",
        "mean_entropy_selected",
        "mean_priority_selected",
        "num_masked_after_step",
        "mean_acc",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_report(path, entropy_summary, accuracy_summary, rows):
    active = [row["mean_entropy_active_block_before"] for row in rows]
    selected = [row["mean_entropy_selected"] for row in rows]
    priority = [row["mean_priority_selected"] for row in rows]
    acc = [row["mean_acc"] for row in rows if row["mean_acc"] is not None]
    peak_selected_idx = max(range(len(selected)), key=lambda i: selected[i])
    min_active_idx = min(range(len(active)), key=lambda i: active[i])
    max_acc_idx = max(range(len(acc)), key=lambda i: acc[i])

    lines = [
        "# M3CoT Entropy Summary",
        "",
        "## Setup",
        "",
        f"- Samples: {entropy_summary['num_samples']}",
        f"- Model: `{entropy_summary['model_path']}`",
        f"- Prompt: `{entropy_summary['prompt']}`",
        f"- Generation: max_new_tokens={entropy_summary['generation']['max_new_tokens']}, block_length={entropy_summary['generation']['block_length']}, step_ratio={entropy_summary['generation']['step_ratio']}",
        f"- Mean elapsed: {entropy_summary['mean_elapsed_sec']:.2f}s / sample",
        "",
        "## Main Patterns",
        "",
        f"- Active masked entropy drops from {active[0]:.3f} at step 1 to {active[-1]:.3f} at step {len(active)} ({pct_change(active[0], active[-1]):.1f}%).",
        f"- The selected-token entropy is lowest at steps 2-3 ({selected[1]:.3f}, {selected[2]:.3f}), then rises through the middle and peaks at step {peak_selected_idx + 1} ({selected[peak_selected_idx]:.3f}).",
        f"- Selection priority moves in the opposite direction: it peaks at step 3 ({priority[2]:.3f}) and ends at {priority[-1]:.3f}.",
        f"- Stepwise accuracy rises quickly early, plateaus around steps 18-23, and reaches its best value at step {max_acc_idx + 1} ({acc[max_acc_idx]:.3f}). Final step accuracy is {acc[-1]:.3f}.",
        "",
        "## Phase View",
        "",
        f"- Early steps 1-8: active entropy stays high, but selected entropy is low after step 1, suggesting the sampler first commits easy/high-confidence positions.",
        f"- Middle steps 9-22: active entropy declines slowly while selected entropy rises, matching the transition from obvious tokens to more ambiguous reasoning tokens.",
        f"- Late steps 23-32: active entropy collapses as few masks remain, but selected entropy remains relatively high; accuracy is already saturated, so late denoising mostly stabilizes the surface form rather than adding much new correctness.",
        "",
        "## Files",
        "",
        "- `combined_entropy_accuracy.svg`",
        "- `entropy_step_table.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    entropy_summary = load_json(args.entropy_summary)
    accuracy_summary = load_json(args.accuracy_summary)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.entropy_summary).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    acc_by_step = {
        int(item["step"]): item.get("mean_acc")
        for item in accuracy_summary.get("step_summary", [])
    }
    rows = []
    for item in entropy_summary["step_summary"]:
        step = int(item["step"])
        rows.append(
            {
                "step": step,
                "mean_entropy_active_block_before": item["mean_entropy_active_block_before"],
                "mean_entropy_selected": item["mean_entropy_selected"],
                "mean_priority_selected": item["mean_priority_selected"],
                "num_masked_after_step": item["num_masked_after_step"],
                "mean_acc": acc_by_step.get(step),
            }
        )

    write_csv(output_dir / "entropy_step_table.csv", rows)
    write_svg(output_dir / "combined_entropy_accuracy.svg", entropy_summary, accuracy_summary, rows)
    write_report(output_dir / "entropy_result_summary.md", entropy_summary, accuracy_summary, rows)
    print(f"Wrote {output_dir / 'entropy_step_table.csv'}")
    print(f"Wrote {output_dir / 'combined_entropy_accuracy.svg'}")
    print(f"Wrote {output_dir / 'entropy_result_summary.md'}")


if __name__ == "__main__":
    main()
