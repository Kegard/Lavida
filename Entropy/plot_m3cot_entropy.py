import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path("Entropy/outputs/m3cot_reason_entropy/summary.json")
DEFAULT_METRIC = "mean_entropy_active_block_before"

SVG_WIDTH = 900
SVG_HEIGHT = 540
PLOT_LEFT = 90
PLOT_RIGHT = 40
PLOT_TOP = 60
PLOT_BOTTOM = 80
BG = "#fcfbf7"
AXIS = "#1f2937"
GRID = "#d6d3d1"
LINE = "#0f766e"
FILL = "#ccfbf1"
TITLE = "#111827"
TEXT = "#374151"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot M3CoT entropy summary to SVG.")
    parser.add_argument("summary_json", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def load_summary(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scale_x(step, max_step):
    plot_width = SVG_WIDTH - PLOT_LEFT - PLOT_RIGHT
    if max_step <= 1:
        return PLOT_LEFT + plot_width / 2
    return PLOT_LEFT + (step - 1) * plot_width / (max_step - 1)


def scale_y(value, y_min, y_max):
    plot_height = SVG_HEIGHT - PLOT_TOP - PLOT_BOTTOM
    ratio = 0.0 if y_max == y_min else (value - y_min) / (y_max - y_min)
    return SVG_HEIGHT - PLOT_BOTTOM - ratio * plot_height


def build_ticks(y_min, y_max, num_ticks=6):
    if y_max <= y_min:
        return [y_min]
    step = (y_max - y_min) / num_ticks
    return [y_min + i * step for i in range(num_ticks + 1)]


def make_svg(summary, metric_name, save_path):
    step_summary = summary["step_summary"]
    filtered = [
        (item["step"], item[metric_name])
        for item in step_summary
        if item.get(metric_name) is not None
    ]
    if not filtered:
        raise ValueError(f"No usable values found for metric {metric_name}.")

    steps = [item[0] for item in filtered]
    values = [item[1] for item in filtered]
    max_step = max(steps)
    min_value = min(values)
    max_value = max(values)
    padding = max((max_value - min_value) * 0.08, 1e-6)
    y_min = min_value - padding
    y_max = max_value + padding

    best_idx = max(range(len(values)), key=lambda i: values[i])
    last_idx = len(values) - 1
    points = [(scale_x(step, max_step), scale_y(value, y_min, y_max)) for step, value in filtered]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    y_ticks = build_ticks(y_min, y_max)
    x_ticks = sorted(set([1, max_step // 4 or 1, max_step // 2 or 1, (3 * max_step) // 4 or 1, max_step]))

    title = f"M3CoT Entropy Curve ({metric_name})"
    subtitle = (
        f"samples={summary['num_samples']}  "
        f"prompt={summary['prompt']}  "
        f"best=step {steps[best_idx]} / {values[best_idx]:.4f}  "
        f"final={values[last_idx]:.4f}"
    )

    elements = []
    elements.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">'
    )
    elements.append(f'<rect width="100%" height="100%" fill="{BG}"/>')
    elements.append(
        f'<text x="{PLOT_LEFT}" y="30" font-size="24" font-family="Arial, Helvetica, sans-serif" fill="{TITLE}" font-weight="700">{escape_xml(title)}</text>'
    )
    elements.append(
        f'<text x="{PLOT_LEFT}" y="52" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="{TEXT}">{escape_xml(subtitle)}</text>'
    )

    for y_tick in y_ticks:
        y = scale_y(y_tick, y_min, y_max)
        elements.append(
            f'<line x1="{PLOT_LEFT}" y1="{y:.2f}" x2="{SVG_WIDTH - PLOT_RIGHT}" y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{PLOT_LEFT - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="{TEXT}">{y_tick:.4f}</text>'
        )

    for x_tick in x_ticks:
        x = scale_x(x_tick, max_step)
        elements.append(
            f'<line x1="{x:.2f}" y1="{PLOT_TOP}" x2="{x:.2f}" y2="{SVG_HEIGHT - PLOT_BOTTOM}" stroke="{GRID}" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{SVG_HEIGHT - PLOT_BOTTOM + 24}" text-anchor="middle" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="{TEXT}">{x_tick}</text>'
        )

    elements.append(
        f'<line x1="{PLOT_LEFT}" y1="{SVG_HEIGHT - PLOT_BOTTOM}" x2="{SVG_WIDTH - PLOT_RIGHT}" y2="{SVG_HEIGHT - PLOT_BOTTOM}" stroke="{AXIS}" stroke-width="2"/>'
    )
    elements.append(
        f'<line x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" y2="{SVG_HEIGHT - PLOT_BOTTOM}" stroke="{AXIS}" stroke-width="2"/>'
    )

    first_x, _ = points[0]
    last_x, _ = points[-1]
    area_points = f"{first_x:.2f},{SVG_HEIGHT - PLOT_BOTTOM} {polyline} {last_x:.2f},{SVG_HEIGHT - PLOT_BOTTOM}"
    elements.append(f'<polygon points="{area_points}" fill="{FILL}" opacity="0.45"/>')
    elements.append(
        f'<polyline points="{polyline}" fill="none" stroke="{LINE}" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>'
    )

    for x, y in points:
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{LINE}"/>')

    best_x, best_y = points[best_idx]
    elements.append(f'<circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="6" fill="#b91c1c"/>')
    elements.append(
        f'<text x="{best_x + 12:.2f}" y="{best_y - 10:.2f}" font-size="12" font-family="Arial, Helvetica, sans-serif" fill="#991b1b">peak: step {steps[best_idx]}, {values[best_idx]:.4f}</text>'
    )

    elements.append(
        f'<text x="{(PLOT_LEFT + SVG_WIDTH - PLOT_RIGHT) / 2:.2f}" y="{SVG_HEIGHT - 24}" text-anchor="middle" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="{TEXT}">Denoising step</text>'
    )
    elements.append(
        f'<text x="24" y="{(PLOT_TOP + SVG_HEIGHT - PLOT_BOTTOM) / 2:.2f}" transform="rotate(-90 24 {(PLOT_TOP + SVG_HEIGHT - PLOT_BOTTOM) / 2:.2f})" text-anchor="middle" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="{TEXT}">{escape_xml(metric_name)}</text>'
    )
    elements.append("</svg>")

    save_path.write_text("\n".join(elements), encoding="utf-8")


def main():
    args = parse_args()
    input_path = Path(args.summary_json)
    summary = load_summary(input_path)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{args.metric}.svg")
    make_svg(summary, args.metric, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
