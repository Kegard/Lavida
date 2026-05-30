import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot LaViDa text/vision causal weights as a grouped bar chart."
    )
    parser.add_argument(
        "--summary-csv",
        default="Causal_Analysis/outputs/debug_run/step_intervention_summary.csv",
    )
    parser.add_argument(
        "--output",
        default="Causal_Analysis/outputs/debug_run/causal_weight_bars.png",
    )
    parser.add_argument(
        "--metric",
        default="abs_delta",
        choices=["abs_delta", "change_rate"],
        help="Use mean absolute causal delta or prediction-change rate for weights.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_rows(summary_csv):
    with Path(summary_csv).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_weights(rows, metric):
    steps = []
    text_values = []
    vision_values = []

    for row in rows:
        steps.append(int(row["step"]))
        if metric == "abs_delta":
            text_value = float(row["text_mean_abs_delta"])
            vision_value = float(row["vision_mean_abs_delta"])
        else:
            text_value = float(row["text_change_rate"])
            vision_value = float(row["vision_change_rate"])

        denom = text_value + vision_value
        if denom == 0:
            text_weight = 0.5
            vision_weight = 0.5
        else:
            text_weight = text_value / denom
            vision_weight = vision_value / denom

        text_values.append(text_weight)
        vision_values.append(vision_weight)

    return steps, np.array(text_values), np.array(vision_values)


def main():
    args = parse_args()
    rows = load_rows(args.summary_csv)
    steps, text_weights, vision_weights = compute_weights(rows, args.metric)

    # Convert step index to denoising progress labels so the x-axis resembles Consis-GCPO.
    max_step = max(steps)
    timestep_labels = [f"{1.0 - (step - 1) / max_step:.2f}" for step in steps]

    x = np.arange(len(steps))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    text_color = "#6BAED6"
    vision_color = "#F2A65A"
    ax.bar(x - width / 2, text_weights, width, label="Text Weight", color=text_color, edgecolor="#3E6E8E")
    ax.bar(x + width / 2, vision_weights, width, label="Vision Weight", color=vision_color, edgecolor="#A96723")

    ax.set_title("Text and Vision Causal Weights over Denoising Steps", fontsize=12)
    ax.set_xlabel("Timestep t (Denoising Process 1 -> 0)")
    ax.set_ylabel("Normalized Weight")
    ax.set_xticks(x)
    ax.set_xticklabels(timestep_labels)
    ax.set_ylim(0, max(1.0, float(max(text_weights.max(), vision_weights.max())) * 1.15))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=True)

    metric_label = "mean absolute causal delta" if args.metric == "abs_delta" else "prediction-change rate"
    ax.text(
        0.99,
        0.02,
        f"Weight source: {metric_label}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
