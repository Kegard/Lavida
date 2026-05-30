import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot bucketed answer unmask-step ratios from MMMU trace JSONL files."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more JSONL trace files.")
    parser.add_argument("--labels", nargs="+", default=None, help="Legend labels for each input file.")
    parser.add_argument("--field", default="answer_unmask_step", help="Record field to plot.")
    parser.add_argument("--output", default="experiment/mmmu_unmask_plot.png")
    parser.add_argument("--title", default=None)
    parser.add_argument("--xlabel", default="Final Answer Generation Step")
    parser.add_argument("--ylabel", default="Ratio (%)")
    parser.add_argument("--first-bucket-end", type=int, default=3)
    parser.add_argument("--bucket-size", type=int, default=4)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--figsize", nargs=2, type=float, default=(8.0, 4.8))
    return parser.parse_args()


def load_steps(path, field):
    steps = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            value = record.get(field)
            if isinstance(value, int) and value > 0:
                steps.append(value)
    return steps


def infer_max_step(step_lists, explicit_max_step):
    if explicit_max_step is not None:
        return explicit_max_step
    max_step = 0
    for steps in step_lists:
        if steps:
            max_step = max(max_step, max(steps))
    return max_step


def build_buckets(max_step, first_bucket_end, bucket_size):
    if max_step < 1:
        raise ValueError("No valid positive steps found to plot.")

    buckets = [(1, min(first_bucket_end, max_step))]
    start = first_bucket_end + 1
    while start <= max_step:
        end = min(start + bucket_size - 1, max_step)
        buckets.append((start, end))
        start = end + 1
    return buckets


def bucket_label(bucket):
    start, end = bucket
    return f"{start}-{end}"


def bucket_ratios(steps, buckets):
    if not steps:
        return [0.0 for _ in buckets]

    total = len(steps)
    ratios = []
    for start, end in buckets:
        count = sum(start <= step <= end for step in steps)
        ratios.append(count * 100.0 / total)
    return ratios


def default_labels(paths):
    return [Path(path).stem for path in paths]


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.inputs):
        raise ValueError("--labels must have the same length as --inputs.")
    if args.first_bucket_end < 1:
        raise ValueError("--first-bucket-end must be >= 1.")
    if args.bucket_size < 1:
        raise ValueError("--bucket-size must be >= 1.")

    labels = args.labels or default_labels(args.inputs)
    step_lists = [load_steps(path, args.field) for path in args.inputs]
    max_step = infer_max_step(step_lists, args.max_step)
    buckets = build_buckets(max_step, args.first_bucket_end, args.bucket_size)
    bucket_names = [bucket_label(bucket) for bucket in buckets]
    ratio_lists = [bucket_ratios(steps, buckets) for steps in step_lists]

    x = np.arange(len(buckets))
    num_series = len(ratio_lists)
    bar_width = 0.8 / max(num_series, 1)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    colors = plt.get_cmap("tab10").colors

    for idx, (label, ratios) in enumerate(zip(labels, ratio_lists)):
        offset = (idx - (num_series - 1) / 2.0) * bar_width
        ax.bar(
            x + offset,
            ratios,
            width=bar_width,
            label=label,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names, rotation=30)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    if args.title:
        ax.set_title(args.title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    if num_series > 1:
        ax.legend(frameon=False)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
