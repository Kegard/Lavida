import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_step_summary(path: Path):
    data = json.loads(path.read_text())
    step_summary = data["step_summary"]
    steps = [item["step"] for item in step_summary]
    accuracy = [item["mean_exact_match"] * 100.0 for item in step_summary]
    num_samples = data.get("num_samples")
    return steps, accuracy, num_samples


def main():
    parser = argparse.ArgumentParser(
        description="Plot TextVQA accuracy at each denoising step."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more summary.json files containing step_summary.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels for each input curve.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path, e.g. VRG/outputs/textvqa_step_accuracy.png",
    )
    args = parser.parse_args()

    if len(args.input) != len(args.labels):
        raise ValueError("--input and --labels must have the same length.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=200)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    for idx, (input_path, label) in enumerate(zip(args.input, args.labels)):
        steps, accuracy, num_samples = load_step_summary(Path(input_path))
        ax.plot(
            steps,
            accuracy,
            marker="o",
            markersize=3.5,
            linewidth=2.2,
            color=colors[idx % len(colors)],
            label=f"{label} (n={num_samples})",
        )

    ax.set_title("TextVQA Accuracy Across Denoising Steps", fontsize=15, pad=10)
    ax.set_xlabel("Denoising Step", fontsize=12)
    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=12)
    ax.set_xlim(1, 32)
    ax.set_xticks(range(1, 33, 2))
    ax.legend(frameon=True, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
