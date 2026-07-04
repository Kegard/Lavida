import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot VRG entropy-gate behavior by denoising step.")
    parser.add_argument(
        "--input-jsonl",
        default="M3CoT/benchmark/vis50_vrg_1p0_0p0_entropy_gate_cot.jsonl",
        help="Prediction jsonl containing vrg_gate_stats.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots. Defaults to <input_jsonl stem>_plots next to the jsonl.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_gate_matrix(input_path):
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            stats = obj.get("vrg_gate_stats")
            if not stats:
                continue
            records.append((obj.get("id", f"sample_{len(records)}"), stats))

    if not records:
        raise ValueError(f"No records with vrg_gate_stats found in {input_path}")

    max_step = max(int(item["step"]) for _, stats in records for item in stats)
    use_vrg = np.full((len(records), max_step), np.nan, dtype=float)
    delta_entropy = np.full((len(records), max_step), np.nan, dtype=float)
    alpha = np.full((len(records), max_step), np.nan, dtype=float)

    for row_idx, (_, stats) in enumerate(records):
        for item in stats:
            step_idx = int(item["step"]) - 1
            use_vrg[row_idx, step_idx] = 1.0 if item["use_vrg"] else 0.0
            delta_entropy[row_idx, step_idx] = float(item["delta_entropy"])
            alpha[row_idx, step_idx] = float(item["alpha_t"])

    return records, use_vrg, delta_entropy, alpha


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)


def save_use_vrg_rate(use_vrg, output_path, dpi):
    steps = np.arange(1, use_vrg.shape[1] + 1)
    rate = np.nanmean(use_vrg, axis=0)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(steps, rate, marker="o", linewidth=2.2, color="#1f6f8b")
    ax.fill_between(steps, 0, rate, color="#1f6f8b", alpha=0.14)
    ax.set_title("VRG Gate Acceptance Rate by Step")
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("use_vrg rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(steps)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_delta_entropy(delta_entropy, output_path, dpi):
    steps = np.arange(1, delta_entropy.shape[1] + 1)
    mean_delta = np.nanmean(delta_entropy, axis=0)
    median_delta = np.nanmedian(delta_entropy, axis=0)
    q25 = np.nanpercentile(delta_entropy, 25, axis=0)
    q75 = np.nanpercentile(delta_entropy, 75, axis=0)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.1, alpha=0.75)
    ax.fill_between(steps, q25, q75, color="#d98c3f", alpha=0.18, label="IQR")
    ax.plot(steps, mean_delta, marker="o", linewidth=2.2, color="#b85c38", label="mean")
    ax.plot(steps, median_delta, marker="s", linewidth=1.8, color="#2d6a4f", label="median")
    ax.set_title("Delta Entropy by Step")
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("entropy_base - entropy_vrg")
    ax.set_xticks(steps)
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_gate_heatmap(records, use_vrg, output_path, dpi):
    fig_height = max(5.0, min(12.0, 0.16 * len(records) + 2.0))
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    masked = np.ma.masked_invalid(use_vrg)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    ax.set_title("VRG Gate Decisions")
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Sample")
    ax.set_xticks(np.arange(use_vrg.shape[1]))
    ax.set_xticklabels(np.arange(1, use_vrg.shape[1] + 1))

    if len(records) <= 60:
        ax.set_yticks(np.arange(len(records)))
        ax.set_yticklabels([sample_id for sample_id, _ in records], fontsize=7)
    else:
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["fallback", "use VRG"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.with_suffix("")
    output_dir = output_dir.parent / f"{output_dir.name}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    records, use_vrg, delta_entropy, _ = load_gate_matrix(input_path)

    save_use_vrg_rate(use_vrg, output_dir / "use_vrg_rate_by_step.png", args.dpi)
    save_delta_entropy(delta_entropy, output_dir / "delta_entropy_by_step.png", args.dpi)
    save_gate_heatmap(records, use_vrg, output_dir / "gate_heatmap.png", args.dpi)

    print(f"Loaded {len(records)} records with {use_vrg.shape[1]} steps.")
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
