import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"Glyph .* missing from font.*",
    category=UserWarning,
    module=r"matplotlib\..*",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize cond/uncond logits differences from VRG trace outputs.")
    parser.add_argument("--trace-json", required=True, help="Trace JSON produced by VRG/debug_timestep_vrg.py --save-trace.")
    parser.add_argument("--output-dir", default="VRG/outputs/logits_trace")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def save_alpha_plot(trace_records, output_path: Path, dpi: int):
    steps = [record["global_step_idx"] for record in trace_records]
    alphas = [record["alpha_t"] for record in trace_records]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(steps, alphas, color="#1f77b4", linewidth=1.8, marker="o", markersize=3.0)
    ax.set_xlabel("Diffusion Step Index")
    ax.set_ylabel("Alpha")
    ax.set_title("VRG Alpha Over Steps")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_delta_plot(trace_records, output_path: Path, dpi: int):
    steps = [record["global_step_idx"] for record in trace_records]
    mean_abs = [record["mean_abs_delta"] for record in trace_records]
    max_abs = [record["max_abs_delta"] for record in trace_records]
    mean_kl = [record["mean_kl_cond_uncond"] for record in trace_records]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(steps, mean_abs, color="#d62728", linewidth=1.8, marker="o", markersize=3.0, label="Mean |delta logit|")
    ax.plot(steps, max_abs, color="#ff7f0e", linewidth=1.6, marker="s", markersize=2.8, label="Max |delta logit|")
    ax.plot(steps, mean_kl, color="#2ca02c", linewidth=1.6, marker="^", markersize=2.8, label="Mean KL(cond||uncond)")
    ax.set_xlabel("Diffusion Step Index")
    ax.set_ylabel("Magnitude")
    ax.set_title("Cond/Uncond Logits Difference Over Steps")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_position_heatmap(trace_records, output_path: Path, dpi: int):
    max_position = 0
    for record in trace_records:
        if record["active_positions"]:
            max_position = max(max_position, max(record["active_positions"]))
    if max_position == 0 and not any(record["active_positions"] for record in trace_records):
        raise ValueError("No active positions found in trace_records.")

    heatmap = np.full((len(trace_records), max_position + 1), np.nan, dtype=np.float32)
    for step_idx, record in enumerate(trace_records):
        for pos, value in zip(record["active_positions"], record["per_position_mean_abs_delta"]):
            heatmap[step_idx, int(pos)] = float(value)

    masked = np.ma.masked_invalid(heatmap)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f3f3f3")

    fig, ax = plt.subplots(figsize=(10, 5.4))
    im = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap)
    fig.colorbar(im, ax=ax, label="Mean |cond - uncond| logit")
    ax.set_xlabel("Generated Token Position")
    ax.set_ylabel("Trace Step Index")
    ax.set_title("Logits Difference Heatmap by Step and Position")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_topk_bar_plot(trace_records, output_path: Path, dpi: int):
    best_record = max(trace_records, key=lambda record: record["mean_abs_delta"])
    if not best_record["representative_positions"]:
        raise ValueError("No representative positions available for top-k delta plot.")

    position_record = best_record["representative_positions"][0]
    negative = list(reversed(position_record["top_negative_deltas"]))
    positive = position_record["top_positive_deltas"]
    entries = negative + positive

    labels = [
        f"{entry['token_text']} ({entry['token_id']})"
        for entry in entries
    ]
    values = [entry["delta_logit"] for entry in entries]
    colors = ["#1f77b4" if value < 0 else "#d62728" for value in values]

    fig, ax = plt.subplots(figsize=(max(10, len(entries) * 0.8), 5.2))
    x = np.arange(len(entries))
    ax.bar(x, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Delta Logit (cond - uncond)")
    ax.set_title(
        "Top Delta Tokens at Step "
        f"{best_record['global_step_idx']} Position {position_record['position']}"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_summary(trace_payload, output_path: Path):
    trace_records = trace_payload["trace_records"]
    best_record = max(trace_records, key=lambda record: record["mean_abs_delta"])
    summary = {
        "image": trace_payload.get("image"),
        "question": trace_payload.get("question"),
        "final_text": trace_payload.get("final_text"),
        "num_trace_steps": len(trace_records),
        "max_mean_abs_delta_step": int(best_record["global_step_idx"]),
        "max_mean_abs_delta": float(best_record["mean_abs_delta"]),
        "max_abs_delta": float(max(record["max_abs_delta"] for record in trace_records)),
        "alpha_start": trace_payload.get("final_meta", {}).get("alpha_start"),
        "alpha_end": trace_payload.get("final_meta", {}).get("alpha_end"),
        "alpha_schedule": trace_payload.get("final_meta", {}).get("alpha_schedule"),
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    trace_path = Path(args.trace_json)
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_records = trace_payload["trace_records"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_alpha_plot(trace_records, output_dir / "alpha_over_steps.png", args.dpi)
    save_delta_plot(trace_records, output_dir / "delta_over_steps.png", args.dpi)
    save_position_heatmap(trace_records, output_dir / "position_delta_heatmap.png", args.dpi)
    save_topk_bar_plot(trace_records, output_dir / "topk_delta_tokens.png", args.dpi)
    save_summary(trace_payload, output_dir / "summary.json")

    print(f"Saved VRG trace plots to {output_dir}")


if __name__ == "__main__":
    main()
