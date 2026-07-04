import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build stepwise summary.json from records.jsonl.")
    parser.add_argument("records_jsonl")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    records_path = Path(args.records_jsonl)
    output_path = Path(args.output) if args.output else records_path.with_name("summary.json")

    num_samples = 0
    elapsed_sum = 0.0
    generation = None
    step_correct = defaultdict(int)
    step_count = defaultdict(int)

    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            num_samples += 1
            elapsed_sum += float(record.get("elapsed_sec", 0.0))

            meta = record.get("meta", {})
            if generation is None:
                generation = {
                    "max_new_tokens": int(meta.get("max_new_tokens", 0)),
                    "block_length": int(meta.get("block_length", 0)),
                    "steps_per_block": int(meta.get("steps_per_block", 0)),
                    "total_denoising_steps": int(meta.get("total_denoising_steps", 0)),
                }

            for step_result in record.get("step_results", []):
                step = int(step_result["step"])
                step_correct[step] += int(bool(step_result.get("correct", False)))
                step_count[step] += 1

    step_summary = []
    for step in sorted(step_count):
        count = step_count[step]
        mean_acc = step_correct[step] / count if count else 0.0
        step_summary.append(
            {
                "step": step,
                "num_samples": count,
                "num_correct": step_correct[step],
                "mean_acc": mean_acc,
            }
        )

    summary = {
        "num_samples": num_samples,
        "mean_elapsed_sec": (elapsed_sum / num_samples) if num_samples else 0.0,
        "generation": generation or {},
        "step_summary": step_summary,
    }

    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
