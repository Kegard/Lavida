import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
M3COT_ROOT = REPO_ROOT / "M3CoT"
if str(M3COT_ROOT) not in sys.path:
    sys.path.insert(0, str(M3COT_ROOT))

from M3CoT.plot_stepwise_summaries import make_svg


ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]


def judge_answer(text, choices, answer):
    if isinstance(answer, int):
        answer = ALPHA_MAP[answer]
    return extract_answer(text, choices) == answer


def extract_answer(text, choices):
    valid_letters = "".join(ALPHA_MAP[: len(choices)])
    scoped_text = text
    if "[Answer]" in scoped_text:
        scoped_text = scoped_text.split("[Answer]")[-1].split("[Rationale]")[0].split("[Context]")[0]

    explicit_patterns = [
        rf"Answer\s*:\s*[\(\[]?\s*([{valid_letters}{valid_letters.lower()}])\b",
        rf"\\boxed\s*\{{?\s*([{valid_letters}{valid_letters.lower()}])\s*\}}?",
        rf"\[Answer\]\s*[\(\[]?\s*([{valid_letters}{valid_letters.lower()}])\b",
        rf"\(([{valid_letters}{valid_letters.lower()}])\)",
    ]
    for pattern in explicit_patterns:
        res = re.findall(pattern, scoped_text)
        if res:
            return res[-1].upper()

    res = []
    for i, choice in enumerate(choices):
        if choice.lower() in scoped_text.lower():
            res.append(ALPHA_MAP[i])
    if res:
        return res[-1]

    normalized_text = re.sub(r"[\n.,!?]", " ", scoped_text)
    tokens = normalized_text.split()
    res = []
    for i, _ in enumerate(choices):
        if ALPHA_MAP[i] in tokens:
            res.append(ALPHA_MAP[i])
    if res:
        return res[-1]

    res = []
    for i, _ in enumerate(choices):
        if ALPHA_MAP[i].lower() in tokens:
            res.append(ALPHA_MAP[i])
    if res:
        return res[-1]

    return "FAILED"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute M3CoT stepwise x0 summary from records.jsonl and redraw the curve."
    )
    parser.add_argument(
        "--records",
        default="M3CoT/outputs/64_stepwise_x0_reason_cot/records.jsonl",
        help="Path to records.jsonl produced by run_m3cot_stepwise_x0.py.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Output summary path. Defaults to summary.reeval.json beside records.",
    )
    parser.add_argument(
        "--svg-output",
        default=None,
        help="Output SVG path. Defaults to stepwise_curve.reeval.svg beside records.",
    )
    return parser.parse_args()


def load_records(records_path):
    records = []
    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {records_path}")
    return records


def recompute_summary(records):
    step_correct = defaultdict(float)
    step_counts = defaultdict(int)

    for record in records:
        choices = record["choices"]
        answer = record["answer"]
        for step_record in record["step_results"]:
            step = int(step_record["step"])
            is_correct = bool(judge_answer(step_record["candidate_text"], choices, answer))
            step_correct[step] += float(is_correct)
            step_counts[step] += 1

    step_summary = []
    for step in sorted(step_counts):
        count = step_counts[step]
        step_summary.append(
            {
                "step": int(step),
                "mean_acc": step_correct[step] / count if count else None,
                "count": int(count),
            }
        )

    first = records[0]
    summary = {
        "dataset_path": "from_records",
        "split": "from_records",
        "start_index": records[0].get("dataset_index"),
        "num_samples": len(records),
        "prompt": "from_records",
        "step_definition": (
            "Recomputed from records.jsonl with the current M3CoT.utils.metric.judge_answer; "
            "each score uses candidate_text at that denoising step."
        ),
        "total_elapsed_sec": sum(float(record.get("elapsed_sec", 0.0)) for record in records),
        "mean_elapsed_sec": (
            sum(float(record.get("elapsed_sec", 0.0)) for record in records) / len(records)
            if records
            else None
        ),
        "generation": dict(first.get("meta", {})),
        "step_summary": step_summary,
    }

    generation = summary["generation"]
    if "max_new_tokens" not in generation:
        generation["max_new_tokens"] = len(step_summary)
    return summary


def main():
    args = parse_args()
    records_path = Path(args.records)
    output_dir = records_path.parent
    summary_output = Path(args.summary_output) if args.summary_output else output_dir / "summary.reeval.json"
    svg_output = Path(args.svg_output) if args.svg_output else output_dir / "stepwise_curve.reeval.svg"

    records = load_records(records_path)
    summary = recompute_summary(records)

    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_svg(summary, svg_output)

    best = max(summary["step_summary"], key=lambda item: item["mean_acc"])
    final = summary["step_summary"][-1]
    print(f"Read {len(records)} records from {records_path}")
    print(f"Wrote {summary_output}")
    print(f"Wrote {svg_output}")
    print(f"Best step: {best['step']} acc={best['mean_acc']:.4f}")
    print(f"Final step: {final['step']} acc={final['mean_acc']:.4f}")


if __name__ == "__main__":
    main()
