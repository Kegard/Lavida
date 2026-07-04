import argparse
import json
import re
from collections import Counter
from pathlib import Path


ALPHA_MAP = "ABCDEF"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze when answers first appear in stepwise x0 records.")
    parser.add_argument("records", nargs="+", help="Path(s) to records.jsonl.")
    parser.add_argument(
        "--explicit-only",
        action="store_true",
        help="Only count explicit answer formats: Answer: X, \\boxed{X}, [Answer] X, or (X).",
    )
    return parser.parse_args()


def extract_answer(text, num_choices, explicit_only):
    valid = ALPHA_MAP[:num_choices]
    scoped = text
    if "[Answer]" in scoped:
        scoped = scoped.split("[Answer]")[-1].split("[Rationale]")[0].split("[Context]")[0]

    patterns = [
        rf"Answer\s*:\s*[\(\[]?\s*([{valid}{valid.lower()}])\b",
        rf"\\boxed\s*\{{?\s*([{valid}{valid.lower()}])\s*\}}?",
        rf"\[Answer\]\s*[\(\[]?\s*([{valid}{valid.lower()}])\b",
        rf"\(([{valid}{valid.lower()}])\)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, scoped)
        if matches:
            return matches[-1].upper()

    if explicit_only:
        return "FAILED"

    return "FAILED"


def summarize(values, total_steps):
    finite = sorted(value for value in values if value is not None)
    if not finite:
        return {
            "found": 0,
            "missing": len(values),
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "mean_progress": None,
        }

    def percentile(q):
        return finite[min(len(finite) - 1, int(round(q * (len(finite) - 1))))]

    return {
        "found": len(finite),
        "missing": len(values) - len(finite),
        "mean": sum(finite) / len(finite),
        "median": percentile(0.5),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p90": percentile(0.9),
        "mean_progress": sum(value / total_steps for value in finite) / len(finite),
    }


def progress_bins(values, total_steps):
    bins = Counter()
    for value in values:
        if value is None:
            key = "missing"
        elif value <= total_steps * 0.25:
            key = "0-25%"
        elif value <= total_steps * 0.5:
            key = "25-50%"
        elif value <= total_steps * 0.75:
            key = "50-75%"
        else:
            key = "75-100%"
        bins[key] += 1
    return bins


def analyze(records_path, explicit_only):
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {records_path}")

    total_steps = len(records[0]["step_results"])
    first_any = []
    first_correct = []

    for record in records:
        first_any_step = None
        first_correct_step = None
        for step_record in record["step_results"]:
            pred = extract_answer(
                step_record["candidate_text"],
                num_choices=len(record["choices"]),
                explicit_only=explicit_only,
            )
            if pred != "FAILED" and first_any_step is None:
                first_any_step = int(step_record["step"])
            if pred == record["answer"] and first_correct_step is None:
                first_correct_step = int(step_record["step"])
        first_any.append(first_any_step)
        first_correct.append(first_correct_step)

    print(f"\n{records_path}")
    print(f"samples={len(records)} steps={total_steps} explicit_only={explicit_only}")
    for name, values in (("first_explicit_answer", first_any), ("first_correct_explicit_answer", first_correct)):
        summary = summarize(values, total_steps)
        bins = progress_bins(values, total_steps)
        mean = "NA" if summary["mean"] is None else f"{summary['mean']:.1f}"
        progress = "NA" if summary["mean_progress"] is None else f"{summary['mean_progress']:.3f}"
        print(
            f"{name}: found={summary['found']} missing={summary['missing']} "
            f"mean_step={mean} median={summary['median']} "
            f"p25/p75={summary['p25']}/{summary['p75']} p90={summary['p90']} "
            f"mean_progress={progress} bins={dict(bins)}"
        )


def main():
    args = parse_args()
    for records in args.records:
        analyze(Path(records), explicit_only=args.explicit_only)


if __name__ == "__main__":
    main()
