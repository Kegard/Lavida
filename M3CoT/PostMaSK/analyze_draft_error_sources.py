import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze why PostMaSK draft_text answers are judged wrong."
    )
    parser.add_argument(
        "--outputs-dir",
        default="M3CoT/PostMaSK/outputs",
        help="Directory containing experiment subdirectories with records.jsonl.",
    )
    parser.add_argument(
        "--output-json",
        default="M3CoT/PostMaSK/draft_error_source_summary.json",
    )
    parser.add_argument(
        "--output-md",
        default="M3CoT/PostMaSK/draft_error_source_summary.md",
    )
    parser.add_argument("--examples-per-type", type=int, default=8)
    return parser.parse_args()


def extract_answer_with_source(text, choices):
    valid_letters = "".join(ALPHA_MAP[: len(choices)])
    scoped_text = text
    if "[Answer]" in scoped_text:
        scoped_text = scoped_text.split("[Answer]")[-1].split("[Rationale]")[0].split("[Context]")[0]

    explicit_patterns = [
        ("answer_colon", rf"Answer\s*:\s*[\(\[]?\s*([{valid_letters}{valid_letters.lower()}])\b"),
        ("boxed", rf"\\boxed\s*\{{?\s*([{valid_letters}{valid_letters.lower()}])\s*\}}?"),
        ("answer_tag", rf"\[Answer\]\s*[\(\[]?\s*([{valid_letters}{valid_letters.lower()}])\b"),
        ("paren_letter", rf"\(([{valid_letters}{valid_letters.lower()}])\)"),
    ]
    for source, pattern in explicit_patterns:
        res = re.findall(pattern, scoped_text)
        if res:
            return res[-1].upper(), source

    res = []
    for idx, choice in enumerate(choices):
        if choice.lower() in scoped_text.lower():
            res.append(ALPHA_MAP[idx])
    if res:
        return res[-1], "choice_text_keyword"

    normalized_text = re.sub(r"[\n.,!?]", " ", scoped_text)
    tokens = normalized_text.split()
    res = []
    for idx, _ in enumerate(choices):
        if ALPHA_MAP[idx] in tokens:
            res.append(ALPHA_MAP[idx])
    if res:
        return res[-1], "letter_token_upper"

    res = []
    for idx, _ in enumerate(choices):
        if ALPHA_MAP[idx].lower() in tokens:
            res.append(ALPHA_MAP[idx])
    if res:
        return res[-1], "letter_token_lower"

    return "FAILED", "failed"


def explicit_mentions(text, choices):
    mentions = Counter()
    explicit = re.findall(
        r"(?:Answer\s*:|\\boxed\s*\{?|\[Answer\]\s*)\s*[\(\[]?\s*([A-Fa-f])",
        text,
    )
    for item in explicit:
        item = item.upper()
        if item in ALPHA_MAP[: len(choices)]:
            mentions[item] += 1
    return dict(mentions)


def choice_mentions(text, choices):
    lowered = text.lower()
    return {
        ALPHA_MAP[idx]: lowered.count(choice.lower())
        for idx, choice in enumerate(choices)
    }


def letter_mentions(text, choices):
    normalized = re.sub(r"[\n.,!?]", " ", text)
    tokens = normalized.split()
    return {
        ALPHA_MAP[idx]: tokens.count(ALPHA_MAP[idx]) + tokens.count(ALPHA_MAP[idx].lower())
        for idx, _ in enumerate(choices)
    }


def classify_draft_error(record):
    choices = record["choices"]
    answer = record["answer"]
    if isinstance(answer, int):
        answer = ALPHA_MAP[answer]
    draft_text = record.get("draft_text", "")
    pred, source = extract_answer_with_source(draft_text, choices)

    if pred == answer:
        error_type = "correct"
    elif source in {"answer_colon", "boxed", "answer_tag", "paren_letter"}:
        error_type = "global_explicit_wrong_answer"
    elif source in {"choice_text_keyword", "letter_token_upper", "letter_token_lower"}:
        error_type = "local_keyword_extraction_error"
    elif source == "failed":
        error_type = "local_incomplete_or_no_answer"
    else:
        error_type = "unknown"

    return {
        "error_type": error_type,
        "pred": pred,
        "gold": answer,
        "source": source,
        "explicit_mentions": explicit_mentions(draft_text, choices),
        "choice_mentions": choice_mentions(draft_text, choices),
        "letter_mentions": letter_mentions(draft_text, choices),
    }


def pct(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def fmt_pct(value):
    if value is None:
        return "NA"
    return f"{value:.3f}"


def analyze_file(path, examples_per_type):
    counts = Counter()
    by_source = Counter()
    by_domain = defaultdict(Counter)
    by_topic = defaultdict(Counter)
    examples = defaultdict(list)

    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            info = classify_draft_error(record)
            error_type = info["error_type"]
            counts[error_type] += 1
            counts["total"] += 1
            by_source[info["source"]] += 1
            by_domain[record.get("domain", "UNKNOWN")][error_type] += 1
            by_domain[record.get("domain", "UNKNOWN")]["total"] += 1
            by_topic[record.get("topic", "UNKNOWN")][error_type] += 1
            by_topic[record.get("topic", "UNKNOWN")]["total"] += 1

            if error_type != "correct" and len(examples[error_type]) < examples_per_type:
                examples[error_type].append(
                    {
                        "dataset_index": record.get("dataset_index"),
                        "id": record.get("id"),
                        "question": record.get("question", "").splitlines()[0],
                        "gold": info["gold"],
                        "pred": info["pred"],
                        "source": info["source"],
                        "explicit_mentions": info["explicit_mentions"],
                        "choice_mentions": info["choice_mentions"],
                        "letter_mentions": info["letter_mentions"],
                        "draft_text": record.get("draft_text", ""),
                    }
                )

    draft_wrong = counts["total"] - counts["correct"]
    return {
        "experiment": path.parent.name,
        "path": str(path),
        "total": counts["total"],
        "draft_correct": counts["correct"],
        "draft_wrong": draft_wrong,
        "draft_accuracy": pct(counts["correct"], counts["total"]),
        "counts": dict(counts),
        "wrong_breakdown": {
            key: {
                "count": counts[key],
                "ratio_among_draft_errors": pct(counts[key], draft_wrong),
            }
            for key in [
                "global_explicit_wrong_answer",
                "local_keyword_extraction_error",
                "local_incomplete_or_no_answer",
                "unknown",
            ]
        },
        "extract_source_counts": dict(by_source),
        "by_domain": {key: dict(value) for key, value in sorted(by_domain.items())},
        "by_topic": {key: dict(value) for key, value in sorted(by_topic.items())},
        "examples": dict(examples),
    }


def write_markdown(results, path):
    lines = [
        "# Draft Text Error Source Summary",
        "",
        "This file analyzes only `draft_text`, before PostMaSK refinement.",
        "",
        "Definitions:",
        "",
        "- `global_explicit_wrong_answer`: the draft explicitly outputs a wrong option through `Answer:`, `\\boxed{}`, `[Answer]`, or `(A)`-style answer syntax.",
        "- `local_keyword_extraction_error`: the draft has no explicit final answer, and the metric picks a wrong option because of a local keyword, option text, or standalone option letter in the draft.",
        "- `local_incomplete_or_no_answer`: the draft has no extractable answer at all, usually because it stops in the middle of reasoning.",
        "",
        "These labels are automatic proxies. `global_explicit_wrong_answer` is the strongest signal that the draft's overall conclusion is wrong; `local_keyword_extraction_error` is the strongest signal that a local phrase or keyword caused the judged answer.",
        "",
        "## Experiment Table",
        "",
        "| Experiment | N | Draft Acc | Draft Wrong | Global Explicit Wrong | Local Keyword | Incomplete/No Answer |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for item in results:
        counts = item["counts"]
        lines.append(
            "| `{experiment}` | {total} | {draft_acc} | {draft_wrong} | {global_count} ({global_ratio}) | {local_count} ({local_ratio}) | {failed_count} ({failed_ratio}) |".format(
                experiment=item["experiment"],
                total=item["total"],
                draft_acc=fmt_pct(item["draft_accuracy"]),
                draft_wrong=item["draft_wrong"],
                global_count=counts.get("global_explicit_wrong_answer", 0),
                global_ratio=fmt_pct(item["wrong_breakdown"]["global_explicit_wrong_answer"]["ratio_among_draft_errors"]),
                local_count=counts.get("local_keyword_extraction_error", 0),
                local_ratio=fmt_pct(item["wrong_breakdown"]["local_keyword_extraction_error"]["ratio_among_draft_errors"]),
                failed_count=counts.get("local_incomplete_or_no_answer", 0),
                failed_ratio=fmt_pct(item["wrong_breakdown"]["local_incomplete_or_no_answer"]["ratio_among_draft_errors"]),
            )
        )

    lines.extend(
        [
            "",
            "## Extract Source Table",
            "",
            "| Experiment | Explicit Sources | Choice Text Keyword | Letter Token | Failed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in results:
        src = item["extract_source_counts"]
        explicit_total = sum(src.get(key, 0) for key in ["answer_colon", "boxed", "answer_tag", "paren_letter"])
        letter_total = src.get("letter_token_upper", 0) + src.get("letter_token_lower", 0)
        lines.append(
            f"| `{item['experiment']}` | {explicit_total} | {src.get('choice_text_keyword', 0)} | {letter_total} | {src.get('failed', 0)} |"
        )

    lines.extend(["", "## Representative Examples", ""])
    for item in results:
        if item["experiment"] != "postmask_sr0p5_d16_p16_conf_r4_seed42_n400":
            continue
        lines.append(f"### `{item['experiment']}`")
        lines.append("")
        for error_type in [
            "global_explicit_wrong_answer",
            "local_keyword_extraction_error",
            "local_incomplete_or_no_answer",
        ]:
            lines.append(f"#### `{error_type}`")
            lines.append("")
            for example in item["examples"].get(error_type, [])[:5]:
                draft_preview = example["draft_text"].replace("\n", " ")[:260]
                lines.append(
                    "- idx={idx}, id={id}, gold={gold}, pred={pred}, source={source}: {question}  \n  `{preview}`".format(
                        idx=example["dataset_index"],
                        id=example["id"],
                        gold=example["gold"],
                        pred=example["pred"],
                        source=example["source"],
                        question=example["question"],
                        preview=draft_preview,
                    )
                )
            lines.append("")
        break

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    record_paths = sorted(outputs_dir.glob("*/records.jsonl"))
    if not record_paths:
        raise FileNotFoundError(f"No records.jsonl files found under {outputs_dir}")

    results = [analyze_file(path, args.examples_per_type) for path in record_paths]
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(results, output_md)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
