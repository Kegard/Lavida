import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]
EXPLICIT_SOURCES = {"answer_colon", "boxed", "answer_tag", "paren_letter"}
KEYWORD_SOURCES = {"choice_text_keyword", "letter_token_upper", "letter_token_lower"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify draft_text errors as semantic local/global errors."
    )
    parser.add_argument(
        "--records",
        default="M3CoT/PostMaSK/outputs/postmask_sr0p5_d16_p16_conf_r4_seed42_n400/records.jsonl",
        help="PostMaSK records.jsonl file to classify.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="If set, classify every */records.jsonl under this directory and write an all-experiment summary.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="M3CoT/PostMaSK/draft_semantic_error_labels.jsonl",
    )
    parser.add_argument(
        "--output-md",
        default="M3CoT/PostMaSK/draft_semantic_error_summary.md",
    )
    return parser.parse_args()


def extract_answer_with_source(text, choices):
    valid_letters = "".join(ALPHA_MAP[: len(choices)])
    explicit_patterns = [
        ("answer_colon", rf"Answer\s*:\s*[\(\[]?\s*([{valid_letters}{valid_letters.lower()}])\b"),
        ("boxed", rf"\\boxed\s*\{{?\s*([{valid_letters}{valid_letters.lower()}])\s*\}}?"),
        ("answer_tag", rf"\[Answer\]\s*[\(\[]?\s*([{valid_letters}{valid_letters.lower()}])\b"),
        ("paren_letter", rf"\(([{valid_letters}{valid_letters.lower()}])\)"),
    ]
    for source, pattern in explicit_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper(), source

    matches = []
    lowered = text.lower()
    for idx, choice in enumerate(choices):
        if choice.lower() in lowered:
            matches.append(ALPHA_MAP[idx])
    if matches:
        return matches[-1], "choice_text_keyword"

    tokens = re.sub(r"[\n.,!?]", " ", text).split()
    matches = [ALPHA_MAP[idx] for idx in range(len(choices)) if ALPHA_MAP[idx] in tokens]
    if matches:
        return matches[-1], "letter_token_upper"

    matches = [ALPHA_MAP[idx] for idx in range(len(choices)) if ALPHA_MAP[idx].lower() in tokens]
    if matches:
        return matches[-1], "letter_token_lower"

    return "FAILED", "failed"


def final_sentences(text, limit=3):
    normalized = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return " ".join(parts[-limit:])


def has_answer_like_conclusion(text):
    conclusion_patterns = [
        r"therefore",
        r"correct answer",
        r"correct choice",
        r"the answer is",
        r"matches option",
        r"corresponds to option",
        r"should be selected",
        r"\\boxed",
    ]
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in conclusion_patterns)


def is_truncated_or_insufficient(text):
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped.split()) < 28:
        return True
    if re.search(r"(:|\$|-\s*|,|\()\s*$", stripped):
        return True
    if "<|mdm_mask|>" in stripped:
        return True
    if not has_answer_like_conclusion(stripped):
        return True
    return False


def is_generic_unsupported_answer(text):
    lowered = text.lower()
    generic_patterns = [
        r"analyze the options provided:\s*-?\s*a\s*-?\s*b\s*-?\s*c\s*-?\s*d",
        r"the correct answer is option one",
    ]
    if any(re.search(pattern, lowered) for pattern in generic_patterns):
        return True
    repeated_fillers = len(re.findall(r"\b(the|option|correct|answer|choice)\b(?:\s+\1\b){2,}", lowered))
    return repeated_fillers > 0


def choose_local_span(text, pred, source):
    normalized = re.sub(r"\s+", " ", text.strip())
    if pred != "FAILED":
        option_patterns = [
            rf"(?:option|answer|choice)\s*{re.escape(pred)}\b",
            rf"\\boxed\s*\{{?\s*{re.escape(pred)}",
            rf"\b{re.escape(pred)}\b",
        ]
        for pattern in option_patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                start = max(0, match.start() - 100)
                end = min(len(normalized), match.end() + 140)
                return normalized[start:end]
    return final_sentences(text)


def classify_semantic_error(record):
    choices = record["choices"]
    gold = record["answer"]
    if isinstance(gold, int):
        gold = ALPHA_MAP[gold]
    draft_text = record.get("draft_text", "")
    pred, source = extract_answer_with_source(draft_text, choices)
    if pred == gold:
        return None

    if source == "failed":
        label = "global"
        subtype = "insufficient_or_incomplete_cot"
        rationale = "The draft has no extractable answer and does not contain enough completed reasoning to recover the answer by editing one local span."
    elif source in KEYWORD_SOURCES and not has_answer_like_conclusion(draft_text):
        label = "global"
        subtype = "metric_keyword_artifact_without_answer"
        rationale = "A local keyword triggered the metric, but the CoT itself has no final answer; deleting or editing that keyword would not recover the answer."
    elif is_generic_unsupported_answer(draft_text):
        label = "global"
        subtype = "unsupported_reasoning_trajectory"
        rationale = "The draft jumps to an answer without enough local evidence; fixing an isolated span is insufficient."
    elif source in EXPLICIT_SOURCES:
        label = "local"
        subtype = "localized_wrong_step_or_span"
        rationale = "The draft has a completed answer-bearing conclusion; the wrong answer can be traced to a local observation, calculation, option mapping, or final reasoning step."
    elif is_truncated_or_insufficient(draft_text):
        label = "global"
        subtype = "insufficient_or_incomplete_cot"
        rationale = "The draft is truncated or lacks a completed answer-bearing reasoning step, so the error is not attributable to a single fixable local step."
    else:
        label = "local"
        subtype = "localized_wrong_step_or_span"
        rationale = "The draft contains a completed but wrong answer-bearing step; replacing that local observation, value, option mapping, or final step could recover the answer."

    return {
        "dataset_index": record.get("dataset_index"),
        "id": record.get("id"),
        "domain": record.get("domain"),
        "topic": record.get("topic"),
        "gold": gold,
        "pred": pred,
        "extract_source": source,
        "label": label,
        "subtype": subtype,
        "rationale": rationale,
        "local_span": choose_local_span(draft_text, pred, source) if label == "local" else "",
        "question": record.get("question", ""),
        "choices": choices,
        "draft_text": draft_text,
    }


def analyze(records_path):
    labels = []
    totals = Counter()
    by_topic = defaultdict(Counter)
    by_source = defaultdict(Counter)
    with records_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            label = classify_semantic_error(record)
            totals["total"] += 1
            if label is None:
                totals["draft_correct"] += 1
                continue
            labels.append(label)
            totals["draft_wrong"] += 1
            totals[label["label"]] += 1
            totals[label["subtype"]] += 1
            by_topic[label.get("topic") or "UNKNOWN"][label["label"]] += 1
            by_topic[label.get("topic") or "UNKNOWN"]["total"] += 1
            by_source[label["extract_source"]][label["label"]] += 1
            by_source[label["extract_source"]]["total"] += 1
    return labels, totals, by_topic, by_source


def pct(count, denom):
    if denom == 0:
        return "NA"
    return f"{count / denom:.3f}"


def write_markdown(path, records_path, labels, totals, by_topic, by_source):
    wrong = totals["draft_wrong"]
    lines = [
        "# Draft Semantic Error Classification",
        "",
        f"Records: `{records_path}`",
        "",
        "Definitions used:",
        "",
        "- `local`: the draft error is attributable to a specific local token span or reasoning step, and a minimal local edit could recover the answer.",
        "- `global`: the draft's semantic trajectory is incomplete, unsupported, or misaligned enough that fixing an isolated span is insufficient.",
        "",
        "Important convention: changing only the final `\\boxed{}` letter is not counted as a local repair unless the preceding CoT already contains a localized answer-bearing mistake that explains the wrong conclusion.",
        "",
        "## Summary",
        "",
        "| Total | Draft Correct | Draft Wrong | Local | Global |",
        "|---:|---:|---:|---:|---:|",
        f"| {totals['total']} | {totals['draft_correct']} | {wrong} | {totals['local']} ({pct(totals['local'], wrong)}) | {totals['global']} ({pct(totals['global'], wrong)}) |",
        "",
        "## Global Subtypes",
        "",
        "| Subtype | Count | Ratio among draft errors |",
        "|---|---:|---:|",
    ]
    for subtype in [
        "insufficient_or_incomplete_cot",
        "metric_keyword_artifact_without_answer",
        "unsupported_reasoning_trajectory",
    ]:
        lines.append(f"| `{subtype}` | {totals[subtype]} | {pct(totals[subtype], wrong)} |")

    lines.extend(
        [
            "",
            "## By Extraction Source",
            "",
            "| Extract Source | Total Wrong | Local | Global |",
            "|---|---:|---:|---:|",
        ]
    )
    for source, counter in sorted(by_source.items()):
        lines.append(
            f"| `{source}` | {counter['total']} | {counter['local']} | {counter['global']} |"
        )

    lines.extend(
        [
            "",
            "## Topic Breakdown",
            "",
            "| Topic | Wrong | Local | Global |",
            "|---|---:|---:|---:|",
        ]
    )
    for topic, counter in sorted(by_topic.items(), key=lambda item: (-item[1]["total"], item[0])):
        lines.append(f"| `{topic}` | {counter['total']} | {counter['local']} | {counter['global']} |")

    lines.extend(["", "## Representative Local Examples", ""])
    for item in [label for label in labels if label["label"] == "local"][:12]:
        preview = item["draft_text"].replace("\n", " ")[:320]
        span = item["local_span"].replace("\n", " ")[:220]
        lines.append(
            f"- idx={item['dataset_index']}, id={item['id']}, gold={item['gold']}, pred={item['pred']}, source={item['extract_source']}: {item['question'].splitlines()[0]}"
        )
        lines.append(f"  - span: `{span}`")
        lines.append(f"  - draft: `{preview}`")

    lines.extend(["", "## Representative Global Examples", ""])
    for item in [label for label in labels if label["label"] == "global"][:12]:
        preview = item["draft_text"].replace("\n", " ")[:320]
        lines.append(
            f"- idx={item['dataset_index']}, id={item['id']}, gold={item['gold']}, pred={item['pred']}, subtype={item['subtype']}: {item['question'].splitlines()[0]}"
        )
        lines.append(f"  - draft: `{preview}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.outputs_dir:
        results = []
        for records_path in sorted(Path(args.outputs_dir).glob("*/records.jsonl")):
            labels, totals, by_topic, by_source = analyze(records_path)
            results.append(
                {
                    "experiment": records_path.parent.name,
                    "records": str(records_path),
                    "total": totals["total"],
                    "draft_correct": totals["draft_correct"],
                    "draft_wrong": totals["draft_wrong"],
                    "local": totals["local"],
                    "global": totals["global"],
                    "subtypes": {
                        "insufficient_or_incomplete_cot": totals["insufficient_or_incomplete_cot"],
                        "metric_keyword_artifact_without_answer": totals["metric_keyword_artifact_without_answer"],
                        "unsupported_reasoning_trajectory": totals["unsupported_reasoning_trajectory"],
                    },
                    "by_source": {key: dict(value) for key, value in sorted(by_source.items())},
                    "by_topic": {key: dict(value) for key, value in sorted(by_topic.items())},
                }
            )
        output_json = Path(args.output_jsonl).with_suffix(".all.json")
        output_md = Path(args.output_md).with_name("draft_semantic_error_summary_all.md")
        output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# Draft Semantic Error Classification All Experiments",
            "",
            "| Experiment | N | Draft Wrong | Local | Global | Incomplete | Keyword Artifact | Unsupported |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in results:
            wrong = item["draft_wrong"]
            lines.append(
                "| `{}` | {} | {} | {} ({}) | {} ({}) | {} | {} | {} |".format(
                    item["experiment"],
                    item["total"],
                    wrong,
                    item["local"],
                    pct(item["local"], wrong),
                    item["global"],
                    pct(item["global"], wrong),
                    item["subtypes"]["insufficient_or_incomplete_cot"],
                    item["subtypes"]["metric_keyword_artifact_without_answer"],
                    item["subtypes"]["unsupported_reasoning_trajectory"],
                )
            )
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {output_json}")
        print(f"Wrote {output_md}")
        return

    records_path = Path(args.records)
    labels, totals, by_topic, by_source = analyze(records_path)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for item in labels:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_markdown(Path(args.output_md), records_path, labels, totals, by_topic, by_source)
    print(f"Wrote {output_jsonl}")
    print(f"Wrote {args.output_md}")
    print(f"Draft wrong={totals['draft_wrong']} local={totals['local']} global={totals['global']}")


if __name__ == "__main__":
    main()
