import argparse
import json
import math
import statistics as st
import string
from collections import Counter, defaultdict
from pathlib import Path


SPECIAL_TOKENS = {"<|eot_id|>", "<|endoftext|>", "<|im_end|>", "<|im_start|>"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build word/span-level summaries from M3CoT transfer visual-gain records."
    )
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--visual-gain-threshold", type=float, default=0.2)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def load_rows(path):
    with open(path, "r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def token_display(token_text):
    return token_text.replace("\\n", "\n")


def is_special_or_format(token):
    text = token["token_text"]
    stripped = token_display(text).strip()
    if text in SPECIAL_TOKENS or stripped.startswith("<|"):
        return True
    if token.get("is_whitespace"):
        return True
    if not stripped:
        return True
    if stripped in {"boxed", "{", "}", "\\", "Answer", ":"}:
        return True
    return False


def is_punctuation_token(token):
    stripped = token_display(token["token_text"]).strip()
    return bool(stripped) and all(ch in string.punctuation for ch in stripped)


def starts_new_word(token_text):
    text = token_display(token_text)
    return text.startswith((" ", "\n", "\t"))


def clean_span_text(text):
    return " ".join(text.replace("\\n", " ").split())


def summarize_members(members):
    gains = [item["visual_gain"] for item in members]
    confidences = [item["confidence"] for item in members]
    return {
        "visual_gain_sum": sum(gains),
        "visual_gain_mean": sum(gains) / len(gains),
        "visual_gain_max": max(gains),
        "confidence_mean": sum(confidences) / len(confidences),
        "confidence_min": min(confidences),
    }


def build_word_spans(row):
    spans = []
    current = None

    def flush():
        nonlocal current
        if current is not None and current["text"]:
            members = current.pop("members")
            current.update(summarize_members(members))
            current["token_texts"] = [item["token_text"] for item in members]
            current["token_ids"] = [item["token_id"] for item in members]
            current["start_answer_position"] = members[0]["answer_position"]
            current["end_answer_position"] = members[-1]["answer_position"]
            current["transfer_step_min"] = min(item["transfer_step"] for item in members)
            current["transfer_step_max"] = max(item["transfer_step"] for item in members)
            current["num_tokens"] = len(members)
            current["is_high_visual_gain"] = current["visual_gain_sum"] > 0.2
            spans.append(current)
        current = None

    for token in row["token_records"]:
        if is_special_or_format(token):
            flush()
            continue
        if is_punctuation_token(token):
            flush()
            continue

        text_piece = token_display(token["token_text"])
        if current is None or starts_new_word(token["token_text"]):
            flush()
            current = {"text": clean_span_text(text_piece), "members": [token]}
        else:
            current["text"] = clean_span_text(current["text"] + text_piece)
            current["members"].append(token)

    flush()
    for span in spans:
        span.update(
            {
                "dataset_index": row["dataset_index"],
                "id": row["id"],
                "topic": row["topic"],
                "domain": row["domain"],
                "final_correct": row["final_correct"],
            }
        )
    return spans


def build_high_gain_runs(row, threshold):
    runs = []
    current = []

    def flush():
        nonlocal current
        if not current:
            return
        text = clean_span_text("".join(token_display(item["token_text"]) for item in current))
        if text:
            summary = summarize_members(current)
            runs.append(
                {
                    "text": text,
                    "dataset_index": row["dataset_index"],
                    "id": row["id"],
                    "topic": row["topic"],
                    "domain": row["domain"],
                    "final_correct": row["final_correct"],
                    "start_answer_position": current[0]["answer_position"],
                    "end_answer_position": current[-1]["answer_position"],
                    "transfer_step_min": min(item["transfer_step"] for item in current),
                    "transfer_step_max": max(item["transfer_step"] for item in current),
                    "num_tokens": len(current),
                    "token_texts": [item["token_text"] for item in current],
                    "token_ids": [item["token_id"] for item in current],
                    **summary,
                }
            )
        current = []

    for token in row["token_records"]:
        if is_special_or_format(token) or is_punctuation_token(token):
            flush()
            continue
        if token["visual_gain"] > threshold:
            current.append(token)
        else:
            flush()
    flush()
    return runs


def aggregate_by_text(spans, min_count, top_k):
    grouped = defaultdict(list)
    for span in spans:
        grouped[span["text"]].append(span)
    rows = []
    for text, items in grouped.items():
        if len(items) < min_count:
            continue
        sums = [item["visual_gain_sum"] for item in items]
        means = [item["visual_gain_mean"] for item in items]
        confs = [item["confidence_mean"] for item in items]
        rows.append(
            {
                "text": text,
                "count": len(items),
                "mean_visual_gain_sum": sum(sums) / len(sums),
                "median_visual_gain_sum": st.median(sums),
                "max_visual_gain_sum": max(sums),
                "mean_visual_gain_mean": sum(means) / len(means),
                "mean_confidence": sum(confs) / len(confs),
                "correct_ratio": sum(item["final_correct"] for item in items) / len(items),
                "topics": Counter(item["topic"] for item in items).most_common(5),
            }
        )
    return sorted(rows, key=lambda item: item["mean_visual_gain_sum"], reverse=True)[:top_k]


def top_occurrences(spans, top_k):
    return sorted(spans, key=lambda item: item["visual_gain_sum"], reverse=True)[:top_k]


def span_group_stats(spans):
    if not spans:
        return {}
    sums = [item["visual_gain_sum"] for item in spans]
    means = [item["visual_gain_mean"] for item in spans]
    return {
        "num_spans": len(spans),
        "mean_visual_gain_sum": sum(sums) / len(sums),
        "median_visual_gain_sum": st.median(sums),
        "mean_visual_gain_mean": sum(means) / len(means),
        "high_gain_span_ratio": sum(value > 0.2 for value in sums) / len(sums),
    }


def make_markdown(summary, top_word_types, top_word_occ, top_run_types, top_run_occ):
    lines = ["# Span-Level Transfer Visual Gain Analysis", ""]
    lines.append(f"- Samples: {summary['num_samples']}")
    lines.append(f"- Word spans: {summary['word_spans']['num_spans']}")
    lines.append(f"- High-gain runs: {summary['high_gain_runs']['num_spans']}")
    lines.append(f"- Word-span mean gain sum: {summary['word_spans']['mean_visual_gain_sum']:.4f}")
    lines.append(f"- Word-span high-gain ratio: {summary['word_spans']['high_gain_span_ratio']:.4f}")
    lines.append("")
    lines.append("## Top Word Span Types")
    lines.append("")
    for item in top_word_types[:30]:
        topics = ", ".join(f"{topic}:{count}" for topic, count in item["topics"])
        lines.append(
            f"- `{item['text']}`: count={item['count']}, mean_sum={item['mean_visual_gain_sum']:.3f}, "
            f"median_sum={item['median_visual_gain_sum']:.3f}, max_sum={item['max_visual_gain_sum']:.3f}, topics={topics}"
        )
    lines.append("")
    lines.append("## Top Word Span Occurrences")
    lines.append("")
    for item in top_word_occ[:30]:
        lines.append(
            f"- idx={item['dataset_index']}, topic={item['topic']}, correct={item['final_correct']}, "
            f"span=`{item['text']}`, gain_sum={item['visual_gain_sum']:.3f}, "
            f"gain_mean={item['visual_gain_mean']:.3f}, conf={item['confidence_mean']:.3f}, "
            f"pos={item['start_answer_position']}-{item['end_answer_position']}"
        )
    lines.append("")
    lines.append("## Top High-Gain Run Types")
    lines.append("")
    for item in top_run_types[:30]:
        topics = ", ".join(f"{topic}:{count}" for topic, count in item["topics"])
        lines.append(
            f"- `{item['text']}`: count={item['count']}, mean_sum={item['mean_visual_gain_sum']:.3f}, "
            f"median_sum={item['median_visual_gain_sum']:.3f}, max_sum={item['max_visual_gain_sum']:.3f}, topics={topics}"
        )
    lines.append("")
    lines.append("## Top High-Gain Run Occurrences")
    lines.append("")
    for item in top_run_occ[:30]:
        lines.append(
            f"- idx={item['dataset_index']}, topic={item['topic']}, correct={item['final_correct']}, "
            f"span=`{item['text']}`, gain_sum={item['visual_gain_sum']:.3f}, "
            f"gain_mean={item['visual_gain_mean']:.3f}, conf={item['confidence_mean']:.3f}, "
            f"pos={item['start_answer_position']}-{item['end_answer_position']}"
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    records_path = Path(args.records_jsonl)
    output_dir = Path(args.output_dir) if args.output_dir else records_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(records_path)
    word_spans = []
    high_gain_runs = []
    for row in rows:
        word_spans.extend(build_word_spans(row))
        high_gain_runs.extend(build_high_gain_runs(row, args.visual_gain_threshold))

    top_word_types = aggregate_by_text(word_spans, args.min_count, args.top_k)
    top_word_occ = top_occurrences(word_spans, args.top_k)
    top_run_types = aggregate_by_text(high_gain_runs, args.min_count, args.top_k)
    top_run_occ = top_occurrences(high_gain_runs, args.top_k)

    summary = {
        "records_jsonl": str(records_path),
        "num_samples": len(rows),
        "visual_gain_threshold": args.visual_gain_threshold,
        "confidence_threshold": args.confidence_threshold,
        "word_spans": span_group_stats(word_spans),
        "high_gain_runs": span_group_stats(high_gain_runs),
        "top_word_span_types": top_word_types,
        "top_word_span_occurrences": top_word_occ,
        "top_high_gain_run_types": top_run_types,
        "top_high_gain_run_occurrences": top_run_occ,
    }

    (output_dir / "span_level_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "span_level_analysis.md").write_text(
        make_markdown(summary, top_word_types, top_word_occ, top_run_types, top_run_occ),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'span_level_stats.json'}")
    print(f"Wrote {output_dir / 'span_level_analysis.md'}")


if __name__ == "__main__":
    main()
