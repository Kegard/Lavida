import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]
DOMAIN_ORDER = ["science", "commonsense", "mathematics"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate M3CoT prediction JSONL with the official M3CoT answer-matching logic."
    )
    parser.add_argument("--prediction-jsonl", "--metric-path", dest="prediction_jsonl", required=True)
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--metric-by", default="topic", choices=["all", "domain", "topic"])
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def normalize_answer(answer):
    if isinstance(answer, int):
        return ALPHA_MAP[answer]
    return str(answer)


def extract_official_answer(text, choices):
    if "[Answer]" in text:
        text = text.split("[Answer]")[-1].split("[Rationale]")[0].split("[Context]")[0]

    pattern = re.compile(r"\(([A-Za-z])\)")
    res = pattern.findall(text)
    if len(res) >= 1:
        return res[-1].upper()

    res = []
    for i, choice in enumerate(choices):
        if choice.lower() in text.lower():
            res.append(ALPHA_MAP[i])
    if len(res) >= 1:
        return res[-1]

    res = []
    for i, _ in enumerate(choices):
        text = re.sub(r"[\n.,!?]", " ", text)
        if ALPHA_MAP[i] in text.split(" "):
            res.append(ALPHA_MAP[i])
    if len(res) >= 1:
        return res[-1]

    res = []
    for i, _ in enumerate(choices):
        text = re.sub(r"[\n.,!?]", " ", text)
        if ALPHA_MAP[i].lower() in text.split(" "):
            res.append(ALPHA_MAP[i])
    if len(res) >= 1:
        return res[-1]

    return "FAILED"


def get_prediction_text(record):
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        return str(messages[-1])
    for key in ("prediction", "text", "answer", "output"):
        if key in record:
            return str(record[key])
    return ""


def load_split_map(dataset_path, split):
    import datasets

    dataset = datasets.load_dataset(dataset_path, split=split)
    return {doc["id"]: doc for doc in dataset}


def sorted_domains(domain_dict):
    known = [domain for domain in DOMAIN_ORDER if domain in domain_dict]
    extra = sorted(domain for domain in domain_dict if domain not in DOMAIN_ORDER)
    return known + extra


def summarize_counts(total, correct):
    summary = {}
    for domain in sorted_domains(total):
        summary[domain] = {}
        for topic in sorted(total[domain]):
            topic_total = total[domain][topic]
            topic_correct = correct[domain][topic]
            summary[domain][topic] = {
                "acc": topic_correct / topic_total if topic_total else 0.0,
                "total": topic_total,
                "correct": topic_correct,
            }
    return summary


def summarize_domains(total, correct):
    summary = {}
    for domain in sorted_domains(total):
        domain_total = sum(total[domain].values())
        domain_correct = sum(correct[domain].values())
        summary[domain] = {
            "acc": domain_correct / domain_total if domain_total else 0.0,
            "total": domain_total,
            "correct": domain_correct,
        }
    return summary


def print_summary(metric_by, topic_summary, domain_summary, overall):
    if metric_by == "topic":
        print("domain\ttopic\tAcc\tCorrect/Total")
        for domain in sorted_domains(topic_summary):
            for topic in sorted(topic_summary[domain]):
                item = topic_summary[domain][topic]
                print(f"{domain}\t{topic}\t{item['acc'] * 100.0:.2f}\t{item['correct']}/{item['total']}")
    elif metric_by == "domain":
        print("domain\tAcc\tCorrect/Total")
        for domain in sorted_domains(domain_summary):
            item = domain_summary[domain]
            print(f"{domain}\t{item['acc'] * 100.0:.2f}\t{item['correct']}/{item['total']}")
    print(f"Total: {overall['total']}, Correct: {overall['correct']}, Acc: {overall['acc'] * 100.0:.2f}%")


def evaluate(records, split_map):
    total = defaultdict(lambda: defaultdict(int))
    correct = defaultdict(lambda: defaultdict(int))
    seen_ids = set()
    duplicate_ids = []
    skipped_ids = []
    evaluated = 0

    for record in records:
        sample_id = record.get("id")
        if sample_id in seen_ids:
            duplicate_ids.append(sample_id)
            continue
        seen_ids.add(sample_id)
        doc = split_map.get(sample_id)
        if doc is None:
            skipped_ids.append(sample_id)
            continue

        choices = list(doc["choices"])
        gold = normalize_answer(doc["answer"])
        pred_text = get_prediction_text(record)
        pred = extract_official_answer(pred_text, choices)
        domain = doc["domain"]
        topic = doc["topic"]
        total[domain][topic] += 1
        if pred == gold:
            correct[domain][topic] += 1
        evaluated += 1

    topic_summary = summarize_counts(total, correct)
    domain_summary = summarize_domains(total, correct)
    total_count = sum(sum(topic_counts.values()) for topic_counts in total.values())
    correct_count = sum(sum(topic_counts.values()) for topic_counts in correct.values())
    overall = {
        "acc": correct_count / total_count if total_count else 0.0,
        "total": total_count,
        "correct": correct_count,
    }
    dataset_missing_ids = sorted(set(split_map) - seen_ids)
    diagnostics = {
        "records": len(records),
        "evaluated": evaluated,
        "duplicates": len(duplicate_ids),
        "duplicate_ids": duplicate_ids,
        "skipped_not_in_split": len(skipped_ids),
        "skipped_ids": skipped_ids,
        "missing_predictions": len(dataset_missing_ids),
        "missing_prediction_ids": dataset_missing_ids,
    }
    return overall, domain_summary, topic_summary, diagnostics


def main():
    args = parse_args()
    prediction_path = Path(args.prediction_jsonl)
    records = read_jsonl(prediction_path)
    split_map = load_split_map(args.dataset_path, args.split)
    overall, domain_summary, topic_summary, diagnostics = evaluate(records, split_map)
    report = {
        "prediction_jsonl": str(prediction_path),
        "dataset_path": args.dataset_path,
        "split": args.split,
        "metric_source": "LightChen233/M3CoT utils/metric.py judge_answer logic, with explicit split selection",
        "overall": overall,
        "domain": domain_summary,
        "topic": topic_summary,
        "diagnostics": diagnostics,
    }
    print_summary(args.metric_by, topic_summary, domain_summary, overall)
    if diagnostics["duplicates"] or diagnostics["skipped_not_in_split"] or diagnostics["missing_predictions"]:
        print(
            "Diagnostics: "
            f"duplicates={diagnostics['duplicates']}, "
            f"skipped_not_in_split={diagnostics['skipped_not_in_split']}, "
            f"missing_predictions={diagnostics['missing_predictions']}"
        )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(report, fout, ensure_ascii=False, indent=2)
        print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
