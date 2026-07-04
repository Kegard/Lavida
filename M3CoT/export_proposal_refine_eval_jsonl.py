import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert M3CoT proposal/refine records into official custom-eval jsonl format."
    )
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--text-field",
        default="final_text",
        choices=["final_text", "proposal_text"],
        help="Which text field from records.jsonl should be evaluated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records_path = Path(args.records_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with records_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prediction = record[args.text_field]
            obj = {
                "id": record["id"],
                "choices": record["choices"],
                "answer": record["answer"],
                "domain": record["domain"],
                "topic": record["topic"],
                "method": "proposal_refine",
                "messages": [record.get("question", ""), prediction],
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} records to {output_path}", flush=True)


if __name__ == "__main__":
    main()
