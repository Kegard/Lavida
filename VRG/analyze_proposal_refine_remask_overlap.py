"""
这个脚本用于比较两次 proposal-refine 实验在“相同样本上到底 remask 了哪些 token”。

典型用途：

1. 保持 proposal-step / remask-ratio / late-refine-steps 不变；
2. 只改变 proposal score 的定义，例如：
   - 一次使用 `confidence`
   - 一次使用 `hybrid_visual_delta`
3. 然后比较两次实验中，每个样本被 remask 的 token 集合是否相同。

这个脚本回答的问题是：

- 两次实验是否真的在“行为上”不同；
- 有多少样本 remask 集合完全相同；
- 平均 overlap / Jaccard 有多高；
- 如果不一样，平均差多少个 token；
- 哪些样本差异最大，方便后续人工分析。

注意：

- 这里比较的是 `records.jsonl` 里的 `remasked_positions`；
- 默认使用 `answer_position` 作为集合元素，因为这个位置最稳定，且直接对应答案内部位置；
- 如果需要，也可以切换成 `sequence_position`。
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare remasked token sets between two proposal-refine runs."
    )
    parser.add_argument("--records-a", required=True, help="First records.jsonl file.")
    parser.add_argument("--records-b", required=True, help="Second records.jsonl file.")
    parser.add_argument(
        "--position-key",
        default="answer_position",
        choices=["answer_position", "sequence_position"],
        help="Which position field inside remasked_positions is used to build the token set.",
    )
    parser.add_argument(
        "--top-k-diff",
        type=int,
        default=20,
        help="Show the top-k samples with the largest remask-set differences.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the full summary JSON.",
    )
    return parser.parse_args()


def load_records(path: Path, position_key: str):
    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            record = json.loads(line)
            dataset_index = int(record["dataset_index"])
            question_id = record.get("question_id")
            remasked_positions = record.get("remasked_positions", [])
            remask_set = {
                int(item[position_key])
                for item in remasked_positions
                if position_key in item
            }
            records[dataset_index] = {
                "dataset_index": dataset_index,
                "question_id": question_id,
                "question": record.get("question"),
                "proposal_text": record.get("proposal_text"),
                "final_text": record.get("final_text"),
                "proposal_exact_match": record.get("proposal_exact_match"),
                "final_exact_match": record.get("final_exact_match"),
                "remask_set": remask_set,
                "remasked_positions": remasked_positions,
                "line_idx": line_idx,
            }
    return records


def jaccard(set_a, set_b):
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def overlap_ratio(set_a, set_b):
    denom = min(len(set_a), len(set_b))
    if denom == 0:
        return 1.0 if len(set_a) == len(set_b) == 0 else 0.0
    return len(set_a & set_b) / denom


def main():
    args = parse_args()
    path_a = Path(args.records_a)
    path_b = Path(args.records_b)

    records_a = load_records(path_a, args.position_key)
    records_b = load_records(path_b, args.position_key)

    shared_indices = sorted(set(records_a) & set(records_b))
    only_a = sorted(set(records_a) - set(records_b))
    only_b = sorted(set(records_b) - set(records_a))

    per_sample = []
    exact_same = 0
    both_empty = 0
    sum_jaccard = 0.0
    sum_overlap = 0.0
    sum_intersection = 0
    sum_union = 0
    sum_size_a = 0
    sum_size_b = 0

    for dataset_index in shared_indices:
        rec_a = records_a[dataset_index]
        rec_b = records_b[dataset_index]
        set_a = rec_a["remask_set"]
        set_b = rec_b["remask_set"]
        inter = set_a & set_b
        only_in_a = sorted(set_a - set_b)
        only_in_b = sorted(set_b - set_a)
        union = set_a | set_b
        jac = jaccard(set_a, set_b)
        ov = overlap_ratio(set_a, set_b)
        same = set_a == set_b

        if same:
            exact_same += 1
        if not set_a and not set_b:
            both_empty += 1

        sum_jaccard += jac
        sum_overlap += ov
        sum_intersection += len(inter)
        sum_union += len(union)
        sum_size_a += len(set_a)
        sum_size_b += len(set_b)

        per_sample.append(
            {
                "dataset_index": dataset_index,
                "question_id": rec_a["question_id"],
                "question": rec_a["question"],
                "set_a": sorted(set_a),
                "set_b": sorted(set_b),
                "intersection": sorted(inter),
                "only_in_a": only_in_a,
                "only_in_b": only_in_b,
                "same_set": same,
                "jaccard": jac,
                "overlap_ratio": ov,
                "size_a": len(set_a),
                "size_b": len(set_b),
                "symmetric_diff_size": len(only_in_a) + len(only_in_b),
                "final_exact_match_a": rec_a["final_exact_match"],
                "final_exact_match_b": rec_b["final_exact_match"],
                "proposal_text_a": rec_a["proposal_text"],
                "proposal_text_b": rec_b["proposal_text"],
                "final_text_a": rec_a["final_text"],
                "final_text_b": rec_b["final_text"],
            }
        )

    shared_count = len(shared_indices)
    summary = {
        "records_a": str(path_a),
        "records_b": str(path_b),
        "position_key": args.position_key,
        "num_records_a": len(records_a),
        "num_records_b": len(records_b),
        "num_shared_samples": shared_count,
        "num_only_in_a": len(only_a),
        "num_only_in_b": len(only_b),
        "num_exact_same_sets": exact_same,
        "ratio_exact_same_sets": (exact_same / shared_count) if shared_count else None,
        "num_both_empty_sets": both_empty,
        "ratio_both_empty_sets": (both_empty / shared_count) if shared_count else None,
        "mean_jaccard": (sum_jaccard / shared_count) if shared_count else None,
        "mean_overlap_ratio": (sum_overlap / shared_count) if shared_count else None,
        "micro_jaccard": (sum_intersection / sum_union) if sum_union > 0 else 1.0,
        "mean_remask_size_a": (sum_size_a / shared_count) if shared_count else None,
        "mean_remask_size_b": (sum_size_b / shared_count) if shared_count else None,
        "shared_dataset_indices_preview": shared_indices[:10],
        "only_in_a_preview": only_a[:10],
        "only_in_b_preview": only_b[:10],
    }

    top_diff_samples = sorted(
        per_sample,
        key=lambda item: (item["symmetric_diff_size"], 1.0 - item["jaccard"], abs(item["size_a"] - item["size_b"])),
        reverse=True,
    )[: args.top_k_diff]
    summary["top_diff_samples"] = top_diff_samples

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"records_a: {path_a}")
    print(f"records_b: {path_b}")
    print(f"position_key: {args.position_key}")
    print(f"shared_samples: {shared_count}")
    print(f"only_in_a: {len(only_a)}")
    print(f"only_in_b: {len(only_b)}")
    print(f"exact_same_sets: {exact_same} ({summary['ratio_exact_same_sets']:.4f})")
    print(f"both_empty_sets: {both_empty} ({summary['ratio_both_empty_sets']:.4f})")
    print(f"mean_jaccard: {summary['mean_jaccard']:.4f}")
    print(f"mean_overlap_ratio: {summary['mean_overlap_ratio']:.4f}")
    print(f"micro_jaccard: {summary['micro_jaccard']:.4f}")
    print(f"mean_remask_size_a: {summary['mean_remask_size_a']:.4f}")
    print(f"mean_remask_size_b: {summary['mean_remask_size_b']:.4f}")

    print("\nTop differing samples:")
    for item in top_diff_samples:
        print(
            f"[dataset_index={item['dataset_index']}] "
            f"qid={item['question_id']} same={item['same_set']} "
            f"jaccard={item['jaccard']:.4f} "
            f"size_a={item['size_a']} size_b={item['size_b']} "
            f"symdiff={item['symmetric_diff_size']}"
        )
        print(f"  only_in_a={item['only_in_a']}")
        print(f"  only_in_b={item['only_in_b']}")
        if item.get("question"):
            print(f"  question={item['question']}")


if __name__ == "__main__":
    main()
