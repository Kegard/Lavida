"""
分析目标：

1. 读取一份原生 stepwise x0 日志，以及一份 proposal-refine 日志。
2. 先找出“same-set”样本：
   - 原生路径在某个 native_compare_step 之后自然剩余的 mask 位置集合；
   - proposal-refine 在 proposal 之后主动 remask 的位置集合；
   - 如果这两个集合完全相同，则视为 same-set。
3. 在 same-set 样本里，继续比较：
   - 原生路径后续每一步真正写回的 state_text；
   - proposal-refine 每一步 refine 后的 refine_state_text；
   - 两条轨迹是否相同；
   - 哪一条轨迹更早到达正确答案。

这个脚本的用途不是重新评估模型，而是对已有日志做“轨迹层面”的对齐分析，
帮助判断 proposal-refine 的提升是否更可能来自后续 refinement dynamics 的变化。
"""

import argparse
import json
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Compare native and proposal-refine trajectories on same-set samples.")
    parser.add_argument("--native-records", required=True, help="Path to native stepwise records.jsonl.")
    parser.add_argument("--proposal-records", required=True, help="Path to proposal-refine records.jsonl.")
    parser.add_argument(
        "--native-compare-step",
        type=int,
        required=True,
        help="Native step after which the remaining mask set is compared with proposal remask set.",
    )
    parser.add_argument(
        "--native-final-step",
        type=int,
        default=None,
        help="Last native step to include when comparing late trajectories. Defaults to total_denoising_steps in each sample.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=10,
        help="How many differing examples to print.",
    )
    return parser.parse_args()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def find_first_correct_step(step_like_records):
    for item in step_like_records:
        if float(item.get("exact_match", 0.0)) > 0.0:
            return int(item.get("step", item.get("refine_step", -1)))
    return None


def normalize_native_late_records(step_results, compare_step, final_step):
    late = []
    for item in step_results:
        step = int(item["step"])
        if compare_step < step <= final_step:
            late.append(
                {
                    "step": step,
                    "text": item.get("state_text", item.get("candidate_text", "")),
                    "exact_match": float(item.get("exact_match", 0.0)),
                    "selected_positions": item.get("selected_positions", []),
                }
            )
    return late


def normalize_refine_records(refine_records):
    late = []
    for item in refine_records:
        late.append(
            {
                "step": int(item["refine_step"]),
                "text": item.get("refine_state_text", item.get("candidate_text", "")),
                "exact_match": float(item.get("exact_match", 0.0)),
                "selected_positions": item.get("selected_positions", []),
            }
        )
    return late


def main():
    args = parse_args()
    native_rows = load_jsonl(args.native_records)
    proposal_rows = load_jsonl(args.proposal_records)
    if len(native_rows) != len(proposal_rows):
        raise ValueError("Native and proposal logs must have the same number of samples.")

    stats = Counter()
    examples = []

    for native_row, proposal_row in zip(native_rows, proposal_rows):
        if native_row["dataset_index"] != proposal_row["dataset_index"]:
            raise ValueError("dataset_index mismatch between native and proposal logs.")

        prefix_length = int(native_row["meta"]["prefix_length"])
        total_native_steps = int(native_row["meta"]["total_denoising_steps"])
        native_final_step = args.native_final_step or total_native_steps

        selected_answer_positions = set()
        for item in native_row["step_results"]:
            if int(item["step"]) <= args.native_compare_step:
                for pos in item["selected_positions"]:
                    selected_answer_positions.add(int(pos) - prefix_length)

        answer_length = len(proposal_row["proposal_answer_ids"])
        native_remaining = set(range(answer_length)) - selected_answer_positions
        proposal_remask = {int(item["answer_position"]) for item in proposal_row["remasked_positions"]}

        if native_remaining == proposal_remask:
            stats["same_set"] += 1
            native_late = normalize_native_late_records(
                native_row["step_results"],
                compare_step=args.native_compare_step,
                final_step=native_final_step,
            )
            refine_late = normalize_refine_records(proposal_row["refine_records"])

            native_texts = [item["text"] for item in native_late]
            refine_texts = [item["text"] for item in refine_late]
            if native_texts == refine_texts:
                stats["same_set_same_trajectory"] += 1
            else:
                stats["same_set_diff_trajectory"] += 1

            native_first_correct = find_first_correct_step(native_late)
            refine_first_correct = find_first_correct_step(refine_late)
            if native_first_correct is None and refine_first_correct is None:
                stats["same_set_neither_correct"] += 1
            elif native_first_correct is None:
                stats["same_set_refine_only_correct"] += 1
            elif refine_first_correct is None:
                stats["same_set_native_only_correct"] += 1
            elif refine_first_correct < native_first_correct - args.native_compare_step:
                stats["same_set_refine_earlier"] += 1
            elif refine_first_correct > native_first_correct - args.native_compare_step:
                stats["same_set_native_earlier"] += 1
            else:
                stats["same_set_tie_first_correct"] += 1

            if (
                len(examples) < args.show_examples
                and native_texts != refine_texts
            ):
                examples.append(
                    {
                        "dataset_index": native_row["dataset_index"],
                        "question_id": native_row["question_id"],
                        "question": native_row["question"].split("\n")[0],
                        "native_remaining": sorted(native_remaining),
                        "proposal_remask": sorted(proposal_remask),
                        "native_late_texts": native_texts,
                        "refine_late_texts": refine_texts,
                        "native_final_text": native_row.get("final_state_text", ""),
                        "proposal_final_text": proposal_row.get("final_text", ""),
                    }
                )
        else:
            stats["diff_set"] += 1

    print("=== Summary ===")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")

    if stats["same_set"] > 0:
        same_set = stats["same_set"]
        diff_traj = stats["same_set_diff_trajectory"]
        print(f"same_set_diff_trajectory_rate: {diff_traj / same_set:.4f}")

    print("\n=== Examples ===")
    for item in examples:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
