"""
这个脚本用于批量 sweep `run_textvqa_proposal_refine.py` 的超参数组合。

目标：

1. 自动枚举 proposal step / remask ratio / late refine steps 的组合；
2. 为每组组合创建独立输出目录，避免手工改路径；
3. 顺序调用 `VRG/run_textvqa_proposal_refine.py`；
4. 跑完后自动读取每组 `summary.json`，汇总成一个总表，方便比较。

推荐用途：

- 初步网格搜索：
  `proposal-step = 8,12,16,20`
  `proposal-remask-ratio = 0.1,0.25,0.5`
  `late-refine-steps = 2,4,8`

- 快速验证：
  先用 `--limit 100` 或 `--limit 200` 跑一轮小规模 sweep；
  再挑表现最好的几组参数扩大到更多样本。

注意：

- 这个脚本默认顺序执行，不做并行调度；
- 它会直接调用当前 Python 解释器（`sys.executable`）去运行主实验脚本；
- 如果输出目录已存在且已经有 `summary.json`，默认会跳过，除非显式传 `--overwrite`。
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_csv_ints(raw_value):
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_csv_floats(raw_value):
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def sanitize_float(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def build_run_name(proposal_step, remask_ratio, late_refine_steps):
    return (
        f"p{proposal_step}"
        f"_rr{sanitize_float(remask_ratio)}"
        f"_r{late_refine_steps}"
    )


def load_summary(summary_path):
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Batch sweep proposal-refine hyperparameters for TextVQA."
    )
    parser.add_argument("--proposal-steps", default="8,12,16,20")
    parser.add_argument("--proposal-remask-ratios", default="0.1,0.25,0.5")
    parser.add_argument("--late-refine-steps-list", default="2,4,8")
    parser.add_argument("--base-output-dir", default="VRG/outputs/textvqa_proposal_refine_sweep")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra raw args appended to each run, e.g. \"--device-map cuda:0 --print-every 20\"",
    )
    args = parser.parse_args()

    proposal_steps = parse_csv_ints(args.proposal_steps)
    remask_ratios = parse_csv_floats(args.proposal_remask_ratios)
    late_refine_steps_list = parse_csv_ints(args.late_refine_steps_list)

    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    sweep_summary_jsonl = base_output_dir / "sweep_summary.jsonl"
    sweep_summary_json = base_output_dir / "sweep_summary.json"

    experiment_rows = []
    extra_args = [item for item in args.extra_args.split(" ") if item]

    combinations = list(
        itertools.product(proposal_steps, remask_ratios, late_refine_steps_list)
    )
    print(f"Total runs: {len(combinations)}")

    for run_index, (proposal_step, remask_ratio, late_refine_steps) in enumerate(combinations, start=1):
        run_name = build_run_name(proposal_step, remask_ratio, late_refine_steps)
        output_dir = base_output_dir / run_name
        summary_path = output_dir / "summary.json"

        command = [
            sys.executable,
            "VRG/run_textvqa_proposal_refine.py",
            "--proposal-step",
            str(proposal_step),
            "--proposal-remask-ratio",
            str(remask_ratio),
            "--late-refine-steps",
            str(late_refine_steps),
            "--limit",
            str(args.limit),
            "--start-index",
            str(args.start_index),
            "--output-dir",
            str(output_dir),
        ] + extra_args

        print(f"[{run_index}/{len(combinations)}] {run_name}")
        print(" ".join(command))

        if summary_path.exists() and not args.overwrite:
            print(f"Skip existing run: {summary_path}")
        elif not args.dry_run:
            subprocess.run(command, check=True)

        if summary_path.exists():
            summary = load_summary(summary_path)
            experiment_rows.append(
                {
                    "run_name": run_name,
                    "output_dir": str(output_dir),
                    "proposal_step": proposal_step,
                    "proposal_remask_ratio": remask_ratio,
                    "late_refine_steps": late_refine_steps,
                    "proposal_mean_exact_match": summary.get("proposal_mean_exact_match"),
                    "final_mean_exact_match": summary.get("final_mean_exact_match"),
                    "mean_gain_from_refine": summary.get("mean_gain_from_refine"),
                    "num_improved_after_refine": summary.get("num_improved_after_refine"),
                    "num_worsened_after_refine": summary.get("num_worsened_after_refine"),
                    "num_unchanged_after_refine": summary.get("num_unchanged_after_refine"),
                    "mean_elapsed_sec": summary.get("mean_elapsed_sec"),
                }
            )

    experiment_rows.sort(
        key=lambda item: (
            -(item.get("final_mean_exact_match") or float("-inf")),
            -(item.get("mean_gain_from_refine") or float("-inf")),
        )
    )

    with sweep_summary_jsonl.open("w", encoding="utf-8") as fout:
        for row in experiment_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    sweep_summary_json.write_text(
        json.dumps(experiment_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote sweep summary to {sweep_summary_jsonl}")
    print(f"Wrote ranked sweep table to {sweep_summary_json}")


if __name__ == "__main__":
    main()
