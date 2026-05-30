"""
这个脚本用于分析 proposal-refine 方法是否真的带来了加速。

它同时报告两类指标：

1. 更通用的推理成本指标（推荐对外报告）
   - 平均单样本延迟：mean latency (sec/sample)
   - 吞吐：throughput (samples/sec)
   - 平均解码迭代数：average decoding iterations
   - 相对迭代压缩率：iteration reduction ratio

2. 速度提升指标（相对 baseline）
   - 理论 step 加速（theoretical step speedup）
   - 实际 wall-clock 加速（wall-clock speedup）

这些指标中：

- latency / throughput 是最通用的系统指标；
- average decoding iterations / iteration reduction 更适合 diffusion/mask-predict 方法；
- speedup 则适合直接和 baseline 比较。

输入：

- 一个 baseline summary（通常来自 `VRG/outputs/textvqa_stepwise_x0/summary.json`）
- 一个或多个 proposal-refine summary，或者一个 sweep 根目录

输出：

- 每个配置的 final EM、相对 baseline 的性能差
- latency / throughput
- average decoding iterations / iteration reduction
- 理论 step speedup / wall-clock speedup
- proposal 阶段平均重 mask 数量
- refine 阶段平均实际步数
"""

import argparse
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_summary_paths(root_dir):
    root = Path(root_dir)
    return sorted(
        path for path in root.rglob("summary.json")
        if path.is_file() and path.name == "summary.json"
    )


def infer_baseline_final_em(summary):
    step_summary = summary.get("step_summary", [])
    if not step_summary:
        return None
    return step_summary[-1]["mean_exact_match"]


def infer_baseline_total_steps(summary):
    step_summary = summary.get("step_summary", [])
    if step_summary:
        return len(step_summary)
    generation = summary.get("generation", {})
    max_new_tokens = generation.get("max_new_tokens")
    if max_new_tokens is not None:
        return int(max_new_tokens)
    return None


def summarize_records(records_path, proposal_step):
    rows = []
    with Path(records_path).open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            meta = record.get("meta", {})
            rows.append(
                {
                    "late_refine_steps_run": int(meta.get("late_refine_steps_run", 0)),
                    "num_remasked_positions": int(meta.get("num_remasked_positions", 0)),
                    "actual_total_steps": int(proposal_step) + int(meta.get("late_refine_steps_run", 0)),
                }
            )
    if not rows:
        return {
            "avg_late_refine_steps_run": None,
            "avg_num_remasked_positions": None,
            "avg_actual_total_steps": None,
        }
    return {
        "avg_late_refine_steps_run": sum(row["late_refine_steps_run"] for row in rows) / len(rows),
        "avg_num_remasked_positions": sum(row["num_remasked_positions"] for row in rows) / len(rows),
        "avg_actual_total_steps": sum(row["actual_total_steps"] for row in rows) / len(rows),
    }


def build_row(baseline_summary, proposal_summary_path):
    proposal_summary = load_json(proposal_summary_path)
    proposal_cfg = proposal_summary.get("proposal_refine", {})
    proposal_step = int(proposal_cfg.get("proposal_step"))
    records_path = Path(proposal_summary_path).with_name("records.jsonl")
    records_stats = summarize_records(records_path, proposal_step) if records_path.exists() else {
        "avg_late_refine_steps_run": None,
        "avg_num_remasked_positions": None,
        "avg_actual_total_steps": None,
    }

    baseline_final_em = infer_baseline_final_em(baseline_summary)
    baseline_total_steps = infer_baseline_total_steps(baseline_summary)
    baseline_mean_elapsed = baseline_summary.get("mean_elapsed_sec")
    baseline_throughput = None
    if baseline_mean_elapsed and baseline_mean_elapsed > 0:
        baseline_throughput = 1.0 / baseline_mean_elapsed

    final_em = proposal_summary.get("final_mean_exact_match")
    proposal_em = proposal_summary.get("proposal_mean_exact_match")
    method_mean_elapsed = proposal_summary.get("mean_elapsed_sec")
    method_throughput = None
    if method_mean_elapsed and method_mean_elapsed > 0:
        method_throughput = 1.0 / method_mean_elapsed
    avg_actual_total_steps = records_stats["avg_actual_total_steps"]

    iteration_reduction_ratio = None
    if baseline_total_steps and avg_actual_total_steps:
        iteration_reduction_ratio = 1.0 - (avg_actual_total_steps / baseline_total_steps)

    theoretical_step_speedup = None
    if baseline_total_steps and avg_actual_total_steps:
        theoretical_step_speedup = baseline_total_steps / avg_actual_total_steps

    wall_clock_speedup = None
    if baseline_mean_elapsed and method_mean_elapsed:
        wall_clock_speedup = baseline_mean_elapsed / method_mean_elapsed

    return {
        "summary_path": str(proposal_summary_path),
        "run_name": Path(proposal_summary_path).parent.name,
        "proposal_step": proposal_step,
        "proposal_remask_ratio": proposal_cfg.get("proposal_remask_ratio"),
        "late_refine_steps_requested": proposal_cfg.get("late_refine_steps"),
        "proposal_mean_exact_match": proposal_em,
        "final_mean_exact_match": final_em,
        "em_gap_vs_baseline": (final_em - baseline_final_em) if baseline_final_em is not None and final_em is not None else None,
        "mean_gain_from_refine": proposal_summary.get("mean_gain_from_refine"),
        "avg_late_refine_steps_run": records_stats["avg_late_refine_steps_run"],
        "avg_num_remasked_positions": records_stats["avg_num_remasked_positions"],
        "avg_decoding_iterations": avg_actual_total_steps,
        "baseline_decoding_iterations": baseline_total_steps,
        "iteration_reduction_ratio": iteration_reduction_ratio,
        "theoretical_step_speedup": theoretical_step_speedup,
        "baseline_mean_latency_sec": baseline_mean_elapsed,
        "method_mean_latency_sec": method_mean_elapsed,
        "baseline_throughput_samples_per_sec": baseline_throughput,
        "method_throughput_samples_per_sec": method_throughput,
        "wall_clock_speedup": wall_clock_speedup,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze speed/accuracy tradeoffs of proposal-refine runs."
    )
    parser.add_argument(
        "--baseline-summary",
        default="VRG/outputs/textvqa_stepwise_x0/summary.json",
    )
    parser.add_argument(
        "--proposal-summaries",
        default="",
        help="Comma-separated summary.json paths for proposal-refine runs.",
    )
    parser.add_argument(
        "--sweep-root",
        default="",
        help="If set, automatically discover proposal-refine summary.json files under this directory.",
    )
    parser.add_argument(
        "--output-json",
        default="VRG/outputs/textvqa_proposal_refine_speed_summary.json",
    )
    args = parser.parse_args()

    baseline_summary = load_json(args.baseline_summary)

    summary_paths = []
    if args.proposal_summaries.strip():
        summary_paths.extend(
            Path(item.strip()) for item in args.proposal_summaries.split(",") if item.strip()
        )
    if args.sweep_root.strip():
        summary_paths.extend(find_summary_paths(args.sweep_root))

    # 去重
    unique_paths = []
    seen = set()
    for path in summary_paths:
        path = Path(path)
        if str(path) not in seen:
            seen.add(str(path))
            unique_paths.append(path)

    if not unique_paths:
        raise ValueError("No proposal-refine summary files found. Pass --proposal-summaries or --sweep-root.")

    rows = [build_row(baseline_summary, path) for path in unique_paths]
    rows.sort(
        key=lambda item: (
            -(item.get("final_mean_exact_match") or float("-inf")),
            -(item.get("theoretical_step_speedup") or float("-inf")),
        )
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote speed summary to {output_path}")
    for row in rows:
        throughput = row.get("method_throughput_samples_per_sec")
        print(
            f"{row['run_name']}: "
            f"final_em={row['final_mean_exact_match']:.4f}, "
            f"em_gap={row['em_gap_vs_baseline']:.4f}, "
            f"latency={row['method_mean_latency_sec']:.3f}s, "
            f"throughput={throughput:.3f}/s, "
            f"iters={row['avg_decoding_iterations']:.2f}, "
            f"step_speedup={row['theoretical_step_speedup']:.3f}, "
            f"wall_clock_speedup={row['wall_clock_speedup']:.3f}"
        )


if __name__ == "__main__":
    main()
