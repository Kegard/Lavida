import argparse
import json
import sys
import time
from pathlib import Path

import datasets
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "eval"
M3COT_ROOT = REPO_ROOT / "M3CoT"
for path in (REPO_ROOT, EVAL_ROOT, M3COT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from M3CoT.StepMask.free_correction import generate_with_free_correction
from M3CoT.run_m3cot_stepwise_x0 import prepare_prefix
from M3CoT.utils.metric import judge_answer
from Scale_Attention.reweight_patch import get_torch_dtype, maybe_disable_torch_compile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Training-Free Self-Correction style stepwise remasking on M3CoT."
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-mode", default="sequential", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="M3CoT/outputs/free_correction")

    parser.add_argument("--pretrained", default="weight/lavida-reason")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--prompt", default="cot", choices=["direct", "cot", "ccot", "dsp"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--step-per-block", type=int, default=32)
    parser.add_argument("--step-ratio", type=float, default=None)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])

    parser.add_argument("--correction-score", default="cumulated", choices=["current", "cumulated"])
    parser.add_argument(
        "--correction-metric",
        default=None,
        choices=["confidence", "time_aggregation", "topk_margin", "kl_divergence"],
        help="Algorithm used to rank generated tokens for remasking.",
    )
    parser.add_argument("--correction-rule", default="deterministic", choices=["deterministic", "stochastic"])
    parser.add_argument(
        "--transfer-per-step",
        type=int,
        default=None,
        help="If set, reveal this many masked tokens each step before self-correction remasking.",
    )
    parser.add_argument("--remask-ratio", type=float, default=0.25)
    parser.add_argument(
        "--remask-per-step",
        type=int,
        default=None,
        help="If set, remask this many generated tokens after each denoising step instead of using remask_ratio.",
    )
    parser.add_argument("--max-remask-per-step", type=int, default=None)
    parser.add_argument("--correction-scope", default="current_block", choices=["current_block", "generated"])
    parser.add_argument("--loo-chunk-size", type=int, default=16)
    parser.add_argument("--stochastic-temperature", type=float, default=1.0)
    parser.add_argument(
        "--record-detailed-metrics",
        action="store_true",
        help="Store all candidate token metrics. This is slower because it computes metrics not needed by the active correction metric.",
    )
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be > 0.")

    restore_compile = maybe_disable_torch_compile()

    from llava.model.builder import load_pretrained_model

    vision_kwargs = dict(
        mm_vision_tower=args.vision_tower,
        mm_resampler_type=None,
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=1152,
        mm_pooler_ratio=2,
        mm_patch_merge_type="spatial_unpad",
        use_mm_proj=True,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        None,
        args.model_name,
        device_map=args.device_map,
        vision_kwargs=vision_kwargs,
        torch_dtype=args.torch_dtype,
    )
    model.eval()
    model.tie_weights()
    model.to(get_torch_dtype(args.torch_dtype))
    core_model = model.get_model()

    dataset = datasets.load_dataset(args.dataset_path, split=args.split)
    if args.sample_mode == "random":
        dataset = dataset.shuffle(seed=args.sample_seed)
    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"

    total_elapsed = 0.0
    correct = 0
    written = 0
    total_remasked = 0

    with records_path.open("w", encoding="utf-8") as fout:
        for dataset_index, doc in enumerate(dataset):
            if doc.get("image") is None:
                continue

            t0 = time.time()
            context, prompt, prefix_embeds, _ = prepare_prefix(
                args,
                model,
                tokenizer,
                image_processor,
                doc,
            )
            run_output = generate_with_free_correction(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                step_per_block=args.step_per_block,
                temperature=args.temperature,
                remasking=args.remasking,
                schedule=args.schedule,
                schedule_shift=args.schedule_shift,
                step_ratio=args.step_ratio,
                correction_score=args.correction_score,
                correction_metric=args.correction_metric,
                correction_rule=args.correction_rule,
                transfer_per_step=args.transfer_per_step,
                remask_ratio=args.remask_ratio,
                remask_per_step=args.remask_per_step,
                max_remask_per_step=args.max_remask_per_step,
                correction_scope=args.correction_scope,
                loo_chunk_size=args.loo_chunk_size,
                stochastic_temperature=args.stochastic_temperature,
                record_detailed_metrics=args.record_detailed_metrics,
            )
            elapsed = time.time() - t0
            total_elapsed += elapsed

            final_text = run_output["final_text"]
            final_correct = bool(judge_answer(final_text, doc["choices"], doc["answer"]))
            correct += int(final_correct)
            remasked_count = sum(len(item["remasked_positions"]) for item in run_output["correction_records"])
            total_remasked += remasked_count

            record = {
                "dataset_index": int(dataset_index),
                "id": doc["id"],
                "question": context,
                "choices": list(doc["choices"]),
                "answer": doc["answer"],
                "domain": doc["domain"],
                "topic": doc["topic"],
                "prompt": prompt,
                "elapsed_sec": elapsed,
                "final_text": final_text,
                "final_correct": final_correct,
                "num_remasked_total": int(remasked_count),
                "final_answer_ids": run_output["final_answer_ids"],
                "correction_records": run_output["correction_records"],
                "meta": run_output["meta"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                print(
                    f"[{written}] dataset_index={dataset_index} id={doc['id']} "
                    f"final={final_correct} remasked={remasked_count} elapsed={elapsed:.2f}s",
                    flush=True,
                )

    summary = {
        "dataset_path": args.dataset_path,
        "split": args.split,
        "start_index": args.start_index,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed if args.sample_mode == "random" else None,
        "num_samples": written,
        "prompt": args.prompt,
        "algorithm": (
            "Stepwise self-correction: after each denoising transfer, score generated tokens "
            "with leave-one-out likelihood and remask the lowest-scoring tokens."
        ),
        "mean_acc": correct / written if written else None,
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": total_elapsed / written if written else None,
        "mean_remasked_tokens": total_remasked / written if written else None,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "step_per_block": args.step_per_block,
            "step_ratio": args.step_ratio,
            "schedule": args.schedule,
            "schedule_shift": args.schedule_shift,
            "remasking": args.remasking,
            "temperature": args.temperature,
        },
        "free_correction": {
            "correction_score": args.correction_score,
            "correction_metric": args.correction_metric,
            "correction_rule": args.correction_rule,
            "transfer_per_step": args.transfer_per_step,
            "remask_ratio": args.remask_ratio,
            "remask_per_step": args.remask_per_step,
            "max_remask_per_step": args.max_remask_per_step,
            "correction_scope": args.correction_scope,
            "loo_chunk_size": args.loo_chunk_size,
            "stochastic_temperature": args.stochastic_temperature,
            "record_detailed_metrics": args.record_detailed_metrics,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote records to {records_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)
    restore_compile()


if __name__ == "__main__":
    main()
