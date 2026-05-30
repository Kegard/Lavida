import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_ROOT, REPO_ROOT / "M3CoT"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from M3CoT.run_m3cot_stepwise_x0 import (
    MASK_TOKEN_ID,
    clean_generated_text,
    compute_remasking_confidence,
    prepare_prefix,
)
from Scale_Attention.reweight_patch import get_torch_dtype, maybe_disable_torch_compile
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate per-step entropy changes of LaViDa on M3CoT."
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default="Entropy/outputs/m3cot_reason_entropy")

    parser.add_argument("--pretrained", default="weight/lavida-reason")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )

    parser.add_argument("--prompt", default="cot", choices=["direct", "cot", "ccot", "dsp"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--step-per-block", type=int, default=None)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument(
        "--schedule",
        default="none",
        choices=["shift", "cosine", "logit_normal", "none"],
    )
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--remasking",
        default="low_confidence",
        choices=["low_confidence", "random", "entrophy", "margin"],
    )
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor):
    masked_values = values[mask]
    if masked_values.numel() == 0:
        return None
    return float(masked_values.mean().item())


@torch.no_grad()
def generate_entropy_records(
    core_model,
    tokenizer,
    prefix_embeds,
    max_new_tokens,
    block_length,
    step_per_block,
    temperature,
    remasking,
    schedule,
    schedule_shift,
    step_ratio,
):
    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")

    device = prefix_embeds.device
    batch_size, prefix_length = prefix_embeds.shape[:2]
    prompt = torch.full((batch_size, prefix_length), 0, dtype=torch.long, device=device)
    x = torch.full(
        (batch_size, prefix_length + max_new_tokens),
        MASK_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    x[:, :prefix_length] = prompt

    num_blocks = max_new_tokens // block_length
    steps = max_new_tokens
    if steps % num_blocks != 0 and step_per_block is None:
        raise ValueError("Native generation requires steps % num_blocks == 0 unless step_per_block is set.")
    steps = steps // num_blocks
    if step_per_block is None and step_ratio is None:
        step_per_block = block_length
    if step_per_block is not None:
        if step_ratio is not None:
            raise ValueError("Do not pass both --step-per-block and --step-ratio.")
        steps = min(int(step_per_block), block_length)
    if step_ratio:
        steps = int(steps * step_ratio)
    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0. Increase step_ratio or max_new_tokens.")

    schedule_value = None if schedule == "none" else schedule
    schedule_kwargs = {"shift": schedule_shift} if schedule_value == "shift" else None
    step_records = []

    for block_idx in range(num_blocks):
        block_start = prefix_length + block_idx * block_length
        block_end = prefix_length + (block_idx + 1) * block_length
        block_slice = slice(block_start, block_end)
        block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
        num_transfer_tokens = get_num_transfer_tokens_sch(
            block_mask_index,
            steps,
            schedule=schedule_value,
            schedule_kwargs=schedule_kwargs,
        )

        for step_idx in range(num_transfer_tokens.shape[1]):
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_slice]
            if block_mask_index.sum().item() == 0:
                continue

            current_embeds = core_model.transformer.wte(x)
            current_embeds[:, :prefix_length] = prefix_embeds
            logits = core_model(None, input_embeddings=current_embeds).logits

            token_entropy = compute_token_entropy(logits)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            remasking_priority = compute_remasking_confidence(logits, x0, remasking)
            remasking_priority[:, prefix_length + (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)

            candidate_token_ids = x0[0, prefix_length:].detach().cpu().tolist()
            candidate_text = clean_generated_text(
                tokenizer.decode(
                    candidate_token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )

            confidence = torch.where(mask_index, remasking_priority, -torch.inf)
            k = int(num_transfer_tokens[0, step_idx].item())
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            _, select_index = torch.topk(confidence[0], k=k)
            transfer_index[0, select_index] = True

            step_entropy_all_masked = masked_mean(token_entropy, mask_index)
            step_entropy_active_block = masked_mean(token_entropy[:, block_slice], block_mask_index)
            selected_entropy = float(token_entropy[0, select_index].mean().item())
            selected_priority = float(remasking_priority[0, select_index].mean().item())

            x[transfer_index] = x0[transfer_index]

            state_answer_ids = x[0, prefix_length:].detach().cpu().tolist()
            state_text = clean_generated_text(
                tokenizer.decode(
                    state_answer_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )

            step_records.append(
                {
                    "step": int(len(step_records) + 1),
                    "block_index": int(block_idx + 1),
                    "step_in_block": int(step_idx + 1),
                    "num_transferred": int(k),
                    "selected_positions": [int(pos) for pos in select_index.detach().cpu().tolist()],
                    "num_masked_after_step": int((x[:, prefix_length:] == MASK_TOKEN_ID).sum().item()),
                    "mean_entropy_all_masked_before": step_entropy_all_masked,
                    "mean_entropy_active_block_before": step_entropy_active_block,
                    "mean_entropy_selected": selected_entropy,
                    "mean_priority_selected": selected_priority,
                    "candidate_text": candidate_text,
                    "state_text": state_text,
                }
            )

    final_state_text = clean_generated_text(
        tokenizer.decode(
            x[0, prefix_length:].detach().cpu().tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )
    meta = {
        "prefix_length": int(prefix_length),
        "num_blocks": int(num_blocks),
        "steps_per_block": int(steps),
        "total_denoising_steps": int(len(step_records)),
        "max_new_tokens": int(max_new_tokens),
        "block_length": int(block_length),
    }
    return step_records, final_state_text, meta


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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"

    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    step_meta = {}
    total_elapsed = 0.0
    written = 0

    with records_path.open("w", encoding="utf-8") as fout:
        for dataset_index in range(args.start_index, len(dataset)):
            if written >= args.limit:
                break

            doc = dataset[dataset_index]
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
            step_records, final_state_text, meta = generate_entropy_records(
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
            )

            for step_record in step_records:
                step = int(step_record["step"])
                step_meta.setdefault(
                    step,
                    {
                        "block_index": int(step_record["block_index"]),
                        "step_in_block": int(step_record["step_in_block"]),
                    },
                )
                for metric_name in (
                    "mean_entropy_all_masked_before",
                    "mean_entropy_active_block_before",
                    "mean_entropy_selected",
                    "mean_priority_selected",
                    "num_masked_after_step",
                ):
                    metric_value = step_record.get(metric_name)
                    if metric_value is None:
                        continue
                    metric_sums[(step, metric_name)] += float(metric_value)
                    metric_counts[(step, metric_name)] += 1

            elapsed = time.time() - t0
            total_elapsed += elapsed
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
                "final_state_text": final_state_text,
                "step_results": step_records,
                "meta": meta,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                preview_step = 1
                entropy_key = (preview_step, "mean_entropy_active_block_before")
                message = f"[{written}] dataset_index={dataset_index} id={doc['id']} elapsed={elapsed:.2f}s"
                if metric_counts[entropy_key]:
                    preview_entropy = metric_sums[entropy_key] / metric_counts[entropy_key]
                    message += f" step1_active_entropy={preview_entropy:.4f}"
                print(message)

    step_summary = []
    for step in sorted(step_meta):
        summary_item = {
            "step": int(step),
            "block_index": int(step_meta[step]["block_index"]),
            "step_in_block": int(step_meta[step]["step_in_block"]),
        }
        for metric_name in (
            "mean_entropy_all_masked_before",
            "mean_entropy_active_block_before",
            "mean_entropy_selected",
            "mean_priority_selected",
            "num_masked_after_step",
        ):
            count = metric_counts[(step, metric_name)]
            summary_item[metric_name] = (
                metric_sums[(step, metric_name)] / count if count else None
            )
        summary_item["count"] = int(metric_counts[(step, "mean_entropy_selected")])
        step_summary.append(summary_item)

    summary = {
        "dataset_path": args.dataset_path,
        "split": args.split,
        "start_index": args.start_index,
        "num_samples": written,
        "prompt": args.prompt,
        "model_path": args.pretrained,
        "metric_definition": {
            "mean_entropy_all_masked_before": "Mean token entropy over all masked answer positions before the current transfer step.",
            "mean_entropy_active_block_before": "Mean token entropy over still-masked positions inside the currently active block before transfer.",
            "mean_entropy_selected": "Mean token entropy of the positions selected for transfer at the current step.",
            "mean_priority_selected": "Mean remasking priority score of the positions selected for transfer at the current step.",
            "num_masked_after_step": "Remaining number of masked answer positions after the current step.",
        },
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": total_elapsed / written if written else None,
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
        "step_summary": step_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote records to {records_path}")
    print(f"Wrote summary to {summary_path}")
    restore_compile()


if __name__ == "__main__":
    main()
