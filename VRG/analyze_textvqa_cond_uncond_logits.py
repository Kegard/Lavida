import argparse
import copy
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Scale_Attention.reweight_patch import build_prefix_from_multimodal_inputs, get_torch_dtype, maybe_disable_torch_compile
from VRG.timestep_vrg import MASK_TOKEN_ID, build_unconditional_prefix_embeds, compute_remasking_confidence
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


TEXTVQA_PROMPT = "Answer the question using a single word or phrase."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace TextVQA cond/uncond logits for normalized correct-answer tokens and final-answer tokens."
    )
    parser.add_argument("--dataset-path", default="lmms-lab/textvqa")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-dir", default="VRG/outputs/textvqa_cond_uncond_logits")
    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--step-ratio", type=float, default=1.0)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument("--null-visual-mode", default="zeros", choices=["zeros", "mask_token"])
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip-plots", action="store_true", help="Only write JSONL/summary outputs.")
    return parser.parse_args()


def construct_textvqa_prompt(doc):
    return f"{doc['question'].capitalize()}\n{TEXTVQA_PROMPT}"


def build_prompt(context, conv_template):
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_textvqa_split(dataset_path, dataset_name, split):
    if dataset_name is None:
        return load_dataset(dataset_path, split=split)
    return load_dataset(dataset_path, dataset_name, split=split)


def choose_normalized_answer(answers, answer_processor):
    normalized = [answer_processor(answer) for answer in answers or [] if answer is not None]
    normalized = [answer for answer in normalized if answer]
    if not normalized:
        return ""
    counts = Counter(normalized)
    best_count = max(counts.values())
    candidates = [answer for answer, count in counts.items() if count == best_count]
    candidates.sort(key=lambda item: (-len(item), item))
    return candidates[0]


def decode_tokens(tokenizer, token_ids):
    return [
        tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]


def clean_generated_text(text):
    return text.replace("<|endoftext|>", "").replace("<|eot_id|>", "").strip()


def token_logit_entry(logits_row, token_id, top5_ids):
    if token_id is None:
        return None
    token_id = int(token_id)
    matches = [idx for idx, candidate_id in enumerate(top5_ids) if int(candidate_id) == token_id]
    return {
        "token_id": token_id,
        "logit": float(logits_row[token_id].item()),
        "top5_rank": int(matches[0] + 1) if matches else None,
        "in_top5": bool(matches),
    }


def topk_entries(logits_row, tokenizer, k):
    values, indices = torch.topk(logits_row, k=min(k, logits_row.shape[-1]))
    return [
        {
            "rank": idx + 1,
            "token_id": int(token_id),
            "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "logit": float(value),
        }
        for idx, (value, token_id) in enumerate(zip(values.tolist(), indices.tolist()))
    ]


def pad_or_none(token_ids, position):
    if position < len(token_ids):
        return int(token_ids[position])
    return None


def build_target_positions(correct_token_ids, final_token_ids, max_new_tokens):
    max_len = min(max(len(correct_token_ids), len(final_token_ids)), max_new_tokens)
    return list(range(max_len))


def init_paired_state(core_model, prefix_embeds, prefix_input_ids_full, null_visual_mode):
    uncond_prefix_embeds, visual_mask = build_unconditional_prefix_embeds(
        core_model=core_model,
        prefix_embeds=prefix_embeds,
        prefix_input_ids_full=prefix_input_ids_full,
        null_visual_mode=null_visual_mode,
    )
    cond_past = core_model(None, input_embeddings=prefix_embeds, use_cache=True).attn_key_values
    uncond_past = core_model(None, input_embeddings=uncond_prefix_embeds, use_cache=True).attn_key_values
    paired_past = []
    for cond_layer, uncond_layer in zip(cond_past, uncond_past):
        cond_key, cond_value = cond_layer
        uncond_key, uncond_value = uncond_layer
        paired_past.append(
            (
                torch.cat([cond_key, uncond_key], dim=0),
                torch.cat([cond_value, uncond_value], dim=0),
            )
        )
    return tuple(paired_past), int(visual_mask.sum().item())


@torch.no_grad()
def generate_final_tokens(
    core_model,
    prefix_embeds,
    prefix_input_ids_full,
    max_new_tokens,
    block_length,
    temperature,
    remasking,
    schedule,
    schedule_shift,
    step_ratio,
    null_visual_mode,
):
    device = prefix_embeds.device
    paired_past, num_visual_tokens = init_paired_state(
        core_model,
        prefix_embeds,
        prefix_input_ids_full,
        null_visual_mode,
    )
    x = torch.full((1, max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    schedule_kwargs = {"shift": schedule_shift} if schedule == "shift" else None
    num_blocks = max_new_tokens // block_length
    steps = int((max_new_tokens // num_blocks) * step_ratio)
    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0.")

    for block_idx in range(num_blocks):
        block_slice = slice(block_idx * block_length, (block_idx + 1) * block_length)
        block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
        num_transfer_tokens = get_num_transfer_tokens_sch(
            block_mask_index,
            steps,
            schedule=None if schedule == "none" else schedule,
            schedule_kwargs=schedule_kwargs,
        )
        for step_idx in range(num_transfer_tokens.shape[1]):
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_slice]
            if block_mask_index.sum().item() == 0:
                continue

            current_embeds = core_model.transformer.wte(x)
            paired_logits = core_model(
                None,
                input_embeddings=torch.cat([current_embeds, current_embeds], dim=0),
                past_key_values=paired_past,
            ).logits
            logits_cond, _ = torch.chunk(paired_logits, 2, dim=0)
            logits_with_noise = add_gumbel_noise(logits_cond, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0_p = compute_remasking_confidence(logits_cond, x0, remasking)
            x0_p[:, (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for batch_idx in range(confidence.shape[0]):
                k = int(num_transfer_tokens[batch_idx, step_idx].item())
                _, select_index = torch.topk(confidence[batch_idx], k=k)
                transfer_index[batch_idx, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x.detach().clone(), {"num_visual_tokens": num_visual_tokens, "steps_per_block": steps}


@torch.no_grad()
def trace_cond_uncond_logits(
    core_model,
    tokenizer,
    prefix_embeds,
    prefix_input_ids_full,
    final_token_ids,
    correct_token_ids,
    target_positions,
    max_new_tokens,
    block_length,
    temperature,
    remasking,
    schedule,
    schedule_shift,
    step_ratio,
    null_visual_mode,
):
    device = prefix_embeds.device
    paired_past, num_visual_tokens = init_paired_state(
        core_model,
        prefix_embeds,
        prefix_input_ids_full,
        null_visual_mode,
    )
    x = torch.full((1, max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    schedule_kwargs = {"shift": schedule_shift} if schedule == "shift" else None
    num_blocks = max_new_tokens // block_length
    steps = int((max_new_tokens // num_blocks) * step_ratio)
    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0.")

    records = []
    global_step = 0
    final_target_positions = set(target_positions)

    for block_idx in range(num_blocks):
        block_slice = slice(block_idx * block_length, (block_idx + 1) * block_length)
        block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
        num_transfer_tokens = get_num_transfer_tokens_sch(
            block_mask_index,
            steps,
            schedule=None if schedule == "none" else schedule,
            schedule_kwargs=schedule_kwargs,
        )
        for step_idx in range(num_transfer_tokens.shape[1]):
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_slice]
            if block_mask_index.sum().item() == 0:
                global_step += 1
                continue

            current_embeds = core_model.transformer.wte(x)
            paired_logits = core_model(
                None,
                input_embeddings=torch.cat([current_embeds, current_embeds], dim=0),
                past_key_values=paired_past,
            ).logits
            logits_cond, logits_uncond = torch.chunk(paired_logits, 2, dim=0)
            logits_with_noise = add_gumbel_noise(logits_cond, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0_p = compute_remasking_confidence(logits_cond, x0, remasking)
            x0_p[:, (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            k = int(num_transfer_tokens[0, step_idx].item())
            _, select_index = torch.topk(confidence[0], k=k)
            transfer_index[0, select_index] = True

            position_records = []
            for position in target_positions:
                correct_id = pad_or_none(correct_token_ids, position)
                final_id = pad_or_none(final_token_ids, position)
                cond_row = logits_cond[0, position]
                uncond_row = logits_uncond[0, position]
                cond_top5_ids = torch.topk(cond_row, k=5).indices.tolist()
                uncond_top5_ids = torch.topk(uncond_row, k=5).indices.tolist()
                position_records.append(
                    {
                        "position": int(position),
                        "masked_before_step": bool(mask_index[0, position].item()),
                        "selected_this_step": bool(transfer_index[0, position].item()),
                        "is_final_answer_position": int(position) in final_target_positions,
                        "correct_token": token_logit_entry(cond_row, correct_id, cond_top5_ids),
                        "correct_token_uncond": token_logit_entry(uncond_row, correct_id, uncond_top5_ids),
                        "final_token": token_logit_entry(cond_row, final_id, cond_top5_ids),
                        "final_token_uncond": token_logit_entry(uncond_row, final_id, uncond_top5_ids),
                        "cond_top3": topk_entries(cond_row, tokenizer, 3),
                        "uncond_top3": topk_entries(uncond_row, tokenizer, 3),
                    }
                )

            records.append(
                {
                    "step": int(global_step + 1),
                    "block_index": int(block_idx + 1),
                    "step_in_block": int(step_idx + 1),
                    "num_transferred": k,
                    "selected_positions": [int(pos) for pos in select_index.detach().cpu().tolist()],
                    "position_records": position_records,
                }
            )
            x[transfer_index] = x0[transfer_index]
            global_step += 1

    return records, {"num_visual_tokens": num_visual_tokens, "steps_per_block": steps}


def prepare_sample(args, model, tokenizer, image_processor, doc):
    image = doc["image"].convert("RGB")
    context = construct_textvqa_prompt(doc)
    prompt = build_prompt(context, args.conv_template)
    image_tensor = process_images([image], image_processor, model.config)
    dtype = get_torch_dtype(args.torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=args.device)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
    prefix_embeds, prefix_input_ids_full = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        attention_mask=attention_mask,
    )
    return context, prompt, prefix_embeds, prefix_input_ids_full


def update_aggregates(aggregates, step_records):
    for step_record in step_records:
        step = step_record["step"]
        for position_record in step_record["position_records"]:
            cond_correct = position_record["correct_token"]
            uncond_correct = position_record["correct_token_uncond"]
            if cond_correct is not None and uncond_correct is not None:
                aggregates[("gap", "correct", "logit")][step].append(
                    cond_correct["logit"] - uncond_correct["logit"]
                )
            for stream, token_key in (("cond", "correct_token"), ("uncond", "correct_token_uncond")):
                entry = position_record[token_key]
                if entry is not None:
                    aggregates[(stream, "correct", "logit")][step].append(entry["logit"])
                    aggregates[(stream, "correct", "top5_hit")][step].append(1.0 if entry["in_top5"] else 0.0)
            for stream, token_key in (("cond", "final_token"), ("uncond", "final_token_uncond")):
                entry = position_record[token_key]
                if entry is not None:
                    aggregates[(stream, "final", "logit")][step].append(entry["logit"])
                    aggregates[(stream, "final", "top5_hit")][step].append(1.0 if entry["in_top5"] else 0.0)


def mean_curve(values_by_step):
    return [
        {"step": int(step), "mean": float(statistics.mean(values))}
        for step, values in sorted(values_by_step.items())
        if values
    ]


def plot_aggregate(summary, output_dir, dpi):
    import matplotlib.pyplot as plt

    plot_specs = [
        ("logit", "Mean Logit", output_dir / "aggregate_logits.png"),
        ("top5_hit", "Top-5 Hit Rate", output_dir / "aggregate_top5_hits.png"),
    ]
    colors = {
        ("cond", "correct"): "#1f77b4",
        ("uncond", "correct"): "#ff7f0e",
        ("cond", "final"): "#2ca02c",
        ("uncond", "final"): "#d62728",
    }
    labels = {
        ("cond", "correct"): "cond correct",
        ("uncond", "correct"): "uncond correct",
        ("cond", "final"): "cond final",
        ("uncond", "final"): "uncond final",
    }
    for metric, ylabel, path in plot_specs:
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        for stream in ("cond", "uncond"):
            for target in ("correct", "final"):
                curve = summary["aggregate_curves"][stream][target][metric]
                if not curve:
                    continue
                ax.plot(
                    [item["step"] for item in curve],
                    [item["mean"] for item in curve],
                    marker="o",
                    linewidth=1.8,
                    markersize=3.0,
                    color=colors[(stream, target)],
                    label=labels[(stream, target)],
                )
        ax.set_xlabel("Denoising Step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"TextVQA Cond/Uncond {ylabel}")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    gap_curve = summary["aggregate_curves"]["gap"]["correct"]["logit"]
    if gap_curve:
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        ax.plot(
            [item["step"] for item in gap_curve],
            [item["mean"] for item in gap_curve],
            marker="o",
            linewidth=1.9,
            markersize=3.2,
            color="#7b2cbf",
            label="cond correct - uncond correct",
        )
        ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xlabel("Denoising Step")
        ax.set_ylabel("Mean Logit Gap")
        ax.set_title("TextVQA Visual Logit Gap for Correct Tokens")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "aggregate_correct_logit_gap.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    if args.max_new_tokens % args.block_length != 0:
        raise ValueError("--max-new-tokens must be divisible by --block-length.")
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

    dataset = load_textvqa_split(args.dataset_path, args.dataset_name, args.split)
    answer_processor = EvalAIAnswerProcessor()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"

    aggregates = defaultdict(lambda: defaultdict(list))
    written = 0
    total_elapsed = 0.0

    with records_path.open("w", encoding="utf-8") as fout:
        for dataset_index in range(args.start_index, len(dataset)):
            if written >= args.limit:
                break
            doc = dataset[dataset_index]
            if doc.get("image") is None:
                continue

            t0 = time.time()
            context, prompt, prefix_embeds, prefix_input_ids_full = prepare_sample(
                args,
                model,
                tokenizer,
                image_processor,
                doc,
            )
            correct_text = choose_normalized_answer(doc.get("answers", []), answer_processor)
            correct_token_ids = tokenizer.encode(correct_text, add_special_tokens=False)[: args.max_new_tokens]

            final_tokens, final_meta = generate_final_tokens(
                core_model=core_model,
                prefix_embeds=prefix_embeds,
                prefix_input_ids_full=prefix_input_ids_full,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                temperature=args.temperature,
                remasking=args.remasking,
                schedule=args.schedule,
                schedule_shift=args.schedule_shift,
                step_ratio=args.step_ratio,
                null_visual_mode=args.null_visual_mode,
            )
            final_token_ids = final_tokens[0].detach().cpu().tolist()
            final_text = clean_generated_text(
                tokenizer.decode(final_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            )
            normalized_final_text = answer_processor(final_text)
            final_token_ids = tokenizer.encode(normalized_final_text, add_special_tokens=False)[: args.max_new_tokens]

            target_positions = build_target_positions(correct_token_ids, final_token_ids, args.max_new_tokens)
            step_records, trace_meta = trace_cond_uncond_logits(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                prefix_input_ids_full=prefix_input_ids_full,
                final_token_ids=final_token_ids,
                correct_token_ids=correct_token_ids,
                target_positions=target_positions,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                temperature=args.temperature,
                remasking=args.remasking,
                schedule=args.schedule,
                schedule_shift=args.schedule_shift,
                step_ratio=args.step_ratio,
                null_visual_mode=args.null_visual_mode,
            )
            elapsed = time.time() - t0
            total_elapsed += elapsed
            update_aggregates(aggregates, step_records)

            record = {
                "dataset_index": int(dataset_index),
                "question_id": doc.get("question_id"),
                "question": context,
                "answers": doc.get("answers"),
                "ocr_tokens": doc.get("ocr_tokens"),
                "normalized_correct_answer": correct_text,
                "normalized_final_answer": normalized_final_text,
                "correct_token_ids": correct_token_ids,
                "correct_token_texts": decode_tokens(tokenizer, correct_token_ids),
                "final_token_ids": final_token_ids,
                "final_token_texts": decode_tokens(tokenizer, final_token_ids),
                "target_positions": target_positions,
                "prompt": prompt,
                "elapsed_sec": elapsed,
                "meta": {
                    "final_generation": final_meta,
                    "trace": trace_meta,
                    "max_new_tokens": args.max_new_tokens,
                    "block_length": args.block_length,
                    "step_ratio": args.step_ratio,
                    "schedule": args.schedule,
                    "schedule_shift": args.schedule_shift,
                    "remasking": args.remasking,
                    "null_visual_mode": args.null_visual_mode,
                },
                "step_records": step_records,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                print(
                    f"[{written}] dataset_index={dataset_index} "
                    f"correct={correct_text!r} final={normalized_final_text!r} "
                    f"positions={len(target_positions)} elapsed={elapsed:.2f}s"
                )

    summary = {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "start_index": args.start_index,
        "num_samples": written,
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": total_elapsed / written if written else None,
        "rank_definition": "top5_rank is 1..5 when the target token is in top5, otherwise null",
        "actual_output_definition": "final tokens are from the cond trajectory, normalized by EvalAIAnswerProcessor",
        "aggregate_curves": {
            stream: {
                target: {
                    metric: mean_curve(aggregates[(stream, target, metric)])
                    for metric in ("logit", "top5_hit")
                }
                for target in ("correct", "final")
            }
            for stream in ("cond", "uncond")
        },
    }
    summary["aggregate_curves"]["gap"] = {
        "correct": {
            "logit": mean_curve(aggregates[("gap", "correct", "logit")]),
        }
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.skip_plots:
        plot_aggregate(summary, output_dir, args.dpi)

    print(f"Wrote records to {records_path}")
    print(f"Wrote summary to {summary_path}")
    if not args.skip_plots:
        print(
            "Wrote plots to "
            f"{output_dir / 'aggregate_logits.png'}, "
            f"{output_dir / 'aggregate_top5_hits.png'}, and "
            f"{output_dir / 'aggregate_correct_logit_gap.png'}"
        )
    restore_compile()


if __name__ == "__main__":
    main()
