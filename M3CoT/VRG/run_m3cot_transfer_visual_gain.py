import argparse
import copy
import json
import math
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from M3CoT.run_m3cot_stepwise_x0 import (
    MASK_TOKEN_ID,
    build_prompt,
    clean_generated_text,
    compute_remasking_confidence,
)
from M3CoT.utils.metric import judge_answer
from Scale_Attention.reweight_patch import (
    build_prefix_from_multimodal_inputs,
    get_torch_dtype,
    maybe_disable_torch_compile,
)
from VRG.timestep_vrg import build_unconditional_prefix_embeds
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace transfer-time token visual gain on M3CoT."
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--sample-mode", default="random", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="M3CoT/VRG/outputs/transfer_visual_gain")

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
    parser.add_argument("--step-per-block", type=int, default=None)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument("--weak-visual-mode", default="null_visual", choices=["null_visual", "diffusion_noise"])
    parser.add_argument("--null-visual-mode", default="zeros", choices=["zeros", "mask_token"])
    parser.add_argument("--vcd-noise-step", type=int, default=500)
    parser.add_argument("--vcd-noise-seed", type=int, default=None)
    parser.add_argument("--vcd-alpha", type=float, default=0.0)

    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--visual-gain-threshold", type=float, default=0.2)
    parser.add_argument("--top-k-tokens", type=int, default=30)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def add_diffusion_noise_tensor(image_tensor, noise_step, seed=None):
    if not 0 <= int(noise_step) < 1000:
        raise ValueError("--vcd-noise-step must be in [0, 999].")

    device = image_tensor.device
    dtype = image_tensor.dtype
    betas = torch.linspace(-6, 6, 1000, device=device, dtype=torch.float32)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alpha_bar = alphas_prod[int(noise_step)]

    if seed is None:
        noise = torch.randn_like(image_tensor)
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            image_tensor.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    return alpha_bar.sqrt().to(dtype) * image_tensor + (1.0 - alpha_bar).sqrt().to(dtype) * noise


def add_diffusion_noise(images, noise_step, seed=None):
    if isinstance(images, list):
        return [
            add_diffusion_noise_tensor(image, noise_step=noise_step, seed=None if seed is None else int(seed) + idx)
            for idx, image in enumerate(images)
        ]
    return add_diffusion_noise_tensor(images, noise_step=noise_step, seed=seed)


def prepare_prefix_pair(args, model, tokenizer, image_processor, doc):
    image = doc["image"].convert("RGB")
    context = build_prompt(doc, args.prompt)

    conv = copy.deepcopy(conv_templates[args.conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

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

    core_model = model.get_model()
    if args.weak_visual_mode == "null_visual":
        weak_prefix_embeds, _ = build_unconditional_prefix_embeds(
            core_model=core_model,
            prefix_embeds=prefix_embeds,
            prefix_input_ids_full=prefix_input_ids_full,
            null_visual_mode=args.null_visual_mode,
        )
        weak_visual_meta = {
            "weak_visual_mode": args.weak_visual_mode,
            "null_visual_mode": args.null_visual_mode,
            "vcd_noise_step": None,
            "vcd_noise_seed": None,
        }
    elif args.weak_visual_mode == "diffusion_noise":
        noisy_image_tensor = add_diffusion_noise(
            image_tensor,
            noise_step=args.vcd_noise_step,
            seed=args.vcd_noise_seed,
        )
        weak_prefix_embeds, _ = build_prefix_from_multimodal_inputs(
            model=model,
            input_ids=input_ids,
            images=noisy_image_tensor,
            image_sizes=[image.size],
            attention_mask=attention_mask,
        )
        weak_visual_meta = {
            "weak_visual_mode": args.weak_visual_mode,
            "null_visual_mode": None,
            "vcd_noise_step": int(args.vcd_noise_step),
            "vcd_noise_seed": int(args.vcd_noise_seed) if args.vcd_noise_seed is not None else None,
        }
    else:
        raise ValueError(f"Unsupported weak visual mode: {args.weak_visual_mode}")

    return context, prompt, prefix_embeds, weak_prefix_embeds, prefix_input_ids_full, weak_visual_meta


def resolve_steps(max_new_tokens, block_length, step_per_block, step_ratio):
    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")
    num_blocks = max_new_tokens // block_length
    steps = max_new_tokens // num_blocks
    if step_per_block is not None:
        if step_ratio is not None:
            raise ValueError("Do not pass both --step-per-block and --step-ratio.")
        steps = min(int(step_per_block), block_length)
    elif step_ratio is not None:
        steps = int(steps * float(step_ratio))
    if steps <= 0:
        raise ValueError("The computed number of steps is 0.")
    return num_blocks, steps


def decode_token(tokenizer, token_id, skip_special_tokens=False):
    text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )
    return text.replace("\n", "\\n")


def classify_token_text(token_text):
    stripped = token_text.replace("\\n", "\n").strip()
    if not stripped:
        return {"is_whitespace": True, "is_punctuation": False, "contains_digit": False}
    return {
        "is_whitespace": False,
        "is_punctuation": all(ch in string.punctuation for ch in stripped),
        "contains_digit": any(ch.isdigit() for ch in stripped),
    }


def token_rank(logits_row, token_id):
    token_logit = logits_row[int(token_id)]
    return int((logits_row > token_logit).sum().item()) + 1


def finite_or_none(value):
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def summarize_token_records(records, confidence_threshold, visual_gain_threshold, top_k):
    if not records:
        return {
            "num_tokens": 0,
            "mean_confidence": None,
            "mean_visual_gain": None,
            "num_high_visual_gain": 0,
            "num_high_conf_low_visual_gain": 0,
            "high_visual_gain_ratio": None,
            "high_conf_low_visual_gain_ratio": None,
            "top_visual_gain_tokens": [],
            "top_high_conf_low_visual_gain_tokens": [],
        }

    high_vis = [item for item in records if item["visual_gain"] > visual_gain_threshold]
    high_conf_low_vis = [
        item
        for item in records
        if item["confidence"] >= confidence_threshold and item["visual_gain"] <= visual_gain_threshold
    ]
    top_vis = sorted(records, key=lambda item: item["visual_gain"], reverse=True)[:top_k]
    top_suspicious = sorted(
        high_conf_low_vis,
        key=lambda item: (item["visual_gain"], -item["confidence"]),
    )[:top_k]
    return {
        "num_tokens": len(records),
        "mean_confidence": sum(item["confidence"] for item in records) / len(records),
        "mean_visual_gain": sum(item["visual_gain"] for item in records) / len(records),
        "num_high_visual_gain": len(high_vis),
        "num_high_conf_low_visual_gain": len(high_conf_low_vis),
        "high_visual_gain_ratio": len(high_vis) / len(records),
        "high_conf_low_visual_gain_ratio": len(high_conf_low_vis) / len(records),
        "top_visual_gain_tokens": top_vis,
        "top_high_conf_low_visual_gain_tokens": top_suspicious,
    }


@torch.no_grad()
def generate_and_trace_transfer_visual_gain(
    core_model,
    tokenizer,
    prefix_embeds,
    weak_prefix_embeds,
    prefix_input_ids_full,
    weak_visual_meta,
    max_new_tokens,
    block_length,
    step_per_block,
    step_ratio,
    temperature,
    remasking,
    vcd_alpha,
    confidence_threshold,
    visual_gain_threshold,
    top_k_tokens,
):
    device = prefix_embeds.device
    batch_size, prefix_length = prefix_embeds.shape[:2]
    if batch_size != 1:
        raise ValueError("This tracer currently expects batch size 1.")

    visual_mask = prefix_input_ids_full.eq(IMAGE_TOKEN_INDEX)
    num_blocks, steps = resolve_steps(
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        step_per_block=step_per_block,
        step_ratio=step_ratio,
    )

    x = torch.full((batch_size, prefix_length + max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :prefix_length] = 0
    answer_slice = slice(prefix_length, prefix_length + max_new_tokens)
    token_records = {}
    step_records = []

    global_step = 0
    for block_idx in range(num_blocks):
        block_start = prefix_length + block_idx * block_length
        block_end = prefix_length + (block_idx + 1) * block_length
        block_slice = slice(block_start, block_end)
        block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
        num_transfer_tokens = get_num_transfer_tokens_sch(
            block_mask_index,
            steps,
            schedule=None,
            schedule_kwargs=None,
        )

        for step_idx in range(num_transfer_tokens.shape[1]):
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_slice]
            if block_mask_index.sum().item() == 0:
                global_step += 1
                continue

            current_embeds = core_model.transformer.wte(x)
            cond_current_embeds = current_embeds.clone()
            weak_current_embeds = current_embeds.clone()
            cond_current_embeds[:, :prefix_length] = prefix_embeds
            weak_current_embeds[:, :prefix_length] = weak_prefix_embeds

            logits_cond = core_model(None, input_embeddings=cond_current_embeds).logits.to(torch.float64)
            logits_weak = core_model(None, input_embeddings=weak_current_embeds).logits.to(torch.float64)
            logits_select = (1.0 + float(vcd_alpha)) * logits_cond - float(vcd_alpha) * logits_weak

            logits_with_noise = add_gumbel_noise(logits_select, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0_confidence = compute_remasking_confidence(logits_select, x0, remasking)
            x0_confidence[:, block_end:] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_confidence, -torch.inf)

            k = int(num_transfer_tokens[0, step_idx].item())
            _, select_index = torch.topk(confidence[0], k=k)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            transfer_index[0, select_index] = True

            cond_log_probs = F.log_softmax(logits_cond, dim=-1)
            weak_log_probs = F.log_softmax(logits_weak, dim=-1)
            select_log_probs = F.log_softmax(logits_select, dim=-1)
            cond_probs = cond_log_probs.exp()
            cond_top1 = torch.argmax(logits_cond, dim=-1)
            weak_top1 = torch.argmax(logits_weak, dim=-1)
            select_top1 = torch.argmax(logits_select, dim=-1)

            selected_answer_positions = []
            for seq_pos in select_index.detach().cpu().tolist():
                if seq_pos < prefix_length:
                    continue
                answer_pos = int(seq_pos - prefix_length)
                if answer_pos in token_records:
                    continue

                token_id = int(x0[0, seq_pos].item())
                cond_lp = float(cond_log_probs[0, seq_pos, token_id].item())
                weak_lp = float(weak_log_probs[0, seq_pos, token_id].item())
                select_lp = float(select_log_probs[0, seq_pos, token_id].item())
                visual_gain = cond_lp - weak_lp
                cond_top1_id = int(cond_top1[0, seq_pos].item())
                weak_top1_id = int(weak_top1[0, seq_pos].item())
                select_top1_id = int(select_top1[0, seq_pos].item())
                token_text = decode_token(tokenizer, token_id, skip_special_tokens=False)
                text_flags = classify_token_text(token_text)
                distribution_kl = float(
                    (
                        cond_probs[0, seq_pos]
                        * (cond_log_probs[0, seq_pos] - weak_log_probs[0, seq_pos])
                    ).sum().item()
                )

                record = {
                    "answer_position": answer_pos,
                    "sequence_position": int(seq_pos),
                    "token_id": token_id,
                    "token_text": token_text,
                    "token_text_clean": decode_token(tokenizer, token_id, skip_special_tokens=True),
                    "transfer_step": int(global_step + 1),
                    "block_index": int(block_idx + 1),
                    "step_in_block": int(step_idx + 1),
                    "confidence": finite_or_none(confidence[0, seq_pos].item()),
                    "conditional_log_prob": finite_or_none(cond_lp),
                    "weak_log_prob": finite_or_none(weak_lp),
                    "select_log_prob": finite_or_none(select_lp),
                    "visual_gain": finite_or_none(visual_gain),
                    "prob_ratio": finite_or_none(math.exp(visual_gain)),
                    "token_logit_delta": finite_or_none(
                        logits_cond[0, seq_pos, token_id].item() - logits_weak[0, seq_pos, token_id].item()
                    ),
                    "distribution_kl": finite_or_none(distribution_kl),
                    "conditional_rank": token_rank(logits_cond[0, seq_pos], token_id),
                    "weak_rank": token_rank(logits_weak[0, seq_pos], token_id),
                    "select_rank": token_rank(logits_select[0, seq_pos], token_id),
                    "conditional_top1_token_id": cond_top1_id,
                    "conditional_top1_token_text": decode_token(tokenizer, cond_top1_id, skip_special_tokens=False),
                    "weak_top1_token_id": weak_top1_id,
                    "weak_top1_token_text": decode_token(tokenizer, weak_top1_id, skip_special_tokens=False),
                    "select_top1_token_id": select_top1_id,
                    "select_top1_token_text": decode_token(tokenizer, select_top1_id, skip_special_tokens=False),
                    "top1_agree": cond_top1_id == weak_top1_id,
                    "select_matches_cond_top1": select_top1_id == cond_top1_id,
                    "select_matches_weak_top1": select_top1_id == weak_top1_id,
                    "is_high_visual_gain": visual_gain > visual_gain_threshold,
                    "is_high_conf_low_visual_gain": (
                        float(confidence[0, seq_pos].item()) >= confidence_threshold
                        and visual_gain <= visual_gain_threshold
                    ),
                }
                record.update(text_flags)
                token_records[answer_pos] = record
                selected_answer_positions.append(answer_pos)

            x[transfer_index] = x0[transfer_index]
            state_answer_ids = x[0, answer_slice].detach().cpu().tolist()
            step_records.append(
                {
                    "step": int(global_step + 1),
                    "block_index": int(block_idx + 1),
                    "step_in_block": int(step_idx + 1),
                    "num_transferred": int(k),
                    "selected_answer_positions": selected_answer_positions,
                    "num_masked_after_step": int((x[:, answer_slice] == MASK_TOKEN_ID).sum().item()),
                    "state_text": clean_generated_text(
                        tokenizer.decode(
                            state_answer_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    ),
                }
            )
            global_step += 1

    ordered_records = [token_records[pos] for pos in sorted(token_records)]
    final_answer_ids = x[0, answer_slice].detach().cpu().tolist()
    final_text = clean_generated_text(
        tokenizer.decode(
            final_answer_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )
    summary = summarize_token_records(
        records=ordered_records,
        confidence_threshold=confidence_threshold,
        visual_gain_threshold=visual_gain_threshold,
        top_k=top_k_tokens,
    )
    meta = {
        "prefix_length": int(prefix_length),
        "num_visual_tokens": int(visual_mask.sum().item()),
        "num_blocks": int(num_blocks),
        "steps_per_block": int(steps),
        "total_denoising_steps": int(global_step),
        "analysis_method": "transfer_time_cond_vs_weak_visual_logprob_delta_with_optional_vcd_select",
        "analysis_metric": "visual_gain = log p(token | text,image,current_state) - log p(token | text,weak_visual,current_state)",
        "select_logits": "(1 + vcd_alpha) * logits(image) - vcd_alpha * logits(weak_visual)",
        "vcd_alpha": float(vcd_alpha),
        **weak_visual_meta,
    }
    return final_answer_ids, final_text, ordered_records, step_records, summary, meta


def update_global_counters(global_stats, token_records):
    for record in token_records:
        token = record["token_text"]
        clean = record["token_text_clean"] or token
        global_stats["token_count"][token] += 1
        global_stats["clean_token_count"][clean] += 1
        if record["is_high_visual_gain"]:
            global_stats["high_visual_gain_token_count"][token] += 1
            global_stats["high_visual_gain_clean_token_count"][clean] += 1
        if record["is_high_conf_low_visual_gain"]:
            global_stats["high_conf_low_visual_gain_token_count"][token] += 1


def counter_top(counter, k):
    return [{"token": token, "count": int(count)} for token, count in counter.most_common(k)]


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
    written = 0
    final_correct_total = 0
    num_tokens_total = 0
    num_high_visual_gain_total = 0
    num_high_conf_low_visual_gain_total = 0
    global_stats = defaultdict(Counter)

    with records_path.open("w", encoding="utf-8") as fout:
        for dataset_index, doc in enumerate(dataset):
            if doc.get("image") is None:
                continue

            t0 = time.time()
            context, prompt, prefix_embeds, weak_prefix_embeds, prefix_input_ids_full, weak_visual_meta = prepare_prefix_pair(
                args,
                model,
                tokenizer,
                image_processor,
                doc,
            )
            final_answer_ids, final_text, token_records, step_records, sample_summary, meta = (
                generate_and_trace_transfer_visual_gain(
                    core_model=core_model,
                    tokenizer=tokenizer,
                    prefix_embeds=prefix_embeds,
                    weak_prefix_embeds=weak_prefix_embeds,
                    prefix_input_ids_full=prefix_input_ids_full,
                    weak_visual_meta=weak_visual_meta,
                    max_new_tokens=args.max_new_tokens,
                    block_length=args.block_length,
                    step_per_block=args.step_per_block,
                    step_ratio=args.step_ratio,
                    temperature=args.temperature,
                    remasking=args.remasking,
                    vcd_alpha=args.vcd_alpha,
                    confidence_threshold=args.confidence_threshold,
                    visual_gain_threshold=args.visual_gain_threshold,
                    top_k_tokens=args.top_k_tokens,
                )
            )
            elapsed = time.time() - t0
            total_elapsed += elapsed

            final_correct = bool(judge_answer(final_text, doc["choices"], doc["answer"]))
            final_correct_total += int(final_correct)
            written += 1
            num_tokens_total += int(sample_summary["num_tokens"])
            num_high_visual_gain_total += int(sample_summary["num_high_visual_gain"])
            num_high_conf_low_visual_gain_total += int(sample_summary["num_high_conf_low_visual_gain"])
            update_global_counters(global_stats, token_records)

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
                "final_answer_ids": final_answer_ids,
                "final_correct": final_correct,
                "token_records": token_records,
                "step_records": step_records,
                "summary": sample_summary,
                "meta": meta,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if args.print_every > 0 and written % args.print_every == 0:
                print(
                    f"[{written}] dataset_index={dataset_index} id={doc['id']} "
                    f"final={final_correct} tokens={sample_summary['num_tokens']} "
                    f"mean_gain={sample_summary['mean_visual_gain']:.4f} elapsed={elapsed:.2f}s",
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
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": total_elapsed / written if written else None,
        "final_accuracy": final_correct_total / written if written else None,
        "num_tokens": int(num_tokens_total),
        "num_high_visual_gain": int(num_high_visual_gain_total),
        "num_high_conf_low_visual_gain": int(num_high_conf_low_visual_gain_total),
        "high_visual_gain_ratio": (
            num_high_visual_gain_total / num_tokens_total if num_tokens_total else None
        ),
        "high_conf_low_visual_gain_ratio": (
            num_high_conf_low_visual_gain_total / num_tokens_total if num_tokens_total else None
        ),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "step_per_block": args.step_per_block,
            "step_ratio": args.step_ratio,
            "temperature": args.temperature,
            "remasking": args.remasking,
            "weak_visual_mode": args.weak_visual_mode,
            "null_visual_mode": args.null_visual_mode if args.weak_visual_mode == "null_visual" else None,
            "vcd_noise_step": args.vcd_noise_step if args.weak_visual_mode == "diffusion_noise" else None,
            "vcd_noise_seed": args.vcd_noise_seed if args.weak_visual_mode == "diffusion_noise" else None,
            "vcd_alpha": args.vcd_alpha,
        },
        "thresholds": {
            "confidence_threshold": args.confidence_threshold,
            "visual_gain_threshold": args.visual_gain_threshold,
        },
        "top_tokens": {
            "high_visual_gain": counter_top(
                global_stats["high_visual_gain_token_count"],
                args.top_k_tokens,
            ),
            "high_visual_gain_clean": counter_top(
                global_stats["high_visual_gain_clean_token_count"],
                args.top_k_tokens,
            ),
            "high_conf_low_visual_gain": counter_top(
                global_stats["high_conf_low_visual_gain_token_count"],
                args.top_k_tokens,
            ),
            "overall": counter_top(global_stats["token_count"], args.top_k_tokens),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote records to {records_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)
    restore_compile()


if __name__ == "__main__":
    main()
