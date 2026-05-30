import argparse
import copy
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import datasets
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Scale_Attention.reweight_patch import (
    build_prefix_from_multimodal_inputs,
    get_torch_dtype,
    maybe_disable_torch_compile,
)
from VRG.timestep_vrg import compute_remasking_confidence
from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.language_model.llada.generate import (
    add_gumbel_noise,
    get_num_transfer_tokens_sch,
)
from llava.model.language_model.llada.log_likelyhood import get_log_likelihood


MASK_TOKEN_ID = 126336
LETTER_MAP = "ABCDEFG"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run step-wise modality causal analysis for LaViDa reasoning on M3CoT."
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-dir", default="Causal_Analysis/outputs/lavida_reasoning_m3cot_causal")

    parser.add_argument("--pretrained", default="weight/lavida-reason")
    parser.add_argument("--model-variant", default="lavida_reasoning")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--vision-projector", default="mlp2x_gelu")
    parser.add_argument("--vision-hidden-size", type=int, default=1152)
    parser.add_argument("--mm-pooler-ratio", type=int, default=1)
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--prompt", default="cot", choices=["direct", "cot", "ccot", "dsp"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--step-per-block", type=int, default=None)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument("--intervention-replacement", default="zeros", choices=["zeros", "pad"])
    parser.add_argument("--intervention-batch-size", type=int, default=8)
    parser.add_argument("--step-stride", type=int, default=1)

    parser.add_argument("--mc-num", type=int, default=32)
    parser.add_argument("--mc-batch-size", type=int, default=16)
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args()


def build_choice_block(choices):
    return "\n".join(f"({LETTER_MAP[i]}) {choice}" for i, choice in enumerate(choices))


def build_base_prompt(doc):
    parts = []
    context = (doc.get("context") or "").strip()
    if context:
        parts.append(f"[Context]\n{context}")
    parts.append(f"[Question]\n{doc['question']}")
    parts.append(f"[Choices]\n{build_choice_block(doc['choices'])}")
    return "\n".join(parts)


def build_prompt(doc, prompt_style):
    base = build_base_prompt(doc)
    if prompt_style == "direct":
        return base + "\n\nAnswer with the option's letter from the given choices directly."
    if prompt_style == "cot":
        return base + "\n\nThink step by step and then provide your final answer in the format [Answer] (X)."
    if prompt_style == "dsp":
        return (
            base
            + "\n\nFirst describe the image information relevant to the question. "
            + "Then reason briefly and provide the final answer in the format [Answer] (X)."
        )
    if prompt_style == "ccot":
        return (
            base
            + "\n\nFirst identify the relevant objects, attributes, and relationships in the image as a compact scene graph. "
            + "Then solve the question and provide the final answer in the format [Answer] (X)."
        )
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def clean_generated_text(text):
    text = text.lstrip("!")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("<|eot_id|>", "")
    text = text.replace("<|im_end|>\n", "")
    text = text.replace("<|im_end|>", "")
    return text.strip()


def compute_generation_schedule(max_new_tokens, block_length, step_per_block, step_ratio):
    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")

    num_blocks = max_new_tokens // block_length
    steps = max_new_tokens // num_blocks

    if step_per_block is not None:
        steps = min(int(step_per_block), block_length)
    elif step_ratio is not None:
        steps = int(steps * float(step_ratio))

    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0. Increase step_ratio or max_new_tokens.")

    return {
        "num_blocks": int(num_blocks),
        "steps_per_block": int(steps),
        "total_denoising_steps": int(num_blocks * steps),
    }


def build_step_list(total_steps, step_stride):
    if step_stride <= 0:
        raise ValueError("--step-stride must be > 0.")
    return list(range(1, total_steps + 1, step_stride))


def build_model(args):
    from llava.model.builder import load_pretrained_model

    os.environ["LLADA_VISION_ENCODER"] = args.vision_tower
    os.environ["LLADA_VISION_PROJECTOR"] = args.vision_projector
    os.environ["LLADA_VISION_ENCODER_HIDDEN_SIZE"] = str(args.vision_hidden_size)
    os.environ["LLADA_MM_POOLER_RATIO"] = str(args.mm_pooler_ratio)

    vision_kwargs = dict(
        mm_vision_tower=args.vision_tower,
        mm_resampler_type=None,
        mm_projector_type=args.vision_projector,
        mm_hidden_size=args.vision_hidden_size,
        mm_pooler_ratio=args.mm_pooler_ratio,
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
    return tokenizer, model, image_processor, model.get_model()


def prepare_prefix(args, model, tokenizer, image_processor, doc):
    image = doc["image"].convert("RGB")
    prompt_text = build_prompt(doc, args.prompt)

    conv = copy.deepcopy(conv_templates[args.conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt_text)
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

    return {
        "prompt_text": prompt_text,
        "prompt": prompt,
        "image_size": image.size,
        "prefix_embeds": prefix_embeds,
        "prefix_input_ids_full": prefix_input_ids_full,
    }


def build_modality_masks(prefix_input_ids_full):
    visual_mask = prefix_input_ids_full.eq(IMAGE_TOKEN_INDEX)
    text_mask = prefix_input_ids_full.ne(IMAGE_TOKEN_INDEX) & prefix_input_ids_full.ne(IGNORE_INDEX)
    return visual_mask, text_mask


def make_intervened_prefix_embeds(core_model, tokenizer, prefix_embeds, prefix_input_ids_full, modality, replacement):
    visual_mask, text_mask = build_modality_masks(prefix_input_ids_full)
    target_mask = visual_mask if modality == "vision" else text_mask

    modified = prefix_embeds.clone()
    if not target_mask.any():
        return modified, int(target_mask.sum().item())

    if replacement == "zeros":
        modified[:, target_mask, :] = 0
    elif replacement == "pad":
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        pad_embed = core_model.transformer.wte(
            torch.tensor([pad_token_id], dtype=torch.long, device=prefix_embeds.device)
        ).to(dtype=prefix_embeds.dtype)
        modified[:, target_mask, :] = pad_embed.view(1, 1, -1)
    else:
        raise ValueError(f"Unsupported replacement: {replacement}")

    return modified, int(target_mask.sum().item())


@torch.no_grad()
def decode_with_optional_intervention(
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
    intervention_step=None,
    intervened_prefix_embeds=None,
):
    schedule_meta = compute_generation_schedule(
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        step_per_block=step_per_block,
        step_ratio=step_ratio,
    )
    num_blocks = schedule_meta["num_blocks"]
    steps = schedule_meta["steps_per_block"]
    total_denoising_steps = schedule_meta["total_denoising_steps"]

    device = prefix_embeds.device
    batch_size, prefix_length = prefix_embeds.shape[:2]
    x = torch.full((batch_size, prefix_length + max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :prefix_length] = 0

    schedule_value = None if schedule == "none" else schedule
    schedule_kwargs = {"shift": schedule_shift} if schedule_value == "shift" else None

    global_step = 0
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
            global_step += 1
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_slice]
            if block_mask_index.sum().item() == 0:
                continue

            current_embeds = core_model.transformer.wte(x)
            active_prefix = prefix_embeds
            if intervention_step is not None and global_step == intervention_step:
                active_prefix = intervened_prefix_embeds
            current_embeds[:, :prefix_length] = active_prefix

            logits = core_model(None, input_embeddings=current_embeds).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0_p = compute_remasking_confidence(logits, x0, remasking)

            x0_p[:, prefix_length + (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for batch_idx in range(confidence.shape[0]):
                k = int(num_transfer_tokens[batch_idx, step_idx].item())
                _, select_index = torch.topk(confidence[batch_idx], k=k)
                transfer_index[batch_idx, select_index] = True
            x[transfer_index] = x0[transfer_index]

    answer_ids = x[0, prefix_length:].detach().cpu().tolist()
    answer_text = clean_generated_text(
        tokenizer.decode(
            answer_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )
    return {
        "answer_ids": answer_ids,
        "answer_text": answer_text,
        "meta": {
            "prefix_length": int(prefix_length),
            "num_blocks": int(num_blocks),
            "steps_per_block": int(steps),
            "total_denoising_steps": int(total_denoising_steps),
        },
    }


@torch.no_grad()
def decode_intervention_batch(
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
    intervention_steps,
    intervened_prefix_embeds,
):
    if not intervention_steps:
        return []

    schedule_meta = compute_generation_schedule(
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        step_per_block=step_per_block,
        step_ratio=step_ratio,
    )
    num_blocks = schedule_meta["num_blocks"]
    steps = schedule_meta["steps_per_block"]
    total_denoising_steps = schedule_meta["total_denoising_steps"]

    device = prefix_embeds.device
    batch_size = len(intervention_steps)
    prefix_length = prefix_embeds.shape[1]
    x = torch.full((batch_size, prefix_length + max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :prefix_length] = 0

    base_prefix_batch = prefix_embeds.expand(batch_size, -1, -1)
    intervened_prefix_batch = intervened_prefix_embeds.expand(batch_size, -1, -1)
    intervention_steps_tensor = torch.tensor(intervention_steps, dtype=torch.long, device=device)

    schedule_value = None if schedule == "none" else schedule
    schedule_kwargs = {"shift": schedule_shift} if schedule_value == "shift" else None

    global_step = 0
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
            global_step += 1
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_slice]
            if block_mask_index.sum().item() == 0:
                continue

            current_embeds = core_model.transformer.wte(x)
            current_embeds[:, :prefix_length] = base_prefix_batch
            apply_mask = intervention_steps_tensor.eq(global_step)
            if apply_mask.any():
                current_embeds[apply_mask, :prefix_length] = intervened_prefix_batch[apply_mask]

            logits = core_model(None, input_embeddings=current_embeds).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0_p = compute_remasking_confidence(logits, x0, remasking)

            x0_p[:, prefix_length + (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for batch_idx in range(batch_size):
                k = int(num_transfer_tokens[batch_idx, step_idx].item())
                _, select_index = torch.topk(confidence[batch_idx], k=k)
                transfer_index[batch_idx, select_index] = True
            x[transfer_index] = x0[transfer_index]

    results = []
    for batch_idx, intervention_step in enumerate(intervention_steps):
        answer_ids = x[batch_idx, prefix_length:].detach().cpu().tolist()
        answer_text = clean_generated_text(
            tokenizer.decode(
                answer_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        results.append(
            {
                "step": int(intervention_step),
                "answer_ids": answer_ids,
                "answer_text": answer_text,
                "meta": {
                    "prefix_length": int(prefix_length),
                    "num_blocks": int(num_blocks),
                    "steps_per_block": int(steps),
                    "total_denoising_steps": int(total_denoising_steps),
                },
            }
        )
    return results


@torch.no_grad()
def decode_interventions_in_chunks(
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
    intervention_steps,
    intervention_batch_size,
    intervened_prefix_embeds,
):
    if intervention_batch_size <= 0:
        raise ValueError("--intervention-batch-size must be > 0.")

    results = []
    for start in range(0, len(intervention_steps), intervention_batch_size):
        chunk_steps = intervention_steps[start : start + intervention_batch_size]
        results.extend(
            decode_intervention_batch(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                max_new_tokens=max_new_tokens,
                block_length=block_length,
                step_per_block=step_per_block,
                temperature=temperature,
                remasking=remasking,
                schedule=schedule,
                schedule_shift=schedule_shift,
                step_ratio=step_ratio,
                intervention_steps=chunk_steps,
                intervened_prefix_embeds=intervened_prefix_embeds,
            )
        )
    return results


@torch.no_grad()
def compute_inverse_perplexity(core_model, prefix_embeds, answer_ids, mc_num, mc_batch_size):
    if mc_num <= 0 or mc_batch_size <= 0:
        raise ValueError("mc_num and mc_batch_size must be positive.")
    if mc_num % mc_batch_size != 0:
        raise ValueError("mc_num must be divisible by mc_batch_size.")

    answer_tensor = torch.tensor(answer_ids, dtype=torch.long, device=prefix_embeds.device).unsqueeze(0)
    log_likelihood = float(
        get_log_likelihood(
            core_model,
            prompt=None,
            answer=answer_tensor,
            mc_num=mc_num,
            batch_size=mc_batch_size,
            inputs_embeds=prefix_embeds,
        )
    )
    inverse_perplexity = math.exp(max(min(log_likelihood, 60.0), -60.0))
    return {
        "log_likelihood": log_likelihood,
        "inverse_perplexity": inverse_perplexity,
    }


def softmax_normalize(delta_dict, temperature=1.0):
    if not delta_dict:
        return {}
    steps = sorted(delta_dict)
    values = torch.tensor([delta_dict[step] for step in steps], dtype=torch.float64)
    values = values / float(temperature)
    values = values - values.max()
    weights = torch.softmax(values, dim=0)
    return {int(step): float(weight.item()) for step, weight in zip(steps, weights)}


def build_step_summary(value_lists, weight_lists):
    steps = sorted(value_lists)
    summary = []
    for step in steps:
        deltas = value_lists[step]
        weights = weight_lists[step]
        summary.append(
            {
                "step": int(step),
                "mean_delta": float(sum(deltas) / len(deltas)),
                "mean_weight": float(sum(weights) / len(weights)),
                "count": int(len(deltas)),
            }
        )
    return summary


def main():
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be > 0.")

    restore_compile = maybe_disable_torch_compile()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"

    tokenizer, model, image_processor, core_model = build_model(args)
    dataset = datasets.load_dataset(args.dataset_path, split=args.split)

    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    main_scores = []
    text_delta_values = defaultdict(list)
    text_weight_values = defaultdict(list)
    vision_delta_values = defaultdict(list)
    vision_weight_values = defaultdict(list)
    total_elapsed = 0.0
    written = 0

    with records_path.open("w", encoding="utf-8") as fout:
        for dataset_offset, doc in enumerate(tqdm(dataset, desc=f"M3CoT {args.split} causal")):
            if doc.get("image") is None:
                continue

            t0 = time.time()
            prepared = prepare_prefix(args, model, tokenizer, image_processor, doc)
            prefix_embeds = prepared["prefix_embeds"]
            prefix_input_ids_full = prepared["prefix_input_ids_full"]

            visual_prefix_embeds, num_visual_tokens = make_intervened_prefix_embeds(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                prefix_input_ids_full=prefix_input_ids_full,
                modality="vision",
                replacement=args.intervention_replacement,
            )
            text_prefix_embeds, num_text_tokens = make_intervened_prefix_embeds(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                prefix_input_ids_full=prefix_input_ids_full,
                modality="text",
                replacement=args.intervention_replacement,
            )

            main_result = decode_with_optional_intervention(
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
            main_score = compute_inverse_perplexity(
                core_model=core_model,
                prefix_embeds=prefix_embeds,
                answer_ids=main_result["answer_ids"],
                mc_num=args.mc_num,
                mc_batch_size=args.mc_batch_size,
            )

            total_steps = main_result["meta"]["total_denoising_steps"]
            evaluated_steps = build_step_list(total_steps, args.step_stride)
            text_interventions = []
            vision_interventions = []
            delta_text = {}
            delta_vision = {}

            text_results = decode_interventions_in_chunks(
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
                intervention_steps=evaluated_steps,
                intervention_batch_size=args.intervention_batch_size,
                intervened_prefix_embeds=text_prefix_embeds,
            )
            for text_result in text_results:
                step = int(text_result["step"])
                text_score = compute_inverse_perplexity(
                    core_model=core_model,
                    prefix_embeds=prefix_embeds,
                    answer_ids=text_result["answer_ids"],
                    mc_num=args.mc_num,
                    mc_batch_size=args.mc_batch_size,
                )
                delta_text[step] = main_score["inverse_perplexity"] - text_score["inverse_perplexity"]
                text_interventions.append(
                    {
                        "step": int(step),
                        "answer_text": text_result["answer_text"],
                        "log_likelihood": text_score["log_likelihood"],
                        "inverse_perplexity": text_score["inverse_perplexity"],
                        "causal_effect": delta_text[step],
                    }
                )

            vision_results = decode_interventions_in_chunks(
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
                intervention_steps=evaluated_steps,
                intervention_batch_size=args.intervention_batch_size,
                intervened_prefix_embeds=visual_prefix_embeds,
            )
            for vision_result in vision_results:
                step = int(vision_result["step"])
                vision_score = compute_inverse_perplexity(
                    core_model=core_model,
                    prefix_embeds=prefix_embeds,
                    answer_ids=vision_result["answer_ids"],
                    mc_num=args.mc_num,
                    mc_batch_size=args.mc_batch_size,
                )
                delta_vision[step] = main_score["inverse_perplexity"] - vision_score["inverse_perplexity"]
                vision_interventions.append(
                    {
                        "step": int(step),
                        "answer_text": vision_result["answer_text"],
                        "log_likelihood": vision_score["log_likelihood"],
                        "inverse_perplexity": vision_score["inverse_perplexity"],
                        "causal_effect": delta_vision[step],
                    }
                )

            omega_text = softmax_normalize(delta_text)
            omega_vision = softmax_normalize(delta_vision)

            for step, delta in delta_text.items():
                text_delta_values[step].append(delta)
                text_weight_values[step].append(omega_text[step])
            for step, delta in delta_vision.items():
                vision_delta_values[step].append(delta)
                vision_weight_values[step].append(omega_vision[step])

            elapsed = time.time() - t0
            total_elapsed += elapsed
            written += 1
            main_scores.append(main_score["inverse_perplexity"])

            record = {
                "dataset_index": int(args.start_index + dataset_offset),
                "id": doc["id"],
                "domain": doc["domain"],
                "topic": doc["topic"],
                "question": prepared["prompt_text"],
                "choices": list(doc["choices"]),
                "answer": doc["answer"],
                "prompt": prepared["prompt"],
                "elapsed_sec": elapsed,
                "meta": {
                    "model_variant": args.model_variant,
                    "pretrained": args.pretrained,
                    "max_new_tokens": args.max_new_tokens,
                    "block_length": args.block_length,
                    "step_ratio": args.step_ratio,
                    "total_denoising_steps": total_steps,
                    "evaluated_steps": evaluated_steps,
                    "step_stride": args.step_stride,
                    "intervention_batch_size": args.intervention_batch_size,
                    "num_visual_tokens": num_visual_tokens,
                    "num_text_tokens": num_text_tokens,
                    "intervention_replacement": args.intervention_replacement,
                    "metric": "inverse_perplexity",
                    "mc_num": args.mc_num,
                    "mc_batch_size": args.mc_batch_size,
                },
                "main": {
                    "answer_text": main_result["answer_text"],
                    "log_likelihood": main_score["log_likelihood"],
                    "inverse_perplexity": main_score["inverse_perplexity"],
                },
                "text_interventions": text_interventions,
                "vision_interventions": vision_interventions,
                "delta_text": {str(step): value for step, value in delta_text.items()},
                "delta_vision": {str(step): value for step, value in delta_vision.items()},
                "omega_text": {str(step): value for step, value in omega_text.items()},
                "omega_vision": {str(step): value for step, value in omega_vision.items()},
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if args.print_every > 0 and written % args.print_every == 0:
                print(
                    f"[{written}] id={doc['id']} elapsed={elapsed:.2f}s "
                    f"main_inv_ppl={main_score['inverse_perplexity']:.6f}"
                )

    summary = {
        "dataset_path": args.dataset_path,
        "split": args.split,
        "start_index": args.start_index,
        "num_samples": written,
        "generation": {
            "model_variant": args.model_variant,
            "pretrained": args.pretrained,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "step_per_block": args.step_per_block,
            "step_ratio": args.step_ratio,
            "schedule": args.schedule,
            "schedule_shift": args.schedule_shift,
            "temperature": args.temperature,
            "remasking": args.remasking,
            "intervention_replacement": args.intervention_replacement,
            "step_stride": args.step_stride,
            "intervention_batch_size": args.intervention_batch_size,
        },
        "metric": {
            "name": "inverse_perplexity",
            "mc_num": args.mc_num,
            "mc_batch_size": args.mc_batch_size,
        },
        "mean_main_inverse_perplexity": float(sum(main_scores) / len(main_scores)) if main_scores else None,
        "mean_elapsed_sec": float(total_elapsed / written) if written else None,
        "text_step_summary": build_step_summary(text_delta_values, text_weight_values),
        "vision_step_summary": build_step_summary(vision_delta_values, vision_weight_values),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if restore_compile is not None:
        restore_compile()

    print(f"Wrote {written} sample records to {records_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
