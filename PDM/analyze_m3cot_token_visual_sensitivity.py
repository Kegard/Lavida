import argparse
import copy
import json
import os
import re
import string
import sys
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_ROOT = REPO_ROOT / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch
from lmms_eval.models.llava_llada import Llava_Llada
from Scale_Attention.reweight_patch import build_prefix_from_multimodal_inputs
from VRG.timestep_vrg import build_unconditional_prefix_embeds, compute_step_vrg_alpha


LETTER_MAP = "ABCDEFG"
MASK_TOKEN_ID = 126336


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze token-level visual sensitivity for LaViDa generations on M3CoT."
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-path", default="/data/jindong_gu/LaViDa/weight/lavida-reason")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--vision-tower", default="/data/jindong_gu/LaViDa/weight/siglip")
    parser.add_argument("--vision-projector", default="mlp2x_gelu")
    parser.add_argument("--vision-hidden-size", type=int, default=1152)
    parser.add_argument("--mm-pooler-ratio", type=int, default=2)
    parser.add_argument("--prompt", default="cot", choices=["direct", "cot", "ccot", "dsp"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-mode", default="random", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--null-visual-mode", default="zeros")
    parser.add_argument(
        "--analysis-mode",
        default="approx_global",
        choices=["approx_global", "final_step_leave_one_out"],
    )
    parser.add_argument(
        "--analysis-selection",
        default="all",
        choices=["all", "content", "last_n_content"],
    )
    parser.add_argument("--analysis-last-n", type=int, default=32)
    parser.add_argument("--analysis-batch-size", type=int, default=8)
    parser.add_argument("--keep-all-token-records", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--sensitivity-threshold", type=float, default=0.2)
    parser.add_argument("--top-suspicious", type=int, default=10)
    parser.add_argument("--postmask-vrg-enable", action="store_true")
    parser.add_argument("--postmask-vrg-alpha-start", type=float, default=0.0)
    parser.add_argument("--postmask-vrg-alpha-end", type=float, default=1.0)
    parser.add_argument(
        "--postmask-vrg-alpha-schedule",
        default="linear",
        choices=["linear", "cosine", "power"],
    )
    parser.add_argument("--postmask-vrg-alpha-power", type=float, default=2.0)
    return parser.parse_args()


def get_torch_dtype(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_choice_block(choices):
    return "\n".join(f"{LETTER_MAP[i]}. {choice}" for i, choice in enumerate(choices))


def build_base_prompt(doc):
    parts = []
    context = (doc.get("context") or "").strip()
    if context:
        parts.append(context)
    parts.append(doc["question"])
    parts.append(build_choice_block(doc["choices"]))
    return "\n".join(parts)


def build_prompt(doc, prompt_style):
    base = build_base_prompt(doc)
    if prompt_style == "direct":
        return base + "\n\nAnswer with the option's letter from the given choices directly."
    if prompt_style == "cot":
        return (
            base
            + "\n\nPlease reason step by step, and answer the question with option letter "
            + "from given choices in the format of Answer: <option letter>."
        )
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


def build_model(args):
    common_kwargs = dict(
        pretrained=args.model_path,
        truncation=True,
        device=args.device,
        batch_size=1,
        model_name="llava_llada",
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        conv_template=args.conv_template,
        use_cache=True,
        truncate_context=False,
        customized_config=None,
        max_frames_num=32,
        mm_spatial_pool_stride=2,
        mm_spatial_pool_mode="bilinear",
        token_strategy="single",
        video_decode_backend="decord",
        mc_num=16,
    )

    os.environ["LLADA_VISION_ENCODER"] = args.vision_tower
    os.environ["LLADA_VISION_PROJECTOR"] = args.vision_projector
    os.environ["LLADA_VISION_ENCODER_HIDDEN_SIZE"] = str(args.vision_hidden_size)
    os.environ["LLADA_MM_POOLER_RATIO"] = str(args.mm_pooler_ratio)
    return Llava_Llada(**common_kwargs)


def prepare_sample_inputs(model_wrapper, doc, prompt_text, torch_dtype):
    tokenizer = model_wrapper.tokenizer
    image = doc["image"].convert("RGB")
    image_tensor = process_images([image], model_wrapper._image_processor, model_wrapper.config)
    dtype = get_torch_dtype(torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=model_wrapper.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=model_wrapper.device)

    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    if "llama_3" in model_wrapper.conv_template or "llada" in model_wrapper.conv_template:
        conv = copy.deepcopy(conv_templates[model_wrapper.conv_template])
    else:
        conv = conv_templates[model_wrapper.conv_template].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model_wrapper.device)
    attention_masks = torch.ones_like(input_ids, dtype=torch.bool, device=model_wrapper.device)
    return image, image_tensor, input_ids, attention_masks


def strip_trailing_special_tokens(tokenizer, token_ids):
    special_ids = set(tokenizer.all_special_ids)
    trimmed = list(token_ids)
    while trimmed and int(trimmed[-1]) in special_ids:
        trimmed.pop()
    return trimmed


def generate_prediction(model_wrapper, input_ids, attention_mask, image_tensor, image_size, args):
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        sequences = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            images=image_tensor,
            image_sizes=[image_size],
            use_cache=True,
            max_new_tokens=args.max_new_tokens,
            block_length=int(args.block_length or args.max_new_tokens),
            step_ratio=args.step_ratio,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            top_p=args.top_p,
            num_beams=args.num_beams,
        )

    continuation_ids = sequences[0, -int(args.max_new_tokens) :].detach().cpu().tolist()
    continuation_ids = strip_trailing_special_tokens(tokenizer, continuation_ids)
    prediction_text = tokenizer.decode(continuation_ids, skip_special_tokens=True).lstrip("!").strip()
    return continuation_ids, prediction_text


def prepare_prefill_states(model, input_ids, attention_mask, images, image_sizes, null_visual_mode):
    core_model = model.get_model()
    cond_inputs_embeds, prefix_input_ids_full = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=images,
        image_sizes=image_sizes,
        attention_mask=attention_mask,
    )
    uncond_inputs_embeds, _ = build_unconditional_prefix_embeds(
        core_model=core_model,
        prefix_embeds=cond_inputs_embeds,
        prefix_input_ids_full=prefix_input_ids_full,
        null_visual_mode=null_visual_mode,
    )

    cond_prefill = core_model(None, input_embeddings=cond_inputs_embeds, use_cache=True)
    uncond_prefill = core_model(None, input_embeddings=uncond_inputs_embeds, use_cache=True)
    return core_model, cond_prefill, uncond_prefill


def decode_token(tokenizer, token_id, skip_special_tokens=False):
    text = tokenizer.decode([int(token_id)], skip_special_tokens=skip_special_tokens)
    return text.replace("\n", "\\n")


def token_rank(logits, token_id):
    token_logit = logits[0, int(token_id)]
    return int((logits[0] > token_logit).sum().item()) + 1


def repeat_past_key_values(past_key_values, batch_size):
    def _repeat(item):
        if torch.is_tensor(item):
            if item.shape[0] != 1:
                raise ValueError(f"Expected cached batch size 1, got shape {tuple(item.shape)}")
            return item.expand(batch_size, *item.shape[1:])
        if isinstance(item, tuple):
            return tuple(_repeat(sub_item) for sub_item in item)
        if isinstance(item, list):
            return [_repeat(sub_item) for sub_item in item]
        raise TypeError(f"Unsupported past_key_values item type: {type(item)!r}")

    return _repeat(past_key_values)


def compute_remasking_confidence(logits: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits.to(torch.float64), dim=-1)
    return torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)


def build_vrg_guided_logits(
    logits_cond: torch.Tensor,
    logits_uncond: torch.Tensor,
    alpha_t: float,
) -> torch.Tensor:
    return logits_uncond + (float(alpha_t) + 1.0) * (logits_cond - logits_uncond)


def classify_token_text(token_text):
    stripped = token_text.strip()
    if not stripped:
        return {"is_whitespace": True, "is_punctuation": False, "contains_digit": False}
    return {
        "is_whitespace": False,
        "is_punctuation": all(ch in string.punctuation for ch in stripped),
        "contains_digit": any(ch.isdigit() for ch in stripped),
    }


@torch.no_grad()
def generate_and_trace_transfer(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    images,
    image_size,
    max_new_tokens,
    block_length,
    step_ratio,
    temperature,
    null_visual_mode,
    confidence_threshold,
    sensitivity_threshold,
    postmask_vrg_enable,
    postmask_vrg_alpha_start,
    postmask_vrg_alpha_end,
    postmask_vrg_alpha_schedule,
    postmask_vrg_alpha_power,
):
    core_model = model.get_model()
    cond_inputs_embeds, prefix_input_ids_full = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=images,
        image_sizes=[image_size],
        attention_mask=attention_mask,
    )
    uncond_inputs_embeds, _ = build_unconditional_prefix_embeds(
        core_model=core_model,
        prefix_embeds=cond_inputs_embeds,
        prefix_input_ids_full=prefix_input_ids_full,
        null_visual_mode=null_visual_mode,
    )

    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")

    bsz = cond_inputs_embeds.shape[0]
    if bsz != 1:
        raise ValueError("Only batch_size=1 is supported for transfer tracing.")

    cond_past_key_values = core_model(None, input_embeddings=cond_inputs_embeds, use_cache=True).attn_key_values
    uncond_past_key_values = core_model(None, input_embeddings=uncond_inputs_embeds, use_cache=True).attn_key_values

    x = torch.full((bsz, max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=cond_inputs_embeds.device)
    num_blocks = max_new_tokens // block_length
    steps = max_new_tokens // num_blocks
    if step_ratio is not None:
        steps = int(steps * float(step_ratio))
    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0. Increase step_ratio or max_new_tokens.")

    transfer_records = {}
    global_step_idx = 0
    total_denoising_steps = num_blocks * steps

    for block_idx in range(num_blocks):
        block_start = block_idx * block_length
        block_end = (block_idx + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == MASK_TOKEN_ID
        num_transfer_tokens = get_num_transfer_tokens_sch(
            block_mask_index,
            steps,
            schedule=None,
            schedule_kwargs=None,
        )

        for step_idx in range(num_transfer_tokens.shape[1]):
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_start:block_end]
            if block_mask_index.sum().item() == 0:
                global_step_idx += 1
                continue

            current_embeds = core_model.transformer.wte(x)
            cond_out = core_model(
                None,
                input_embeddings=current_embeds,
                past_key_values=cond_past_key_values,
            )
            uncond_out = core_model(
                None,
                input_embeddings=current_embeds,
                past_key_values=uncond_past_key_values,
            )
            logits_cond = cond_out.logits.to(torch.float64)
            logits_uncond = uncond_out.logits.to(torch.float64)
            if postmask_vrg_enable:
                alpha_t = compute_step_vrg_alpha(
                    global_step_idx=global_step_idx,
                    total_steps=total_denoising_steps,
                    alpha_start=postmask_vrg_alpha_start,
                    alpha_end=postmask_vrg_alpha_end,
                    schedule=postmask_vrg_alpha_schedule,
                    power=postmask_vrg_alpha_power,
                )
                logits_select = build_vrg_guided_logits(
                    logits_cond=logits_cond,
                    logits_uncond=logits_uncond,
                    alpha_t=alpha_t,
                )
            else:
                alpha_t = 0.0
                logits_select = logits_cond

            logits_with_noise = add_gumbel_noise(logits_select, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0_conf = compute_remasking_confidence(logits_select, x0)

            x0_conf[:, block_end:] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_conf, -torch.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            k = int(num_transfer_tokens[0, step_idx].item())
            _, select_index = torch.topk(confidence[0], k=k)
            transfer_index[0, select_index] = True

            cond_log_probs = torch.log_softmax(logits_cond, dim=-1)
            uncond_log_probs = torch.log_softmax(logits_uncond, dim=-1)
            guided_log_probs = torch.log_softmax(logits_select, dim=-1)
            cond_probs = cond_log_probs.exp()
            cond_top1 = torch.argmax(logits_cond, dim=-1)
            uncond_top1 = torch.argmax(logits_uncond, dim=-1)
            guided_top1 = torch.argmax(logits_select, dim=-1)

            for pos in select_index.tolist():
                if pos in transfer_records:
                    continue
                token_id = int(x0[0, pos].item())
                cond_lp = float(cond_log_probs[0, pos, token_id].item())
                uncond_lp = float(uncond_log_probs[0, pos, token_id].item())
                guided_lp = float(guided_log_probs[0, pos, token_id].item())
                cond_conf = float(confidence[0, pos].item())
                sensitivity = cond_lp - uncond_lp
                cond_top1_id = int(cond_top1[0, pos].item())
                uncond_top1_id = int(uncond_top1[0, pos].item())
                guided_top1_id = int(guided_top1[0, pos].item())
                token_text = decode_token(tokenizer, token_id, skip_special_tokens=False)
                text_flags = classify_token_text(token_text.replace("\\n", "\n"))
                distribution_kl = float(
                    (
                        cond_probs[0, pos]
                        * (cond_log_probs[0, pos] - uncond_log_probs[0, pos])
                    ).sum().item()
                )

                record = {
                    "answer_position": int(pos),
                    "token_id": token_id,
                    "token_text": token_text,
                    "token_text_clean": decode_token(tokenizer, token_id, skip_special_tokens=True),
                    "masked_context": "transfer_time",
                    "transfer_step": int(global_step_idx + 1),
                    "block_index": int(block_idx + 1),
                    "step_in_block": int(step_idx + 1),
                    "conditional_log_prob": cond_lp,
                    "unconditional_log_prob": uncond_lp,
                    "guided_log_prob": guided_lp,
                    "confidence": cond_conf,
                    "visual_sensitivity": sensitivity,
                    "prob_ratio": float(torch.exp(cond_log_probs[0, pos, token_id] - uncond_log_probs[0, pos, token_id]).item()),
                    "distribution_kl": distribution_kl,
                    "postmask_vrg_alpha_t": float(alpha_t),
                    "conditional_rank": token_rank(logits_cond[0, pos].unsqueeze(0), token_id),
                    "unconditional_rank": token_rank(logits_uncond[0, pos].unsqueeze(0), token_id),
                    "guided_rank": token_rank(logits_select[0, pos].unsqueeze(0), token_id),
                    "conditional_top1_token_id": cond_top1_id,
                    "conditional_top1_token_text": decode_token(tokenizer, cond_top1_id, skip_special_tokens=False),
                    "conditional_top1_log_prob": float(cond_log_probs[0, pos, cond_top1_id].item()),
                    "unconditional_top1_token_id": uncond_top1_id,
                    "unconditional_top1_token_text": decode_token(tokenizer, uncond_top1_id, skip_special_tokens=False),
                    "unconditional_top1_log_prob": float(uncond_log_probs[0, pos, uncond_top1_id].item()),
                    "guided_top1_token_id": guided_top1_id,
                    "guided_top1_token_text": decode_token(tokenizer, guided_top1_id, skip_special_tokens=False),
                    "guided_top1_log_prob": float(guided_log_probs[0, pos, guided_top1_id].item()),
                    "top1_agree": cond_top1_id == uncond_top1_id,
                    "guided_matches_cond_top1": guided_top1_id == cond_top1_id,
                    "guided_matches_uncond_top1": guided_top1_id == uncond_top1_id,
                    "is_high_conf_low_sensitivity": cond_conf >= confidence_threshold and sensitivity <= sensitivity_threshold,
                }
                record.update(text_flags)
                transfer_records[pos] = record

            x[transfer_index] = x0[transfer_index]
            global_step_idx += 1

    ordered_positions = sorted(transfer_records)
    ordered_records = [transfer_records[pos] for pos in ordered_positions]
    generated_token_ids = [record["token_id"] for record in ordered_records]
    prediction_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).lstrip("!").strip()
    return generated_token_ids, prediction_text, ordered_records


def select_token_records(token_records, selection, last_n):
    if selection == "all":
        selected = list(token_records)
    else:
        content = [
            record
            for record in token_records
            if not record.get("is_punctuation") and not record.get("is_whitespace")
        ]
        if selection == "content":
            selected = content
        elif selection == "last_n_content":
            selected = content[-max(1, int(last_n)) :]
        else:
            raise ValueError(f"Unsupported analysis selection: {selection}")

    return {
        "selected_records": selected,
        "num_total_records": len(token_records),
        "num_content_records": sum(
            1
            for record in token_records
            if not record.get("is_punctuation") and not record.get("is_whitespace")
        ),
        "num_selected_records": len(selected),
    }


def summarize_token_records(token_records, top_suspicious, confidence_threshold, sensitivity_threshold):
    if not token_records:
        return {
            "num_tokens": 0,
            "mean_confidence": None,
            "mean_visual_sensitivity": None,
            "mean_distribution_kl": None,
            "num_high_conf_low_sensitivity": 0,
            "high_conf_low_sensitivity_ratio": None,
            "quadrant_counts": {
                "high_conf_high_vis": 0,
                "high_conf_low_vis": 0,
                "low_conf_high_vis": 0,
                "low_conf_low_vis": 0,
            },
            "quadrant_ratios": {
                "high_conf_high_vis": None,
                "high_conf_low_vis": None,
                "low_conf_high_vis": None,
                "low_conf_low_vis": None,
            },
            "suspicious_positions": [],
            "top_suspicious_tokens": [],
        }

    suspicious = [record for record in token_records if record["is_high_conf_low_sensitivity"]]
    suspicious_sorted = sorted(
        suspicious,
        key=lambda item: (item["visual_sensitivity"], -item["confidence"]),
    )
    quadrant_counts = {
        "high_conf_high_vis": 0,
        "high_conf_low_vis": 0,
        "low_conf_high_vis": 0,
        "low_conf_low_vis": 0,
    }
    for record in token_records:
        high_conf = record["confidence"] >= confidence_threshold
        high_vis = record["visual_sensitivity"] > sensitivity_threshold
        if high_conf and high_vis:
            quadrant_counts["high_conf_high_vis"] += 1
        elif high_conf and not high_vis:
            quadrant_counts["high_conf_low_vis"] += 1
        elif not high_conf and high_vis:
            quadrant_counts["low_conf_high_vis"] += 1
        else:
            quadrant_counts["low_conf_low_vis"] += 1

    return {
        "num_tokens": len(token_records),
        "mean_confidence": sum(item["confidence"] for item in token_records) / len(token_records),
        "mean_visual_sensitivity": sum(item["visual_sensitivity"] for item in token_records) / len(token_records),
        "mean_distribution_kl": sum(item["distribution_kl"] for item in token_records) / len(token_records),
        "num_high_conf_low_sensitivity": len(suspicious),
        "high_conf_low_sensitivity_ratio": len(suspicious) / len(token_records),
        "quadrant_counts": quadrant_counts,
        "quadrant_ratios": {
            key: value / len(token_records)
            for key, value in quadrant_counts.items()
        },
        "suspicious_positions": [item["answer_position"] for item in suspicious_sorted],
        "top_suspicious_tokens": suspicious_sorted[:top_suspicious],
    }


def parse_choice_letter(prediction_text):
    patterns = [
        r"Answer:\s*([A-G])",
        r"\[Answer\]\s*\(?([A-G])\)?",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, prediction_text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    fallback = re.findall(r"\b([A-G])\b", prediction_text, flags=re.IGNORECASE)
    if fallback:
        return fallback[-1].upper()
    return None


def load_existing_ids(output_path):
    if not output_path.exists():
        return set()
    ids = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = obj.get("id")
            if sample_id is not None:
                ids.add(sample_id)
    return ids


def load_dataset_split(args):
    dataset = datasets.load_dataset(args.dataset_path, split=args.split)
    if args.doc_id is not None:
        for index, doc in enumerate(dataset):
            if doc["id"] == args.doc_id:
                return [(index, doc)]
        raise ValueError(f"Could not find doc_id={args.doc_id!r} in {args.dataset_path}:{args.split}.")

    if args.sample_mode == "random":
        dataset = dataset.shuffle(seed=args.sample_seed)
    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    return list(enumerate(dataset))


def main():
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        done_ids = load_existing_ids(output_path)
        file_mode = "a"
    else:
        done_ids = set()
        file_mode = "w"

    dataset_entries = load_dataset_split(args)
    model_wrapper = build_model(args)

    num_written = 0
    with output_path.open(file_mode, encoding="utf-8") as fout:
        for dataset_index, doc in tqdm(dataset_entries, desc=f"M3CoT {args.split}"):
            if doc["id"] in done_ids:
                continue

            prompt_text = build_prompt(doc, args.prompt)
            image, image_tensor, input_ids, attention_mask = prepare_sample_inputs(
                model_wrapper,
                doc,
                prompt_text,
                args.torch_dtype,
            )
            generated_token_ids, prediction_text, token_records = generate_and_trace_transfer(
                model=model_wrapper.model,
                tokenizer=model_wrapper.tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                image_size=image.size,
                max_new_tokens=args.max_new_tokens,
                block_length=int(args.block_length or args.max_new_tokens),
                step_ratio=args.step_ratio,
                temperature=args.temperature,
                null_visual_mode=args.null_visual_mode,
                confidence_threshold=args.confidence_threshold,
                sensitivity_threshold=args.sensitivity_threshold,
                postmask_vrg_enable=args.postmask_vrg_enable,
                postmask_vrg_alpha_start=args.postmask_vrg_alpha_start,
                postmask_vrg_alpha_end=args.postmask_vrg_alpha_end,
                postmask_vrg_alpha_schedule=args.postmask_vrg_alpha_schedule,
                postmask_vrg_alpha_power=args.postmask_vrg_alpha_power,
            )
            selection_info = select_token_records(
                token_records=token_records,
                selection=args.analysis_selection,
                last_n=args.analysis_last_n,
            )
            selected_token_records = selection_info["selected_records"]
            summary = summarize_token_records(
                selected_token_records,
                args.top_suspicious,
                args.confidence_threshold,
                args.sensitivity_threshold,
            )
            parsed_answer = parse_choice_letter(prediction_text)

            record = {
                "id": doc["id"],
                "dataset_index": dataset_index,
                "question": doc["question"],
                "choices": list(doc["choices"]),
                "gold_answer": doc["answer"],
                "domain": doc["domain"],
                "topic": doc["topic"],
                "prompt_style": args.prompt,
                "prompt_text": prompt_text,
                "prediction_text": prediction_text,
                "parsed_prediction": parsed_answer,
                "is_correct_by_letter": parsed_answer == doc["answer"] if parsed_answer is not None else None,
                "generated_token_ids": generated_token_ids,
                "analysis_method": "transfer_time_confidence_with_synced_cond_uncond_vis",
                "analysis_selection": args.analysis_selection,
                "analysis_last_n": args.analysis_last_n,
                "analysis_metric": "visual_sensitivity = log p(token | image, transfer_state) - log p(token | null_visual, transfer_state)",
                "postmask_vrg_enable": bool(args.postmask_vrg_enable),
                "postmask_vrg_alpha_start": float(args.postmask_vrg_alpha_start),
                "postmask_vrg_alpha_end": float(args.postmask_vrg_alpha_end),
                "postmask_vrg_alpha_schedule": args.postmask_vrg_alpha_schedule,
                "postmask_vrg_alpha_power": float(args.postmask_vrg_alpha_power),
                "analysis_batch_size": args.analysis_batch_size,
                "confidence_threshold": args.confidence_threshold,
                "sensitivity_threshold": args.sensitivity_threshold,
                "num_total_token_records": selection_info["num_total_records"],
                "num_content_token_records": selection_info["num_content_records"],
                "num_selected_token_records": selection_info["num_selected_records"],
                "summary": summary,
                "token_records": selected_token_records,
            }
            if args.keep_all_token_records:
                record["all_token_records"] = token_records
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1
            if num_written % max(1, args.save_every) == 0:
                fout.flush()

    print(f"Wrote {num_written} records to {output_path}")


if __name__ == "__main__":
    main()
