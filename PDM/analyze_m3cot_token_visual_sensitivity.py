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
from lmms_eval.models.llava_llada import Llava_Llada
from Scale_Attention.reweight_patch import build_prefix_from_multimodal_inputs
from VRG.timestep_vrg import build_unconditional_prefix_embeds


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
    parser.add_argument("--analysis-batch-size", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--sensitivity-threshold", type=float, default=0.2)
    parser.add_argument("--top-suspicious", type=int, default=10)
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

    if sequences.shape[1] > input_ids.shape[1]:
        continuation_ids = sequences[0, input_ids.shape[1] :].detach().cpu().tolist()
    else:
        continuation_ids = sequences[0].detach().cpu().tolist()
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


def classify_token_text(token_text):
    stripped = token_text.strip()
    if not stripped:
        return {"is_whitespace": True, "is_punctuation": False, "contains_digit": False}
    return {
        "is_whitespace": False,
        "is_punctuation": all(ch in string.punctuation for ch in stripped),
        "contains_digit": any(ch.isdigit() for ch in stripped),
    }


def analyze_tokens(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    image_tensor,
    image_size,
    generated_token_ids,
    null_visual_mode,
    analysis_mode,
    analysis_batch_size,
    confidence_threshold,
    sensitivity_threshold,
):
    core_model, cond_prefill, uncond_prefill = prepare_prefill_states(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        image_sizes=[image_size],
        null_visual_mode=null_visual_mode,
    )
    records = []
    if not generated_token_ids:
        return records

    answer_ids = torch.tensor(generated_token_ids, dtype=torch.long, device=input_ids.device)
    cond_past = cond_prefill.attn_key_values
    uncond_past = uncond_prefill.attn_key_values

    if analysis_mode == "approx_global":
        answer_embeds = core_model.transformer.wte(answer_ids.unsqueeze(0))
        with torch.inference_mode():
            cond_out = core_model(
                None,
                input_embeddings=answer_embeds,
                past_key_values=cond_past,
            )
            uncond_out = core_model(
                None,
                input_embeddings=answer_embeds,
                past_key_values=uncond_past,
            )

        cond_logits = cond_out.logits[0].to(torch.float64)
        uncond_logits = uncond_out.logits[0].to(torch.float64)
        cond_log_probs = torch.log_softmax(cond_logits, dim=-1)
        uncond_log_probs = torch.log_softmax(uncond_logits, dim=-1)
        cond_probs = cond_log_probs.exp()

        for answer_position, token_id in enumerate(generated_token_ids):
            cond_lp = float(cond_log_probs[answer_position, token_id].item())
            uncond_lp = float(uncond_log_probs[answer_position, token_id].item())
            confidence = float(cond_probs[answer_position, token_id].item())
            sensitivity = cond_lp - uncond_lp
            cond_top1_id = int(torch.argmax(cond_logits[answer_position]).item())
            uncond_top1_id = int(torch.argmax(uncond_logits[answer_position]).item())
            cond_top1_lp = float(cond_log_probs[answer_position, cond_top1_id].item())
            uncond_top1_lp = float(uncond_log_probs[answer_position, uncond_top1_id].item())
            token_text = decode_token(tokenizer, token_id, skip_special_tokens=False)
            text_flags = classify_token_text(token_text.replace("\\n", "\n"))

            record = {
                "answer_position": answer_position,
                "token_id": int(token_id),
                "token_text": token_text,
                "token_text_clean": decode_token(tokenizer, token_id, skip_special_tokens=True),
                "masked_context": "approx_global_no_leave_one_out",
                "conditional_log_prob": cond_lp,
                "unconditional_log_prob": uncond_lp,
                "confidence": confidence,
                "visual_sensitivity": sensitivity,
                "prob_ratio": float(torch.exp(cond_log_probs[answer_position, token_id] - uncond_log_probs[answer_position, token_id]).item()),
                "distribution_kl": float((cond_probs[answer_position] * (cond_log_probs[answer_position] - uncond_log_probs[answer_position])).sum().item()),
                "conditional_rank": token_rank(cond_logits[answer_position].unsqueeze(0), token_id),
                "unconditional_rank": token_rank(uncond_logits[answer_position].unsqueeze(0), token_id),
                "conditional_top1_token_id": cond_top1_id,
                "conditional_top1_token_text": decode_token(tokenizer, cond_top1_id, skip_special_tokens=False),
                "conditional_top1_log_prob": cond_top1_lp,
                "unconditional_top1_token_id": uncond_top1_id,
                "unconditional_top1_token_text": decode_token(tokenizer, uncond_top1_id, skip_special_tokens=False),
                "unconditional_top1_log_prob": uncond_top1_lp,
                "top1_agree": cond_top1_id == uncond_top1_id,
                "is_high_conf_low_sensitivity": confidence >= confidence_threshold and sensitivity <= sensitivity_threshold,
            }
            record.update(text_flags)
            records.append(record)
        return records

    with torch.inference_mode():
        for start in range(0, len(generated_token_ids), max(1, analysis_batch_size)):
            end = min(start + max(1, analysis_batch_size), len(generated_token_ids))
            positions = list(range(start, end))
            batch_size = len(positions)

            masked_answer_ids = answer_ids.unsqueeze(0).repeat(batch_size, 1)
            row_index = torch.arange(batch_size, device=input_ids.device)
            pos_index = torch.tensor(positions, dtype=torch.long, device=input_ids.device)
            masked_answer_ids[row_index, pos_index] = MASK_TOKEN_ID
            masked_embeds = core_model.transformer.wte(masked_answer_ids)

            cond_out = core_model(
                None,
                input_embeddings=masked_embeds,
                past_key_values=repeat_past_key_values(cond_past, batch_size),
            )
            uncond_out = core_model(
                None,
                input_embeddings=masked_embeds,
                past_key_values=repeat_past_key_values(uncond_past, batch_size),
            )

            cond_logits = cond_out.logits[row_index, pos_index, :].to(torch.float64)
            uncond_logits = uncond_out.logits[row_index, pos_index, :].to(torch.float64)
            cond_log_probs = torch.log_softmax(cond_logits, dim=-1)
            uncond_log_probs = torch.log_softmax(uncond_logits, dim=-1)
            cond_probs = cond_log_probs.exp()

            target_ids = answer_ids[pos_index]
            cond_target_lp = cond_log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
            uncond_target_lp = uncond_log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
            cond_target_prob = cond_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
            cond_top1_ids = torch.argmax(cond_logits, dim=-1)
            uncond_top1_ids = torch.argmax(uncond_logits, dim=-1)
            cond_top1_lp = cond_log_probs.gather(1, cond_top1_ids.unsqueeze(-1)).squeeze(-1)
            uncond_top1_lp = uncond_log_probs.gather(1, uncond_top1_ids.unsqueeze(-1)).squeeze(-1)
            kl_values = (cond_probs * (cond_log_probs - uncond_log_probs)).sum(dim=-1)
            cond_ranks = (cond_logits > cond_logits.gather(1, target_ids.unsqueeze(-1))).sum(dim=-1) + 1
            uncond_ranks = (uncond_logits > uncond_logits.gather(1, target_ids.unsqueeze(-1))).sum(dim=-1) + 1

            for local_idx, answer_position in enumerate(positions):
                token_id = int(target_ids[local_idx].item())
                cond_lp = float(cond_target_lp[local_idx].item())
                uncond_lp = float(uncond_target_lp[local_idx].item())
                confidence = float(cond_target_prob[local_idx].item())
                sensitivity = cond_lp - uncond_lp
                cond_top1_id = int(cond_top1_ids[local_idx].item())
                uncond_top1_id = int(uncond_top1_ids[local_idx].item())
                token_text = decode_token(tokenizer, token_id, skip_special_tokens=False)
                text_flags = classify_token_text(token_text.replace("\\n", "\n"))

                record = {
                    "answer_position": answer_position,
                    "token_id": token_id,
                    "token_text": token_text,
                    "token_text_clean": decode_token(tokenizer, token_id, skip_special_tokens=True),
                    "masked_context": "final_step_leave_one_out",
                    "conditional_log_prob": cond_lp,
                    "unconditional_log_prob": uncond_lp,
                    "confidence": confidence,
                    "visual_sensitivity": sensitivity,
                    "prob_ratio": float(torch.exp(cond_target_lp[local_idx] - uncond_target_lp[local_idx]).item()),
                    "distribution_kl": float(kl_values[local_idx].item()),
                    "conditional_rank": int(cond_ranks[local_idx].item()),
                    "unconditional_rank": int(uncond_ranks[local_idx].item()),
                    "conditional_top1_token_id": cond_top1_id,
                    "conditional_top1_token_text": decode_token(tokenizer, cond_top1_id, skip_special_tokens=False),
                    "conditional_top1_log_prob": float(cond_top1_lp[local_idx].item()),
                    "unconditional_top1_token_id": uncond_top1_id,
                    "unconditional_top1_token_text": decode_token(tokenizer, uncond_top1_id, skip_special_tokens=False),
                    "unconditional_top1_log_prob": float(uncond_top1_lp[local_idx].item()),
                    "top1_agree": cond_top1_id == uncond_top1_id,
                    "is_high_conf_low_sensitivity": confidence >= confidence_threshold and sensitivity <= sensitivity_threshold,
                }
                record.update(text_flags)
                records.append(record)

    return records


def summarize_token_records(token_records, top_suspicious):
    if not token_records:
        return {
            "num_tokens": 0,
            "mean_confidence": None,
            "mean_visual_sensitivity": None,
            "mean_distribution_kl": None,
            "num_high_conf_low_sensitivity": 0,
            "high_conf_low_sensitivity_ratio": None,
            "suspicious_positions": [],
            "top_suspicious_tokens": [],
        }

    suspicious = [record for record in token_records if record["is_high_conf_low_sensitivity"]]
    suspicious_sorted = sorted(
        suspicious,
        key=lambda item: (item["visual_sensitivity"], -item["confidence"]),
    )
    return {
        "num_tokens": len(token_records),
        "mean_confidence": sum(item["confidence"] for item in token_records) / len(token_records),
        "mean_visual_sensitivity": sum(item["visual_sensitivity"] for item in token_records) / len(token_records),
        "mean_distribution_kl": sum(item["distribution_kl"] for item in token_records) / len(token_records),
        "num_high_conf_low_sensitivity": len(suspicious),
        "high_conf_low_sensitivity_ratio": len(suspicious) / len(token_records),
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
            generated_token_ids, prediction_text = generate_prediction(
                model_wrapper=model_wrapper,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_tensor=image_tensor,
                image_size=image.size,
                args=args,
            )
            token_records = analyze_tokens(
                model=model_wrapper.model,
                tokenizer=model_wrapper.tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_tensor=image_tensor,
                image_size=image.size,
            generated_token_ids=generated_token_ids,
            null_visual_mode=args.null_visual_mode,
            analysis_mode=args.analysis_mode,
            analysis_batch_size=args.analysis_batch_size,
            confidence_threshold=args.confidence_threshold,
            sensitivity_threshold=args.sensitivity_threshold,
            )
            summary = summarize_token_records(token_records, args.top_suspicious)
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
                "analysis_method": args.analysis_mode,
                "analysis_metric": (
                    "visual_sensitivity = log p(token | image, final_answer) - log p(token | null_visual, final_answer)"
                    if args.analysis_mode == "approx_global"
                    else "visual_sensitivity = log p(token | image, final_answer_-i) - log p(token | null_visual, final_answer_-i)"
                ),
                "analysis_batch_size": args.analysis_batch_size,
                "confidence_threshold": args.confidence_threshold,
                "sensitivity_threshold": args.sensitivity_threshold,
                "summary": summary,
                "token_records": token_records,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1
            if num_written % max(1, args.save_every) == 0:
                fout.flush()

    print(f"Wrote {num_written} records to {output_path}")


if __name__ == "__main__":
    main()
