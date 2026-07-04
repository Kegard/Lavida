import argparse
import copy
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
for path in (REPO_ROOT, EVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from M3CoT.utils.metric import judge_answer
from Scale_Attention.reweight_patch import (
    build_prefix_from_multimodal_inputs,
    get_torch_dtype,
    maybe_disable_torch_compile,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch


MASK_TOKEN_ID = 126336
LETTER_MAP = "ABCDEFG"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate each LaViDa denoising step's x0 decode on M3CoT."
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sample-mode", default="sequential", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="M3CoT/outputs/stepwise_x0")

    parser.add_argument("--pretrained", default="weight/lavida-reason")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--prompt", default="cot", choices=["direct", "cot", "ccot", "dsp"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--step-per-block", type=int, default=None)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


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


def clean_generated_text(text):
    text = text.lstrip("!")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("<|eot_id|>", "")
    text = text.replace("<|im_end|>\n", "")
    text = text.replace("<|im_end|>", "")
    return text.strip()


def compute_remasking_confidence(logits, x0, remasking):
    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    if remasking == "random":
        return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    if remasking == "entrophy":
        epsilon = 1e-10
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        log_probs = torch.log(probs + epsilon)
        return torch.sum(probs * log_probs, dim=-1)
    if remasking == "margin":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        return sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
    raise NotImplementedError(remasking)


def prepare_prefix(args, model, tokenizer, image_processor, doc):
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
    return context, prompt, prefix_embeds, prefix_input_ids_full


@torch.no_grad()
def generate_stepwise_x0_records(
    core_model,
    tokenizer,
    prefix_embeds,
    max_new_tokens,
    block_length,
    step_per_block,
    cfg_scale,
    temperature,
    remasking,
    schedule,
    schedule_shift,
    step_ratio,
):
    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")
    if cfg_scale > 0.0:
        raise NotImplementedError("cfg_scale > 0.0 is not supported in the native path.")

    device = prefix_embeds.device
    batch_size, prefix_length = prefix_embeds.shape[:2]
    prompt = torch.full((batch_size, prefix_length), 0, dtype=torch.long, device=device)
    x = torch.full((batch_size, prefix_length + max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
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

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0_p = compute_remasking_confidence(logits, x0, remasking)
            x0_p[:, prefix_length + (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)

            candidate_token_ids = x0[0, prefix_length:].detach().cpu().tolist()
            candidate_text = clean_generated_text(
                tokenizer.decode(
                    candidate_token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )

            confidence = torch.where(mask_index, x0_p, -torch.inf)
            k = int(num_transfer_tokens[0, step_idx].item())
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            _, select_index = torch.topk(confidence[0], k=k)
            transfer_index[0, select_index] = True
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
                    "candidate_token_ids": candidate_token_ids,
                    "candidate_text": candidate_text,
                    "state_answer_ids": state_answer_ids,
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

    step_correct = defaultdict(float)
    step_counts = defaultdict(int)
    total_elapsed = 0.0
    written = 0

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
            step_records, final_state_text, meta = generate_stepwise_x0_records(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                step_per_block=args.step_per_block,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                remasking=args.remasking,
                schedule=args.schedule,
                schedule_shift=args.schedule_shift,
                step_ratio=args.step_ratio,
            )

            scored_steps = []
            for step_record in step_records:
                is_correct = bool(judge_answer(step_record["candidate_text"], doc["choices"], doc["answer"]))
                step = int(step_record["step"])
                step_correct[step] += float(is_correct)
                step_counts[step] += 1

                scored_step = dict(step_record)
                scored_step["correct"] = is_correct
                scored_steps.append(scored_step)

            elapsed = time.time() - t0
            total_elapsed += elapsed
            final_correct = bool(judge_answer(final_state_text, doc["choices"], doc["answer"]))

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
                "final_correct": final_correct,
                "step_results": scored_steps,
                "meta": meta,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                preview = ", ".join(
                    f"step={step}: {step_correct[step] / step_counts[step]:.3f}"
                    for step in sorted(step_counts)[: min(5, len(step_counts))]
                )
                print(
                    f"[{written}] dataset_index={dataset_index} id={doc['id']} "
                    f"elapsed={elapsed:.2f}s final={final_correct} {preview}",
                    flush=True,
                )

    step_summary = []
    for step in sorted(step_counts):
        count = step_counts[step]
        step_summary.append(
            {
                "step": int(step),
                "mean_acc": step_correct[step] / count if count else None,
                "count": int(count),
            }
        )

    summary = {
        "dataset_path": args.dataset_path,
        "split": args.split,
        "start_index": args.start_index,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed if args.sample_mode == "random" else None,
        "num_samples": written,
        "prompt": args.prompt,
        "step_definition": "1-based native denoising steps; each score is computed from the x0 decode at that step without any extra refine pass.",
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": total_elapsed / written if written else None,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "step_per_block": args.step_per_block,
            "step_ratio": args.step_ratio,
            "schedule": args.schedule,
            "schedule_shift": args.schedule_shift,
            "cfg_scale": args.cfg_scale,
            "remasking": args.remasking,
            "temperature": args.temperature,
        },
        "step_summary": step_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote records to {records_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)
    for item in step_summary:
        print(f"step={item['step']} mean_acc={item['mean_acc']:.4f} count={item['count']}", flush=True)
    restore_compile()


if __name__ == "__main__":
    main()
