import argparse
import copy
import json
import re
import statistics
import sys
import time
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


TEXTVQA_SHORT_PROMPT = "Answer the question using a single word or phrase."
TEXTVQA_REASONING_PROMPT = (
    "Please reason step by step, and answer the question using a single word or phrase "
    "in the format of Answer: <answer>."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TextVQA visual warmup decoding: use image-conditioned logits for early steps, then switch to unconditioned logits."
    )
    parser.add_argument("--dataset-path", default="lmms-lab/textvqa")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--cutoffs",
        default="0,4,8,16,24,32",
        help="Comma-separated 1-based warmup step cutoffs. 0 means all-uncond; total steps means all-cond.",
    )
    parser.add_argument("--output-dir", default="VRG/outputs/textvqa_visual_warmup")
    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--prompt-mode", default="auto", choices=["auto", "short", "reasoning"])
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
    parser.add_argument(
        "--score-mode",
        default="final_candidate",
        choices=["cutoff_candidate", "final_candidate"],
        help="Score x0 at the warmup cutoff step, or score the last x0 after all denoising steps.",
    )
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def parse_cutoffs(raw_cutoffs):
    cutoffs = []
    for item in raw_cutoffs.split(","):
        item = item.strip()
        if not item:
            continue
        cutoffs.append(int(item))
    if not cutoffs:
        raise ValueError("--cutoffs must contain at least one integer.")
    return sorted(set(cutoffs))


def load_textvqa_split(dataset_path, dataset_name, split):
    if dataset_name is None:
        return load_dataset(dataset_path, split=split)
    return load_dataset(dataset_path, dataset_name, split=split)


def resolve_textvqa_prompt_mode(prompt_mode, pretrained_path):
    if prompt_mode == "auto":
        pretrained_text = (pretrained_path or "").lower()
        return "reasoning" if "reason" in pretrained_text else "short"
    return prompt_mode


def construct_textvqa_prompt(doc, prompt_mode="auto", pretrained_path=None):
    resolved_mode = resolve_textvqa_prompt_mode(prompt_mode, pretrained_path)
    prompt_suffix = TEXTVQA_REASONING_PROMPT if resolved_mode == "reasoning" else TEXTVQA_SHORT_PROMPT
    return f"{doc['question'].capitalize()}\n{prompt_suffix}"


def build_prompt(context, conv_template):
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def clean_generated_text(text):
    return text.replace("<|endoftext|>", "").replace("<|eot_id|>", "").strip()


def _clean_extracted_answer(text):
    text = text.strip().strip("\"'`*")
    text = re.sub(r"^[-*]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(".,;:!?")
    return text


def _short_candidate_or_none(text, max_words=8):
    candidate = _clean_extracted_answer(text)
    if not candidate:
        return None
    if len(candidate.split()) > max_words:
        return None
    return candidate


def extract_textvqa_final_answer(text):
    cleaned = clean_generated_text(text)
    for pattern in (r"Answer\s*:\s*(.+)", r"Final answer\s*:\s*(.+)"):
        matches = re.findall(pattern, cleaned, flags=re.IGNORECASE)
        if matches:
            candidate = _short_candidate_or_none(matches[-1].splitlines()[0])
            if candidate is not None:
                return candidate

    boxed_matches = re.findall(r"\\boxed\s*{([^{}]+)}", cleaned)
    if boxed_matches:
        candidate = _short_candidate_or_none(boxed_matches[-1])
        if candidate is not None:
            return candidate

    nonempty_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if nonempty_lines:
        last_line = nonempty_lines[-1]
        candidate = _short_candidate_or_none(last_line)
        if candidate is not None and not re.fullmatch(r"(therefore|thus|so)[,:]?", candidate, flags=re.IGNORECASE):
            return candidate

    quoted_matches = re.findall(r'"([^"\n]{1,80})"|\'([^\'\n]{1,80})\'', cleaned)
    if quoted_matches:
        flat_matches = [a or b for a, b in quoted_matches]
        candidate = _short_candidate_or_none(flat_matches[-1])
        if candidate is not None:
            return candidate

    sentence_patterns = (
        r"(?:therefore|thus|so)[^.\n]*?\b(?:is|are|was|were)\s+([^.\n]+)",
        r"\b(?:answer|number|brand|word|time|title|name|value|type|state|color|event|measurement)\b[^.\n]*?\b(?:is|are|was|were)\s+([^.\n]+)",
    )
    for pattern in sentence_patterns:
        matches = re.findall(pattern, cleaned, flags=re.IGNORECASE)
        if matches:
            candidate = _short_candidate_or_none(matches[-1])
            if candidate is not None:
                return candidate

    tail = cleaned.splitlines()[-1] if cleaned.splitlines() else cleaned
    return _clean_extracted_answer(tail)


def normalize_answers(doc, answer_processor):
    answers = doc.get("answers")
    if not answers:
        return []
    return [answer_processor(answer) for answer in answers if answer is not None]


def compute_textvqa_score(normalized_answers, prediction, answer_processor):
    normalized_prediction = answer_processor(extract_textvqa_final_answer(prediction))
    if not normalized_answers:
        return 0.0, normalized_prediction

    gt_acc = []
    for idx in range(len(normalized_answers)):
        other_answers = [normalized_answers[j] for j in range(len(normalized_answers)) if j != idx]
        matching = [answer for answer in other_answers if answer == normalized_prediction]
        gt_acc.append(min(1.0, float(len(matching)) / 3.0))
    return statistics.mean(gt_acc), normalized_prediction


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


def prepare_prefix(args, model, tokenizer, image_processor, doc):
    image = doc["image"].convert("RGB")
    context = construct_textvqa_prompt(
        doc,
        prompt_mode=getattr(args, "prompt_mode", "auto"),
        pretrained_path=getattr(args, "pretrained", None),
    )
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


@torch.no_grad()
def generate_with_visual_warmup(
    core_model,
    tokenizer,
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
    warmup_cutoff,
):
    device = prefix_embeds.device
    paired_past, num_visual_tokens = init_paired_state(
        core_model,
        prefix_embeds,
        prefix_input_ids_full,
        null_visual_mode,
    )
    x = torch.full((1, max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    num_blocks = max_new_tokens // block_length
    steps = int((max_new_tokens // num_blocks) * step_ratio)
    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0.")
    schedule_kwargs = {"shift": schedule_shift} if schedule == "shift" else None

    trace = []
    global_step = 0
    last_candidate_text = ""
    cutoff_candidate_text = None
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
            step_number = global_step + 1
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
            use_visual = step_number <= warmup_cutoff
            logits = logits_cond if use_visual else logits_uncond

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0_p = compute_remasking_confidence(logits, x0, remasking)
            x0_p[:, (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            last_candidate_text = clean_generated_text(
                tokenizer.decode(
                    x0[0].detach().cpu().tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            if step_number == warmup_cutoff:
                cutoff_candidate_text = last_candidate_text
            confidence = torch.where(mask_index, x0_p, -torch.inf)

            k = int(num_transfer_tokens[0, step_idx].item())
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            _, select_index = torch.topk(confidence[0], k=k)
            transfer_index[0, select_index] = True
            x[transfer_index] = x0[transfer_index]

            trace.append(
                {
                    "step": int(step_number),
                    "block_index": int(block_idx + 1),
                    "step_in_block": int(step_idx + 1),
                    "mode": "cond" if use_visual else "uncond",
                    "num_transferred": k,
                    "selected_positions": [int(pos) for pos in select_index.detach().cpu().tolist()],
                    "num_masked_after_step": int((x == MASK_TOKEN_ID).sum().item()),
                    "candidate_text": last_candidate_text,
                }
            )
            global_step += 1

    final_state_text = clean_generated_text(tokenizer.decode(
        x[0].detach().cpu().tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ))
    if cutoff_candidate_text is None:
        cutoff_candidate_text = last_candidate_text
    return (
        cutoff_candidate_text,
        last_candidate_text,
        final_state_text,
        trace,
        {"num_visual_tokens": num_visual_tokens, "steps_per_block": steps},
    )


def main():
    args = parse_args()
    if args.max_new_tokens % args.block_length != 0:
        raise ValueError("--max-new-tokens must be divisible by --block-length.")
    if args.limit <= 0:
        raise ValueError("--limit must be > 0.")

    cutoffs = parse_cutoffs(args.cutoffs)
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
    simple_records_path = output_dir / "records_simple.json"
    summary_path = output_dir / "summary.json"

    cutoff_totals = {cutoff: 0.0 for cutoff in cutoffs}
    cutoff_counts = {cutoff: 0 for cutoff in cutoffs}
    total_elapsed = 0.0
    written = 0
    simple_records = []

    with records_path.open("w", encoding="utf-8") as fout:
        for dataset_index in range(args.start_index, len(dataset)):
            if written >= args.limit:
                break
            doc = dataset[dataset_index]
            if doc.get("image") is None:
                continue

            t0 = time.time()
            context, prompt, prefix_embeds, prefix_input_ids_full = prepare_prefix(
                args,
                model,
                tokenizer,
                image_processor,
                doc,
            )
            normalized_answers = normalize_answers(doc, answer_processor)
            cutoff_results = []
            for cutoff in cutoffs:
                cutoff_candidate_text, final_candidate_text, final_state_text, trace, meta = generate_with_visual_warmup(
                    core_model=core_model,
                    tokenizer=tokenizer,
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
                    warmup_cutoff=cutoff,
                )
                scored_text = cutoff_candidate_text if args.score_mode == "cutoff_candidate" else final_candidate_text
                exact_match, normalized_prediction = compute_textvqa_score(
                    normalized_answers,
                    scored_text,
                    answer_processor,
                )
                cutoff_totals[cutoff] += exact_match
                cutoff_counts[cutoff] += 1
                cutoff_results.append(
                    {
                        "cutoff": int(cutoff),
                        "score_mode": args.score_mode,
                        "scored_text": scored_text,
                        "cutoff_candidate_text": cutoff_candidate_text,
                        "final_candidate_text": final_candidate_text,
                        "final_state_text": final_state_text,
                        "normalized_prediction": normalized_prediction,
                        "exact_match": exact_match,
                        "num_cond_steps": sum(1 for item in trace if item["mode"] == "cond"),
                        "num_uncond_steps": sum(1 for item in trace if item["mode"] == "uncond"),
                        "trace": trace,
                        "meta": meta,
                    }
                )

            elapsed = time.time() - t0
            total_elapsed += elapsed
            record = {
                "dataset_index": int(dataset_index),
                "question_id": doc.get("question_id"),
                "question": context,
                "answers": doc.get("answers"),
                "normalized_answers": normalized_answers,
                "ocr_tokens": doc.get("ocr_tokens"),
                "prompt": prompt,
                "elapsed_sec": elapsed,
                "cutoff_results": cutoff_results,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            simple_records.append(
                {
                    "dataset_index": int(dataset_index),
                    "question_id": doc.get("question_id"),
                    "question": context,
                    "answers": doc.get("answers"),
                    "normalized_answers": normalized_answers,
                    "cutoff_results": [
                        {
                            "cutoff": item["cutoff"],
                            "prediction": item["scored_text"],
                            "normalized_prediction": item["normalized_prediction"],
                            "exact_match": item["exact_match"],
                        }
                        for item in cutoff_results
                    ],
                }
            )
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                scores = ", ".join(
                    f"k={cutoff}: {cutoff_totals[cutoff] / max(1, cutoff_counts[cutoff]):.3f}"
                    for cutoff in cutoffs
                )
                print(f"[{written}] dataset_index={dataset_index} elapsed={elapsed:.2f}s {scores}")

    cutoff_summary = []
    for cutoff in cutoffs:
        count = cutoff_counts[cutoff]
        cutoff_summary.append(
            {
                "cutoff": int(cutoff),
                "mean_exact_match": cutoff_totals[cutoff] / count if count else None,
                "count": count,
            }
        )

    summary = {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "start_index": args.start_index,
        "num_samples": written,
        "cutoffs": cutoffs,
        "cutoff_definition": "1-based denoising steps: step <= cutoff uses cond logits, later steps use uncond logits.",
        "score_mode": args.score_mode,
        "score_definition": "final_candidate scores the last-step x0 after all denoising steps; cutoff_candidate scores x0 at the cutoff step.",
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": total_elapsed / written if written else None,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "step_ratio": args.step_ratio,
            "schedule": args.schedule,
            "schedule_shift": args.schedule_shift,
            "remasking": args.remasking,
            "null_visual_mode": args.null_visual_mode,
            "temperature": args.temperature,
        },
        "cutoff_summary": cutoff_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    simple_records_path.write_text(json.dumps(simple_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote records to {records_path}")
    print(f"Wrote simple records to {simple_records_path}")
    print(f"Wrote summary to {summary_path}")
    for item in cutoff_summary:
        print(f"cutoff={item['cutoff']} mean_exact_match={item['mean_exact_match']:.4f} count={item['count']}")
    restore_compile()


if __name__ == "__main__":
    main()
