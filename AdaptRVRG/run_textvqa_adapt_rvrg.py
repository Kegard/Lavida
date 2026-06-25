import argparse
import copy
import importlib.util
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


MASK_TOKEN_ID = 126336
REGION_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
METHODS = ("native", "global", "regional_weighted")
TEXTVQA_SHORT_PROMPT = "Answer the question using a single word or phrase."
TEXTVQA_REASONING_PROMPT = (
    "Please reason step by step, and answer the question using a single word or phrase "
    "in the format of Answer: <answer>."
)


def get_torch_dtype(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def maybe_disable_torch_compile():
    original_compile = getattr(torch, "compile", None)
    if original_compile is None:
        return lambda: None

    def eager_compile(fn=None, *compile_args, **compile_kwargs):
        if fn is None:
            return lambda inner_fn: inner_fn
        return fn

    torch.compile = eager_compile

    def restore():
        torch.compile = original_compile

    return restore


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for batch_idx in range(mask_num.size(0)):
        num_transfer_tokens[batch_idx, : remainder[batch_idx]] += 1
    return num_transfer_tokens


def get_num_transfer_tokens_sch(mask_index, steps, schedule=None, schedule_kwargs=None):
    if schedule is None:
        return get_num_transfer_tokens(mask_index, steps)
    if schedule_kwargs is None:
        schedule_kwargs = {}

    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = int(min(steps, mask_num[0]))
    t = torch.linspace(0, 1, steps + 1, device=mask_index.device)
    if schedule == "logit_normal":
        safe_t = torch.clamp(t, 1e-6, 1 - 1e-6)
        logit_t = torch.log(safe_t / (1 - safe_t))
        sigmas = 0.5 * (1 + torch.erf(logit_t / torch.sqrt(torch.tensor(2.0, device=mask_index.device))))
    elif schedule == "shift":
        shift = float(schedule_kwargs.get("shift", 3))
        sigmas = shift * t / (1 + (shift - 1) * t)
    elif schedule == "cosine":
        sigmas = 1 - 0.5 * (1 + torch.cos(torch.pi * t))
    else:
        sigmas = t

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    for batch_idx in range(mask_num.size(0)):
        sigmas_sample = (sigmas * mask_num[batch_idx]).to(torch.int64)
        sigmas_sample = sigmas_sample[1:] - sigmas_sample[:-1]
        sigmas_sample = torch.clamp(sigmas_sample, 1, None)
        delta = sigmas_sample.sum() - mask_num[batch_idx]
        cursor = 0
        while delta > 0:
            cursor = cursor % len(sigmas_sample)
            if sigmas_sample[cursor] == 1:
                cursor += 1
                continue
            delta -= 1
            sigmas_sample[cursor] -= 1
            cursor += 1
        num_transfer_tokens[batch_idx] = sigmas_sample
    return num_transfer_tokens.flip(-1)


def compute_remasking_confidence(logits, x0, remasking):
    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    if remasking == "random":
        return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    if remasking == "entrophy":
        epsilon = 1e-10
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.sum(probs * torch.log(probs + epsilon), dim=-1)
    if remasking == "margin":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        return sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
    raise NotImplementedError(remasking)


def load_evalai_answer_processor():
    metric_path = EVAL_ROOT / "lmms_eval" / "tasks" / "_task_utils" / "vqa_eval_metric.py"
    spec = importlib.util.spec_from_file_location("adapt_rvrg_vqa_eval_metric", metric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EvalAIAnswerProcessor


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
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal no-training Adaptive Regional VRG experiment on TextVQA. "
            "Runs native, global blur VRG, and regional weighted blur VRG for direct comparison."
        )
    )
    parser.add_argument("--dataset-path", default="lmms-lab/textvqa")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output-dir", default="AdaptRVRG/outputs/textvqa_minimal")
    parser.add_argument("--methods", default="native,global,regional_weighted")
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
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--guide-lambda", type=float, default=0.2)
    parser.add_argument("--regional-tau", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--blur-radius", type=float, default=10.0)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def parse_methods(raw_methods):
    methods = []
    for item in raw_methods.split(","):
        method = item.strip()
        if not method:
            continue
        if method not in METHODS:
            raise ValueError(f"Unsupported method {method!r}. Choose from {METHODS}.")
        methods.append(method)
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    return methods


def region_boxes(width, height):
    mid_x = width // 2
    mid_y = height // 2
    return (
        (0, 0, mid_x, mid_y),
        (mid_x, 0, width, mid_y),
        (0, mid_y, mid_x, height),
        (mid_x, mid_y, width, height),
    )


def build_blurred_images(image, blur_radius):
    full_blur = image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    regional = []
    for box in region_boxes(*image.size):
        patched = image.copy()
        patched.paste(full_blur.crop(box), box)
        regional.append(patched)
    return full_blur, regional


def build_prefix_for_image(args, model, tokenizer, image_processor, image, prompt):
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token

    image_tensor = process_images([image], image_processor, model.config)
    dtype = get_torch_dtype(args.torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=args.device)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
    ret = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=image_tensor,
        modalities=["image"],
        image_sizes=[image.size],
        return_inputs=True,
    )
    return ret[4], ret[-1][0]


def prepare_prefix_bundle(args, model, tokenizer, image_processor, doc, need_global, need_regional):
    image = doc["image"].convert("RGB")
    context = construct_textvqa_prompt(
        doc,
        prompt_mode=args.prompt_mode,
        pretrained_path=args.pretrained,
    )
    prompt = build_prompt(context, args.conv_template)
    full_prefix, prefix_input_ids_full = build_prefix_for_image(args, model, tokenizer, image_processor, image, prompt)

    global_prefix = None
    regional_prefixes = None
    if need_global or need_regional:
        global_blur, regional_blurs = build_blurred_images(image, args.blur_radius)
        if need_global:
            global_prefix, _ = build_prefix_for_image(args, model, tokenizer, image_processor, global_blur, prompt)
            if global_prefix.shape != full_prefix.shape:
                raise ValueError("Global blurred prefix shape differs from the full-image prefix shape.")
        if need_regional:
            regional_prefixes = []
            for regional_blur in regional_blurs:
                regional_prefix, _ = build_prefix_for_image(args, model, tokenizer, image_processor, regional_blur, prompt)
                if regional_prefix.shape != full_prefix.shape:
                    raise ValueError("Regional blurred prefix shape differs from the full-image prefix shape.")
                regional_prefixes.append(regional_prefix)

    return {
        "context": context,
        "prompt": prompt,
        "prefix_embeds": full_prefix,
        "prefix_input_ids_full": prefix_input_ids_full,
        "global_prefix_embeds": global_prefix,
        "regional_prefix_embeds": regional_prefixes,
    }


def resolve_steps(max_new_tokens, block_length, step_ratio):
    if max_new_tokens % block_length != 0:
        raise ValueError("--max-new-tokens must be divisible by --block-length.")
    num_blocks = max_new_tokens // block_length
    steps = int((max_new_tokens // num_blocks) * step_ratio)
    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0.")
    return num_blocks, steps


def forward_with_prefix(core_model, x, prefix_embeds):
    prefix_length = prefix_embeds.shape[1]
    current_embeds = core_model.transformer.wte(x)
    current_embeds[:, :prefix_length] = prefix_embeds
    return core_model(None, input_embeddings=current_embeds).logits


def topk_guided_logits(logits_full, deltas, active_positions, top_k, guide_lambda):
    guided = logits_full.clone()
    if active_positions.numel() == 0:
        return guided
    vocab_size = logits_full.shape[-1]
    k = min(int(top_k), int(vocab_size))
    top_indices = torch.topk(logits_full[0, active_positions, :], k=k, dim=-1).indices
    for row_idx, pos in enumerate(active_positions.tolist()):
        indices = top_indices[row_idx]
        guided[0, pos, indices] = guided[0, pos, indices] + float(guide_lambda) * deltas[row_idx].to(guided.dtype)
    return guided


def build_global_guided_logits(core_model, x, prefix_embeds_global, logits_full, active_positions, top_k, guide_lambda):
    logits_blur = forward_with_prefix(core_model, x, prefix_embeds_global)
    top_indices = torch.topk(
        logits_full[0, active_positions, :],
        k=min(int(top_k), int(logits_full.shape[-1])),
        dim=-1,
    ).indices
    delta_top = torch.gather(
        (logits_full[0, active_positions, :] - logits_blur[0, active_positions, :]).to(torch.float32),
        dim=-1,
        index=top_indices,
    )
    return topk_guided_logits(logits_full, delta_top, active_positions, top_k, guide_lambda), {
        "mean_delta": float(delta_top.mean().item()) if delta_top.numel() else 0.0,
    }


def build_regional_guided_logits(
    core_model,
    x,
    regional_prefixes,
    logits_full,
    active_positions,
    top_k,
    guide_lambda,
    tau,
):
    if tau <= 0:
        raise ValueError("--regional-tau must be > 0.")
    k = min(int(top_k), int(logits_full.shape[-1]))
    top_indices = torch.topk(logits_full[0, active_positions, :], k=k, dim=-1).indices
    deltas = []
    scores = []
    for regional_prefix in regional_prefixes:
        logits_blur = forward_with_prefix(core_model, x, regional_prefix)
        delta_top = torch.gather(
            (logits_full[0, active_positions, :] - logits_blur[0, active_positions, :]).to(torch.float32),
            dim=-1,
            index=top_indices,
        )
        deltas.append(delta_top)
        scores.append(delta_top.mean())
    score_tensor = torch.stack(scores)
    weights = F.softmax(score_tensor / float(tau), dim=0)
    weighted_delta = torch.zeros_like(deltas[0])
    for weight, delta_top in zip(weights, deltas):
        weighted_delta = weighted_delta + weight * delta_top
    guided = topk_guided_logits(logits_full, weighted_delta, active_positions, top_k, guide_lambda)
    best_region_idx = int(torch.argmax(score_tensor).item())
    return guided, {
        "region_scores": [float(value) for value in score_tensor.detach().cpu().tolist()],
        "region_weights": [float(value) for value in weights.detach().cpu().tolist()],
        "best_region": REGION_NAMES[best_region_idx],
    }


@torch.no_grad()
def generate_with_adaptive_guide(
    core_model,
    tokenizer,
    prefix_embeds,
    method,
    max_new_tokens,
    block_length,
    step_ratio,
    schedule,
    schedule_shift,
    temperature,
    remasking,
    confidence_threshold,
    top_k,
    guide_lambda,
    global_prefix_embeds=None,
    regional_prefix_embeds=None,
    regional_tau=1.0,
):
    if method == "global" and global_prefix_embeds is None:
        raise ValueError("global_prefix_embeds is required for global method.")
    if method == "regional_weighted" and not regional_prefix_embeds:
        raise ValueError("regional_prefix_embeds is required for regional_weighted method.")

    device = prefix_embeds.device
    batch_size, prefix_length = prefix_embeds.shape[:2]
    prompt = torch.full((batch_size, prefix_length), 0, dtype=torch.long, device=device)
    x = torch.full((batch_size, prefix_length + max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :prefix_length] = prompt

    num_blocks, steps = resolve_steps(max_new_tokens, block_length, step_ratio)
    schedule_value = None if schedule == "none" else schedule
    schedule_kwargs = {"shift": schedule_shift} if schedule_value == "shift" else None

    trace = []
    num_guided_steps = 0
    total_steps = 0
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

            total_steps += 1
            logits_full = forward_with_prefix(core_model, x, prefix_embeds)
            logits_with_noise = add_gumbel_noise(logits_full, temperature=temperature)
            x0_full = torch.argmax(logits_with_noise, dim=-1)
            x0_p = compute_remasking_confidence(logits_full, x0_full, remasking)
            x0_p[:, prefix_length + (block_idx + 1) * block_length :] = -torch.inf
            x0_full = torch.where(mask_index, x0_full, x)

            confidence = torch.where(mask_index, x0_p, -torch.inf)
            k_transfer = int(num_transfer_tokens[0, step_idx].item())
            _, select_index = torch.topk(confidence[0], k=k_transfer)
            selected_conf = confidence[0, select_index]
            finite_selected_conf = selected_conf[torch.isfinite(selected_conf)]
            step_conf = float(finite_selected_conf.mean().item()) if finite_selected_conf.numel() else 0.0

            use_guide = method != "native" and step_conf < float(confidence_threshold)
            guide_meta = {}
            logits_for_x0 = logits_full
            if use_guide:
                active_positions = torch.nonzero(block_mask_index[0], as_tuple=False).squeeze(-1) + int(block_slice.start)
                if method == "global":
                    logits_for_x0, guide_meta = build_global_guided_logits(
                        core_model=core_model,
                        x=x,
                        prefix_embeds_global=global_prefix_embeds,
                        logits_full=logits_full,
                        active_positions=active_positions,
                        top_k=top_k,
                        guide_lambda=guide_lambda,
                    )
                elif method == "regional_weighted":
                    logits_for_x0, guide_meta = build_regional_guided_logits(
                        core_model=core_model,
                        x=x,
                        regional_prefixes=regional_prefix_embeds,
                        logits_full=logits_full,
                        active_positions=active_positions,
                        top_k=top_k,
                        guide_lambda=guide_lambda,
                        tau=regional_tau,
                    )
                num_guided_steps += 1

            guided_logits_with_noise = add_gumbel_noise(logits_for_x0, temperature=temperature)
            x0 = torch.argmax(guided_logits_with_noise, dim=-1)
            x0 = torch.where(mask_index, x0, x)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            transfer_index[0, select_index] = True
            x[transfer_index] = x0[transfer_index]

            trace_item = {
                "step": int(total_steps),
                "block_index": int(block_idx + 1),
                "step_in_block": int(step_idx + 1),
                "num_transferred": int(k_transfer),
                "step_confidence": step_conf,
                "guided": bool(use_guide),
                "selected_positions": [int(pos) for pos in select_index.detach().cpu().tolist()],
                "num_masked_after_step": int((x[:, prefix_length:] == MASK_TOKEN_ID).sum().item()),
            }
            trace_item.update(guide_meta)
            trace.append(trace_item)

    final_text = clean_generated_text(
        tokenizer.decode(
            x[0, prefix_length:].detach().cpu().tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )
    meta = {
        "method": method,
        "prefix_length": int(prefix_length),
        "num_blocks": int(num_blocks),
        "steps_per_block": int(steps),
        "total_steps": int(total_steps),
        "num_guided_steps": int(num_guided_steps),
    }
    return final_text, trace, meta


def correct_from_score(score):
    return bool(score > 0.0)


def main():
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be > 0.")
    methods = parse_methods(args.methods)
    need_global = "global" in methods
    need_regional = "regional_weighted" in methods

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
    answer_processor = load_evalai_answer_processor()()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"

    method_totals = defaultdict(float)
    method_counts = defaultdict(int)
    guided_step_totals = defaultdict(int)
    elapsed_totals = defaultdict(float)
    transition_counts = {
        method: {"wrong_to_correct": 0, "correct_to_wrong": 0, "same_correct": 0, "same_wrong": 0}
        for method in methods
        if method != "native"
    }

    total_elapsed = 0.0
    written = 0
    with records_path.open("w", encoding="utf-8") as fout:
        for dataset_index in range(args.start_index, len(dataset)):
            if written >= args.limit:
                break
            doc = dataset[dataset_index]
            if doc.get("image") is None:
                continue

            sample_t0 = time.time()
            bundle = prepare_prefix_bundle(
                args=args,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                doc=doc,
                need_global=need_global,
                need_regional=need_regional,
            )
            normalized_answers = normalize_answers(doc, answer_processor)

            method_results = {}
            for method in methods:
                method_t0 = time.time()
                prediction, trace, meta = generate_with_adaptive_guide(
                    core_model=core_model,
                    tokenizer=tokenizer,
                    prefix_embeds=bundle["prefix_embeds"],
                    method=method,
                    max_new_tokens=args.max_new_tokens,
                    block_length=args.block_length,
                    step_ratio=args.step_ratio,
                    schedule=args.schedule,
                    schedule_shift=args.schedule_shift,
                    temperature=args.temperature,
                    remasking=args.remasking,
                    confidence_threshold=args.confidence_threshold,
                    top_k=args.top_k,
                    guide_lambda=args.guide_lambda,
                    global_prefix_embeds=bundle["global_prefix_embeds"],
                    regional_prefix_embeds=bundle["regional_prefix_embeds"],
                    regional_tau=args.regional_tau,
                )
                elapsed = time.time() - method_t0
                exact_match, normalized_prediction = compute_textvqa_score(
                    normalized_answers,
                    prediction,
                    answer_processor,
                )
                method_totals[method] += exact_match
                method_counts[method] += 1
                guided_step_totals[method] += int(meta["num_guided_steps"])
                elapsed_totals[method] += elapsed
                method_results[method] = {
                    "prediction": prediction,
                    "normalized_prediction": normalized_prediction,
                    "exact_match": exact_match,
                    "correct": correct_from_score(exact_match),
                    "elapsed_sec": elapsed,
                    "num_guided_steps": int(meta["num_guided_steps"]),
                    "trace": trace,
                    "meta": meta,
                }

            if "native" in method_results:
                native_correct = method_results["native"]["correct"]
                for method, counts in transition_counts.items():
                    guided_correct = method_results[method]["correct"]
                    if not native_correct and guided_correct:
                        counts["wrong_to_correct"] += 1
                    elif native_correct and not guided_correct:
                        counts["correct_to_wrong"] += 1
                    elif native_correct and guided_correct:
                        counts["same_correct"] += 1
                    else:
                        counts["same_wrong"] += 1

            sample_elapsed = time.time() - sample_t0
            total_elapsed += sample_elapsed
            record = {
                "dataset_index": int(dataset_index),
                "question_id": doc.get("question_id"),
                "question": bundle["context"],
                "answers": doc.get("answers"),
                "normalized_answers": normalized_answers,
                "ocr_tokens": doc.get("ocr_tokens"),
                "prompt": bundle["prompt"],
                "elapsed_sec": sample_elapsed,
                "method_results": method_results,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                scores = ", ".join(
                    f"{method}={method_totals[method] / max(1, method_counts[method]):.3f}"
                    for method in methods
                )
                print(f"[{written}] dataset_index={dataset_index} elapsed={sample_elapsed:.2f}s {scores}")

    method_summary = []
    for method in methods:
        count = method_counts[method]
        method_summary.append(
            {
                "method": method,
                "mean_exact_match": method_totals[method] / count if count else None,
                "count": int(count),
                "mean_guided_steps": guided_step_totals[method] / count if count else None,
                "mean_elapsed_sec": elapsed_totals[method] / count if count else None,
            }
        )

    summary = {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "start_index": args.start_index,
        "num_samples": written,
        "methods": methods,
        "total_elapsed_sec": total_elapsed,
        "mean_sample_elapsed_sec": total_elapsed / written if written else None,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "step_ratio": args.step_ratio,
            "schedule": args.schedule,
            "schedule_shift": args.schedule_shift,
            "remasking": args.remasking,
            "temperature": args.temperature,
        },
        "guide": {
            "confidence_threshold": args.confidence_threshold,
            "guide_lambda": args.guide_lambda,
            "regional_tau": args.regional_tau,
            "top_k": args.top_k,
            "blur_radius": args.blur_radius,
            "trigger": "step_confidence < confidence_threshold",
            "scope": "top-k logits only; unmask position selection remains native/full-image confidence",
        },
        "method_summary": method_summary,
        "transitions_vs_native": transition_counts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote records to {records_path}")
    print(f"Wrote summary to {summary_path}")
    for item in method_summary:
        print(
            f"{item['method']}: mean_exact_match={item['mean_exact_match']:.4f} "
            f"mean_guided_steps={item['mean_guided_steps']:.2f} "
            f"mean_elapsed_sec={item['mean_elapsed_sec']:.2f} count={item['count']}"
        )
    if transition_counts:
        print(json.dumps(transition_counts, ensure_ascii=False, indent=2))
    restore_compile()


if __name__ == "__main__":
    main()
