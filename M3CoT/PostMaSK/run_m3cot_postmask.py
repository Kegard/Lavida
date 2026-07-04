import argparse
import copy
import json
import math
import re
import string
import sys
import time
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
    prepare_prefix,
)
from M3CoT.StepMask.free_correction import compute_leave_one_out_metrics
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
from llava.model.language_model.llada.generate import add_gumbel_noise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a simple two-stage PostMask decoding experiment on M3CoT."
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--domain-filter", default=None)
    parser.add_argument("--sample-mode", default="sequential", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="M3CoT/PostMaSK/outputs/simple_postmask")

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

    parser.add_argument(
        "--draft-steps",
        type=int,
        default=None,
        help="Number of early steps used to reveal the full answer. Defaults to total_steps // 2.",
    )
    parser.add_argument(
        "--postmask-steps",
        type=int,
        default=None,
        help="Number of late remask-refill steps. Defaults to total_steps - draft_steps.",
    )
    parser.add_argument(
        "--remask-per-step",
        type=int,
        default=None,
        help="How many generated answer tokens to remask at each PostMask step. Defaults to the draft reveal count per step.",
    )
    parser.add_argument(
        "--remask-selection",
        default="cached_confidence",
        choices=[
            "cached_confidence",
            "proposal_confidence",
            "proposal_confidence_position_boost",
            "history_confidence",
            "mean_after_fill",
            "topk_margin",
            "kl_divergence",
            "full_confidence",
            "visual_gain",
            "high_visual_gain",
            "conf_low_visual_gain",
            "proposal_then_visual_gain",
            "rank_conf_visual_gain",
            "quadrant_conf_visual_gain",
            "norm_conf_visual_gain",
            "highgain_lowconf",
            "highgain_lowconf_negboost",
            "rule_content_confidence",
            "random",
        ],
        help="How to choose which generated tokens to remask during the PostMask phase.",
    )
    parser.add_argument(
        "--null-visual-mode",
        default="zeros",
        choices=["zeros", "mask_token"],
        help="How to build the weakened-visual prefix when computing visual_gain-based remask scores.",
    )
    parser.add_argument(
        "--selector-weak-visual-mode",
        default="null_visual",
        choices=["null_visual", "diffusion_noise"],
        help="Weak visual condition used only for visual_gain-based PostMask token selection.",
    )
    parser.add_argument(
        "--vcd-noise-step",
        type=int,
        default=500,
        help="Forward-diffusion timestep for --selector-weak-visual-mode diffusion_noise.",
    )
    parser.add_argument(
        "--vcd-noise-seed",
        type=int,
        default=None,
        help="Optional noise seed for --selector-weak-visual-mode diffusion_noise.",
    )
    parser.add_argument(
        "--refill-guidance",
        default="none",
        choices=["none", "vcd", "prompt_contrast"],
        help="Optional logits guidance used only during PostMask refill.",
    )
    parser.add_argument(
        "--draft-guidance",
        default="none",
        choices=["none", "vcd"],
        help="Optional logits guidance used during the draft generation stage.",
    )
    parser.add_argument(
        "--draft-weak-visual-mode",
        default="diffusion_noise",
        choices=["null_visual", "diffusion_noise"],
        help="Weak visual condition for VCD-guided draft logits.",
    )
    parser.add_argument(
        "--vcd-draft-alpha",
        type=float,
        default=1.0,
        help="Alpha in guided draft logits: (1 + alpha) * logits(image) - alpha * logits(weak_visual).",
    )
    parser.add_argument(
        "--draft-guidance-ratio",
        type=float,
        default=1.0,
        help="Fraction of draft steps that use --draft-guidance vcd, starting from the first draft step.",
    )
    parser.add_argument(
        "--refill-weak-visual-mode",
        default="diffusion_noise",
        choices=["null_visual", "diffusion_noise"],
        help="Weak visual condition for VCD-guided refill logits.",
    )
    parser.add_argument(
        "--vcd-refill-alpha",
        type=float,
        default=1.0,
        help="Alpha in guided refill logits: (1 + alpha) * logits(image) - alpha * logits(weak_visual).",
    )
    parser.add_argument(
        "--refill-vrg-calibration",
        default="none",
        choices=["none", "soft_confidence", "hard_confidence"],
        help="Only for VCD refill: calibrate token-wise VRG strength from proposal confidence.",
    )
    parser.add_argument(
        "--refill-vrg-confidence-threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for --refill-vrg-calibration hard_confidence.",
    )
    parser.add_argument(
        "--refill-vrg-confidence-gate-tau",
        type=float,
        default=None,
        help=(
            "Optional VCD refill gate. If set, use guided logits at an answer position only when "
            "max_softmax(guided) - max_softmax(cond) > tau; otherwise fall back to cond logits."
        ),
    )
    parser.add_argument(
        "--refill-guidance-steps",
        type=int,
        default=None,
        help="Only for refill guidance: apply VRG/prompt-contrast on the first k postmask steps, then fall back to normal refill.",
    )
    parser.add_argument(
        "--prompt-contrast-alpha",
        type=float,
        default=0.5,
        help="Alpha in prompt-contrast refill logits: (1 + alpha) * logits(original question) - alpha * logits(perturbed question).",
    )
    parser.add_argument(
        "--prompt-contrast-mode",
        default="confused_option_mapping",
        choices=["option_direct", "confused_option_mapping"],
        help="How to perturb only the question branch for prompt-contrast refill guidance.",
    )
    parser.add_argument(
        "--selector-candidate-size",
        type=int,
        default=64,
        help="Candidate pool size for two-stage selector modes.",
    )
    parser.add_argument(
        "--position-boost-lambda",
        type=float,
        default=0.0,
        help="Scale of the piecewise late-position boost for proposal_confidence_position_boost.",
    )
    parser.add_argument(
        "--position-boost-q2",
        type=float,
        default=0.0,
        help="Relative boost assigned to positions in [0.25, 0.5) for proposal_confidence_position_boost.",
    )
    parser.add_argument(
        "--position-boost-q3",
        type=float,
        default=0.5,
        help="Relative boost assigned to positions in [0.5, 0.75) for proposal_confidence_position_boost.",
    )
    parser.add_argument(
        "--position-boost-q4",
        type=float,
        default=1.0,
        help="Relative boost assigned to positions in [0.75, 1.0) for proposal_confidence_position_boost.",
    )
    parser.add_argument(
        "--rank-visual-weight",
        type=float,
        default=1.0,
        help="Weight on visual-gain rank for rank_conf_visual_gain.",
    )
    parser.add_argument(
        "--quadrant-confidence-threshold",
        type=float,
        default=0.9,
        help="High-confidence threshold for quadrant_conf_visual_gain.",
    )
    parser.add_argument(
        "--quadrant-visual-threshold",
        type=float,
        default=0.2,
        help="Low-visual-gain threshold for quadrant_conf_visual_gain.",
    )
    parser.add_argument(
        "--normalized-visual-alpha",
        type=float,
        default=1.0,
        help="Alpha weight on normalized visual_gain for norm_conf_visual_gain.",
    )
    parser.add_argument(
        "--normalized-confidence-beta",
        type=float,
        default=1.0,
        help="Beta weight on normalized confidence for norm_conf_visual_gain.",
    )
    parser.add_argument(
        "--negative-gain-lambda",
        type=float,
        default=1.0,
        help="Lambda weight on I(visual_gain < 0) for highgain_lowconf_negboost.",
    )
    parser.add_argument(
        "--postmask-mode",
        default="dynamic",
        choices=["dynamic", "fixed_set", "span_fixed_set"],
        help="Whether to reselect remasked positions every PostMask step or fix the remask set once.",
    )
    parser.add_argument(
        "--fixed-set-size",
        type=int,
        default=None,
        help="Only used when --postmask-mode fixed_set. If set, choose this many positions for the fixed remask set.",
    )
    parser.add_argument(
        "--fixed-refill-per-step",
        type=int,
        default=None,
        help="Only used when --postmask-mode fixed_set/span_fixed_set. If set, refill exactly this many masked positions per step.",
    )
    parser.add_argument(
        "--span-num-spans",
        type=int,
        default=4,
        help="Only used when --postmask-mode span_fixed_set. Number of fixed confidence spans to remask.",
    )
    parser.add_argument(
        "--span-length",
        type=int,
        default=8,
        help="Only used when --postmask-mode span_fixed_set. Length of each fixed remask span.",
    )
    parser.add_argument(
        "--loo-chunk-size",
        type=int,
        default=16,
        help="Chunk size for full-confidence leave-one-out rescoring.",
    )
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def resolve_total_steps(max_new_tokens, block_length, step_per_block, step_ratio):
    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")
    num_blocks = max_new_tokens // block_length
    if num_blocks != 1:
        raise ValueError("The simple PostMask runner currently expects block_length == max_new_tokens.")

    steps = max_new_tokens
    if step_per_block is not None:
        if step_ratio is not None:
            raise ValueError("Do not pass both --step-per-block and --step-ratio.")
        steps = min(int(step_per_block), block_length)
    elif step_ratio is not None:
        steps = int(steps * float(step_ratio))
    if steps <= 0:
        raise ValueError("The computed total step count is 0.")
    return steps


def build_even_schedule(total_tokens, num_steps):
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0.")
    base = total_tokens // num_steps
    remainder = total_tokens % num_steps
    schedule = []
    for step_idx in range(num_steps):
        schedule.append(base + (1 if step_idx < remainder else 0))
    return schedule


def build_piecewise_position_boost(
    max_new_tokens,
    device,
    q2_boost,
    q3_boost,
    q4_boost,
):
    positions = torch.arange(max_new_tokens, device=device, dtype=torch.float64)
    pos_ratio = positions / float(max_new_tokens)
    boost = torch.zeros_like(positions, dtype=torch.float64)
    boost[(pos_ratio >= 0.25) & (pos_ratio < 0.5)] = float(q2_boost)
    boost[(pos_ratio >= 0.5) & (pos_ratio < 0.75)] = float(q3_boost)
    boost[pos_ratio >= 0.75] = float(q4_boost)
    return boost


def decode_answer(tokenizer, answer_ids):
    return clean_generated_text(
        tokenizer.decode(
            answer_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def forward_logits(core_model, x, prefix_embeds, prefix_length):
    current_embeds = core_model.transformer.wte(x)
    current_embeds[:, :prefix_length] = prefix_embeds
    return core_model(None, input_embeddings=current_embeds).logits


def forward_aligned_answer_logits(
    core_model,
    x,
    prefix_embeds,
    answer_prefix_length,
):
    prefix_length = int(prefix_embeds.shape[1])
    if prefix_length == int(answer_prefix_length):
        return forward_logits(core_model, x, prefix_embeds, prefix_length)

    answer_ids = x[:, answer_prefix_length:]
    aligned_x = torch.full(
        (x.shape[0], prefix_length + answer_ids.shape[1]),
        MASK_TOKEN_ID,
        dtype=x.dtype,
        device=x.device,
    )
    aligned_x[:, :prefix_length] = 0
    aligned_x[:, prefix_length:] = answer_ids
    aligned_logits = forward_logits(core_model, aligned_x, prefix_embeds, prefix_length)

    output = torch.zeros(
        (x.shape[0], x.shape[1], aligned_logits.shape[-1]),
        dtype=aligned_logits.dtype,
        device=aligned_logits.device,
    )
    output[:, answer_prefix_length:] = aligned_logits[:, prefix_length:]
    return output


def add_diffusion_noise_tensor(image_tensor, noise_step, seed=None):
    if not 0 <= int(noise_step) < 1000:
        raise ValueError("--vcd-noise-step must be in [0, 999].")

    device = image_tensor.device
    dtype = image_tensor.dtype
    betas = torch.linspace(-6, 6, 1000, device=device, dtype=torch.float32)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)[int(noise_step)]

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
            add_diffusion_noise_tensor(
                image,
                noise_step=noise_step,
                seed=None if seed is None else int(seed) + idx,
            )
            for idx, image in enumerate(images)
        ]
    return add_diffusion_noise_tensor(images, noise_step=noise_step, seed=seed)


def build_diffusion_noise_prefix(args, model, tokenizer, image_processor, doc):
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
    noisy_image_tensor = add_diffusion_noise(
        image_tensor,
        noise_step=args.vcd_noise_step,
        seed=args.vcd_noise_seed,
    )

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
    weak_prefix_embeds, _ = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=noisy_image_tensor,
        image_sizes=[image.size],
        attention_mask=attention_mask,
    )
    return weak_prefix_embeds


def build_prompt_contrast_context(doc, prompt_type, mode):
    if mode == "confused_option_mapping":
        return "You are a confused option-mapping assistant.\n\n" + build_prompt(
            doc,
            prompt_type,
        )
    if mode != "option_direct":
        raise ValueError(f"Unsupported prompt contrast mode: {mode}")
    contrast_doc = dict(doc)
    contrast_doc["question"] = (
        "Identify the option or answer shown inside the image and answer directly. "
        "Focus on the image-internal option label or content."
    )
    return build_prompt(contrast_doc, prompt_type)


def build_prompt_contrast_prefix(args, model, tokenizer, image_processor, doc):
    context = build_prompt_contrast_context(
        doc,
        args.prompt,
        args.prompt_contrast_mode,
    )
    image = doc["image"].convert("RGB")

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

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
    contrast_prefix_embeds, _ = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        attention_mask=attention_mask,
    )
    return contrast_prefix_embeds


def choose_positions(scores, k, mode):
    if k <= 0:
        return scores.new_empty(0, dtype=torch.long)
    if mode in {
        "cached_confidence",
        "proposal_confidence",
        "proposal_confidence_position_boost",
        "history_confidence",
        "mean_after_fill",
        "topk_margin",
        "kl_divergence",
        "full_confidence",
        "visual_gain",
        "high_visual_gain",
        "conf_low_visual_gain",
        "proposal_then_visual_gain",
        "rank_conf_visual_gain",
        "quadrant_conf_visual_gain",
        "norm_conf_visual_gain",
        "highgain_lowconf",
        "highgain_lowconf_negboost",
        "rule_content_confidence",
    }:
        return torch.topk(scores, k=k, largest=False).indices
    if mode == "random":
        return torch.randperm(scores.numel(), device=scores.device)[:k]
    raise ValueError(f"Unsupported remask selection mode: {mode}")


def choose_low_score_spans(scores, num_spans, span_length):
    if num_spans <= 0 or span_length <= 0:
        raise ValueError("num_spans and span_length must be > 0.")
    span_length = min(int(span_length), int(scores.numel()))
    candidates = []
    for start in range(0, int(scores.numel()) - span_length + 1):
        end = start + span_length
        span_scores = scores[start:end]
        if not torch.isfinite(span_scores).all():
            continue
        candidates.append((float(span_scores.mean().item()), start, end - 1))

    selected = []
    occupied = set()
    for score, start, end in sorted(candidates, key=lambda item: (item[0], item[1])):
        positions = set(range(start, end + 1))
        if occupied & positions:
            continue
        selected.append({"start": start, "end": end, "score": score})
        occupied.update(positions)
        if len(selected) >= int(num_spans):
            break

    selected = sorted(selected, key=lambda item: item["start"])
    flat_positions = [
        pos
        for span in selected
        for pos in range(int(span["start"]), int(span["end"]) + 1)
    ]
    positions = torch.tensor(flat_positions, dtype=torch.long, device=scores.device)
    return selected, positions


RULE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "correct",
    "determine",
    "for",
    "from",
    "in",
    "is",
    "it",
    "look",
    "need",
    "of",
    "on",
    "or",
    "that",
    "the",
    "therefore",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
}

RULE_VISUAL_WORDS = {
    "above",
    "behind",
    "below",
    "black",
    "blue",
    "bottom",
    "chart",
    "circle",
    "decrease",
    "front",
    "graph",
    "green",
    "higher",
    "image",
    "increase",
    "larger",
    "left",
    "line",
    "lower",
    "object",
    "person",
    "point",
    "red",
    "right",
    "shorter",
    "smaller",
    "square",
    "table",
    "top",
    "upper",
    "white",
    "yellow",
}

RULE_UNIT_WORDS = {
    "%",
    "$",
    "cm",
    "degree",
    "degrees",
    "hour",
    "hours",
    "kg",
    "km",
    "m",
    "mm",
    "percent",
    "year",
    "years",
}


def normalize_rule_token(token_text):
    return token_text.strip().lower().strip(string.punctuation)


def classify_rule_token(token_text, token_id):
    if int(token_id) == MASK_TOKEN_ID:
        return "non_content"
    raw = token_text.strip()
    token = normalize_rule_token(token_text)
    if not raw or not token:
        return "non_content"
    if all(ch in string.punctuation for ch in raw):
        return "non_content"
    if token in RULE_STOPWORDS:
        return "non_content"
    option = raw.strip().strip(".):]").upper()
    if option in {"A", "B", "C", "D", "E"}:
        return "option"
    if re.search(r"\d", raw):
        return "number"
    if token in RULE_VISUAL_WORDS:
        return "visual_word"
    if token in RULE_UNIT_WORDS:
        return "unit"
    if len(token) >= 4:
        return "lexical_content"
    return "non_content"


def classify_answer_tokens(tokenizer, token_ids):
    categories = []
    token_texts = []
    for token_id in token_ids:
        token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        token_texts.append(token_text)
        categories.append(classify_rule_token(token_text, int(token_id)))
    return token_texts, categories


def compute_rule_content_confidence_scores(tokenizer, x, prefix_length, proposal_confidence):
    answer_token_ids = x[0, prefix_length:].detach().cpu().tolist()
    _, categories = classify_answer_tokens(tokenizer, answer_token_ids)
    scores = proposal_confidence.clone().to(torch.float64)
    penalty = torch.full_like(scores, 1_000_000.0, dtype=torch.float64)
    content_mask = torch.tensor(
        [category != "non_content" for category in categories],
        dtype=torch.bool,
        device=scores.device,
    )
    return torch.where(content_mask, scores, scores + penalty)


def compute_current_answer_logprobs(core_model, x, prefix_embeds, prefix_length):
    logits = forward_logits(core_model, x, prefix_embeds, prefix_length)
    row_index = torch.arange(x.shape[0], device=x.device)
    target_ids = x[row_index, prefix_length:]
    selected_logits = logits[row_index, prefix_length:].to(torch.float64)
    log_probs = F.log_softmax(selected_logits, dim=-1)
    vocab_size = log_probs.shape[-1]
    valid_targets = (target_ids >= 0) & (target_ids < vocab_size) & (target_ids != MASK_TOKEN_ID)
    safe_target_ids = torch.where(valid_targets, target_ids, torch.zeros_like(target_ids))
    gathered = log_probs.gather(dim=-1, index=safe_target_ids.unsqueeze(-1)).squeeze(-1)
    return torch.where(valid_targets, gathered, torch.full_like(gathered, -torch.inf))


def compute_current_answer_probs(core_model, x, prefix_embeds, prefix_length):
    return compute_current_answer_logprobs(
        core_model=core_model,
        x=x,
        prefix_embeds=prefix_embeds,
        prefix_length=prefix_length,
    ).squeeze(0).exp()


def compute_visual_gain_scores(
    core_model,
    x,
    prefix_embeds,
    uncond_prefix_embeds,
    prefix_length,
):
    logits_cond = forward_logits(core_model, x, prefix_embeds, prefix_length)
    logits_uncond = forward_logits(core_model, x, uncond_prefix_embeds, prefix_length)
    answer_token_ids = x[0, prefix_length:]
    cond_log_probs = F.log_softmax(logits_cond[0, prefix_length:].to(torch.float64), dim=-1)
    uncond_log_probs = F.log_softmax(logits_uncond[0, prefix_length:].to(torch.float64), dim=-1)
    vocab_size = cond_log_probs.shape[-1]
    valid_targets = (answer_token_ids >= 0) & (answer_token_ids < vocab_size) & (answer_token_ids != MASK_TOKEN_ID)
    safe_answer_token_ids = torch.where(valid_targets, answer_token_ids, torch.zeros_like(answer_token_ids))
    gather_index = safe_answer_token_ids.unsqueeze(-1)
    cond_token_log_probs = cond_log_probs.gather(dim=-1, index=gather_index).squeeze(-1)
    uncond_token_log_probs = uncond_log_probs.gather(dim=-1, index=gather_index).squeeze(-1)
    visual_gain = cond_token_log_probs - uncond_token_log_probs
    return torch.where(valid_targets, visual_gain, torch.full_like(visual_gain, torch.inf))


def tensor_values_at(tensor, positions):
    if tensor is None:
        return None
    values = []
    for pos in positions.detach().cpu().tolist():
        value = float(tensor[int(pos)].item())
        values.append(value if math.isfinite(value) else None)
    return values


def tensor_to_float_list(tensor):
    if tensor is None:
        return None
    values = []
    for value in tensor.detach().cpu().tolist():
        value = float(value)
        values.append(value if math.isfinite(value) else None)
    return values


def gather_token_logprobs(logits, token_ids):
    log_probs = F.log_softmax(logits.to(torch.float64), dim=-1)
    return log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def gather_topk_margin(logits):
    probs = F.softmax(logits.to(torch.float64), dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def compute_tokenwise_kl_from_logprobs(current_log_probs, previous_log_probs):
    previous_log_probs = previous_log_probs.to(
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    )
    current_probs = current_log_probs.exp()
    return (current_probs * (current_log_probs - previous_log_probs)).sum(dim=-1)


def build_mean_scores(conf_sum, conf_count):
    scores = torch.full_like(conf_sum, float("inf"), dtype=torch.float64)
    seen = conf_count > 0
    if seen.any():
        scores[seen] = conf_sum[seen] / conf_count[seen].to(torch.float64)
    return scores, seen


def rank_scores_ascending(scores):
    finite = torch.isfinite(scores)
    ranks = torch.full_like(scores, float("inf"), dtype=torch.float64)
    if finite.any():
        order = torch.argsort(scores[finite], descending=False)
        finite_positions = torch.nonzero(finite, as_tuple=False).view(-1)
        ranked_positions = finite_positions[order]
        ranks[ranked_positions] = torch.arange(
            ranked_positions.numel(),
            device=scores.device,
            dtype=torch.float64,
        )
    return ranks


def minmax_normalize(scores):
    finite = torch.isfinite(scores)
    normalized = torch.full_like(scores, float("inf"), dtype=torch.float64)
    if finite.any():
        values = scores[finite].to(torch.float64)
        min_value = values.min()
        max_value = values.max()
        denom = max_value - min_value
        if float(denom.item()) == 0.0:
            normalized[finite] = 0.0
        else:
            normalized[finite] = (values - min_value) / denom
    return normalized


def max_softmax_probability(logits):
    logits = logits.to(torch.float64)
    return torch.exp(logits.max(dim=-1).values - torch.logsumexp(logits, dim=-1))


def apply_confidence_gated_refill_logits(cond_logits, guided_logits, answer_slice, tau):
    cond_answer_logits = cond_logits[:, answer_slice]
    guided_answer_logits = guided_logits[:, answer_slice]
    use_guided = (
        max_softmax_probability(guided_answer_logits)
        - max_softmax_probability(cond_answer_logits)
    ) > float(tau)
    logits = cond_logits.clone()
    logits[:, answer_slice] = torch.where(
        use_guided.unsqueeze(-1),
        guided_answer_logits,
        cond_answer_logits,
    )
    return logits


def compute_selection_scores(
    remask_selection,
    tokenizer,
    x,
    prefix_embeds,
    uncond_prefix_embeds,
    prefix_length,
    max_new_tokens,
    device,
    core_model,
    loo_chunk_size,
    cached_confidence,
    proposal_confidence,
    mean_after_fill,
    topk_margin,
    kl_divergence,
    history_confidence,
    precomputed_visual_gain=None,
    precomputed_confidence=None,
    selector_candidate_size=64,
    position_boost_lambda=0.0,
    position_boost_q2=0.0,
    position_boost_q3=0.5,
    position_boost_q4=1.0,
    rank_visual_weight=1.0,
    quadrant_confidence_threshold=0.9,
    quadrant_visual_threshold=0.2,
    normalized_visual_alpha=1.0,
    normalized_confidence_beta=1.0,
    negative_gain_lambda=1.0,
):
    if remask_selection == "full_confidence":
        candidate_positions = torch.arange(
            prefix_length,
            prefix_length + max_new_tokens,
            device=device,
            dtype=torch.long,
        )
        loo_metrics = compute_leave_one_out_metrics(
            core_model=core_model,
            x=x,
            prefix_embeds=prefix_embeds,
            prefix_length=prefix_length,
            candidate_positions=candidate_positions,
            chunk_size=loo_chunk_size,
            previous_log_probs_by_pos={},
            correction_metric="confidence",
            record_detailed_metrics=False,
        )
        return -loo_metrics["log_likelihood"]
    if remask_selection == "proposal_confidence":
        return proposal_confidence
    if remask_selection == "proposal_confidence_position_boost":
        position_boost = build_piecewise_position_boost(
            max_new_tokens=max_new_tokens,
            device=device,
            q2_boost=position_boost_q2,
            q3_boost=position_boost_q3,
            q4_boost=position_boost_q4,
        )
        return proposal_confidence - float(position_boost_lambda) * position_boost
    if remask_selection == "rule_content_confidence":
        return compute_rule_content_confidence_scores(
            tokenizer=tokenizer,
            x=x,
            prefix_length=prefix_length,
            proposal_confidence=proposal_confidence,
        )
    if remask_selection == "mean_after_fill":
        return mean_after_fill
    if remask_selection == "topk_margin":
        return topk_margin
    if remask_selection == "kl_divergence":
        return kl_divergence
    if remask_selection == "history_confidence":
        return history_confidence
    if remask_selection == "visual_gain":
        if precomputed_visual_gain is not None:
            return precomputed_visual_gain
        visual_gain = compute_visual_gain_scores(
            core_model=core_model,
            x=x,
            prefix_embeds=prefix_embeds,
            uncond_prefix_embeds=uncond_prefix_embeds,
            prefix_length=prefix_length,
        )
        return visual_gain
    if remask_selection == "high_visual_gain":
        if precomputed_visual_gain is not None:
            return -precomputed_visual_gain
        visual_gain = compute_visual_gain_scores(
            core_model=core_model,
            x=x,
            prefix_embeds=prefix_embeds,
            uncond_prefix_embeds=uncond_prefix_embeds,
            prefix_length=prefix_length,
        )
        return -visual_gain
    if remask_selection == "conf_low_visual_gain":
        if precomputed_confidence is None:
            current_confidence = compute_current_answer_probs(
                core_model=core_model,
                x=x,
                prefix_embeds=prefix_embeds,
                prefix_length=prefix_length,
            ).to(torch.float64)
        else:
            current_confidence = precomputed_confidence
        if precomputed_visual_gain is None:
            visual_gain = compute_visual_gain_scores(
                core_model=core_model,
                x=x,
                prefix_embeds=prefix_embeds,
                uncond_prefix_embeds=uncond_prefix_embeds,
                prefix_length=prefix_length,
            )
        else:
            visual_gain = precomputed_visual_gain
        return visual_gain - current_confidence
    if remask_selection == "proposal_then_visual_gain":
        if precomputed_visual_gain is None:
            visual_gain = compute_visual_gain_scores(
                core_model=core_model,
                x=x,
                prefix_embeds=prefix_embeds,
                uncond_prefix_embeds=uncond_prefix_embeds,
                prefix_length=prefix_length,
            )
        else:
            visual_gain = precomputed_visual_gain
        candidate_size = min(int(selector_candidate_size), int(max_new_tokens))
        candidate_size = max(candidate_size, 1)
        candidate_positions = torch.topk(
            proposal_confidence,
            k=candidate_size,
            largest=False,
        ).indices
        scores = torch.full_like(visual_gain, float("inf"), dtype=torch.float64)
        scores[candidate_positions] = visual_gain[candidate_positions]
        return scores
    if remask_selection == "rank_conf_visual_gain":
        if precomputed_visual_gain is None:
            visual_gain = compute_visual_gain_scores(
                core_model=core_model,
                x=x,
                prefix_embeds=prefix_embeds,
                uncond_prefix_embeds=uncond_prefix_embeds,
                prefix_length=prefix_length,
            )
        else:
            visual_gain = precomputed_visual_gain
        confidence_rank = rank_scores_ascending(proposal_confidence)
        visual_rank = rank_scores_ascending(visual_gain)
        return confidence_rank + float(rank_visual_weight) * visual_rank
    if remask_selection == "quadrant_conf_visual_gain":
        if precomputed_confidence is None:
            current_confidence = compute_current_answer_probs(
                core_model=core_model,
                x=x,
                prefix_embeds=prefix_embeds,
                prefix_length=prefix_length,
            ).to(torch.float64)
        else:
            current_confidence = precomputed_confidence
        if precomputed_visual_gain is None:
            visual_gain = compute_visual_gain_scores(
                core_model=core_model,
                x=x,
                prefix_embeds=prefix_embeds,
                uncond_prefix_embeds=uncond_prefix_embeds,
                prefix_length=prefix_length,
            )
        else:
            visual_gain = precomputed_visual_gain

        valid = torch.isfinite(current_confidence) & torch.isfinite(visual_gain)
        high_conf = current_confidence >= float(quadrant_confidence_threshold)
        low_vis = visual_gain <= float(quadrant_visual_threshold)
        bucket = torch.full_like(visual_gain, 4.0, dtype=torch.float64)
        bucket[high_conf & low_vis] = 0.0
        bucket[(~high_conf) & low_vis] = 1.0
        bucket[(~high_conf) & (~low_vis)] = 2.0
        bucket[high_conf & (~low_vis)] = 3.0
        bucket[~valid] = float("inf")

        visual_rank = rank_scores_ascending(visual_gain)
        confidence_rank = rank_scores_ascending(-current_confidence)
        return bucket * float(max_new_tokens + 1) * 2.0 + visual_rank + 0.25 * confidence_rank
    if remask_selection == "norm_conf_visual_gain":
        if precomputed_confidence is None:
            current_confidence = compute_current_answer_probs(
                core_model=core_model,
                x=x,
                prefix_embeds=prefix_embeds,
                prefix_length=prefix_length,
            ).to(torch.float64)
        else:
            current_confidence = precomputed_confidence
        if precomputed_visual_gain is None:
            visual_gain = compute_visual_gain_scores(
                core_model=core_model,
                x=x,
                prefix_embeds=prefix_embeds,
                uncond_prefix_embeds=uncond_prefix_embeds,
                prefix_length=prefix_length,
            )
        else:
            visual_gain = precomputed_visual_gain
        norm_visual = minmax_normalize(visual_gain)
        norm_confidence = minmax_normalize(current_confidence)
        valid = torch.isfinite(norm_visual) & torch.isfinite(norm_confidence)
        score = (
            float(normalized_visual_alpha) * norm_visual
            - float(normalized_confidence_beta) * norm_confidence
        )
        return torch.where(valid, score, torch.full_like(score, float("inf")))
    if remask_selection in {"highgain_lowconf", "highgain_lowconf_negboost"}:
        if precomputed_confidence is None:
            current_confidence = compute_current_answer_probs(
                core_model=core_model,
                x=x,
                prefix_embeds=prefix_embeds,
                prefix_length=prefix_length,
            ).to(torch.float64)
        else:
            current_confidence = precomputed_confidence
        if precomputed_visual_gain is None:
            visual_gain = compute_visual_gain_scores(
                core_model=core_model,
                x=x,
                prefix_embeds=prefix_embeds,
                uncond_prefix_embeds=uncond_prefix_embeds,
                prefix_length=prefix_length,
            )
        else:
            visual_gain = precomputed_visual_gain
        valid = torch.isfinite(current_confidence) & torch.isfinite(visual_gain)
        uncertainty = (1.0 - current_confidence).clamp_min(0.0)
        priority = uncertainty * torch.relu(visual_gain)
        if remask_selection == "highgain_lowconf_negboost":
            priority = priority + float(negative_gain_lambda) * (visual_gain < 0).to(torch.float64)
        score = -priority
        return torch.where(valid, score, torch.full_like(score, float("inf")))
    return cached_confidence


@torch.no_grad()
def generate_with_postmask(
    core_model,
    tokenizer,
    prefix_embeds,
    prefix_input_ids_full,
    selector_weak_prefix_embeds,
    draft_weak_prefix_embeds,
    refill_weak_prefix_embeds,
    max_new_tokens,
    block_length,
    step_per_block,
    step_ratio,
    temperature,
    remasking,
    draft_steps,
    postmask_steps,
    remask_per_step,
    remask_selection,
    postmask_mode,
    fixed_set_size,
    fixed_refill_per_step,
    loo_chunk_size,
    null_visual_mode,
    selector_weak_visual_mode,
    draft_guidance,
    draft_weak_visual_mode,
    vcd_draft_alpha,
    draft_guidance_ratio,
    refill_guidance,
    refill_weak_visual_mode,
    vcd_refill_alpha,
    refill_vrg_calibration,
    refill_vrg_confidence_threshold,
    refill_vrg_confidence_gate_tau,
    refill_guidance_steps,
    prompt_contrast_alpha,
    prompt_contrast_mode,
    vcd_noise_step,
    vcd_noise_seed,
    selector_candidate_size,
    position_boost_lambda,
    position_boost_q2,
    position_boost_q3,
    position_boost_q4,
    rank_visual_weight,
    quadrant_confidence_threshold,
    quadrant_visual_threshold,
    normalized_visual_alpha,
    normalized_confidence_beta,
    negative_gain_lambda,
    span_num_spans=4,
    span_length=8,
):
    total_steps = resolve_total_steps(
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        step_per_block=step_per_block,
        step_ratio=step_ratio,
    )
    if draft_steps is None and postmask_steps is None:
        draft_steps = total_steps // 2
        postmask_steps = total_steps - draft_steps
    elif draft_steps is None:
        postmask_steps = int(postmask_steps)
        draft_steps = total_steps - postmask_steps
    elif postmask_steps is None:
        draft_steps = int(draft_steps)
        postmask_steps = total_steps - draft_steps
    else:
        draft_steps = int(draft_steps)
        postmask_steps = int(postmask_steps)

    if draft_steps <= 0:
        raise ValueError("draft_steps must be > 0.")
    if postmask_steps < 0:
        raise ValueError("postmask_steps must be >= 0.")
    if draft_steps + postmask_steps != total_steps:
        raise ValueError("draft_steps + postmask_steps must equal the total decoding steps.")

    draft_schedule = build_even_schedule(max_new_tokens, draft_steps)
    default_remask = int(math.ceil(max_new_tokens / float(draft_steps)))
    if remask_per_step is None:
        remask_per_step = default_remask
    if remask_per_step <= 0:
        raise ValueError("remask_per_step must be > 0.")
    if loo_chunk_size <= 0:
        raise ValueError("loo_chunk_size must be > 0.")
    if postmask_mode not in {"dynamic", "fixed_set", "span_fixed_set"}:
        raise ValueError("postmask_mode must be 'dynamic', 'fixed_set', or 'span_fixed_set'.")
    if fixed_set_size is not None and fixed_set_size <= 0:
        raise ValueError("fixed_set_size must be > 0.")
    if fixed_refill_per_step is not None and fixed_refill_per_step <= 0:
        raise ValueError("fixed_refill_per_step must be > 0.")
    if span_num_spans <= 0:
        raise ValueError("span_num_spans must be > 0.")
    if span_length <= 0:
        raise ValueError("span_length must be > 0.")
    if draft_guidance_ratio < 0.0 or draft_guidance_ratio > 1.0:
        raise ValueError("draft_guidance_ratio must be in [0, 1].")
    if refill_guidance_steps is not None and refill_guidance_steps <= 0:
        raise ValueError("refill_guidance_steps must be > 0 when set.")
    draft_guidance_steps = 0
    if draft_guidance == "vcd":
        draft_guidance_steps = int(math.ceil(float(draft_steps) * float(draft_guidance_ratio)))

    device = prefix_embeds.device
    batch_size, prefix_length = prefix_embeds.shape[:2]
    if batch_size != 1:
        raise ValueError("generate_with_postmask currently expects batch size 1.")
    uncond_prefix_embeds = None
    visual_selector_modes = {
        "visual_gain",
        "high_visual_gain",
        "conf_low_visual_gain",
        "proposal_then_visual_gain",
        "rank_conf_visual_gain",
        "quadrant_conf_visual_gain",
        "norm_conf_visual_gain",
        "highgain_lowconf",
        "highgain_lowconf_negboost",
    }
    if remask_selection in visual_selector_modes:
        if selector_weak_visual_mode == "null_visual":
            uncond_prefix_embeds, _ = build_unconditional_prefix_embeds(
                core_model=core_model,
                prefix_embeds=prefix_embeds,
                prefix_input_ids_full=prefix_input_ids_full,
                null_visual_mode=null_visual_mode,
            )
        elif selector_weak_visual_mode == "diffusion_noise":
            if selector_weak_prefix_embeds is None:
                raise ValueError("diffusion_noise selector requires selector_weak_prefix_embeds.")
            uncond_prefix_embeds = selector_weak_prefix_embeds
        else:
            raise ValueError(f"Unsupported selector weak visual mode: {selector_weak_visual_mode}")

    def resolve_guidance_prefix(guidance, weak_visual_mode, weak_prefix_embeds, phase_name):
        if guidance == "none":
            return None
        if guidance == "prompt_contrast":
            if weak_prefix_embeds is None:
                raise ValueError(f"Prompt-contrast {phase_name} guidance requires weak_prefix_embeds.")
            return weak_prefix_embeds
        if guidance != "vcd":
            raise ValueError(f"Unsupported {phase_name} guidance: {guidance}")
        if weak_visual_mode == "null_visual":
            prefix, _ = build_unconditional_prefix_embeds(
                core_model=core_model,
                prefix_embeds=prefix_embeds,
                prefix_input_ids_full=prefix_input_ids_full,
                null_visual_mode=null_visual_mode,
            )
            return prefix
        if weak_visual_mode == "diffusion_noise":
            if weak_prefix_embeds is None:
                raise ValueError(f"VCD {phase_name} guidance with diffusion_noise requires weak_prefix_embeds.")
            return weak_prefix_embeds
        raise ValueError(f"Unsupported {phase_name} weak visual mode: {weak_visual_mode}")

    def apply_vcd_guidance(logits, x_state, weak_prefix_embeds, alpha):
        weak_logits = forward_aligned_answer_logits(
            core_model,
            x_state,
            weak_prefix_embeds,
            prefix_length,
        )
        if torch.is_tensor(alpha):
            guided = logits.clone()
            answer_alpha = alpha.to(
                device=logits.device,
                dtype=logits.dtype,
            ).view(1, -1, 1)
            guided[:, answer_slice] = (
                logits[:, answer_slice]
                + answer_alpha * (logits[:, answer_slice] - weak_logits[:, answer_slice])
            )
            return guided
        return (1.0 + float(alpha)) * logits - float(alpha) * weak_logits

    def build_refill_vrg_alpha():
        if refill_vrg_calibration == "none":
            return float(vcd_refill_alpha)
        proposal = proposal_confidence.to(torch.float64)
        valid = torch.isfinite(proposal)
        confidence = torch.where(valid, proposal.clamp(0.0, 1.0), torch.ones_like(proposal))
        if refill_vrg_calibration == "soft_confidence":
            return float(vcd_refill_alpha) * (1.0 - confidence)
        if refill_vrg_calibration == "hard_confidence":
            enabled = confidence < float(refill_vrg_confidence_threshold)
            return torch.where(
                enabled & valid,
                torch.full_like(confidence, float(vcd_refill_alpha)),
                torch.zeros_like(confidence),
            )
        raise ValueError(f"Unsupported refill VRG calibration: {refill_vrg_calibration}")

    draft_uncond_prefix_embeds = resolve_guidance_prefix(
        draft_guidance,
        draft_weak_visual_mode,
        draft_weak_prefix_embeds,
        "draft",
    )
    refill_uncond_prefix_embeds = resolve_guidance_prefix(
        refill_guidance,
        refill_weak_visual_mode,
        refill_weak_prefix_embeds,
        "refill",
    )

    x = torch.full((batch_size, prefix_length + max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :prefix_length] = 0
    answer_slice = slice(prefix_length, prefix_length + max_new_tokens)
    cached_confidence = torch.full((max_new_tokens,), float("inf"), dtype=torch.float64, device=device)
    proposal_confidence = torch.full((max_new_tokens,), float("inf"), dtype=torch.float64, device=device)
    history_conf_sum = torch.zeros((max_new_tokens,), dtype=torch.float64, device=device)
    history_conf_count = torch.zeros((max_new_tokens,), dtype=torch.long, device=device)
    after_fill_conf_sum = torch.zeros((max_new_tokens,), dtype=torch.float64, device=device)
    after_fill_conf_count = torch.zeros((max_new_tokens,), dtype=torch.long, device=device)
    margin_sum = torch.zeros((max_new_tokens,), dtype=torch.float64, device=device)
    margin_count = torch.zeros((max_new_tokens,), dtype=torch.long, device=device)
    kl_sum = torch.zeros((max_new_tokens,), dtype=torch.float64, device=device)
    kl_count = torch.zeros((max_new_tokens,), dtype=torch.long, device=device)
    previous_after_fill_log_probs = {}

    draft_records = []
    for step_idx, num_to_fill in enumerate(draft_schedule, start=1):
        logits = forward_logits(core_model, x, prefix_embeds, prefix_length)
        draft_guidance_used = draft_guidance == "vcd" and step_idx <= draft_guidance_steps
        if draft_guidance_used:
            logits = apply_vcd_guidance(
                logits,
                x,
                draft_uncond_prefix_embeds,
                vcd_draft_alpha,
            )
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        confidence = compute_remasking_confidence(logits, x0, remasking)
        token_logprobs = gather_token_logprobs(logits, x0)
        token_margin = gather_topk_margin(logits)
        token_full_log_probs = F.log_softmax(logits[0, answer_slice].to(torch.float64), dim=-1)
        proposal_confidence = confidence[0, answer_slice].detach().to(torch.float64)

        mask_index = x == MASK_TOKEN_ID
        answer_mask = mask_index[:, answer_slice][0]
        if answer_mask.any():
            history_conf_sum[answer_mask] += token_logprobs[0, answer_slice][answer_mask].to(torch.float64)
            history_conf_count[answer_mask] += 1
        filled_answer = ~answer_mask
        if filled_answer.any():
            filled_token_ids = x[0, answer_slice][filled_answer]
            filled_logits = logits[0, answer_slice][filled_answer].to(torch.float64)
            filled_log_probs = F.log_softmax(filled_logits, dim=-1)
            filled_log_conf = filled_log_probs.gather(dim=-1, index=filled_token_ids.unsqueeze(-1)).squeeze(-1)
            after_fill_conf_sum[filled_answer] += filled_log_conf
            after_fill_conf_count[filled_answer] += 1
            filled_margin = token_margin[0, answer_slice][filled_answer].to(torch.float64)
            margin_sum[filled_answer] += filled_margin
            margin_count[filled_answer] += 1
            for answer_pos in torch.nonzero(filled_answer, as_tuple=False).view(-1).detach().cpu().tolist():
                current_pos_log_probs = token_full_log_probs[answer_pos]
                previous_pos_log_probs = previous_after_fill_log_probs.get(int(answer_pos))
                if previous_pos_log_probs is not None:
                    kl_value = compute_tokenwise_kl_from_logprobs(
                        current_pos_log_probs.unsqueeze(0),
                        previous_pos_log_probs.unsqueeze(0),
                    )[0]
                    kl_sum[answer_pos] += kl_value
                    kl_count[answer_pos] += 1
                previous_after_fill_log_probs[int(answer_pos)] = current_pos_log_probs.detach().to(device="cpu", dtype=torch.float32)
        x0 = torch.where(mask_index, x0, x)
        masked_confidence = torch.where(mask_index, confidence, -torch.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
        _, select_index = torch.topk(masked_confidence[0], k=int(num_to_fill))
        transfer_index[0, select_index] = True
        x[transfer_index] = x0[transfer_index]

        for seq_pos in select_index.detach().cpu().tolist():
            if seq_pos < prefix_length:
                continue
            answer_pos = int(seq_pos - prefix_length)
            cached_confidence[answer_pos] = confidence[0, seq_pos].to(torch.float64)
            after_fill_conf_sum[answer_pos] += token_logprobs[0, seq_pos].to(torch.float64)
            after_fill_conf_count[answer_pos] += 1
            margin_sum[answer_pos] += token_margin[0, seq_pos].to(torch.float64)
            margin_count[answer_pos] += 1
            current_pos_log_probs = token_full_log_probs[answer_pos]
            previous_pos_log_probs = previous_after_fill_log_probs.get(int(answer_pos))
            if previous_pos_log_probs is not None:
                kl_value = compute_tokenwise_kl_from_logprobs(
                    current_pos_log_probs.unsqueeze(0),
                    previous_pos_log_probs.unsqueeze(0),
                )[0]
                kl_sum[answer_pos] += kl_value
                kl_count[answer_pos] += 1
            previous_after_fill_log_probs[int(answer_pos)] = current_pos_log_probs.detach().to(device="cpu", dtype=torch.float32)

        state_ids = x[0, answer_slice].detach().cpu().tolist()
        draft_records.append(
            {
                "step": int(step_idx),
                "phase": "draft",
                "guidance_used": bool(draft_guidance_used),
                "num_filled": int(num_to_fill),
                "selected_positions": [int(pos) for pos in select_index.detach().cpu().tolist()],
                "state_text": decode_answer(tokenizer, state_ids),
                "num_masked_after_step": int((x[:, answer_slice] == MASK_TOKEN_ID).sum().item()),
            }
        )

    draft_answer_ids = x[0, answer_slice].detach().cpu().tolist()
    draft_text = decode_answer(tokenizer, draft_answer_ids)
    history_confidence, seen_history = build_mean_scores(history_conf_sum, history_conf_count)
    mean_after_fill, seen_after_fill = build_mean_scores(after_fill_conf_sum, after_fill_conf_count)
    topk_margin, seen_margin = build_mean_scores(margin_sum, margin_count)
    mean_kl_scores, seen_kl = build_mean_scores(kl_sum, kl_count)
    kl_divergence = torch.full_like(mean_kl_scores, float("inf"), dtype=torch.float64)
    if seen_kl.any():
        kl_divergence[seen_kl] = -mean_kl_scores[seen_kl]

    fixed_remask_positions = None
    fixed_remask_scores = None
    fixed_remask_visual_gains = None
    fixed_remask_token_ids = None
    fixed_remask_token_texts = None
    fixed_remask_rule_categories = None
    fixed_remask_spans = None
    draft_visual_gain_scores = None
    if postmask_steps > 0 and postmask_mode in {"fixed_set", "span_fixed_set"}:
        effective_fixed_set_size = min(
            int(fixed_set_size) if fixed_set_size is not None else int(remask_per_step),
            int(max_new_tokens),
        )
        if uncond_prefix_embeds is not None:
            draft_visual_gain_scores = compute_visual_gain_scores(
                core_model=core_model,
                x=x,
                prefix_embeds=prefix_embeds,
                uncond_prefix_embeds=uncond_prefix_embeds,
                prefix_length=prefix_length,
            )
        draft_confidence_scores = None
        if remask_selection in {
            "conf_low_visual_gain",
            "quadrant_conf_visual_gain",
            "norm_conf_visual_gain",
            "highgain_lowconf",
            "highgain_lowconf_negboost",
        }:
            draft_confidence_scores = compute_current_answer_probs(
                core_model=core_model,
                x=x,
                prefix_embeds=prefix_embeds,
                prefix_length=prefix_length,
            ).to(torch.float64)
        fixed_selection_scores = compute_selection_scores(
            remask_selection=remask_selection,
            tokenizer=tokenizer,
            x=x,
            prefix_embeds=prefix_embeds,
            uncond_prefix_embeds=uncond_prefix_embeds,
            prefix_length=prefix_length,
            max_new_tokens=max_new_tokens,
            device=device,
            core_model=core_model,
            loo_chunk_size=loo_chunk_size,
            cached_confidence=cached_confidence,
            proposal_confidence=proposal_confidence,
            mean_after_fill=mean_after_fill,
            topk_margin=topk_margin,
            kl_divergence=kl_divergence,
            history_confidence=history_confidence,
            precomputed_visual_gain=draft_visual_gain_scores,
            precomputed_confidence=draft_confidence_scores,
            selector_candidate_size=selector_candidate_size,
            position_boost_lambda=position_boost_lambda,
            position_boost_q2=position_boost_q2,
            position_boost_q3=position_boost_q3,
            position_boost_q4=position_boost_q4,
            rank_visual_weight=rank_visual_weight,
            quadrant_confidence_threshold=quadrant_confidence_threshold,
            quadrant_visual_threshold=quadrant_visual_threshold,
            normalized_visual_alpha=normalized_visual_alpha,
            normalized_confidence_beta=normalized_confidence_beta,
            negative_gain_lambda=negative_gain_lambda,
        )
        if postmask_mode == "span_fixed_set":
            fixed_remask_spans, fixed_remask_positions = choose_low_score_spans(
                scores=fixed_selection_scores,
                num_spans=span_num_spans,
                span_length=span_length,
            )
        else:
            fixed_remask_positions = choose_positions(
                scores=fixed_selection_scores,
                k=effective_fixed_set_size,
                mode=remask_selection,
            )
        fixed_remask_scores = fixed_selection_scores[fixed_remask_positions].detach().clone()
        fixed_remask_visual_gains = (
            draft_visual_gain_scores[fixed_remask_positions].detach().clone()
            if draft_visual_gain_scores is not None
            else None
        )
        fixed_remask_token_ids = x[0, fixed_remask_positions + prefix_length].detach().cpu().tolist()
        fixed_remask_token_texts, fixed_remask_rule_categories = classify_answer_tokens(
            tokenizer,
            fixed_remask_token_ids,
        )
        if fixed_remask_positions.numel() > 0:
            x[0, fixed_remask_positions + prefix_length] = MASK_TOKEN_ID
            cached_confidence[fixed_remask_positions] = float("inf")

    postmask_records = []
    for local_step in range(1, postmask_steps + 1):
        remask_count = min(int(remask_per_step), int(max_new_tokens))
        if remask_count <= 0:
            break

        if postmask_mode in {"fixed_set", "span_fixed_set"}:
            selected_answer_positions = fixed_remask_positions
            selection_scores = fixed_remask_scores
            selected_token_texts = fixed_remask_token_texts
            selected_rule_categories = fixed_remask_rule_categories
            selected_visual_gains = (
                [float(value.item()) for value in fixed_remask_visual_gains.detach().cpu()]
                if fixed_remask_visual_gains is not None
                else None
            )
        else:
            current_visual_gain_scores = None
            current_confidence_scores = None
            if uncond_prefix_embeds is not None:
                current_visual_gain_scores = compute_visual_gain_scores(
                    core_model=core_model,
                    x=x,
                    prefix_embeds=prefix_embeds,
                    uncond_prefix_embeds=uncond_prefix_embeds,
                    prefix_length=prefix_length,
                )
                if remask_selection in {
                    "conf_low_visual_gain",
                    "quadrant_conf_visual_gain",
                    "norm_conf_visual_gain",
                    "highgain_lowconf",
                    "highgain_lowconf_negboost",
                }:
                    current_confidence_scores = compute_current_answer_probs(
                        core_model=core_model,
                        x=x,
                        prefix_embeds=prefix_embeds,
                        prefix_length=prefix_length,
                    ).to(torch.float64)
            selection_scores = compute_selection_scores(
                remask_selection=remask_selection,
                tokenizer=tokenizer,
                x=x,
                prefix_embeds=prefix_embeds,
                uncond_prefix_embeds=uncond_prefix_embeds,
                prefix_length=prefix_length,
                max_new_tokens=max_new_tokens,
                device=device,
                core_model=core_model,
                loo_chunk_size=loo_chunk_size,
                cached_confidence=cached_confidence,
                proposal_confidence=proposal_confidence,
                mean_after_fill=mean_after_fill,
                topk_margin=topk_margin,
                kl_divergence=kl_divergence,
                history_confidence=history_confidence,
                precomputed_visual_gain=current_visual_gain_scores,
                precomputed_confidence=current_confidence_scores,
                selector_candidate_size=selector_candidate_size,
                position_boost_lambda=position_boost_lambda,
                position_boost_q2=position_boost_q2,
                position_boost_q3=position_boost_q3,
                position_boost_q4=position_boost_q4,
                rank_visual_weight=rank_visual_weight,
                quadrant_confidence_threshold=quadrant_confidence_threshold,
                quadrant_visual_threshold=quadrant_visual_threshold,
                normalized_visual_alpha=normalized_visual_alpha,
                normalized_confidence_beta=normalized_confidence_beta,
                negative_gain_lambda=negative_gain_lambda,
            )
            selected_answer_positions = choose_positions(
                scores=selection_scores,
                k=remask_count,
                mode=remask_selection,
            )
            selected_visual_gains = tensor_values_at(current_visual_gain_scores, selected_answer_positions)
            seq_positions = selected_answer_positions + prefix_length
            remasked_before_ids = x[0, seq_positions].detach().cpu().tolist()
            selected_token_texts, selected_rule_categories = classify_answer_tokens(
                tokenizer,
                remasked_before_ids,
            )
            x[0, seq_positions] = MASK_TOKEN_ID
            cached_confidence[selected_answer_positions] = float("inf")

        if postmask_mode in {"fixed_set", "span_fixed_set"}:
            seq_positions = selected_answer_positions + prefix_length
            remasked_before_ids = None if local_step > 1 else fixed_remask_token_ids

        cond_logits = forward_logits(core_model, x, prefix_embeds, prefix_length)
        logits = cond_logits
        guidance_active = (
            refill_guidance in {"vcd", "prompt_contrast"}
            and (
                refill_guidance_steps is None
                or local_step <= int(refill_guidance_steps)
            )
        )
        if guidance_active:
            guidance_alpha = (
                prompt_contrast_alpha
                if refill_guidance == "prompt_contrast"
                else build_refill_vrg_alpha()
            )
            guided_logits = apply_vcd_guidance(
                cond_logits,
                x,
                refill_uncond_prefix_embeds,
                guidance_alpha,
            )
            if refill_guidance == "vcd" and refill_vrg_confidence_gate_tau is not None:
                logits = apply_confidence_gated_refill_logits(
                    cond_logits=cond_logits,
                    guided_logits=guided_logits,
                    answer_slice=answer_slice,
                    tau=refill_vrg_confidence_gate_tau,
                )
            else:
                logits = guided_logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        confidence = compute_remasking_confidence(logits, x0, remasking)
        token_logprobs = gather_token_logprobs(logits, x0)
        token_margin = gather_topk_margin(logits)
        token_full_log_probs = F.log_softmax(logits[0, answer_slice].to(torch.float64), dim=-1)

        mask_index = x == MASK_TOKEN_ID
        x0 = torch.where(mask_index, x0, x)
        masked_confidence = torch.where(mask_index, confidence, -torch.inf)
        masked_remaining = int(mask_index[:, answer_slice].sum().item())
        if postmask_mode in {"fixed_set", "span_fixed_set"} and fixed_refill_per_step is not None:
            refill_count = min(int(fixed_refill_per_step), masked_remaining)
        else:
            refill_count = masked_remaining
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
        _, refill_index = torch.topk(masked_confidence[0], k=refill_count)
        transfer_index[0, refill_index] = True
        x[transfer_index] = x0[transfer_index]

        refilled_answer_positions = []
        for seq_pos in refill_index.detach().cpu().tolist():
            if seq_pos < prefix_length:
                continue
            answer_pos = int(seq_pos - prefix_length)
            cached_confidence[answer_pos] = confidence[0, seq_pos].to(torch.float64)
            regenerated_logprob = token_logprobs[0, seq_pos].to(torch.float64)
            history_conf_sum[answer_pos] += regenerated_logprob
            history_conf_count[answer_pos] += 1
            after_fill_conf_sum[answer_pos] += regenerated_logprob
            after_fill_conf_count[answer_pos] += 1
            margin_sum[answer_pos] += token_margin[0, seq_pos].to(torch.float64)
            margin_count[answer_pos] += 1
            current_pos_log_probs = token_full_log_probs[answer_pos]
            previous_pos_log_probs = previous_after_fill_log_probs.get(int(answer_pos))
            if previous_pos_log_probs is not None:
                kl_value = compute_tokenwise_kl_from_logprobs(
                    current_pos_log_probs.unsqueeze(0),
                    previous_pos_log_probs.unsqueeze(0),
                )[0]
                kl_sum[answer_pos] += kl_value
                kl_count[answer_pos] += 1
            previous_after_fill_log_probs[int(answer_pos)] = current_pos_log_probs.detach().to(device="cpu", dtype=torch.float32)
            refilled_answer_positions.append(answer_pos)

        current_logprobs = compute_current_answer_logprobs(
            core_model=core_model,
            x=x,
            prefix_embeds=prefix_embeds,
            prefix_length=prefix_length,
        )
        for answer_pos in refilled_answer_positions:
            cached_confidence[answer_pos] = current_logprobs[0, answer_pos].exp().to(torch.float64)
        history_confidence, seen_history = build_mean_scores(history_conf_sum, history_conf_count)
        mean_after_fill, seen_after_fill = build_mean_scores(after_fill_conf_sum, after_fill_conf_count)
        topk_margin, seen_margin = build_mean_scores(margin_sum, margin_count)
        mean_kl_scores, seen_kl = build_mean_scores(kl_sum, kl_count)
        kl_divergence = torch.full_like(mean_kl_scores, float("inf"), dtype=torch.float64)
        if seen_kl.any():
            kl_divergence[seen_kl] = -mean_kl_scores[seen_kl]

        state_ids = x[0, answer_slice].detach().cpu().tolist()
        postmask_records.append(
            {
                "step": int(draft_steps + local_step),
                "phase": "postmask",
                "remasked_answer_positions": [int(pos) for pos in selected_answer_positions.detach().cpu().tolist()],
                "remasked_token_ids": [int(token_id) for token_id in remasked_before_ids] if remasked_before_ids is not None else None,
                "remasked_token_texts": selected_token_texts,
                "remasked_rule_categories": selected_rule_categories,
                "refilled_answer_positions": [int(pos) for pos in refilled_answer_positions],
                "selection_scores": (
                    (
                        [float(score.item()) for score in selection_scores.detach().cpu()]
                        if postmask_mode in {"fixed_set", "span_fixed_set"}
                        else [float(selection_scores[int(pos)].item()) for pos in selected_answer_positions.detach().cpu().tolist()]
                    )
                    if remask_selection != "random"
                    else None
                ),
                "selected_visual_gains": selected_visual_gains,
                "remasked_spans": fixed_remask_spans if postmask_mode == "span_fixed_set" else None,
                "state_text": decode_answer(tokenizer, state_ids),
            }
        )

    final_answer_ids = x[0, answer_slice].detach().cpu().tolist()
    final_text = decode_answer(tokenizer, final_answer_ids)
    meta = {
        "max_new_tokens": int(max_new_tokens),
        "block_length": int(block_length),
        "total_steps": int(total_steps),
        "draft_steps": int(draft_steps),
        "postmask_steps": int(postmask_steps),
        "draft_schedule": [int(item) for item in draft_schedule],
        "remask_per_step": int(remask_per_step),
        "remask_selection": remask_selection,
        "postmask_mode": postmask_mode,
        "fixed_set_size": int(fixed_set_size) if fixed_set_size is not None else None,
        "fixed_refill_per_step": int(fixed_refill_per_step) if fixed_refill_per_step is not None else None,
        "span_num_spans": int(span_num_spans) if postmask_mode == "span_fixed_set" else None,
        "span_length": int(span_length) if postmask_mode == "span_fixed_set" else None,
        "span_total_tokens": (
            int(len(fixed_remask_positions)) if postmask_mode == "span_fixed_set" and fixed_remask_positions is not None else None
        ),
        "loo_chunk_size": int(loo_chunk_size),
        "selector_weak_visual_mode": selector_weak_visual_mode,
        "null_visual_mode": null_visual_mode,
        "draft_guidance": draft_guidance,
        "draft_weak_visual_mode": draft_weak_visual_mode if draft_guidance == "vcd" else None,
        "vcd_draft_alpha": float(vcd_draft_alpha) if draft_guidance == "vcd" else None,
        "draft_guidance_ratio": float(draft_guidance_ratio) if draft_guidance == "vcd" else None,
        "draft_guidance_steps": int(draft_guidance_steps) if draft_guidance == "vcd" else 0,
        "refill_guidance": refill_guidance,
        "refill_weak_visual_mode": refill_weak_visual_mode if refill_guidance == "vcd" else None,
        "vcd_refill_alpha": float(vcd_refill_alpha) if refill_guidance == "vcd" else None,
        "refill_vrg_calibration": refill_vrg_calibration if refill_guidance == "vcd" else None,
        "refill_vrg_confidence_gate_tau": (
            float(refill_vrg_confidence_gate_tau)
            if refill_guidance == "vcd" and refill_vrg_confidence_gate_tau is not None
            else None
        ),
        "refill_vrg_confidence_threshold": (
            float(refill_vrg_confidence_threshold)
            if refill_guidance == "vcd" and refill_vrg_calibration == "hard_confidence"
            else None
        ),
        "refill_guidance_steps": int(refill_guidance_steps) if refill_guidance_steps is not None else None,
        "prompt_contrast_alpha": float(prompt_contrast_alpha) if refill_guidance == "prompt_contrast" else None,
        "prompt_contrast_mode": prompt_contrast_mode if refill_guidance == "prompt_contrast" else None,
        "vcd_noise_step": (
            int(vcd_noise_step)
            if (
                selector_weak_visual_mode == "diffusion_noise"
                or (
                    draft_guidance == "vcd"
                    and draft_weak_visual_mode == "diffusion_noise"
                )
                or (
                    refill_guidance == "vcd"
                    and refill_weak_visual_mode == "diffusion_noise"
                )
            )
            else None
        ),
        "vcd_noise_seed": (
            int(vcd_noise_seed)
            if (
                vcd_noise_seed is not None
                and (
                    selector_weak_visual_mode == "diffusion_noise"
                    or (
                        draft_guidance == "vcd"
                        and draft_weak_visual_mode == "diffusion_noise"
                    )
                    or (
                        refill_guidance == "vcd"
                        and refill_weak_visual_mode == "diffusion_noise"
                    )
                )
            )
            else None
        ),
        "selector_candidate_size": int(selector_candidate_size),
        "position_boost_lambda": float(position_boost_lambda),
        "position_boost_q2": float(position_boost_q2),
        "position_boost_q3": float(position_boost_q3),
        "position_boost_q4": float(position_boost_q4),
        "rank_visual_weight": float(rank_visual_weight),
        "quadrant_confidence_threshold": float(quadrant_confidence_threshold),
        "quadrant_visual_threshold": float(quadrant_visual_threshold),
        "normalized_visual_alpha": float(normalized_visual_alpha),
        "normalized_confidence_beta": float(normalized_confidence_beta),
        "negative_gain_lambda": float(negative_gain_lambda),
        "visual_gain_source": (
            "draft_final_state"
            if postmask_mode == "fixed_set" and remask_selection in visual_selector_modes
            else None
        ),
        "history_confidence_mean": float(history_confidence[seen_history].mean().item()) if seen_history.any() else None,
        "mean_after_fill_mean": float(mean_after_fill[seen_after_fill].mean().item()) if seen_after_fill.any() else None,
        "proposal_confidence_mean": float(proposal_confidence.mean().item()),
        "topk_margin_mean": float(topk_margin[seen_margin].mean().item()) if seen_margin.any() else None,
        "kl_divergence_mean": float((-kl_divergence[seen_kl]).mean().item()) if seen_kl.any() else None,
    }
    return {
        "draft_text": draft_text,
        "draft_answer_ids": draft_answer_ids,
        "draft_records": draft_records,
        "postmask_records": postmask_records,
        "final_text": final_text,
        "final_answer_ids": final_answer_ids,
        "history_confidence": history_confidence.detach().cpu().tolist(),
        "history_confidence_count": history_conf_count.detach().cpu().tolist(),
        "proposal_confidence": proposal_confidence.detach().cpu().tolist(),
        "mean_after_fill": mean_after_fill.detach().cpu().tolist(),
        "mean_after_fill_count": after_fill_conf_count.detach().cpu().tolist(),
        "topk_margin": topk_margin.detach().cpu().tolist(),
        "topk_margin_count": margin_count.detach().cpu().tolist(),
        "kl_divergence": kl_divergence.detach().cpu().tolist(),
        "kl_divergence_count": kl_count.detach().cpu().tolist(),
        "draft_visual_gain": tensor_to_float_list(draft_visual_gain_scores),
        "meta": meta,
    }


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
    if args.domain_filter:
        dataset = dataset.filter(lambda row: row.get("domain") == args.domain_filter)
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
    draft_correct_total = 0
    final_correct_total = 0
    improved_total = 0
    worsened_total = 0

    with records_path.open("w", encoding="utf-8") as fout:
        for dataset_index, doc in enumerate(dataset):
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
            selector_weak_prefix_embeds = None
            draft_weak_prefix_embeds = None
            refill_weak_prefix_embeds = None
            if (
                args.remask_selection
                in {
                    "visual_gain",
                    "high_visual_gain",
                    "conf_low_visual_gain",
                    "proposal_then_visual_gain",
                    "rank_conf_visual_gain",
                    "quadrant_conf_visual_gain",
                    "norm_conf_visual_gain",
                    "highgain_lowconf",
                    "highgain_lowconf_negboost",
                }
                and args.selector_weak_visual_mode == "diffusion_noise"
            ):
                selector_weak_prefix_embeds = build_diffusion_noise_prefix(
                    args,
                    model,
                    tokenizer,
                    image_processor,
                    doc,
                )
            if (
                args.draft_guidance == "vcd"
                and args.draft_weak_visual_mode == "diffusion_noise"
            ):
                if (
                    selector_weak_prefix_embeds is not None
                    and args.selector_weak_visual_mode == args.draft_weak_visual_mode
                ):
                    draft_weak_prefix_embeds = selector_weak_prefix_embeds
                else:
                    draft_weak_prefix_embeds = build_diffusion_noise_prefix(
                        args,
                        model,
                        tokenizer,
                        image_processor,
                        doc,
                    )
            if (
                args.refill_guidance == "vcd"
                and args.refill_weak_visual_mode == "diffusion_noise"
            ):
                if (
                    selector_weak_prefix_embeds is not None
                    and args.selector_weak_visual_mode == args.refill_weak_visual_mode
                ):
                    refill_weak_prefix_embeds = selector_weak_prefix_embeds
                elif (
                    draft_weak_prefix_embeds is not None
                    and args.draft_weak_visual_mode == args.refill_weak_visual_mode
                ):
                    refill_weak_prefix_embeds = draft_weak_prefix_embeds
                else:
                    refill_weak_prefix_embeds = build_diffusion_noise_prefix(
                        args,
                        model,
                        tokenizer,
                        image_processor,
                        doc,
                    )
            if args.refill_guidance == "prompt_contrast":
                refill_weak_prefix_embeds = build_prompt_contrast_prefix(
                    args,
                    model,
                    tokenizer,
                    image_processor,
                    doc,
                )
            run_output = generate_with_postmask(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                prefix_input_ids_full=prefix_input_ids_full,
                selector_weak_prefix_embeds=selector_weak_prefix_embeds,
                draft_weak_prefix_embeds=draft_weak_prefix_embeds,
                refill_weak_prefix_embeds=refill_weak_prefix_embeds,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                step_per_block=args.step_per_block,
                step_ratio=args.step_ratio,
                temperature=args.temperature,
                remasking=args.remasking,
                draft_steps=args.draft_steps,
                postmask_steps=args.postmask_steps,
                remask_per_step=args.remask_per_step,
                remask_selection=args.remask_selection,
                postmask_mode=args.postmask_mode,
                fixed_set_size=args.fixed_set_size,
                fixed_refill_per_step=args.fixed_refill_per_step,
                loo_chunk_size=args.loo_chunk_size,
                null_visual_mode=args.null_visual_mode,
                selector_weak_visual_mode=args.selector_weak_visual_mode,
                draft_guidance=args.draft_guidance,
                draft_weak_visual_mode=args.draft_weak_visual_mode,
                vcd_draft_alpha=args.vcd_draft_alpha,
                draft_guidance_ratio=args.draft_guidance_ratio,
                refill_guidance=args.refill_guidance,
                refill_weak_visual_mode=args.refill_weak_visual_mode,
                vcd_refill_alpha=args.vcd_refill_alpha,
                refill_vrg_calibration=args.refill_vrg_calibration,
                refill_vrg_confidence_threshold=args.refill_vrg_confidence_threshold,
                refill_vrg_confidence_gate_tau=args.refill_vrg_confidence_gate_tau,
                refill_guidance_steps=args.refill_guidance_steps,
                prompt_contrast_alpha=args.prompt_contrast_alpha,
                prompt_contrast_mode=args.prompt_contrast_mode,
                vcd_noise_step=args.vcd_noise_step,
                vcd_noise_seed=args.vcd_noise_seed,
                selector_candidate_size=args.selector_candidate_size,
                position_boost_lambda=args.position_boost_lambda,
                position_boost_q2=args.position_boost_q2,
                position_boost_q3=args.position_boost_q3,
                position_boost_q4=args.position_boost_q4,
                rank_visual_weight=args.rank_visual_weight,
                quadrant_confidence_threshold=args.quadrant_confidence_threshold,
                quadrant_visual_threshold=args.quadrant_visual_threshold,
                normalized_visual_alpha=args.normalized_visual_alpha,
                normalized_confidence_beta=args.normalized_confidence_beta,
                negative_gain_lambda=args.negative_gain_lambda,
                span_num_spans=args.span_num_spans,
                span_length=args.span_length,
            )
            elapsed = time.time() - t0
            total_elapsed += elapsed

            draft_correct = bool(judge_answer(run_output["draft_text"], doc["choices"], doc["answer"]))
            final_correct = bool(judge_answer(run_output["final_text"], doc["choices"], doc["answer"]))
            if final_correct and not draft_correct:
                improved_total += 1
            if draft_correct and not final_correct:
                worsened_total += 1
            draft_correct_total += int(draft_correct)
            final_correct_total += int(final_correct)

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
                "draft_text": run_output["draft_text"],
                "draft_answer_ids": run_output["draft_answer_ids"],
                "draft_correct": draft_correct,
                "final_text": run_output["final_text"],
                "final_answer_ids": run_output["final_answer_ids"],
                "final_correct": final_correct,
                "draft_records": run_output["draft_records"],
                "postmask_records": run_output["postmask_records"],
                "history_confidence": run_output["history_confidence"],
                "history_confidence_count": run_output["history_confidence_count"],
                "proposal_confidence": run_output["proposal_confidence"],
                "mean_after_fill": run_output["mean_after_fill"],
                "mean_after_fill_count": run_output["mean_after_fill_count"],
                "topk_margin": run_output["topk_margin"],
                "topk_margin_count": run_output["topk_margin_count"],
                "kl_divergence": run_output["kl_divergence"],
                "kl_divergence_count": run_output["kl_divergence_count"],
                "draft_visual_gain": run_output["draft_visual_gain"],
                "meta": run_output["meta"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                print(
                    f"[{written}] dataset_index={dataset_index} id={doc['id']} "
                    f"draft={draft_correct} final={final_correct} elapsed={elapsed:.2f}s",
                    flush=True,
                )

    summary = {
        "dataset_path": args.dataset_path,
        "split": args.split,
        "start_index": args.start_index,
        "domain_filter": args.domain_filter,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed if args.sample_mode == "random" else None,
        "num_samples": written,
        "prompt": args.prompt,
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": total_elapsed / written if written else None,
        "draft_accuracy": draft_correct_total / written if written else None,
        "final_accuracy": final_correct_total / written if written else None,
        "improved_after_postmask": int(improved_total),
        "worsened_after_postmask": int(worsened_total),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "step_per_block": args.step_per_block,
            "step_ratio": args.step_ratio,
            "temperature": args.temperature,
            "remasking": args.remasking,
            "draft_steps": args.draft_steps,
            "postmask_steps": args.postmask_steps,
            "remask_per_step": args.remask_per_step,
            "remask_selection": args.remask_selection,
            "position_boost_lambda": args.position_boost_lambda,
            "position_boost_q2": args.position_boost_q2,
            "position_boost_q3": args.position_boost_q3,
            "position_boost_q4": args.position_boost_q4,
            "postmask_mode": args.postmask_mode,
            "fixed_set_size": args.fixed_set_size,
            "fixed_refill_per_step": args.fixed_refill_per_step,
            "span_num_spans": args.span_num_spans if args.postmask_mode == "span_fixed_set" else None,
            "span_length": args.span_length if args.postmask_mode == "span_fixed_set" else None,
            "loo_chunk_size": args.loo_chunk_size,
            "selector_weak_visual_mode": args.selector_weak_visual_mode,
            "null_visual_mode": args.null_visual_mode,
            "draft_guidance": args.draft_guidance,
            "draft_weak_visual_mode": args.draft_weak_visual_mode if args.draft_guidance == "vcd" else None,
            "vcd_draft_alpha": args.vcd_draft_alpha if args.draft_guidance == "vcd" else None,
            "draft_guidance_ratio": args.draft_guidance_ratio if args.draft_guidance == "vcd" else None,
            "refill_guidance": args.refill_guidance,
            "refill_weak_visual_mode": args.refill_weak_visual_mode if args.refill_guidance == "vcd" else None,
            "vcd_refill_alpha": args.vcd_refill_alpha if args.refill_guidance == "vcd" else None,
            "refill_vrg_calibration": args.refill_vrg_calibration if args.refill_guidance == "vcd" else None,
            "refill_vrg_confidence_gate_tau": (
                args.refill_vrg_confidence_gate_tau
                if args.refill_guidance == "vcd" and args.refill_vrg_confidence_gate_tau is not None
                else None
            ),
            "refill_vrg_confidence_threshold": (
                args.refill_vrg_confidence_threshold
                if args.refill_guidance == "vcd" and args.refill_vrg_calibration == "hard_confidence"
                else None
            ),
            "refill_guidance_steps": args.refill_guidance_steps,
            "prompt_contrast_alpha": args.prompt_contrast_alpha if args.refill_guidance == "prompt_contrast" else None,
            "prompt_contrast_mode": args.prompt_contrast_mode if args.refill_guidance == "prompt_contrast" else None,
            "vcd_noise_step": (
                args.vcd_noise_step
                if (
                    args.selector_weak_visual_mode == "diffusion_noise"
                    or (
                        args.draft_guidance == "vcd"
                        and args.draft_weak_visual_mode == "diffusion_noise"
                    )
                    or (
                        args.refill_guidance == "vcd"
                        and args.refill_weak_visual_mode == "diffusion_noise"
                    )
                )
                else None
            ),
            "vcd_noise_seed": (
                args.vcd_noise_seed
                if (
                    args.vcd_noise_seed is not None
                    and (
                        args.selector_weak_visual_mode == "diffusion_noise"
                        or (
                            args.draft_guidance == "vcd"
                            and args.draft_weak_visual_mode == "diffusion_noise"
                        )
                        or (
                            args.refill_guidance == "vcd"
                            and args.refill_weak_visual_mode == "diffusion_noise"
                        )
                    )
                )
                else None
            ),
            "selector_candidate_size": args.selector_candidate_size,
            "rank_visual_weight": args.rank_visual_weight,
            "quadrant_confidence_threshold": args.quadrant_confidence_threshold,
            "quadrant_visual_threshold": args.quadrant_visual_threshold,
            "normalized_visual_alpha": args.normalized_visual_alpha,
            "normalized_confidence_beta": args.normalized_confidence_beta,
            "negative_gain_lambda": args.negative_gain_lambda,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote records to {records_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)
    restore_compile()


if __name__ == "__main__":
    main()
