import argparse
import contextlib
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import datasets
import torch


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
from M3CoT.PostVRG.dataset_adapters import add_dataset_adapter_args, load_postvrg_dataset
from M3CoT.utils.metric import judge_answer
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.language_model.llada.generate import add_gumbel_noise



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


def build_prefix_from_multimodal_inputs(
    model,
    input_ids: torch.Tensor,
    images,
    image_sizes,
    attention_mask: torch.Tensor = None,
):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
    else:
        attention_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)

    ret = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=None,
        images=images,
        modalities=["image"],
        image_sizes=image_sizes,
        return_inputs=True,
    )
    return ret[4], ret[-1][0]


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]

def cli_value(flag, default):
    if flag not in sys.argv:
        return default
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


def add_default(flag, value):
    if value is None:
        return
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])


def safe_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def apply_postvrg_defaults():
    noise_step = cli_value("--vcd-noise-step", "200")
    sample_seed = cli_value("--sample-seed", "42")
    limit = cli_value("--limit", "400")
    draft_mode = cli_value("--draft-visual-mode", "full")
    refine_mode = cli_value("--refine-visual-mode", "crop")

    draft_tag = draft_mode if draft_mode == "full" else f"{draft_mode}{noise_step}"
    default_output = (
        "M3CoT/PostVRG/outputs/"
        f"postvrg_draft-{draft_tag}_refine-{refine_mode}_seed{sample_seed}_n{limit}"
    )

    defaults = {
        "--prompt": "cot",
        "--max-new-tokens": 64,
        "--block-length": 64,
        "--step-ratio": 0.5,
        "--limit": limit,
        "--sample-mode": "random",
        "--sample-seed": sample_seed,
        "--draft-steps": 16,
        "--postmask-steps": 16,
        "--fixed-set-size": 32,
        "--fixed-refill-per-step": 2,
        "--vcd-noise-step": noise_step,
        "--vcd-noise-seed": 42,
        "--output-dir": default_output,
    }
    for flag, value in defaults.items():
        add_default(flag, value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a two-stage PostVRG (draft + refine) decoding experiment on M3CoT."
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test")
    add_dataset_adapter_args(parser)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--domain-filter", default=None)
    parser.add_argument("--sample-mode", default="sequential", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="M3CoT/PostVRG/outputs/postvrg")

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
        help="Number of late remask-refill (refine) steps. Defaults to total_steps - draft_steps.",
    )
    parser.add_argument(
        "--vcd-noise-step",
        type=int,
        default=200,
        help="Forward-diffusion timestep for the draft edge_noise/random_noise corruption (0-999; lower=lighter).",
    )
    parser.add_argument(
        "--vcd-noise-seed",
        type=int,
        default=42,
        help="Noise seed for the draft-image corruption (reproducible).",
    )
    parser.add_argument("--region-num", type=int, default=16,
                        help="Number of adaptive regions for edge_noise region selection.")
    parser.add_argument("--region-quantile", type=float, default=0.5,
                        help="Regions above this edge-density quantile are corrupted (0.5 = top half).")
    parser.add_argument("--region-weight", type=float, default=2.5,
                        help="Entropy exponent in the region-splitting complexity score.")
    parser.add_argument("--region-feather", type=float, default=0.02,
                        help="Region-mask feather (Gaussian sigma as fraction of image width).")
    parser.add_argument("--region-blur-frac", type=float, default=0.03,
                        help="Blur sigma (fraction of image width); used only by spotlight blur.")
    parser.add_argument("--region-min-highlight", type=int, default=4,
                        help="Minimum number of highlight regions (fallback if threshold selects fewer).")
    parser.add_argument("--refine-visual-mode", default="full",
                        choices=["full", "crop", "spotlight", "random_crop"],
                        help="Image the REFINE stage conditions on (all single-branch, no VCD). "
                        "full = same as draft; crop = zoom into the edge-entropy foreground region; "
                        "spotlight = whole image with the foreground kept and the background noised; "
                        "random_crop = fixed-size crop at a random position (control).")
    parser.add_argument("--crop-frac", type=float, default=0.6,
                        help="Crop size (fraction of each side) for random_crop and the random_noise box.")
    parser.add_argument("--crop-margin", type=float, default=0.1,
                        help="Margin (fraction of box size) around the edge-crop foreground box.")
    parser.add_argument("--draft-visual-mode", default="full",
                        choices=["full", "edge_noise", "random_noise"],
                        help="Image the DRAFT stage conditions on. full = original; "
                        "edge_noise = noise the edge-entropy detail regions (draft focuses on global); "
                        "random_noise = noise ONE random box (same box the random_crop fill-in zooms into).")
    parser.add_argument(
        "--fixed-set-size",
        type=int,
        default=None,
        help="Size of the fixed remask set chosen once after the draft stage. Defaults to max_new_tokens // 2.",
    )
    parser.add_argument(
        "--fixed-refill-per-step",
        type=int,
        default=None,
        help="If set, refill exactly this many masked positions per refine step.",
    )
    parser.add_argument(
        "--refill-confidence-gate",
        action="store_true",
        help="Keep the DRAFT token when a refilled token is less confident than the draft "
        "was at that position (protects against the fill-in making it worse).",
    )
    parser.add_argument(
        "--position-penalty",
        type=float,
        default=0.0,
        help="Position & Step Penalty gamma (arXiv:2604.05497 eq.3-4). >0 penalizes late-position "
        "tokens' commit confidence in early steps so the model reasons before answering (paper uses 0.5). "
        "0 = off. Applied in the DRAFT stage only (refine call is commented out in generate_with_postmask).",
    )
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--no-records",
        action="store_true",
        help="Do not write per-sample records.jsonl; only accumulate scores and write summary.json.",
    )
    return parser.parse_args()


def resolve_total_steps(max_new_tokens, block_length, step_per_block, step_ratio):
    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")
    num_blocks = max_new_tokens // block_length
    if num_blocks != 1:
        raise ValueError("The simple PostVRG runner currently expects block_length == max_new_tokens.")

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


def decode_answer(tokenizer, answer_ids):
    return clean_generated_text(
        tokenizer.decode(
            answer_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def forward_logits(core_model, x, prefix_embeds, prefix_length, prefix_kv_cache=None):
    if prefix_kv_cache is not None:
        answer_embeds = core_model.transformer.wte(x[:, prefix_length:])
        output = core_model(None, input_embeddings=answer_embeds, past_key_values=prefix_kv_cache)
        # With device_map="auto" the model is sharded and logits land on the last
        # shard; move them back to x's device so downstream manual ops stay on one
        # device. On a single device this is a no-op.
        full_logits = torch.zeros(x.shape[0], x.shape[1], output.logits.shape[-1], dtype=output.logits.dtype, device=x.device)
        full_logits[:, prefix_length:] = output.logits.to(x.device)
        return full_logits
    current_embeds = core_model.transformer.wte(x)
    current_embeds[:, :prefix_length] = prefix_embeds
    return core_model(None, input_embeddings=current_embeds).logits.to(x.device)


def forward_answer_logits(core_model, x, prefix_embeds, prefix_length, prefix_kv_cache=None):
    if prefix_kv_cache is not None:
        answer_embeds = core_model.transformer.wte(x[:, prefix_length:])
        output = core_model(None, input_embeddings=answer_embeds, past_key_values=prefix_kv_cache)
        return output.logits.to(x.device)
    current_embeds = core_model.transformer.wte(x)
    current_embeds[:, :prefix_length] = prefix_embeds
    return core_model(None, input_embeddings=current_embeds).logits[:, prefix_length:].to(x.device)


def compute_prefix_kv_cache(core_model, prefix_embeds):
    output = core_model(None, input_embeddings=prefix_embeds, use_cache=True)
    return output.attn_key_values












def build_processed_draft_prefix(args, model, tokenizer, image_processor, doc, mode, regions=None):
    """DRAFT prefix from a processed image so the draft focuses on the rest:
      edge_noise/edge_blur -> corrupt the edge-entropy detail regions;
      random_noise -> noise ONE random box (the same box the random_crop fill-in
      will zoom into, seeded by the sample id).
    `regions` = a precomputed selection tuple reused across stages (edge_noise only).
    Returns (prefix_embeds, prefix_input_ids_full)."""
    from region_utils_final import apply_region_corruption, apply_box_noise, random_box

    image = doc["image"].convert("RGB")
    context = build_prompt(doc, args.prompt)

    conv = copy.deepcopy(conv_templates[args.conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if mode == "random_noise":
        W, H = image.size
        box = random_box(doc.get("id", ""), W, H, args.crop_frac)   # SAME box as random_crop
        proc_image = apply_box_noise(image, box, noise_step=args.vcd_noise_step,
                                     seed=args.vcd_noise_seed, feather_frac=args.region_feather)
    else:
        proc_image = apply_region_corruption(
            image, gps_num=args.region_num, quantile=args.region_quantile,
            weight=args.region_weight, feather_frac=args.region_feather,
            noise_step=args.vcd_noise_step, seed=args.vcd_noise_seed,
            min_highlight=args.region_min_highlight, regions=regions,
        )

    image_tensor = process_images([proc_image], image_processor, model.config)
    dtype = get_torch_dtype(args.torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=args.device)

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
    prefix_embeds, prefix_input_ids_full = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        attention_mask=attention_mask,
    )
    return prefix_embeds, prefix_input_ids_full


def build_crop_prefix(args, model, tokenizer, image_processor, doc, regions=None):
    """Prefix built from a CROP of the edge-entropy (foreground) region, resized
    back to the original image size (so token count matches the draft prefix).
    Used to condition the refine stage on a zoomed-in, higher-effective-resolution
    view of the detail. Single-branch (no VCD).
    `regions` = a precomputed selection tuple reused across stages."""
    from region_utils_final import select_highlight_regions, image_to_gray_norm

    image = doc["image"].convert("RGB")
    context = build_prompt(doc, args.prompt)

    conv = copy.deepcopy(conv_templates[args.conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    W, H = image.size
    if regions is not None:
        regions_px, _, hi, _ = regions
    else:
        regions_px, _, hi, _ = select_highlight_regions(
            image_to_gray_norm(image), gps_num=args.region_num, quantile=args.region_quantile,
            weight=args.region_weight, min_highlight=args.region_min_highlight)
    hi_boxes = [regions_px[i] for i in range(len(regions_px)) if hi[i]]
    if not hi_boxes:
        hi_boxes = regions_px
    x1 = min(b[0] for b in hi_boxes); y1 = min(b[1] for b in hi_boxes)
    x2 = max(b[2] for b in hi_boxes); y2 = max(b[3] for b in hi_boxes)
    mx = int(args.crop_margin * (x2 - x1)); my = int(args.crop_margin * (y2 - y1))
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(W - 1, x2 + mx); y2 = min(H - 1, y2 + my)
    # crop the foreground box, then resize back to the ORIGINAL size -> same anyres
    # tiling -> same visual token count -> refine prefix length matches draft.
    crop_image = image.crop((x1, y1, x2 + 1, y2 + 1)).resize((W, H))

    image_tensor = process_images([crop_image], image_processor, model.config)
    dtype = get_torch_dtype(args.torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=args.device)

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
    crop_prefix_embeds, _ = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[crop_image.size],
        attention_mask=attention_mask,
    )
    return crop_prefix_embeds


def build_spotlight_prefix(args, model, tokenizer, image_processor, doc, regions=None):
    """Prefix from the WHOLE image with the edge-entropy (detail) region kept at
    full brightness and the rest darkened. Keeps global context (unlike crop) but
    emphasises the detail. Same size -> matches draft prefix length. No VCD.
    `regions` = a precomputed selection tuple reused across stages."""
    from region_utils_final import apply_region_spotlight

    image = doc["image"].convert("RGB")
    context = build_prompt(doc, args.prompt)

    conv = copy.deepcopy(conv_templates[args.conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    spot_image = apply_region_spotlight(
        image, gps_num=args.region_num, quantile=args.region_quantile,
        weight=args.region_weight, feather_frac=args.region_feather,
        noise_step=args.vcd_noise_step, seed=args.vcd_noise_seed,
        min_highlight=args.region_min_highlight, regions=regions)

    image_tensor = process_images([spot_image], image_processor, model.config)
    dtype = get_torch_dtype(args.torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=args.device)

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
    spot_prefix_embeds, _ = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[spot_image.size],
        attention_mask=attention_mask,
    )
    return spot_prefix_embeds




def build_fixed_crop_prefix(args, model, tokenizer, image_processor, doc, random):
    """Content-agnostic crop CONTROL for the edge crop: crop a fixed-size box
    (centered, or randomly placed) and resize back to the original size. No edge
    selection, no VCD. `--crop-frac` sets the box size (fraction of each side).
    random=True places the box randomly (seeded per-sample id for reproducibility)."""
    import zlib
    import random as _random

    image = doc["image"].convert("RGB")
    context = build_prompt(doc, args.prompt)

    conv = copy.deepcopy(conv_templates[args.conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    W, H = image.size
    cw = max(1, int(args.crop_frac * W))
    ch = max(1, int(args.crop_frac * H))
    if random:
        from region_utils_final import random_box
        bx1, by1, bx2, by2 = random_box(doc.get("id", ""), W, H, args.crop_frac)  # SAME box as random_noise draft
        crop_image = image.crop((bx1, by1, bx2 + 1, by2 + 1)).resize((W, H))
    else:  # center crop
        x1 = (W - cw) // 2
        y1 = (H - ch) // 2
        crop_image = image.crop((x1, y1, x1 + cw, y1 + ch)).resize((W, H))

    image_tensor = process_images([crop_image], image_processor, model.config)
    dtype = get_torch_dtype(args.torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=args.device)

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
    crop_prefix_embeds, _ = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[crop_image.size],
        attention_mask=attention_mask,
    )
    return crop_prefix_embeds


def choose_positions(scores, k):
    if k <= 0:
        return scores.new_empty(0, dtype=torch.long)
    return torch.topk(scores, k=k, largest=False).indices






@torch.no_grad()
def generate_with_postmask(
    core_model,
    tokenizer,
    prefix_embeds,
    max_new_tokens,
    block_length,
    step_per_block,
    step_ratio,
    temperature,
    remasking,
    draft_steps,
    postmask_steps,
    fixed_set_size,
    fixed_refill_per_step,
    refine_prefix_embeds=None,
    refill_confidence_gate=False,
    position_penalty=0.0,
):
    """Two-stage masked-diffusion decode: DRAFT (fill all answer tokens) then
    REFINE (re-mask the lowest-confidence set, then refill). Single branch, no VCD.
    If refine_prefix_embeds is given (crop / spotlight / random_crop), the refine
    stage conditions on it instead of the draft image (same length required).
    If refill_confidence_gate is True, a refilled token is only kept when its
    confidence is >= the draft's confidence at that position; otherwise the draft
    token is restored (protects against the fill-in replacing a token with a
    less-confident one)."""
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
    if fixed_set_size is not None and fixed_set_size <= 0:
        raise ValueError("fixed_set_size must be > 0.")
    if fixed_refill_per_step is not None and fixed_refill_per_step <= 0:
        raise ValueError("fixed_refill_per_step must be > 0.")

    device = prefix_embeds.device
    batch_size, prefix_length = prefix_embeds.shape[:2]
    if batch_size != 1:
        raise ValueError("generate_with_postmask currently expects batch size 1.")

    x = torch.full((batch_size, prefix_length + max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :prefix_length] = 0
    answer_slice = slice(prefix_length, prefix_length + max_new_tokens)
    proposal_confidence = torch.full((max_new_tokens,), float("inf"), dtype=torch.float64, device=device)
    cond_prefix_kv = None

    # Position & Step Penalty (PSP, arXiv:2604.05497 eq. 3-4): reweight the commit
    # confidence C by  C * [1 - gamma*(1 - tau)*rel(j)]  where rel(j) is the token's
    # normalized position (0..1, later = larger) and tau = step/total_steps. Late
    # positions (the answer) are suppressed in early steps -> reason first, answer last.
    rel = torch.arange(max_new_tokens, device=device, dtype=torch.float64) / max(1, max_new_tokens - 1)

    def apply_position_penalty(masked_conf, step_index):
        if position_penalty <= 0.0:
            return masked_conf
        tau = float(step_index) / float(total_steps)
        factor = (1.0 - position_penalty * (1.0 - tau) * rel).clamp_min(1e-6)   # [max_new_tokens]
        ans = masked_conf[0]
        masked_conf[0] = torch.where(
            torch.isfinite(ans), ans * factor.to(ans.dtype), ans)              # keep -inf (committed) as-is
        return masked_conf

    # ---- DRAFT: fill all answer tokens over draft_steps (prefix served from KV cache) ----
    draft_records = []
    for step_idx, num_to_fill in enumerate(draft_schedule, start=1):
        if cond_prefix_kv is None:
            cond_prefix_kv = compute_prefix_kv_cache(core_model, prefix_embeds)
        answer_logits = forward_answer_logits(core_model, x, prefix_embeds, prefix_length, prefix_kv_cache=cond_prefix_kv)
        logits_with_noise = add_gumbel_noise(answer_logits, temperature=temperature)
        answer_x0 = torch.argmax(logits_with_noise, dim=-1)
        answer_confidence = compute_remasking_confidence(answer_logits, answer_x0, remasking)
        proposal_confidence = answer_confidence[0].detach().to(torch.float64)
        answer_mask_index = x[:, answer_slice] == MASK_TOKEN_ID
        answer_x0 = torch.where(answer_mask_index, answer_x0, x[:, answer_slice])
        masked_confidence = torch.where(answer_mask_index, answer_confidence, -torch.inf)
        masked_confidence = apply_position_penalty(masked_confidence, step_idx)
        _, select_answer_index = torch.topk(masked_confidence[0], k=int(num_to_fill))
        x[0, prefix_length + select_answer_index] = answer_x0[0, select_answer_index]
        state_ids = x[0, answer_slice].detach().cpu().tolist()
        draft_records.append({
            "step": int(step_idx),
            "phase": "draft",
            "num_filled": int(num_to_fill),
            "selected_positions": [int(prefix_length + pos) for pos in select_answer_index.detach().cpu().tolist()],
            "state_text": decode_answer(tokenizer, state_ids),
            "num_masked_after_step": int((x[:, answer_slice] == MASK_TOKEN_ID).sum().item()),
        })

    draft_answer_ids = x[0, answer_slice].detach().cpu().tolist()
    draft_text = decode_answer(tokenizer, draft_answer_ids)

    # keep the draft's per-answer-token values + confidence for the refill gate
    draft_answer_tokens = x[0, answer_slice].clone()          # [max_new_tokens] draft token ids
    draft_answer_confidence = proposal_confidence.clone()     # [max_new_tokens] draft confidence

    # ---- pick the fixed re-mask set once (lowest draft confidence) ----
    fixed_remask_positions = None
    if postmask_steps > 0:
        effective_fixed_set_size = min(
            int(fixed_set_size) if fixed_set_size is not None else int(max_new_tokens) // 2,
            int(max_new_tokens),
        )
        fixed_remask_positions = choose_positions(scores=proposal_confidence, k=effective_fixed_set_size)
        if fixed_remask_positions.numel() > 0:
            x[0, fixed_remask_positions + prefix_length] = MASK_TOKEN_ID

    # ---- refine conditioning: optionally a different (crop/spotlight/random_crop) prefix ----
    if refine_prefix_embeds is not None:
        if refine_prefix_embeds.shape[1] != prefix_length:
            raise ValueError(
                f"refine_prefix_embeds length {refine_prefix_embeds.shape[1]} != draft "
                f"prefix length {prefix_length}; the refine image must be resized to the original size."
            )
        cond_prefix_kv = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        refine_prefix = refine_prefix_embeds
        refine_cond_kv = compute_prefix_kv_cache(core_model, refine_prefix_embeds)
    else:
        refine_prefix = prefix_embeds
        refine_cond_kv = cond_prefix_kv

    # ---- REFINE: refill the re-masked set over postmask_steps (single branch) ----
    postmask_records = []
    for local_step in range(1, postmask_steps + 1):
        if fixed_remask_positions is None or fixed_remask_positions.numel() == 0:
            break
        selected_answer_positions = fixed_remask_positions
        answer_logits = forward_answer_logits(core_model, x, refine_prefix, prefix_length, prefix_kv_cache=refine_cond_kv)
        logits_with_noise = add_gumbel_noise(answer_logits, temperature=temperature)
        answer_x0 = torch.argmax(logits_with_noise, dim=-1)
        answer_confidence = compute_remasking_confidence(answer_logits, answer_x0, remasking)
        answer_mask_index = x[:, answer_slice] == MASK_TOKEN_ID
        answer_x0 = torch.where(answer_mask_index, answer_x0, x[:, answer_slice])
        masked_confidence = torch.where(answer_mask_index, answer_confidence, -torch.inf)
        # PSP disabled in refine (draft-only). To re-enable, uncomment (consider refine-local tau):
        # masked_confidence = apply_position_penalty(masked_confidence, draft_steps + local_step)
        masked_remaining = int(answer_mask_index.sum().item())
        if fixed_refill_per_step is not None:
            refill_count = min(int(fixed_refill_per_step), masked_remaining)
        else:
            refill_count = masked_remaining
        _, refill_answer_index = torch.topk(masked_confidence[0], k=refill_count)
        if refill_confidence_gate and refill_answer_index.numel() > 0:
            # keep the DRAFT token where the refill is LESS confident than the draft
            ans_pos = refill_answer_index
            refill_conf = answer_confidence[0, refill_answer_index]
            draft_conf = draft_answer_confidence[ans_pos].to(refill_conf.dtype)
            chosen = torch.where(refill_conf >= draft_conf,
                                 answer_x0[0, refill_answer_index],
                                 draft_answer_tokens[ans_pos])
            x[0, prefix_length + refill_answer_index] = chosen
        else:
            x[0, prefix_length + refill_answer_index] = answer_x0[0, refill_answer_index]
        refilled_answer_positions = [int(pos) for pos in refill_answer_index.detach().cpu().tolist()]
        state_ids = x[0, answer_slice].detach().cpu().tolist()
        postmask_records.append({
            "step": int(draft_steps + local_step),
            "phase": "refine",
            "remasked_answer_positions": [int(pos) for pos in selected_answer_positions.detach().cpu().tolist()],
            "refilled_answer_positions": refilled_answer_positions,
            "state_text": decode_answer(tokenizer, state_ids),
        })

    final_answer_ids = x[0, answer_slice].detach().cpu().tolist()
    final_text = decode_answer(tokenizer, final_answer_ids)
    meta = {
        "max_new_tokens": int(max_new_tokens),
        "block_length": int(block_length),
        "total_steps": int(total_steps),
        "draft_steps": int(draft_steps),
        "postmask_steps": int(postmask_steps),
        "draft_schedule": [int(item) for item in draft_schedule],
        "fixed_set_size": int(fixed_set_size) if fixed_set_size is not None else None,
        "fixed_refill_per_step": int(fixed_refill_per_step) if fixed_refill_per_step is not None else None,
        "proposal_confidence_mean": float(proposal_confidence.mean().item()),
    }
    return {
        "draft_text": draft_text,
        "draft_answer_ids": draft_answer_ids,
        "draft_records": draft_records,
        "postmask_records": postmask_records,
        "final_text": final_text,
        "final_answer_ids": final_answer_ids,
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
    if args.conv_template in conv_templates:
        conv_templates[args.conv_template].tokenizer = tokenizer
    model.eval()
    model.tie_weights()
    model.to(get_torch_dtype(args.torch_dtype))
    core_model = model.get_model()
    # Inference is a single forward with no backward, so DeepSpeed activation
    # checkpointing (enabled in the model __init__) only adds recompute overhead
    # and requires an initialized distributed backend we don't set up here.
    # Disabling it does not change outputs.
    if hasattr(core_model, "set_activation_checkpointing"):
        core_model.set_activation_checkpointing(None)

    dataset = load_postvrg_dataset(args)
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
    write_records = not args.no_records

    total_elapsed = 0.0
    written = 0
    draft_correct_total = 0
    final_correct_total = 0
    improved_total = 0
    worsened_total = 0

    with (records_path.open("w", encoding="utf-8") if write_records else contextlib.nullcontext()) as fout:
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
            # Edge-entropy selection is deterministic per image, so compute it ONCE
            # here and reuse it for every stage that needs it (edge_noise draft,
            # crop/spotlight refine) instead of recomputing per builder.
            edge_regions = None
            if (args.draft_visual_mode == "edge_noise"
                    or args.refine_visual_mode in ("crop", "spotlight")):
                from region_utils_final import select_highlight_regions, image_to_gray_norm
                edge_regions = select_highlight_regions(
                    image_to_gray_norm(doc["image"].convert("RGB")),
                    gps_num=args.region_num, quantile=args.region_quantile,
                    weight=args.region_weight, min_highlight=args.region_min_highlight,
                )

            # Optionally condition the DRAFT on an edge-processed image (detail
            # corrupted -> draft focuses on global semantics). Same token count.
            if args.draft_visual_mode != "full":
                prefix_embeds, prefix_input_ids_full = build_processed_draft_prefix(
                    args, model, tokenizer, image_processor, doc, args.draft_visual_mode,
                    regions=edge_regions,
                )

            refine_prefix_embeds = None
            if args.refine_visual_mode == "crop":
                refine_prefix_embeds = build_crop_prefix(
                    args, model, tokenizer, image_processor, doc, regions=edge_regions
                )
            elif args.refine_visual_mode == "spotlight":
                refine_prefix_embeds = build_spotlight_prefix(
                    args, model, tokenizer, image_processor, doc, regions=edge_regions
                )
            elif args.refine_visual_mode == "random_crop":
                refine_prefix_embeds = build_fixed_crop_prefix(
                    args, model, tokenizer, image_processor, doc, random=True
                )
            run_output = generate_with_postmask(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                step_per_block=args.step_per_block,
                step_ratio=args.step_ratio,
                temperature=args.temperature,
                remasking=args.remasking,
                draft_steps=args.draft_steps,
                postmask_steps=args.postmask_steps,
                fixed_set_size=args.fixed_set_size,
                fixed_refill_per_step=args.fixed_refill_per_step,
                refine_prefix_embeds=refine_prefix_embeds,
                refill_confidence_gate=args.refill_confidence_gate,
                position_penalty=args.position_penalty,
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

            if write_records:
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
            "fixed_set_size": args.fixed_set_size,
            "fixed_refill_per_step": args.fixed_refill_per_step,
            "refill_confidence_gate": args.refill_confidence_gate,
            "position_penalty": args.position_penalty,
            "draft_visual_mode": args.draft_visual_mode,
            "refine_visual_mode": args.refine_visual_mode,
            "vcd_noise_step": args.vcd_noise_step if args.draft_visual_mode != "full" else None,
            "vcd_noise_seed": args.vcd_noise_seed if args.draft_visual_mode != "full" else None,
            "crop_margin": args.crop_margin if args.refine_visual_mode == "crop" else None,
            "crop_frac": args.crop_frac if args.refine_visual_mode in ("random_crop",) or args.draft_visual_mode == "random_noise" else None,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote records to {records_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)
    restore_compile()


if __name__ == "__main__":
    apply_postvrg_defaults()
    main()
