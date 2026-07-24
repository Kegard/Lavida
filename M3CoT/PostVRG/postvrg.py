import argparse
import contextlib
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Tuple

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
from M3CoT.utils.metric import judge_answer
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.language_model.llada.generate import add_gumbel_noise

def build_unconditional_prefix_embeds(
    core_model,
    prefix_embeds: torch.Tensor,
    prefix_input_ids_full: torch.Tensor,
    null_visual_mode: str = "zeros",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prefix_input_ids_full.dim() != 1:
        raise ValueError("prefix_input_ids_full must be a 1D tensor for timestep VRG.")

    visual_mask = prefix_input_ids_full.eq(IMAGE_TOKEN_INDEX)
    uncond_prefix_embeds = prefix_embeds.clone()
    if not visual_mask.any():
        return uncond_prefix_embeds, visual_mask

    if null_visual_mode == "zeros":
        null_visual = torch.zeros(
            (1, 1, prefix_embeds.shape[-1]),
            dtype=prefix_embeds.dtype,
            device=prefix_embeds.device,
        )
    elif null_visual_mode == "mask_token":
        null_visual = core_model.transformer.wte(
            torch.tensor([MASK_TOKEN_ID], dtype=torch.long, device=prefix_embeds.device)
        ).view(1, 1, -1).to(dtype=prefix_embeds.dtype)
    else:
        raise ValueError(f"Unsupported null_visual_mode: {null_visual_mode}")

    uncond_prefix_embeds[:, visual_mask, :] = null_visual
    return uncond_prefix_embeds, visual_mask


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
    alpha = cli_value("--vcd-refill-alpha", "1.0")
    calibration = cli_value("--refill-vrg-calibration", "none")
    confidence_threshold = cli_value("--refill-vrg-confidence-threshold", "0.9")
    confidence_gate_tau = cli_value("--refill-vrg-confidence-gate-tau", None)
    guidance_steps = cli_value("--refill-guidance-steps", None)
    refine_prior = cli_value("--refine-prior", "none")
    prior_top_k = cli_value("--prior-top-k", "8")
    prior_alpha = cli_value("--prior-alpha", "0.5")
    cond_mode = cli_value("--refill-cond-visual-mode", "crop")
    weak_mode = cli_value("--refill-weak-visual-mode", "diffusion_noise")
    refill_guidance = cli_value("--refill-guidance", "none")
    noise_step = cli_value("--vcd-noise-step", "500")
    sample_seed = cli_value("--sample-seed", "42")
    limit = cli_value("--limit", "400")

    if weak_mode in ("diffusion_noise", "full_noise"):
        weak_tag = f"fullnoise{noise_step}"
    elif weak_mode == "spotlight":
        weak_tag = f"spotlight_noise{noise_step}"
    else:
        weak_tag = safe_name(weak_mode)
    if calibration == "none":
        calib_tag = f"alpha{safe_name(alpha)}"
    elif calibration == "soft_confidence":
        calib_tag = f"softconf_alpha{safe_name(alpha)}"
    elif calibration == "hard_confidence":
        calib_tag = f"hardconf_tau{safe_name(confidence_threshold)}_alpha{safe_name(alpha)}"
    else:
        calib_tag = f"{calibration}_alpha{safe_name(alpha)}"
    step_tag = "" if guidance_steps is None else f"_k{safe_name(guidance_steps)}"
    gate_tag = "" if confidence_gate_tau is None else f"_gate{safe_name(confidence_gate_tau)}"
    prior_tag = ""
    if refine_prior != "none":
        prior_tag = (
            f"_prior{safe_name(refine_prior)}"
            f"_topk{safe_name(prior_top_k)}"
            f"_alpha{safe_name(prior_alpha)}"
        )
    default_output = (
        "M3CoT/PostVRG/outputs/"
        f"postvrg_proposalconf_refill-{safe_name(refill_guidance)}_"
        f"{calib_tag}{step_tag}{gate_tag}{prior_tag}_cond{safe_name(cond_mode)}"
        f"_vs_{weak_tag}_seed{sample_seed}_n{limit}"
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
        "--refill-guidance": refill_guidance,
        "--refill-cond-visual-mode": cond_mode,
        "--refill-weak-visual-mode": weak_mode,
        "--vcd-refill-alpha": alpha,
        "--refill-vrg-calibration": calibration,
        "--refill-vrg-confidence-threshold": confidence_threshold,
        "--refill-vrg-confidence-gate-tau": confidence_gate_tau,
        "--refill-guidance-steps": guidance_steps,
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
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
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
        default=500,
        help="Forward-diffusion timestep for diffusion-noise weak visual guidance.",
    )
    parser.add_argument(
        "--vcd-noise-seed",
        type=int,
        default=None,
        help="Optional noise seed for diffusion-noise weak visual guidance.",
    )
    parser.add_argument(
        "--refill-guidance",
        default="none",
        choices=["none", "vcd"],
        help="Optional logits guidance (VRG) used only during the refine refill stage.",
    )
    parser.add_argument(
        "--draft-guidance",
        default="none",
        choices=["none", "vcd"],
        help="Optional logits guidance (VRG) used during the draft generation stage.",
    )
    parser.add_argument(
        "--draft-weak-visual-mode",
        default="diffusion_noise",
        choices=["diffusion_noise", "null_visual"],
        help="Weak visual condition for VCD-guided draft logits. "
        "diffusion_noise = noised image; null_visual = no image (zero out image-token embeds).",
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
        "--refill-cond-visual-mode",
        default="full",
        choices=["full", "crop", "spotlight"],
        help=(
            "Image used as the positive/conditional branch during refine. "
            "full keeps the original image; crop/spotlight use the selected edge-region view."
        ),
    )
    parser.add_argument(
        "--refill-weak-visual-mode",
        default="diffusion_noise",
        choices=["diffusion_noise", "full_noise", "null_visual", "crop", "spotlight"],
        help="Weak visual condition for VCD-guided refill logits. "
        "full_noise/diffusion_noise = noised full image; null_visual = no image "
        "(zero out image-token embeds); crop/spotlight = edge-region image used as the comparison branch.",
    )
    parser.add_argument("--region-num", type=int, default=16,
                        help="Number of adaptive regions for crop/spotlight comparison image selection.")
    parser.add_argument("--region-quantile", type=float, default=0.5,
                        help="Regions above this edge-density quantile are selected (0.5 = top half).")
    parser.add_argument("--region-weight", type=float, default=2.5,
                        help="Entropy exponent in the region-splitting complexity score.")
    parser.add_argument("--region-feather", type=float, default=0.02,
                        help="Region-mask feather (Gaussian sigma as fraction of image width).")
    parser.add_argument("--region-min-highlight", type=int, default=4,
                        help="Minimum number of highlight regions (fallback if threshold selects fewer).")
    parser.add_argument("--crop-margin", type=float, default=0.1,
                        help="Margin (fraction of box size) around the edge-crop foreground box.")
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
            "Optional VCD refill gate (token gate). If set, use guided logits at an answer position only when "
            "max_softmax(guided) - max_softmax(cond) > tau; otherwise fall back to cond logits."
        ),
    )
    parser.add_argument(
        "--refill-guidance-steps",
        type=int,
        default=None,
        help="Only for refill guidance: apply VRG on the first k refine steps, then fall back to normal refill.",
    )
    parser.add_argument(
        "--refine-prior",
        default="none",
        choices=["none", "draft_topk_margin"],
        help="Optional draft prior used during refine. draft_topk_margin adds a sparse top-k margin bias.",
    )
    parser.add_argument(
        "--prior-top-k",
        type=int,
        default=8,
        help="Number of committed draft logits to cache per answer position for --refine-prior draft_topk_margin.",
    )
    parser.add_argument(
        "--prior-alpha",
        type=float,
        default=0.5,
        help="Strength of the draft top-k margin bias added to refine logits.",
    )
    parser.add_argument(
        "--record-prior-events",
        action="store_true",
        help="Record lightweight per-position logits/top-k diagnostics for draft prior analysis.",
    )
    parser.add_argument(
        "--prior-event-top-k",
        type=int,
        default=8,
        help="Number of crop/fused top logits to record when --record-prior-events is set.",
    )
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
        full_logits = torch.zeros(x.shape[0], x.shape[1], output.logits.shape[-1], dtype=output.logits.dtype, device=output.logits.device)
        full_logits[:, prefix_length:] = output.logits
        return full_logits
    current_embeds = core_model.transformer.wte(x)
    current_embeds[:, :prefix_length] = prefix_embeds
    return core_model(None, input_embeddings=current_embeds).logits


def forward_answer_logits(core_model, x, prefix_embeds, prefix_length, prefix_kv_cache=None):
    if prefix_kv_cache is not None:
        answer_embeds = core_model.transformer.wte(x[:, prefix_length:])
        output = core_model(None, input_embeddings=answer_embeds, past_key_values=prefix_kv_cache)
        return output.logits
    current_embeds = core_model.transformer.wte(x)
    current_embeds[:, :prefix_length] = prefix_embeds
    return core_model(None, input_embeddings=current_embeds).logits[:, prefix_length:]


def compute_prefix_kv_cache(core_model, prefix_embeds):
    output = core_model(None, input_embeddings=prefix_embeds, use_cache=True)
    return output.attn_key_values


def forward_aligned_answer_only_logits(
    core_model,
    x,
    prefix_embeds,
    answer_prefix_length,
    prefix_kv_cache=None,
):
    prefix_length = int(prefix_embeds.shape[1])
    if prefix_length == int(answer_prefix_length):
        return forward_answer_logits(core_model, x, prefix_embeds, prefix_length, prefix_kv_cache=prefix_kv_cache)

    answer_ids = x[:, answer_prefix_length:]
    aligned_x = torch.full(
        (x.shape[0], prefix_length + answer_ids.shape[1]),
        MASK_TOKEN_ID,
        dtype=x.dtype,
        device=x.device,
    )
    aligned_x[:, :prefix_length] = 0
    aligned_x[:, prefix_length:] = answer_ids
    return forward_answer_logits(core_model, aligned_x, prefix_embeds, prefix_length, prefix_kv_cache=prefix_kv_cache)


def forward_aligned_answer_logits(
    core_model,
    x,
    prefix_embeds,
    answer_prefix_length,
    prefix_kv_cache=None,
):
    prefix_length = int(prefix_embeds.shape[1])
    if prefix_length == int(answer_prefix_length):
        return forward_logits(core_model, x, prefix_embeds, prefix_length, prefix_kv_cache=prefix_kv_cache)

    answer_ids = x[:, answer_prefix_length:]
    aligned_x = torch.full(
        (x.shape[0], prefix_length + answer_ids.shape[1]),
        MASK_TOKEN_ID,
        dtype=x.dtype,
        device=x.device,
    )
    aligned_x[:, :prefix_length] = 0
    aligned_x[:, prefix_length:] = answer_ids
    aligned_logits = forward_logits(core_model, aligned_x, prefix_embeds, prefix_length, prefix_kv_cache=prefix_kv_cache)

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


def build_prompt_inputs(args, tokenizer, doc):
    image = doc["image"].convert("RGB")
    context = build_prompt(doc, args.prompt)

    conv = copy.deepcopy(conv_templates[args.conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
    return image, input_ids, attention_mask


def process_single_image_prefix(args, model, tokenizer, image_processor, doc, proc_image):
    image, input_ids, attention_mask = build_prompt_inputs(args, tokenizer, doc)
    image_tensor = process_images([proc_image], image_processor, model.config)
    dtype = get_torch_dtype(args.torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=args.device)

    prefix_embeds, _ = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        attention_mask=attention_mask,
    )
    return prefix_embeds


def select_edge_regions(args, image):
    from M3CoT.PostVRG.region_utils_final import image_to_gray_norm, select_highlight_regions

    return select_highlight_regions(
        image_to_gray_norm(image),
        gps_num=args.region_num,
        quantile=args.region_quantile,
        weight=args.region_weight,
        min_highlight=args.region_min_highlight,
    )


def build_crop_comparison_prefix(args, model, tokenizer, image_processor, doc, regions=None):
    image = doc["image"].convert("RGB")
    W, H = image.size
    if regions is None:
        regions = select_edge_regions(args, image)
    regions_px, _, hi, _ = regions
    hi_boxes = [regions_px[i] for i in range(len(regions_px)) if hi[i]]
    if not hi_boxes:
        hi_boxes = regions_px
    x1 = min(b[0] for b in hi_boxes)
    y1 = min(b[1] for b in hi_boxes)
    x2 = max(b[2] for b in hi_boxes)
    y2 = max(b[3] for b in hi_boxes)
    mx = int(args.crop_margin * (x2 - x1))
    my = int(args.crop_margin * (y2 - y1))
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(W - 1, x2 + mx)
    y2 = min(H - 1, y2 + my)
    crop_image = image.crop((x1, y1, x2 + 1, y2 + 1)).resize((W, H))
    return process_single_image_prefix(args, model, tokenizer, image_processor, doc, crop_image)


def build_spotlight_comparison_prefix(args, model, tokenizer, image_processor, doc, regions=None):
    from M3CoT.PostVRG.region_utils_final import apply_region_spotlight

    image = doc["image"].convert("RGB")
    if regions is None:
        regions = select_edge_regions(args, image)
    spot_image = apply_region_spotlight(
        image,
        gps_num=args.region_num,
        quantile=args.region_quantile,
        weight=args.region_weight,
        feather_frac=args.region_feather,
        noise_step=args.vcd_noise_step,
        seed=args.vcd_noise_seed,
        min_highlight=args.region_min_highlight,
        regions=regions,
    )
    return process_single_image_prefix(args, model, tokenizer, image_processor, doc, spot_image)


def choose_positions(scores, k):
    if k <= 0:
        return scores.new_empty(0, dtype=torch.long)
    return torch.topk(scores, k=k, largest=False).indices


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


def apply_confidence_gated_answer_logits(cond_answer_logits, guided_answer_logits, tau):
    use_guided = (
        max_softmax_probability(guided_answer_logits)
        - max_softmax_probability(cond_answer_logits)
    ) > float(tau)
    return torch.where(
        use_guided.unsqueeze(-1),
        guided_answer_logits,
        cond_answer_logits,
    )


def build_topk_logits_debug(logits, answer_pos, top_k, tokenizer=None):
    vocab_size = int(logits.shape[-1])
    k = min(max(int(top_k), 1), vocab_size)
    values, ids = torch.topk(logits[answer_pos], k=k, dim=-1)
    token_ids = [int(item) for item in ids.detach().cpu().tolist()]
    entry = {
        "topk_ids": token_ids,
        "topk_logits": [float(item) for item in values.detach().cpu().tolist()],
        "top1_id": token_ids[0] if token_ids else None,
        "top1_logit": float(values[0].item()) if values.numel() else None,
    }
    if values.numel() >= 2:
        entry["top1_top2_margin"] = float((values[0] - values[1]).item())
    if values.numel() >= 1:
        entry["top1_topk_margin"] = float((values[0] - values[-1]).item())
    if tokenizer is not None:
        entry["topk_tokens"] = [
            tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for token_id in token_ids
        ]
    return entry


@torch.no_grad()
def generate_with_postmask(
    core_model,
    tokenizer,
    prefix_embeds,
    refill_cond_prefix_embeds,
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
    fixed_set_size,
    fixed_refill_per_step,
    draft_guidance,
    draft_weak_visual_mode,
    vcd_draft_alpha,
    draft_guidance_ratio,
    refill_guidance,
    refill_cond_visual_mode,
    refill_weak_visual_mode,
    vcd_refill_alpha,
    refill_vrg_calibration,
    refill_vrg_confidence_threshold,
    refill_vrg_confidence_gate_tau,
    refill_guidance_steps,
    refine_prior,
    prior_top_k,
    prior_alpha,
    record_prior_events,
    prior_event_top_k,
    vcd_noise_step,
    vcd_noise_seed,
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
    if fixed_set_size is not None and fixed_set_size <= 0:
        raise ValueError("fixed_set_size must be > 0.")
    if fixed_refill_per_step is not None and fixed_refill_per_step <= 0:
        raise ValueError("fixed_refill_per_step must be > 0.")
    if draft_guidance_ratio < 0.0 or draft_guidance_ratio > 1.0:
        raise ValueError("draft_guidance_ratio must be in [0, 1].")
    if refill_guidance_steps is not None and refill_guidance_steps <= 0:
        raise ValueError("refill_guidance_steps must be > 0 when set.")
    if refine_prior == "draft_topk_margin" and prior_top_k <= 0:
        raise ValueError("prior_top_k must be > 0 when --refine-prior draft_topk_margin is used.")
    if record_prior_events and refine_prior != "draft_topk_margin":
        raise ValueError("--record-prior-events requires --refine-prior draft_topk_margin.")
    if record_prior_events and prior_event_top_k <= 0:
        raise ValueError("prior_event_top_k must be > 0 when --record-prior-events is used.")
    draft_guidance_steps = 0
    if draft_guidance == "vcd":
        draft_guidance_steps = int(math.ceil(float(draft_steps) * float(draft_guidance_ratio)))

    device = prefix_embeds.device
    batch_size, prefix_length = prefix_embeds.shape[:2]
    if batch_size != 1:
        raise ValueError("generate_with_postmask currently expects batch size 1.")
    if refill_cond_prefix_embeds is None:
        refill_cond_prefix_embeds = prefix_embeds
    if int(refill_cond_prefix_embeds.shape[1]) != int(prefix_length):
        raise ValueError(
            "refill_cond_prefix_embeds length must match the draft prefix length; "
            "crop/spotlight images should be resized to the original size."
        )

    def resolve_guidance_prefix(guidance, weak_prefix_embeds, phase_name):
        if guidance == "none":
            return None
        if guidance != "vcd":
            raise ValueError(f"Unsupported {phase_name} guidance: {guidance}")
        if weak_prefix_embeds is None:
            raise ValueError(f"VCD {phase_name} guidance requires weak_prefix_embeds.")
        return weak_prefix_embeds

    def apply_vcd_guidance(answer_logits, x_state, weak_prefix_embeds, alpha, weak_kv_cache=None):
        weak_answer_logits = forward_aligned_answer_only_logits(
            core_model,
            x_state,
            weak_prefix_embeds,
            prefix_length,
            prefix_kv_cache=weak_kv_cache,
        )
        if torch.is_tensor(alpha):
            answer_alpha = alpha.to(
                device=answer_logits.device,
                dtype=answer_logits.dtype,
            ).view(1, -1, 1)
            return answer_logits + answer_alpha * (answer_logits - weak_answer_logits)
        return (1.0 + float(alpha)) * answer_logits - float(alpha) * weak_answer_logits

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

    def apply_draft_topk_margin_prior(answer_logits, answer_mask_index):
        if refine_prior != "draft_topk_margin":
            return answer_logits, answer_mask_index.new_empty(0, dtype=torch.long)
        valid_positions = torch.nonzero(
            answer_mask_index[0] & draft_prior_seen,
            as_tuple=False,
        ).view(-1)
        if valid_positions.numel() == 0:
            return answer_logits, valid_positions
        guided_logits = answer_logits.clone()
        prior_ids = draft_prior_ids[valid_positions]
        prior_margin = draft_prior_margin[valid_positions].to(
            device=answer_logits.device,
            dtype=answer_logits.dtype,
        )
        bias = torch.zeros_like(guided_logits[0, valid_positions])
        bias.scatter_add_(dim=-1, index=prior_ids, src=prior_margin)
        guided_logits[0, valid_positions] += float(prior_alpha) * bias
        return guided_logits, valid_positions

    draft_uncond_prefix_embeds = resolve_guidance_prefix(
        draft_guidance,
        draft_weak_prefix_embeds,
        "draft",
    )
    refill_uncond_prefix_embeds = resolve_guidance_prefix(
        refill_guidance,
        refill_weak_prefix_embeds,
        "refill",
    )

    x = torch.full((batch_size, prefix_length + max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :prefix_length] = 0
    answer_slice = slice(prefix_length, prefix_length + max_new_tokens)
    proposal_confidence = torch.full((max_new_tokens,), float("inf"), dtype=torch.float64, device=device)
    prior_k = min(max(int(prior_top_k), 1), int(core_model.transformer.wte.weight.shape[0]))
    event_k = min(max(int(prior_event_top_k), 1), int(core_model.transformer.wte.weight.shape[0]))
    draft_prior_ids = torch.zeros((max_new_tokens, prior_k), dtype=torch.long, device=device)
    draft_prior_margin = torch.zeros((max_new_tokens, prior_k), dtype=torch.float32, device=device)
    draft_prior_seen = torch.zeros((max_new_tokens,), dtype=torch.bool, device=device)
    prior_events = []

    cond_prefix_kv = None
    refill_cond_kv = None
    draft_weak_kv = None
    refill_weak_kv = None

    draft_records = []
    for step_idx, num_to_fill in enumerate(draft_schedule, start=1):
        if cond_prefix_kv is None:
            cond_prefix_kv = compute_prefix_kv_cache(core_model, prefix_embeds)
        answer_logits = forward_answer_logits(core_model, x, prefix_embeds, prefix_length, prefix_kv_cache=cond_prefix_kv)
        draft_guidance_used = draft_guidance == "vcd" and step_idx <= draft_guidance_steps
        if draft_guidance_used:
            if draft_weak_kv is None and draft_uncond_prefix_embeds is not None:
                weak_prefix_len = int(draft_uncond_prefix_embeds.shape[1])
                if weak_prefix_len == prefix_length:
                    draft_weak_kv = compute_prefix_kv_cache(core_model, draft_uncond_prefix_embeds)
            answer_logits = apply_vcd_guidance(
                answer_logits, x, draft_uncond_prefix_embeds, vcd_draft_alpha,
                weak_kv_cache=draft_weak_kv,
            )
        answer_x0 = torch.argmax(add_gumbel_noise(answer_logits, temperature=temperature), dim=-1)
        answer_confidence = compute_remasking_confidence(answer_logits, answer_x0, remasking)
        proposal_confidence = answer_confidence[0].detach().to(torch.float64)

        answer_mask_index = x[:, answer_slice] == MASK_TOKEN_ID
        answer_x0 = torch.where(answer_mask_index, answer_x0, x[:, answer_slice])
        masked_confidence = torch.where(answer_mask_index, answer_confidence, -torch.inf)

        _, select_answer_index = torch.topk(masked_confidence[0], k=int(num_to_fill))
        if refine_prior == "draft_topk_margin" and select_answer_index.numel() > 0:
            selected_logits = answer_logits[0, select_answer_index]
            topk_values, topk_ids = torch.topk(selected_logits, k=prior_k, dim=-1)
            draft_prior_ids[select_answer_index] = topk_ids
            draft_prior_margin[select_answer_index] = (
                topk_values - topk_values[:, -1:].detach()
            ).to(dtype=draft_prior_margin.dtype)
            draft_prior_seen[select_answer_index] = True
        x[0, prefix_length + select_answer_index] = answer_x0[0, select_answer_index]

        state_ids = x[0, answer_slice].detach().cpu().tolist()
        draft_records.append(
            {
                "step": int(step_idx),
                "phase": "draft",
                "guidance_used": bool(draft_guidance_used),
                "num_filled": int(num_to_fill),
                "selected_positions": [int(pos) for pos in select_answer_index.detach().cpu().tolist()],
                "state_text": decode_answer(tokenizer, state_ids),
                "num_masked_after_step": int((x[:, answer_slice] == MASK_TOKEN_ID).sum().item()),
            }
        )

    draft_answer_ids = x[0, answer_slice].detach().cpu().tolist()
    draft_text = decode_answer(tokenizer, draft_answer_ids)

    # Refine stage: choose a fixed remask set once (by lowest proposal confidence),
    # then iteratively remask-and-refill those positions.
    fixed_remask_positions = None
    if postmask_steps > 0:
        effective_fixed_set_size = min(
            int(fixed_set_size) if fixed_set_size is not None else int(max_new_tokens) // 2,
            int(max_new_tokens),
        )
        fixed_remask_positions = choose_positions(
            scores=proposal_confidence,
            k=effective_fixed_set_size,
        )
        if fixed_remask_positions.numel() > 0:
            x[0, fixed_remask_positions + prefix_length] = MASK_TOKEN_ID

    postmask_records = []
    for local_step in range(1, postmask_steps + 1):
        if fixed_remask_positions is None or fixed_remask_positions.numel() == 0:
            break

        selected_answer_positions = fixed_remask_positions

        if refill_cond_kv is None:
            if refill_cond_prefix_embeds is prefix_embeds:
                refill_cond_kv = cond_prefix_kv
            else:
                refill_cond_kv = compute_prefix_kv_cache(core_model, refill_cond_prefix_embeds)
        cond_answer_logits = forward_answer_logits(
            core_model,
            x,
            refill_cond_prefix_embeds,
            prefix_length,
            prefix_kv_cache=refill_cond_kv,
        )
        answer_logits = cond_answer_logits
        guidance_active = (
            refill_guidance == "vcd"
            and (
                refill_guidance_steps is None
                or local_step <= int(refill_guidance_steps)
            )
        )
        if guidance_active:
            if refill_weak_kv is None and refill_uncond_prefix_embeds is not None:
                weak_prefix_len = int(refill_uncond_prefix_embeds.shape[1])
                if weak_prefix_len == prefix_length:
                    refill_weak_kv = compute_prefix_kv_cache(core_model, refill_uncond_prefix_embeds)
            guided_answer_logits = apply_vcd_guidance(
                cond_answer_logits,
                x,
                refill_uncond_prefix_embeds,
                build_refill_vrg_alpha(),
                weak_kv_cache=refill_weak_kv,
            )
            if refill_vrg_confidence_gate_tau is not None:
                answer_logits = apply_confidence_gated_answer_logits(
                    cond_answer_logits=cond_answer_logits,
                    guided_answer_logits=guided_answer_logits,
                    tau=refill_vrg_confidence_gate_tau,
                )
            else:
                answer_logits = guided_answer_logits
        answer_mask_index = x[:, answer_slice] == MASK_TOKEN_ID
        prior_base_logits = answer_logits
        answer_logits, prior_valid_positions = apply_draft_topk_margin_prior(answer_logits, answer_mask_index)
        answer_x0 = torch.argmax(add_gumbel_noise(answer_logits, temperature=temperature), dim=-1)
        answer_confidence = compute_remasking_confidence(answer_logits, answer_x0, remasking)

        answer_x0 = torch.where(answer_mask_index, answer_x0, x[:, answer_slice])
        masked_confidence = torch.where(answer_mask_index, answer_confidence, -torch.inf)
        masked_remaining = int(answer_mask_index.sum().item())
        if fixed_refill_per_step is not None:
            refill_count = min(int(fixed_refill_per_step), masked_remaining)
        else:
            refill_count = masked_remaining
        _, refill_answer_index = torch.topk(masked_confidence[0], k=refill_count)
        if record_prior_events and prior_valid_positions.numel() > 0:
            refilled_set = {int(pos) for pos in refill_answer_index.detach().cpu().tolist()}
            for answer_pos in prior_valid_positions.detach().cpu().tolist():
                answer_pos = int(answer_pos)
                draft_ids = [int(item) for item in draft_prior_ids[answer_pos].detach().cpu().tolist()]
                draft_margin = [float(item) for item in draft_prior_margin[answer_pos].detach().cpu().tolist()]
                base_entry = build_topk_logits_debug(
                    prior_base_logits[0],
                    answer_pos,
                    event_k,
                    tokenizer=tokenizer,
                )
                fused_entry = build_topk_logits_debug(
                    answer_logits[0],
                    answer_pos,
                    event_k,
                    tokenizer=tokenizer,
                )
                prior_events.append({
                    "step": int(draft_steps + local_step),
                    "local_step": int(local_step),
                    "answer_position": answer_pos,
                    "refilled": answer_pos in refilled_set,
                    "draft_prior_ids": draft_ids,
                    "draft_prior_margin": draft_margin,
                    "draft_prior_top1_id": draft_ids[0] if draft_ids else None,
                    "base": base_entry,
                    "fused": fused_entry,
                    "base_top1_in_draft_topk": (
                        base_entry["top1_id"] in draft_ids
                        if base_entry["top1_id"] is not None
                        else None
                    ),
                    "fused_top1_in_draft_topk": (
                        fused_entry["top1_id"] in draft_ids
                        if fused_entry["top1_id"] is not None
                        else None
                    ),
                    "changed_top1_by_prior": base_entry["top1_id"] != fused_entry["top1_id"],
                    "selected_x0_id": int(answer_x0[0, answer_pos].item()),
                    "selected_confidence": float(answer_confidence[0, answer_pos].item()),
                })
        x[0, prefix_length + refill_answer_index] = answer_x0[0, refill_answer_index]

        refilled_answer_positions = [int(pos) for pos in refill_answer_index.detach().cpu().tolist()]

        state_ids = x[0, answer_slice].detach().cpu().tolist()
        postmask_records.append(
            {
                "step": int(draft_steps + local_step),
                "phase": "refine",
                "remasked_answer_positions": [int(pos) for pos in selected_answer_positions.detach().cpu().tolist()],
                "refilled_answer_positions": refilled_answer_positions,
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
        "fixed_set_size": int(fixed_set_size) if fixed_set_size is not None else None,
        "fixed_refill_per_step": int(fixed_refill_per_step) if fixed_refill_per_step is not None else None,
        "draft_guidance": draft_guidance,
        "draft_weak_visual_mode": draft_weak_visual_mode if draft_guidance == "vcd" else None,
        "vcd_draft_alpha": float(vcd_draft_alpha) if draft_guidance == "vcd" else None,
        "draft_guidance_ratio": float(draft_guidance_ratio) if draft_guidance == "vcd" else None,
        "draft_guidance_steps": int(draft_guidance_steps) if draft_guidance == "vcd" else 0,
        "refill_guidance": refill_guidance,
        "refill_cond_visual_mode": refill_cond_visual_mode,
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
        "refine_prior": refine_prior,
        "prior_top_k": int(prior_k) if refine_prior == "draft_topk_margin" else None,
        "prior_alpha": float(prior_alpha) if refine_prior == "draft_topk_margin" else None,
        "record_prior_events": bool(record_prior_events),
        "prior_event_top_k": int(event_k) if record_prior_events else None,
        "vcd_noise_step": (
            int(vcd_noise_step)
            if draft_guidance == "vcd" or refill_guidance == "vcd"
            else None
        ),
        "vcd_noise_seed": (
            int(vcd_noise_seed)
            if (
                vcd_noise_seed is not None
                and (draft_guidance == "vcd" or refill_guidance == "vcd")
            )
            else None
        ),
        "proposal_confidence_mean": float(proposal_confidence.mean().item()),
    }
    output = {
        "draft_text": draft_text,
        "draft_answer_ids": draft_answer_ids,
        "draft_records": draft_records,
        "postmask_records": postmask_records,
        "final_text": final_text,
        "final_answer_ids": final_answer_ids,
        "meta": meta,
    }
    if record_prior_events:
        output["prior_events"] = prior_events
    return output


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

            edge_regions = None

            def get_edge_regions():
                nonlocal edge_regions
                if edge_regions is None:
                    edge_regions = select_edge_regions(args, doc["image"].convert("RGB"))
                return edge_regions

            def build_weak_prefix(weak_visual_mode):
                if weak_visual_mode in ("diffusion_noise", "full_noise"):
                    return build_diffusion_noise_prefix(
                        args,
                        model,
                        tokenizer,
                        image_processor,
                        doc,
                    )
                if weak_visual_mode == "null_visual":
                    weak_prefix_embeds, _ = build_unconditional_prefix_embeds(
                        core_model=core_model,
                        prefix_embeds=prefix_embeds,
                        prefix_input_ids_full=prefix_input_ids_full,
                        null_visual_mode="zeros",
                    )
                    return weak_prefix_embeds
                if weak_visual_mode == "crop":
                    return build_crop_comparison_prefix(
                        args,
                        model,
                        tokenizer,
                        image_processor,
                        doc,
                        regions=get_edge_regions(),
                    )
                if weak_visual_mode == "spotlight":
                    return build_spotlight_comparison_prefix(
                        args,
                        model,
                        tokenizer,
                        image_processor,
                        doc,
                        regions=get_edge_regions(),
                    )
                raise ValueError(f"Unsupported weak visual mode: {weak_visual_mode}")

            def build_visual_prefix(visual_mode):
                if visual_mode == "full":
                    return prefix_embeds
                if visual_mode == "crop":
                    return build_crop_comparison_prefix(
                        args,
                        model,
                        tokenizer,
                        image_processor,
                        doc,
                        regions=get_edge_regions(),
                    )
                if visual_mode == "spotlight":
                    return build_spotlight_comparison_prefix(
                        args,
                        model,
                        tokenizer,
                        image_processor,
                        doc,
                        regions=get_edge_regions(),
                    )
                raise ValueError(f"Unsupported visual mode: {visual_mode}")

            draft_weak_prefix_embeds = None
            refill_cond_prefix_embeds = build_visual_prefix(args.refill_cond_visual_mode)
            refill_weak_prefix_embeds = None
            if args.draft_guidance == "vcd":
                draft_weak_prefix_embeds = build_weak_prefix(args.draft_weak_visual_mode)
            if args.refill_guidance == "vcd":
                if (
                    draft_weak_prefix_embeds is not None
                    and args.draft_weak_visual_mode == args.refill_weak_visual_mode
                ):
                    refill_weak_prefix_embeds = draft_weak_prefix_embeds
                else:
                    refill_weak_prefix_embeds = build_weak_prefix(args.refill_weak_visual_mode)
            run_output = generate_with_postmask(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                refill_cond_prefix_embeds=refill_cond_prefix_embeds,
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
                fixed_set_size=args.fixed_set_size,
                fixed_refill_per_step=args.fixed_refill_per_step,
                draft_guidance=args.draft_guidance,
                draft_weak_visual_mode=args.draft_weak_visual_mode,
                vcd_draft_alpha=args.vcd_draft_alpha,
                draft_guidance_ratio=args.draft_guidance_ratio,
                refill_guidance=args.refill_guidance,
                refill_cond_visual_mode=args.refill_cond_visual_mode,
                refill_weak_visual_mode=args.refill_weak_visual_mode,
                vcd_refill_alpha=args.vcd_refill_alpha,
                refill_vrg_calibration=args.refill_vrg_calibration,
                refill_vrg_confidence_threshold=args.refill_vrg_confidence_threshold,
                refill_vrg_confidence_gate_tau=args.refill_vrg_confidence_gate_tau,
                refill_guidance_steps=args.refill_guidance_steps,
                refine_prior=args.refine_prior,
                prior_top_k=args.prior_top_k,
                prior_alpha=args.prior_alpha,
                record_prior_events=args.record_prior_events,
                prior_event_top_k=args.prior_event_top_k,
                vcd_noise_step=args.vcd_noise_step,
                vcd_noise_seed=args.vcd_noise_seed,
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
                if args.record_prior_events:
                    record["prior_events"] = run_output.get("prior_events", [])
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
            "draft_guidance": args.draft_guidance,
            "draft_weak_visual_mode": args.draft_weak_visual_mode if args.draft_guidance == "vcd" else None,
            "vcd_draft_alpha": args.vcd_draft_alpha if args.draft_guidance == "vcd" else None,
            "draft_guidance_ratio": args.draft_guidance_ratio if args.draft_guidance == "vcd" else None,
            "refill_guidance": args.refill_guidance,
            "refill_cond_visual_mode": args.refill_cond_visual_mode,
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
            "refine_prior": args.refine_prior,
            "prior_top_k": args.prior_top_k if args.refine_prior == "draft_topk_margin" else None,
            "prior_alpha": args.prior_alpha if args.refine_prior == "draft_topk_margin" else None,
            "record_prior_events": args.record_prior_events,
            "prior_event_top_k": args.prior_event_top_k if args.record_prior_events else None,
            "vcd_noise_step": (
                args.vcd_noise_step
                if args.draft_guidance == "vcd" or args.refill_guidance == "vcd"
                else None
            ),
            "vcd_noise_seed": (
                args.vcd_noise_seed
                if (
                    args.vcd_noise_seed is not None
                    and (args.draft_guidance == "vcd" or args.refill_guidance == "vcd")
                )
                else None
            ),
            "region_num": args.region_num,
            "region_quantile": args.region_quantile,
            "region_weight": args.region_weight,
            "region_feather": args.region_feather,
            "region_min_highlight": args.region_min_highlight,
            "crop_margin": args.crop_margin,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote records to {records_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)
    restore_compile()


if __name__ == "__main__":
    apply_postvrg_defaults()
    main()
