import argparse
import copy
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


MASK_TOKEN_ID = 126336


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Monkey-patch LaViDa attention with a TACA-style logit scaling rule. "
            "This first version applies the scaling at all denoising steps."
        )
    )
    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--image", default='images/dog.png')
    parser.add_argument("--question", default="Describe the image in detail.")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument("--gamma-prompt", type=float, default=1.5, help="TACA scaling factor for prompt-text logits.")
    parser.add_argument("--gamma-visual", type=float, default=1.5, help="TACA scaling factor for visual-token logits.")
    parser.add_argument("--scale-generated", type=float, default=1.0, help="Optional scaling factor for generated-text logits.")
    parser.add_argument("--output", default="Scale_Attention/taca_lavida_output.json")
    return parser.parse_args()


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def parse_bool_like(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_prompt(question: str, conv_template: str) -> str:
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def compute_visual_token_info(model):
    vision_tower = model.get_vision_tower()
    if hasattr(vision_tower, "num_patches_per_side"):
        patches_per_side = int(vision_tower.num_patches_per_side)
    elif hasattr(vision_tower, "num_patches"):
        num_patches = int(vision_tower.num_patches)
        patches_per_side = int(math.sqrt(num_patches))
    else:
        raise ValueError("Cannot infer visual token grid from vision tower.")

    return {
        "height": patches_per_side,
        "width": patches_per_side,
        "total_tokens": patches_per_side * patches_per_side,
    }


def resolve_visual_positions(image_positions, vis_info):
    expected = int(vis_info["total_tokens"])
    if expected > 0 and expected <= len(image_positions):
        return image_positions[:expected]
    return image_positions


def prepare_multimodal_prefix_from_image(model, tokenizer, image_processor, image, prompt_text, device, dtype):
    image = image.convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=dtype, device=device) for _image in image_tensor]

    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)

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
    prefix_embeds, prefix_input_ids = ret[4], ret[-1]

    valid_prefix = prefix_input_ids[0] != IGNORE_INDEX
    prefix_embeds = prefix_embeds[:, valid_prefix, :]
    prefix_input_ids = prefix_input_ids[0, valid_prefix]

    image_positions_raw = torch.where(prefix_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
    vis_info = compute_visual_token_info(model)
    visual_positions = resolve_visual_positions(image_positions_raw, vis_info)

    visual_mask = torch.zeros_like(prefix_input_ids, dtype=torch.bool)
    if visual_positions:
        visual_mask[torch.tensor(visual_positions, device=prefix_input_ids.device, dtype=torch.long)] = True

    prompt_text_mask = (~visual_mask) & (prefix_input_ids != IGNORE_INDEX)
    return prefix_embeds, prefix_input_ids, visual_mask, prompt_text_mask, image_tensor, input_ids


def build_taca_scales_from_multimodal_inputs(
    model,
    input_ids: torch.Tensor,
    images,
    image_sizes,
    gen_len: int,
    gamma_prompt: float,
    gamma_visual: float,
    scale_generated: float = 1.0,
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
    prefix_input_ids_full = ret[-1][0]

    visual_mask = prefix_input_ids_full == IMAGE_TOKEN_INDEX
    prompt_text_mask = (~visual_mask) & (prefix_input_ids_full != IGNORE_INDEX)

    prefix_len = int(prefix_input_ids_full.shape[0])
    total_len = prefix_len + int(gen_len)
    scales = torch.ones(total_len, dtype=torch.float32, device=prefix_input_ids_full.device)
    if prefix_len > 0:
        scales[:prefix_len][visual_mask] = float(gamma_visual)
        scales[:prefix_len][prompt_text_mask] = float(gamma_prompt)
    if int(gen_len) > 0 and float(scale_generated) != 1.0:
        scales[prefix_len:] = float(scale_generated)

    meta = {
        "prefix_len": prefix_len,
        "num_visual_tokens": int(visual_mask.sum().item()),
        "num_prompt_text_tokens": int(prompt_text_mask.sum().item()),
        "num_ignored_prefix_tokens": int((prefix_input_ids_full == IGNORE_INDEX).sum().item()),
        "num_generated_tokens": int(gen_len),
    }
    return scales, meta


def build_taca_scales(prefix_len: int, gen_len: int, visual_mask: torch.Tensor, prompt_text_mask: torch.Tensor, device: torch.device, args):
    total_len = prefix_len + gen_len
    scales = torch.ones(total_len, dtype=torch.float32, device=device)
    if prefix_len > 0:
        scales[:prefix_len][visual_mask.to(device=device)] = float(args.gamma_visual)
        scales[:prefix_len][prompt_text_mask.to(device=device)] = float(args.gamma_prompt)
    if gen_len > 0 and float(args.scale_generated) != 1.0:
        scales[prefix_len:] = float(args.scale_generated)
    return scales


@contextmanager
def patch_taca_attention(model, scales: torch.Tensor):
    blocks = model.get_model().transformer.blocks
    original_methods = {}
    runtime_state = {"layer_state": {}}

    for layer_idx, block in enumerate(blocks):
        original_methods[layer_idx] = block.attention
        layer_name = f"layer_{layer_idx}"
        runtime_state["layer_state"][layer_name] = {"call_id": 0, "decode_timestep": -1}

        def make_patched_attention(current_block, original_attention, layer_name_local):
            def patched_attention(self_block, q, k, v, attention_bias=None, layer_past=None, use_cache=False, block_mask=None):
                if block_mask is not None:
                    return original_attention(
                        q,
                        k,
                        v,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        block_mask=block_mask,
                    )

                layer_state = runtime_state["layer_state"][layer_name_local]
                call_id = int(layer_state["call_id"])
                is_decode_step = bool(layer_past is not None) or (call_id >= 1)
                layer_state["call_id"] = call_id + 1
                if is_decode_step:
                    layer_state["decode_timestep"] = int(layer_state["decode_timestep"]) + 1

                # Keep prefill untouched to match the original LaViDa path.
                if not is_decode_step:
                    return original_attention(
                        q,
                        k,
                        v,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        block_mask=block_mask,
                    )

                B, T, C = q.size()
                dtype = k.dtype

                if self_block.q_norm is not None and self_block.k_norm is not None:
                    q = self_block.q_norm(q).to(dtype=dtype)
                    k = self_block.k_norm(k).to(dtype=dtype)

                q = q.view(B, T, self_block.config.n_heads, C // self_block.config.n_heads).transpose(1, 2)
                k = k.view(B, T, self_block.config.effective_n_kv_heads, C // self_block.config.n_heads).transpose(1, 2)
                v = v.view(B, T, self_block.config.effective_n_kv_heads, C // self_block.config.n_heads).transpose(1, 2)

                if layer_past is not None:
                    past_key, past_value = layer_past
                    k = torch.cat((past_key, k), dim=-2)
                    v = torch.cat((past_value, v), dim=-2)

                present = (k, v) if use_cache else None
                query_len, key_len = q.shape[-2], k.shape[-2]

                if self_block.config.rope:
                    q, k = self_block.rotary_emb(q, k)

                if q.shape[1] != k.shape[1]:
                    repeat_factor = q.shape[1] // k.shape[1]
                    k = k.repeat_interleave(repeat_factor, dim=1)
                    v = v.repeat_interleave(repeat_factor, dim=1)

                scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))

                # Match the stable intervention script behavior:
                # keep decode-time intervention simple and avoid introducing
                # additional attention-bias handling differences.
                key_scales = scales.to(device=scores.device, dtype=scores.dtype)
                if key_scales.numel() < key_len:
                    pad = torch.ones(key_len - key_scales.numel(), dtype=scores.dtype, device=scores.device)
                    key_scales = torch.cat([key_scales, pad], dim=0)
                else:
                    key_scales = key_scales[:key_len]
                scores = scores * key_scales.view(1, 1, 1, key_len)

                probs = F.softmax(scores, dim=-1).to(dtype=v.dtype)
                att = torch.matmul(probs, v)
                att = att.transpose(1, 2).contiguous().view(B, T, C)
                return self_block.attn_out(att), present

            return patched_attention

        block.attention = MethodType(make_patched_attention(block, original_methods[layer_idx], layer_name), block)

    try:
        yield
    finally:
        for layer_idx, original_attention in original_methods.items():
            blocks[layer_idx].attention = original_attention


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


def main():
    args = parse_args()
    restore_compile = maybe_disable_torch_compile()

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
        device_map=f"{args.device}:0" if args.device.startswith("cuda") else args.device,
        vision_kwargs=vision_kwargs,
        torch_dtype=args.torch_dtype,
    )
    model.eval()
    model.tie_weights()
    model.to(get_torch_dtype(args.torch_dtype))

    image = Image.open(args.image).convert("RGB")
    prompt = build_prompt(args.question, args.conv_template)
    prefix_embeds, prefix_input_ids, visual_mask, prompt_text_mask, image_tensor, input_ids = prepare_multimodal_prefix_from_image(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image=image,
        prompt_text=prompt,
        device=args.device,
        dtype=get_torch_dtype(args.torch_dtype),
    )

    scales = build_taca_scales(
        prefix_len=int(prefix_input_ids.shape[0]),
        gen_len=int(args.max_new_tokens),
        visual_mask=visual_mask,
        prompt_text_mask=prompt_text_mask,
        device=prefix_embeds.device,
        args=args,
    )

    with patch_taca_attention(model, scales):
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                max_new_tokens=int(args.max_new_tokens),
                block_length=int(args.block_length),
                temperature=float(args.temperature),
                remasking=args.remasking,
                prefix_lm=True,
                schedule=args.schedule,
                schedule_kwargs={"shift": args.schedule_shift} if args.schedule == "shift" else None,
                step_ratio=float(args.step_ratio),
                tokenizer=tokenizer,
                verbose=False,
            )

    sequences = outputs[0] if isinstance(outputs, tuple) else outputs
    final_text = tokenizer.batch_decode(
        sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].replace("<|endoftext|>", "").strip()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": args.image,
        "question": args.question,
        "final_text": final_text,
        "gamma_prompt": float(args.gamma_prompt),
        "gamma_visual": float(args.gamma_visual),
        "scale_generated": float(args.scale_generated),
        "prefix_len": int(prefix_input_ids.shape[0]),
        "num_visual_tokens": int(visual_mask.sum().item()),
        "num_prompt_text_tokens": int(prompt_text_mask.sum().item()),
        "num_generated_tokens": int(args.max_new_tokens),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    restore_compile()
    print(f"Saved summary to {output_path}")
    print(f"Final text: {final_text}")


if __name__ == "__main__":
    main()
