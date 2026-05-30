#!/usr/bin/env python3
"""
Prefill visual logit boost for LaViDa.

Algorithm:
- Build the multimodal prefix with the shared helper from reweight_patch.py.
- Locate the visual token span and all text query positions in the expanded prefix.
- Monkey-patch attention so that, only on the first prefill attention call of
  each layer, the text-query -> visual-key attention logits are multiplied by
  gamma before softmax.
- All later decode calls remain unchanged.

The script follows the same single-run shape as reweight_patch.py.
"""

import argparse
import copy
import json
import math
import os,sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
from Scale_Attention.reweight_patch import build_prefix_from_multimodal_inputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monkey-patch LaViDa attention with prefill-time visual logit boost."
    )
    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--image", default="images/dog.png")
    parser.add_argument("--question", default="Describe the image in detail.")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="Multiplicative factor for prefill text-query -> visual-key logits.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--output", default="Scale_Attention/reweight_prefill_output.json")
    return parser.parse_args()


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_prompt(question: str, conv_template: str) -> str:
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _infer_prefill_layout(prefix_input_ids_full: torch.Tensor):
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

    valid_ids = prefix_input_ids_full[prefix_input_ids_full != IGNORE_INDEX]
    vis_pos = (valid_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if vis_pos.numel() == 0:
        raise RuntimeError("No visual token positions found in the expanded multimodal prefix.")

    vis_start = int(vis_pos[0].item())
    vis_end = int(vis_pos[-1].item()) + 1
    text_query_positions = [int(i) for i, tid in enumerate(valid_ids.tolist()) if tid != IMAGE_TOKEN_INDEX]
    return vis_start, vis_end, text_query_positions


def _prefill_attention_boost(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_name: str,
    layer_state: Dict[str, Dict[str, int]],
    vis_start: int,
    vis_end: int,
    text_query_positions: List[int],
    gamma: float,
    has_layer_past: bool,
    attention_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    current_layer_state = layer_state.setdefault(layer_name, {"call_id": 0})
    call_id = int(current_layer_state["call_id"])
    is_prefill_step = (not has_layer_past) and (call_id == 0)

    num_q_heads = q.size(1)
    num_kv_heads = k.size(1)
    if num_q_heads != num_kv_heads:
        k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

    scale = 1.0 / math.sqrt(q.size(-1))
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if attention_bias is not None:
        q_len, k_len = attn_scores.shape[-2], attn_scores.shape[-1]
        bias = attention_bias[:, :, k_len - q_len : k_len, :k_len]
        attn_scores = attn_scores + bias.to(dtype=attn_scores.dtype, device=attn_scores.device)

    vs = max(0, min(int(vis_start), attn_scores.shape[-1]))
    ve = max(vs, min(int(vis_end), attn_scores.shape[-1]))
    text_q = [idx for idx in text_query_positions if 0 <= int(idx) < attn_scores.shape[-2]]

    if is_prefill_step and vs < ve and text_q:
        q_idx = torch.as_tensor(text_q, device=attn_scores.device, dtype=torch.long)
        attn_scores[:, :, q_idx, vs:ve] = attn_scores[:, :, q_idx, vs:ve] * float(gamma)

    attn_weights = F.softmax(attn_scores, dim=-1)
    current_layer_state["call_id"] = call_id + 1
    return torch.matmul(attn_weights, v)


@contextmanager
def patch_attention_prefill_boost(
    model,
    layers_to_patch: List[int],
    vis_start: int,
    vis_end: int,
    text_query_positions: List[int],
    gamma: float,
):
    layer_state: Dict[str, Dict[str, int]] = {}
    blocks = model.get_model().transformer.blocks
    original_methods: Dict[int, Any] = {}

    for layer_idx in layers_to_patch:
        block = blocks[layer_idx]
        original_methods[layer_idx] = block.attention
        layer_name = f"layer_{layer_idx}"

        def make_patched_attention(block_module, layer_name_local, original_attention):
            def patched_attention(self_block, q, k, v, attention_bias=None, layer_past=None, use_cache=False, block_mask=None):
                if layer_past is not None:
                    # Only the first prefill call is modified. Decode stays untouched.
                    return original_attention(
                        q,
                        k,
                        v,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        block_mask=block_mask,
                    )

                current_layer_state = layer_state.setdefault(layer_name_local, {"call_id": 0})
                if int(current_layer_state["call_id"]) > 0:
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
                if block_module.q_norm is not None and block_module.k_norm is not None:
                    q = block_module.q_norm(q).to(dtype=dtype)
                    k = block_module.k_norm(k).to(dtype=dtype)

                q = q.view(B, T, block_module.config.n_heads, C // block_module.config.n_heads).transpose(1, 2)
                k = k.view(B, T, block_module.config.effective_n_kv_heads, C // block_module.config.n_heads).transpose(1, 2)
                v = v.view(B, T, block_module.config.effective_n_kv_heads, C // block_module.config.n_heads).transpose(1, 2)

                present = (k, v) if use_cache else None
                if block_module.config.rope:
                    q, k = block_module.rotary_emb(q, k)

                # First prefill call: use the manual attention path so we can
                # bias only the text->visual submatrix before softmax.
                att = _prefill_attention_boost(
                    q=q,
                    k=k,
                    v=v,
                    layer_name=layer_name_local,
                    layer_state=layer_state,
                    vis_start=vis_start,
                    vis_end=vis_end,
                    text_query_positions=text_query_positions,
                    gamma=gamma,
                    has_layer_past=False,
                    attention_bias=attention_bias,
                )
                att = att.transpose(1, 2).contiguous().view(B, T, C)
                return block_module.attn_out(att), present

            return patched_attention

        block.attention = MethodType(make_patched_attention(block, layer_name, original_methods[layer_idx]), block)

    try:
        yield {"layer_state": layer_state}
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


def sanitize_for_json(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def main():
    args = parse_args()
    restore_compile = maybe_disable_torch_compile()

    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    vision_kwargs = dict(
        mm_vision_tower=args.vision_tower,
        mm_resampler_type=None,
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=1152,
        use_mm_proj=True,
    )
    device_map = f"{args.device}:0" if args.device.startswith("cuda") else args.device
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        None,
        args.model_name,
        device_map=device_map,
        vision_kwargs=vision_kwargs,
        torch_dtype=args.torch_dtype,
    )
    model.eval()
    model.tie_weights()
    model.to(get_torch_dtype(args.torch_dtype))

    image = Image.open(args.image).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=get_torch_dtype(args.torch_dtype), device=args.device) for _image in image_tensor]

    prompt = build_prompt(args.question, args.conv_template)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)

    prefix_embeds, prefix_input_ids_full = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        attention_mask=attention_mask,
    )
    vis_start, vis_end, text_query_positions = _infer_prefill_layout(prefix_input_ids_full)
    print(f"[info] visual span: [{vis_start}, {vis_end}), text_query_count={len(text_query_positions)}")

    with patch_attention_prefill_boost(
        model,
        list(range(len(model.get_model().transformer.blocks))),
        vis_start=vis_start,
        vis_end=vis_end,
        text_query_positions=text_query_positions,
        gamma=float(args.gamma),
    ) as runtime_state:
        with torch.no_grad():
            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                temperature=float(args.temperature),
                max_new_tokens=int(args.max_new_tokens),
                block_length=int(args.block_length),
                step_ratio=float(args.step_ratio),
                tokenizer=tokenizer,
                prefix_lm=True,
                verbose=True,
                schedule=args.schedule,
                schedule__shift=float(args.schedule_shift),
            )
        history = runtime_state

    if isinstance(cont, tuple):
        cont, generated_history = cont
        if isinstance(history, dict) and "gamma_is_one" in history:
            history["generate_history"] = generated_history
        else:
            history = {"runtime_state": history, "generate_history": generated_history}

    final_text = tokenizer.batch_decode(
        cont,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].replace("<|endoftext|>", "").strip()

    payload = {
        "image": args.image,
        "question": args.question,
        "final_text": final_text,
        "gamma": float(args.gamma),
        "vis_span": {"start": vis_start, "end": vis_end},
        "text_query_count": len(text_query_positions),
        "history": sanitize_for_json(history),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    restore_compile()
    print(f"Saved summary to {output_path}")
    print(f"Final text: {final_text}")


if __name__ == "__main__":
    main()
