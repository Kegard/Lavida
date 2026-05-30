import argparse
import copy
import json
import math
import os
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
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch


MASK_TOKEN_ID = 126336


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def env_flag_default(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def summarize_tensor(name: str, tensor: torch.Tensor) -> str:
    values = tensor.detach().to(dtype=torch.float32)
    finite = torch.isfinite(values)
    nonfinite = int((~finite).sum().item())
    if finite.any():
        finite_values = values[finite]
        return (
            f"{name}: mean={finite_values.mean().item():.4f} "
            f"std={finite_values.std(unbiased=False).item():.4f} "
            f"min={finite_values.min().item():.4f} "
            f"max={finite_values.max().item():.4f} "
            f"max_abs={finite_values.abs().max().item():.4f} "
            f"nonfinite={nonfinite}"
        )
    return f"{name}: all_nonfinite={values.numel()} nonfinite={nonfinite}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Monkey-patch LaViDa attention with post-softmax category reweighting. "
            "This version directly changes visual/prompt/mask/normal/special attention mass."
        )
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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument("--alpha-prompt", type=float, default=1.0, help="Post-softmax reweight factor for prompt-text attention mass.")
    parser.add_argument("--alpha-visual", type=float, default=1.0, help="Post-softmax reweight factor for visual attention mass.")
    parser.add_argument("--alpha-generated", type=float, default=1.0, help="Default post-softmax reweight factor for generated-token attention mass.")
    parser.add_argument("--alpha-mask", type=float, default=1.0, help="Post-softmax reweight factor for generated mask-token attention mass.")
    parser.add_argument("--alpha-normal", type=float, default=1.0, help="Post-softmax reweight factor for generated normal-token attention mass.")
    parser.add_argument("--alpha-special", type=float, default=1.0, help="Post-softmax reweight factor for generated special-token attention mass.")
    parser.add_argument("--output", default="Scale_Attention/category_reweight_lavida_output.json")
    return parser.parse_args()


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_prompt(question: str, conv_template: str) -> str:
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def get_special_token_ids(tokenizer):
    special_ids = getattr(tokenizer, "all_special_ids", None)
    if special_ids is None:
        return []
    return [int(token_id) for token_id in special_ids if token_id is not None]


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


def build_fine_category_weights(
    prefix_input_ids_full: torch.Tensor,
    gen_tokens: torch.Tensor,
    special_token_ids,
    alpha_prompt: float,
    alpha_visual: float,
    alpha_mask: float,
    alpha_normal: float,
    alpha_special: float,
):
    visual_mask = prefix_input_ids_full == IMAGE_TOKEN_INDEX
    prompt_text_mask = (~visual_mask) & (prefix_input_ids_full != IGNORE_INDEX)

    gen_tokens = gen_tokens.to(device=prefix_input_ids_full.device, dtype=torch.long).view(-1)
    generated_is_mask = gen_tokens == MASK_TOKEN_ID
    generated_is_special = torch.zeros_like(generated_is_mask)
    special_token_ids = [token_id for token_id in special_token_ids if token_id != MASK_TOKEN_ID]
    if special_token_ids:
        special_ids = torch.tensor(special_token_ids, dtype=torch.long, device=gen_tokens.device)
        generated_is_special = torch.isin(gen_tokens, special_ids) & (~generated_is_mask)
    generated_is_normal = (~generated_is_mask) & (~generated_is_special)

    prefix_len = int(prefix_input_ids_full.shape[0])
    gen_len = int(gen_tokens.shape[0])
    weights = torch.ones(prefix_len + gen_len, dtype=torch.float32, device=prefix_input_ids_full.device)
    if prefix_len > 0:
        weights[:prefix_len][visual_mask] = float(alpha_visual)
        weights[:prefix_len][prompt_text_mask] = float(alpha_prompt)
    if gen_len > 0:
        gen_weights = weights[prefix_len:]
        gen_weights[generated_is_mask] = float(alpha_mask)
        gen_weights[generated_is_special] = float(alpha_special)
        gen_weights[generated_is_normal] = float(alpha_normal)
    if torch.any(weights < 0):
        raise ValueError("Category alpha values must be non-negative for attention-logit biasing.")

    meta = {
        "prefix_len": prefix_len,
        "num_visual_tokens": int(visual_mask.sum().item()),
        "num_prompt_text_tokens": int(prompt_text_mask.sum().item()),
        "num_ignored_prefix_tokens": int((prefix_input_ids_full == IGNORE_INDEX).sum().item()),
        "num_generated_tokens": gen_len,
        "num_generated_mask_tokens": int(generated_is_mask.sum().item()),
        "num_generated_special_tokens": int(generated_is_special.sum().item()),
        "num_generated_normal_tokens": int(generated_is_normal.sum().item()),
    }
    return weights, meta


def compute_remasking_confidence(logits: torch.Tensor, x0: torch.Tensor, remasking: str) -> torch.Tensor:
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


def generate_with_dynamic_category_reweighting(
    core_model,
    prefix_embeds: torch.Tensor,
    prefix_input_ids_full: torch.Tensor,
    category_weight_state: dict,
    special_token_ids,
    max_new_tokens: int,
    block_length: int,
    temperature: float,
    remasking: str,
    schedule: str,
    schedule_shift: float,
    step_ratio: float,
    alpha_prompt: float,
    alpha_visual: float,
    alpha_mask: float,
    alpha_normal: float,
    alpha_special: float,
):
    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")

    device = prefix_embeds.device
    past_key_values = core_model(None, input_embeddings=prefix_embeds, use_cache=True).attn_key_values
    x = torch.full((prefix_embeds.shape[0], max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)

    num_blocks = max_new_tokens // block_length
    steps = max_new_tokens // num_blocks
    if step_ratio:
        steps = int(steps * step_ratio)
    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0. Increase step_ratio or max_new_tokens.")

    schedule_kwargs = {"shift": schedule_shift} if schedule == "shift" else None
    latest_meta = {}
    for block_idx in range(num_blocks):
        block_slice = slice(block_idx * block_length, (block_idx + 1) * block_length)
        block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
        num_transfer_tokens = get_num_transfer_tokens_sch(
            block_mask_index,
            steps,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs,
        )

        for step_idx in range(num_transfer_tokens.shape[1]):
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_slice]
            if block_mask_index.sum().item() == 0:
                continue

            weights, latest_meta = build_fine_category_weights(
                prefix_input_ids_full=prefix_input_ids_full,
                gen_tokens=x[0],
                special_token_ids=special_token_ids,
                alpha_prompt=alpha_prompt,
                alpha_visual=alpha_visual,
                alpha_mask=alpha_mask,
                alpha_normal=alpha_normal,
                alpha_special=alpha_special,
            )
            category_weight_state["weights"] = weights
            category_weight_state["query_is_mask"] = mask_index[0]

            current_embeds = core_model.transformer.wte(x)
            logits = core_model(None, input_embeddings=current_embeds, past_key_values=past_key_values).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0_p = compute_remasking_confidence(logits, x0, remasking)

            x0_p[:, (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for batch_idx in range(confidence.shape[0]):
                k = int(num_transfer_tokens[batch_idx, step_idx].item())
                _, select_index = torch.topk(confidence[batch_idx], k=k)
                transfer_index[batch_idx, select_index] = True
            x[transfer_index] = x0[transfer_index]

    final_weights, final_meta = build_fine_category_weights(
        prefix_input_ids_full=prefix_input_ids_full,
        gen_tokens=x[0],
        special_token_ids=special_token_ids,
        alpha_prompt=alpha_prompt,
        alpha_visual=alpha_visual,
        alpha_mask=alpha_mask,
        alpha_normal=alpha_normal,
        alpha_special=alpha_special,
    )
    category_weight_state["weights"] = final_weights
    category_weight_state["query_is_mask"] = x[0] == MASK_TOKEN_ID
    return x, latest_meta, final_meta


@contextmanager
def patch_category_reweight_attention(model, category_weight_state: dict):
    blocks = model.get_model().transformer.blocks
    original_methods = {}
    runtime_state = {"layer_state": {}}
    debug_scores = env_flag_default("REWEIGHT_DEBUG_SCORES", False)
    debug_score_limit = int(os.environ.get("REWEIGHT_DEBUG_SCORE_LIMIT", "1"))

    for layer_idx, block in enumerate(blocks):
        original_methods[layer_idx] = block.attention
        layer_name = f"layer_{layer_idx}"
        runtime_state["layer_state"][layer_name] = {"call_id": 0, "score_debug_prints": 0}

        def make_patched_attention(original_attention, layer_name_local):
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

                # Keep prefill untouched.
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
                if self_block.config.rope:
                    q, k = self_block.rotary_emb(q, k)

                if q.shape[1] != k.shape[1]:
                    repeat_factor = q.shape[1] // k.shape[1]
                    k = k.repeat_interleave(repeat_factor, dim=1)
                    v = v.repeat_interleave(repeat_factor, dim=1)

                scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
                raw_scores = scores
                key_len = scores.shape[-1]
                weights = category_weight_state["weights"].to(device=scores.device, dtype=scores.dtype)
                if weights.numel() < key_len:
                    pad = torch.ones(key_len - weights.numel(), dtype=scores.dtype, device=scores.device)
                    weights = torch.cat([weights, pad], dim=0)
                else:
                    weights = weights[:key_len]

                if torch.any(weights < 0):
                    raise ValueError("Category alpha values must be non-negative for attention-logit biasing.")
                if torch.max(weights) <= 0:
                    raise ValueError("At least one active attention category must have alpha > 0.")

                neg_inf = torch.full_like(weights, -torch.inf)
                logit_bias = torch.where(weights > 0, torch.log(weights), neg_inf)
                query_is_mask = category_weight_state.get("query_is_mask")
                active_query_count = T
                if query_is_mask is None:
                    scores = scores + logit_bias.view(1, 1, 1, key_len)
                else:
                    query_is_mask = query_is_mask.to(device=scores.device, dtype=torch.bool).view(-1)
                    if query_is_mask.numel() < T:
                        query_pad = torch.zeros(T - query_is_mask.numel(), dtype=torch.bool, device=scores.device)
                        query_is_mask = torch.cat([query_is_mask, query_pad], dim=0)
                    else:
                        query_is_mask = query_is_mask[:T]
                    active_query_count = int(query_is_mask.sum().item())
                    row_bias = torch.where(
                        query_is_mask.view(T, 1),
                        logit_bias.view(1, key_len),
                        torch.zeros((T, key_len), dtype=logit_bias.dtype, device=logit_bias.device),
                    )
                    scores = scores + row_bias.view(1, 1, T, key_len)
                if debug_scores and layer_state["score_debug_prints"] < debug_score_limit:
                    layer_state["score_debug_prints"] += 1
                    print(
                        "[ReweightDebug] "
                        f"{layer_name_local} call={call_id} B={B} H={q.shape[1]} T={T} K={key_len} "
                        f"active_mask_queries={active_query_count}/{T} "
                        f"{summarize_tensor('raw_scores', raw_scores)} "
                        f"{summarize_tensor('logit_bias', logit_bias)} "
                        f"{summarize_tensor('biased_scores', scores)}",
                        flush=True,
                    )
                probs = F.softmax(scores, dim=-1)
                attention_store = category_weight_state.get("attention_store")
                if attention_store is not None:
                    attention_store.setdefault(layer_name_local, []).append(probs.detach().cpu())
                probs = probs.to(dtype=v.dtype)

                att = torch.matmul(probs, v)
                att = att.transpose(1, 2).contiguous().view(B, T, C)
                return self_block.attn_out(att), present

            return patched_attention

        block.attention = MethodType(make_patched_attention(original_methods[layer_idx], layer_name), block)

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
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=get_torch_dtype(args.torch_dtype), device=args.device) for _image in image_tensor]

    prompt = build_prompt(args.question, args.conv_template)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)

    alpha_mask = float(args.alpha_generated if args.alpha_mask is None else args.alpha_mask)
    alpha_normal = float(args.alpha_generated if args.alpha_normal is None else args.alpha_normal)
    alpha_special = float(args.alpha_generated if args.alpha_special is None else args.alpha_special)

    prefix_embeds, prefix_input_ids_full = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        attention_mask=attention_mask,
    )
    special_token_ids = get_special_token_ids(tokenizer)
    initial_weights, initial_meta = build_fine_category_weights(
        prefix_input_ids_full=prefix_input_ids_full,
        gen_tokens=torch.full((int(args.max_new_tokens),), MASK_TOKEN_ID, dtype=torch.long, device=args.device),
        special_token_ids=special_token_ids,
        alpha_prompt=float(args.alpha_prompt),
        alpha_visual=float(args.alpha_visual),
        alpha_mask=alpha_mask,
        alpha_normal=alpha_normal,
        alpha_special=alpha_special,
    )
    category_weight_state = {"weights": initial_weights}

    core_model = model.get_model()
    with patch_category_reweight_attention(model, category_weight_state):
        with torch.no_grad():
            sequences, last_step_meta, final_meta = generate_with_dynamic_category_reweighting(
                core_model=core_model,
                prefix_embeds=prefix_embeds,
                prefix_input_ids_full=prefix_input_ids_full,
                category_weight_state=category_weight_state,
                special_token_ids=special_token_ids,
                max_new_tokens=int(args.max_new_tokens),
                block_length=int(args.block_length),
                temperature=float(args.temperature),
                remasking=args.remasking,
                schedule=args.schedule,
                schedule_shift=float(args.schedule_shift),
                step_ratio=float(args.step_ratio),
                alpha_prompt=float(args.alpha_prompt),
                alpha_visual=float(args.alpha_visual),
                alpha_mask=alpha_mask,
                alpha_normal=alpha_normal,
                alpha_special=alpha_special,
            )

    final_text = tokenizer.batch_decode(
        sequences,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].replace("<|endoftext|>", "").strip()

    payload = {
        "image": args.image,
        "question": args.question,
        "final_text": final_text,
        "alpha_prompt": float(args.alpha_prompt),
        "alpha_visual": float(args.alpha_visual),
        "alpha_generated": float(args.alpha_generated),
        "alpha_mask": alpha_mask,
        "alpha_normal": alpha_normal,
        "alpha_special": alpha_special,
        "initial_token_counts": initial_meta,
        "last_step_token_counts": last_step_meta,
        "final_token_counts": final_meta,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    restore_compile()
    print(f"Saved summary to {output_path}")
    print(f"Final text: {final_text}")


if __name__ == "__main__":
    main()
