import argparse
import copy
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MASK_TOKEN_ID = 126336


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize TextVQA visual attention before/after direction_debias."
    )
    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--image", default='images/dog.png')
    parser.add_argument("--question", default='Please Decribe the image in detail.')
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--target-step", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--layer", default="last", help="Layer to visualize: last or a zero-based layer index.")
    parser.add_argument("--selector", default="top_attn", choices=["top_attn", "top_cos", "positive_cos"])
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument(
        "--cos-threshold",
        type=float,
        default=0.0,
        help="For --selector positive_cos, debias visual keys whose mean query-key cosine is above this value.",
    )
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--query-scope", default="all_decode", choices=["selected", "all_decode"])
    parser.add_argument("--no-visual-renorm", action="store_true")
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output", default="Sink/direction_debias_heatmap.png")
    return parser.parse_args()


def get_torch_dtype(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_prompt(question, conv_template):
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def compute_visual_token_info(model):
    vision_tower = model.get_vision_tower()
    if hasattr(vision_tower, "num_patches_per_side"):
        patches_per_side = int(vision_tower.num_patches_per_side)
        total_tokens = patches_per_side * patches_per_side
    elif hasattr(vision_tower, "num_patches"):
        total_tokens = int(vision_tower.num_patches)
        patches_per_side = int(math.sqrt(total_tokens))
        if patches_per_side * patches_per_side != total_tokens:
            raise ValueError(f"Vision tower has non-square visual token count: {total_tokens}")
    else:
        raise ValueError("Cannot infer visual token grid from vision tower.")
    return {"height": patches_per_side, "width": patches_per_side, "total_tokens": total_tokens}


def resolve_patch_visual_positions(image_positions, vis_info):
    expected = int(vis_info["total_tokens"])
    if len(image_positions) not in {expected, expected + 1}:
        raise ValueError(
            f"Cannot project visual tokens to a fixed image grid: expanded prompt has {len(image_positions)} "
            f"IMAGE_TOKEN_INDEX positions, but the vision tower reports {expected} patch tokens. "
            "This script supports flat 729-token single-image inputs and spatial_unpad single-image inputs "
            "with one extra newline token."
        )
    if image_positions != list(range(image_positions[0], image_positions[0] + len(image_positions))):
        raise ValueError("Visual token positions are not contiguous; refusing to reshape them into an image grid.")
    return {
        "patch_positions": image_positions[:expected],
        "excluded_positions": image_positions[expected:],
    }


def prepare_multimodal_prefix_from_image(model, tokenizer, image_processor, image, prompt_text, device, dtype):
    from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token

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

    image_positions = torch.where(prefix_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
    vis_info = compute_visual_token_info(model)
    visual_layout = resolve_patch_visual_positions(image_positions, vis_info)
    return prefix_embeds, visual_layout, vis_info


def resolve_layer_index(model, layer_spec):
    blocks = model.get_model().transformer.blocks
    if layer_spec == "last":
        return len(blocks) - 1
    layer_idx = int(layer_spec)
    if layer_idx < 0:
        layer_idx = len(blocks) + layer_idx
    if layer_idx < 0 or layer_idx >= len(blocks):
        raise ValueError(f"Layer index out of range: {layer_spec}")
    return layer_idx


def repeat_kv(q, k, v):
    if q.shape[1] == k.shape[1]:
        return q, k, v
    repeat = q.shape[1] // k.shape[1]
    return q, k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)


def reconstruct_qkv(block, raw_q, raw_k, raw_v, attention_bias=None, layer_past=None):
    bsz, q_len, hidden = raw_q.shape
    cfg = block.config
    dtype = raw_k.dtype
    q = raw_q
    k = raw_k
    v = raw_v

    if getattr(block, "q_norm", None) is not None and getattr(block, "k_norm", None) is not None:
        q = block.q_norm(q).to(dtype=dtype)
        k = block.k_norm(k).to(dtype=dtype)

    head_dim = hidden // cfg.n_heads
    q = q.view(bsz, q_len, cfg.n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, k.shape[1], cfg.effective_n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, v.shape[1], cfg.effective_n_kv_heads, head_dim).transpose(1, 2)
    if layer_past is not None:
        past_key, past_value = layer_past
        k = torch.cat((past_key, k), dim=-2)
        v = torch.cat((past_value, v), dim=-2)

    if cfg.rope and hasattr(block, "rotary_emb"):
        q, k = block.rotary_emb(q, k)

    q, k, v = repeat_kv(q, k, v)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    if attention_bias is not None:
        q_now, k_now = scores.shape[-2], scores.shape[-1]
        scores = scores + attention_bias[:, :, k_now - q_now : k_now, :k_now].to(dtype=scores.dtype, device=scores.device)
    return q, k, v, scores


@contextmanager
def capture_layer_tensors(model, layer_idx):
    core = model.get_model()
    block = core.transformer.blocks[layer_idx]
    state = {"raw_q": None, "raw_k": None, "raw_v": None, "attention_bias": None, "layer_past": None}
    original_attention = block.attention

    def wrapped_attention(
        self_block,
        q,
        k,
        v,
        attention_bias=None,
        layer_past=None,
        use_cache=False,
        block_mask=None,
    ):
        state["raw_q"] = q.detach()
        state["raw_k"] = k.detach()
        state["raw_v"] = v.detach()
        state["attention_bias"] = attention_bias.detach() if torch.is_tensor(attention_bias) else attention_bias
        state["layer_past"] = layer_past
        att, present = original_attention(
            q,
            k,
            v,
            attention_bias=attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
            block_mask=block_mask,
        )
        return att, present

    block.attention = MethodType(wrapped_attention, block)

    try:
        yield block, state
    finally:
        block.attention = original_attention


def normalize_map(attn_map):
    attn_map = attn_map.astype(np.float32)
    lo = float(attn_map.min())
    hi = float(attn_map.max())
    if hi - lo <= 1e-8:
        return np.zeros_like(attn_map, dtype=np.float32)
    return (attn_map - lo) / (hi - lo)


def resize_map_to_image(attn_map, image):
    heat_img = Image.fromarray((normalize_map(attn_map) * 255.0).astype(np.uint8))
    heat_img = heat_img.resize(image.size, Image.BILINEAR)
    return np.array(heat_img, dtype=np.float32) / 255.0


def select_sink_positions(scores, q, k, selected_queries, visual_positions, selector, topk, cos_threshold):
    query_idx = torch.as_tensor(selected_queries, dtype=torch.long, device=scores.device)
    visual_idx = torch.as_tensor(visual_positions, dtype=torch.long, device=scores.device)
    visual_scores = scores.index_select(2, query_idx).index_select(3, visual_idx)
    visual_attn = F.softmax(visual_scores.to(torch.float32), dim=-1)
    attn_mean = visual_attn.mean(dim=(0, 1, 2))

    q_sel = q.index_select(2, query_idx)
    k_sel = k.index_select(2, visual_idx)
    q_unit = F.normalize(q_sel.to(torch.float32), p=2, dim=-1, eps=1e-12)
    k_unit = F.normalize(k_sel.to(torch.float32), p=2, dim=-1, eps=1e-12)
    cos_mean = (q_unit.unsqueeze(-2) * k_unit.unsqueeze(-3)).sum(dim=-1).mean(dim=(0, 1, 2))

    if selector == "positive_cos":
        local_indices = torch.where(cos_mean > float(cos_threshold))[0]
        if local_indices.numel() == 0:
            local_indices = torch.argsort(cos_mean, descending=True)[:1]
    else:
        rank_source = attn_mean if selector == "top_attn" else cos_mean
        local_indices = torch.argsort(rank_source, descending=True)[: min(topk, int(rank_source.numel()))]
    selected_positions = visual_idx.index_select(0, local_indices)
    return selected_positions, local_indices.detach().cpu(), attn_mean.detach().cpu(), cos_mean.detach().cpu()


def apply_direction_debias(q, k, selected_positions, alpha):
    mean_q = q.to(torch.float32).mean(dim=-2, keepdim=True)
    mean_q = F.normalize(mean_q, p=2, dim=-1, eps=1e-12).to(dtype=k.dtype)
    selected_k = k.index_select(-2, selected_positions)
    projection = (selected_k * mean_q).sum(dim=-1, keepdim=True) * mean_q
    debiased_k = selected_k - float(alpha) * projection
    updated_k = k.clone()
    updated_k.index_copy_(-2, selected_positions, debiased_k)
    return updated_k


def visual_heatmap_from_scores(scores, selected_queries, visual_positions, grid_shape, visual_renorm):
    query_idx = torch.as_tensor(selected_queries, dtype=torch.long, device=scores.device)
    visual_idx = torch.as_tensor(visual_positions, dtype=torch.long, device=scores.device)
    probs = F.softmax(scores.to(torch.float32), dim=-1)
    values = probs[0, :, :, :].index_select(1, query_idx).index_select(2, visual_idx)
    if visual_renorm:
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return values.mean(dim=(0, 1)).view(grid_shape).detach().cpu().numpy()


def summarize_attention_shift(before, after, selected_sink_indices, topn=10):
    before_flat = before.reshape(-1).astype(np.float64)
    after_flat = after.reshape(-1).astype(np.float64)
    before_sum = max(float(before_flat.sum()), 1e-12)
    after_sum = max(float(after_flat.sum()), 1e-12)
    before_prob = before_flat / before_sum
    after_prob = after_flat / after_sum

    topn = min(int(topn), int(before_flat.size))
    before_top = np.argsort(-before_prob)[:topn]
    after_top = np.argsort(-after_prob)[:topn]
    before_top_set = set(int(x) for x in before_top.tolist())
    after_top_set = set(int(x) for x in after_top.tolist())
    overlap = sorted(before_top_set & after_top_set)
    selected_sink = [int(x) for x in selected_sink_indices]
    after_new_top = [int(x) for x in after_top.tolist() if int(x) not in selected_sink]

    def entropy(prob):
        prob = prob[prob > 0]
        return float(-(prob * np.log(prob)).sum())

    before_selected_mass = float(before_prob[selected_sink].sum()) if selected_sink else 0.0
    after_selected_mass = float(after_prob[selected_sink].sum()) if selected_sink else 0.0
    before_top_mass = float(before_prob[before_top].sum())
    after_top_mass = float(after_prob[after_top].sum())
    after_new_top1 = after_new_top[0] if after_new_top else int(after_top[0])

    return {
        "before_top10_indices": [int(x) for x in before_top.tolist()],
        "after_top10_indices": [int(x) for x in after_top.tolist()],
        "top10_overlap_indices": overlap,
        "top10_overlap_count": len(overlap),
        "before_entropy": entropy(before_prob),
        "after_entropy": entropy(after_prob),
        "entropy_delta": entropy(after_prob) - entropy(before_prob),
        "before_selected_sink_mass": before_selected_mass,
        "after_selected_sink_mass": after_selected_mass,
        "selected_sink_mass_delta": after_selected_mass - before_selected_mass,
        "before_top10_mass": before_top_mass,
        "after_top10_mass": after_top_mass,
        "after_new_top1_index": int(after_new_top1),
        "after_new_top1_mass": float(after_prob[after_new_top1]),
    }


def get_confidence(logits, x0, remasking):
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
    raise ValueError(f"Unsupported remasking: {remasking}")


def extract_debias_comparison(model, tokenizer, image_processor, args):
    from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch

    device = args.device
    dtype = get_torch_dtype(args.torch_dtype)
    image = Image.open(args.image).convert("RGB")
    prompt = build_prompt(args.question, args.conv_template)
    prefix_embeds, visual_layout, vis_info = prepare_multimodal_prefix_from_image(
        model, tokenizer, image_processor, image, prompt, device, dtype
    )
    visual_positions = visual_layout["patch_positions"]
    layer_idx = resolve_layer_index(model, args.layer)
    core = model.get_model()

    with torch.no_grad():
        past_key_values = core(None, input_embeddings=prefix_embeds, use_cache=True).attn_key_values
        x = torch.full((1, args.max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
        if args.max_new_tokens % args.block_length != 0:
            raise ValueError("max_new_tokens must be divisible by block_length.")

        num_blocks = args.max_new_tokens // args.block_length
        steps = args.max_new_tokens // num_blocks
        if args.step_ratio is not None:
            steps = int(steps * args.step_ratio)
        if steps <= 0:
            raise ValueError("The computed number of steps per block is 0.")

        schedule_kwargs = {"shift": args.schedule_shift} if args.schedule == "shift" else None
        total_decode_steps = 0

        for block_idx in range(num_blocks):
            block_slice = slice(block_idx * args.block_length, (block_idx + 1) * args.block_length)
            block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
            num_transfer_tokens = get_num_transfer_tokens_sch(
                block_mask_index, steps, schedule=args.schedule, schedule_kwargs=schedule_kwargs
            )

            for step_idx in range(num_transfer_tokens.shape[1]):
                mask_index = x == MASK_TOKEN_ID
                if mask_index[:, block_slice].sum().item() == 0:
                    continue

                current_embeds = core.transformer.wte(x)
                logits = core(None, input_embeddings=current_embeds, past_key_values=past_key_values).logits
                logits_with_noise = add_gumbel_noise(logits, temperature=args.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                x0_p = get_confidence(logits, x0, args.remasking)
                x0_p[:, (block_idx + 1) * args.block_length :] = -torch.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -torch.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for batch_idx in range(confidence.shape[0]):
                    k_transfer = int(num_transfer_tokens[batch_idx, step_idx].item())
                    _, select_index = torch.topk(confidence[batch_idx], k=k_transfer)
                    transfer_index[batch_idx, select_index] = True

                transfer_queries = torch.where(transfer_index[0])[0].tolist()
                selected_queries = transfer_queries
                if args.query_scope == "all_decode":
                    selected_queries = list(range(x.shape[1]))

                total_decode_steps += 1
                if total_decode_steps == args.target_step:
                    with capture_layer_tensors(model, layer_idx) as (block, state):
                        _ = core(None, input_embeddings=current_embeds, past_key_values=past_key_values).logits

                    if state["raw_q"] is None or state["raw_k"] is None or state["raw_v"] is None:
                        raise RuntimeError("Failed to capture Q/K/V tensors.")
                    q, k_attn, _, scores_before = reconstruct_qkv(
                        block,
                        state["raw_q"],
                        state["raw_k"],
                        state["raw_v"],
                        state["attention_bias"],
                        layer_past=state["layer_past"],
                    )
                    sink_positions, sink_local_indices, attn_mean, cos_mean = select_sink_positions(
                        scores_before,
                        q,
                        k_attn,
                        selected_queries,
                        visual_positions,
                        args.selector,
                        args.topk,
                        args.cos_threshold,
                    )
                    k_after = apply_direction_debias(q, k_attn, sink_positions, args.alpha)
                    scores_after = torch.matmul(q, k_after.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
                    if state["attention_bias"] is not None:
                        q_now, k_now = scores_after.shape[-2], scores_after.shape[-1]
                        scores_after = scores_after + state["attention_bias"][:, :, k_now - q_now : k_now, :k_now].to(
                            dtype=scores_after.dtype, device=scores_after.device
                        )

                    grid_shape = (vis_info["height"], vis_info["width"])
                    before = visual_heatmap_from_scores(
                        scores_before, selected_queries, visual_positions, grid_shape, not args.no_visual_renorm
                    )
                    after = visual_heatmap_from_scores(
                        scores_after, selected_queries, visual_positions, grid_shape, not args.no_visual_renorm
                    )
                    shift_stats = summarize_attention_shift(before, after, sink_local_indices.tolist(), topn=10)
                    token_text = tokenizer.batch_decode(
                        x0[0, transfer_queries].detach().cpu().unsqueeze(0),
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )[0].replace("<|endoftext|>", "").strip()
                    meta = {
                        "layer_idx": layer_idx,
                        "step": total_decode_steps,
                        "visual_tokens": len(visual_positions),
                        "excluded_visual_tokens": len(visual_layout["excluded_positions"]),
                        "excluded_visual_positions": visual_layout["excluded_positions"],
                        "grid": grid_shape,
                        "query_scope": args.query_scope,
                        "selector": args.selector,
                        "cos_threshold": args.cos_threshold,
                        "selected_sink_count": int(len(sink_local_indices)),
                        "sink_local_indices": sink_local_indices.tolist(),
                        "sink_positions": sink_positions.detach().cpu().tolist(),
                        "sink_attn_mean": attn_mean[sink_local_indices].tolist(),
                        "sink_cos_mean": cos_mean[sink_local_indices].tolist(),
                        "l1_delta": float(np.abs(after - before).sum()),
                        "max_abs_delta": float(np.abs(after - before).max()),
                        "generated_text": token_text,
                    }
                    meta.update(shift_stats)
                    return image, before, after, meta

                x[transfer_index] = x0[transfer_index]

    raise ValueError(f"target step {args.target_step} is out of range. Actual decode steps: {total_decode_steps}.")


def render_figure(image, before, after, meta, args):
    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    diff = after - before
    before_norm = normalize_map(before)
    after_norm = normalize_map(after)
    overlay_before = resize_map_to_image(before, image)
    overlay_after = resize_map_to_image(after, image)
    vmax = max(float(np.abs(diff).max()), 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), facecolor="white")
    axes[0, 0].imshow(image_array)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image_array)
    axes[0, 1].imshow(overlay_before, cmap=args.cmap, alpha=args.overlay_alpha, interpolation="bilinear")
    axes[0, 1].set_title("Before Debias Overlay")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(image_array)
    axes[0, 2].imshow(overlay_after, cmap=args.cmap, alpha=args.overlay_alpha, interpolation="bilinear")
    axes[0, 2].set_title(f"After Debias Overlay alpha={args.alpha:g}")
    axes[0, 2].axis("off")

    im0 = axes[1, 0].imshow(before_norm, cmap=args.cmap, interpolation="nearest")
    axes[1, 0].set_title("Before Debias Patch Heatmap")
    axes[1, 0].axis("off")
    fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im1 = axes[1, 1].imshow(after_norm, cmap=args.cmap, interpolation="nearest")
    axes[1, 1].set_title("After Debias Patch Heatmap")
    axes[1, 1].axis("off")
    fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 2].imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    axes[1, 2].set_title("After - Before")
    axes[1, 2].axis("off")
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Direction Debias Attention Change\n"
        f"layer={meta['layer_idx']}, step={meta['step']}, visual={meta['visual_tokens']} "
        f"({meta['grid'][0]}x{meta['grid'][1]} patches), excluded_newline={meta['excluded_visual_tokens']}, "
        f"query_scope={meta['query_scope']}, selector={meta['selector']}, "
        f"cos_threshold={meta['cos_threshold']:.3f}, selected={meta['selected_sink_count']}, "
        f"sink_local={meta['sink_local_indices'][:10]}, "
        f"L1 delta={meta['l1_delta']:.6f}, max delta={meta['max_abs_delta']:.6f}\n"
        f"top10 overlap={meta['top10_overlap_count']}/10, "
        f"entropy delta={meta['entropy_delta']:.4f}, "
        f"selected sink mass {meta['before_selected_sink_mass']:.4f}->{meta['after_selected_sink_mass']:.4f}, "
        f"new top1={meta['after_new_top1_index']}\n"
        f"Selected tokens: {meta['generated_text'] or '[empty]'}",
        fontsize=11,
        y=1.02,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
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
        device_map=f"{args.device}:0" if args.device.startswith("cuda") else args.device,
        vision_kwargs=vision_kwargs,
        torch_dtype=args.torch_dtype,
    )
    model.eval()
    model.tie_weights()
    model.to(get_torch_dtype(args.torch_dtype))

    image, before, after, meta = extract_debias_comparison(model, tokenizer, image_processor, args)
    render_figure(image, before, after, meta, args)

    print(f"Saved figure to {args.output}")
    print(
        f"Visual grid: {meta['visual_tokens']} patch tokens -> "
        f"{meta['grid'][0]}x{meta['grid'][1]}; "
        f"excluded newline/non-patch tokens: {meta['excluded_visual_tokens']} "
        f"{meta['excluded_visual_positions']}"
    )
    print(f"Query scope: {meta['query_scope']}")
    print(
        f"Layer: {meta['layer_idx']}; step: {meta['step']}; "
        f"selector={meta['selector']}; cos_threshold={meta['cos_threshold']}; "
        f"selected sink count={meta['selected_sink_count']}"
    )
    print(f"Selected sink local indices: {meta['sink_local_indices']}")
    print(f"L1 delta: {meta['l1_delta']:.8f}; max abs delta: {meta['max_abs_delta']:.8f}")
    print(f"Before top10 patch indices: {meta['before_top10_indices']}")
    print(f"After top10 patch indices: {meta['after_top10_indices']}")
    print(
        f"Top10 overlap: {meta['top10_overlap_count']}/10 "
        f"indices={meta['top10_overlap_indices']}"
    )
    print(
        f"Entropy before/after/delta: "
        f"{meta['before_entropy']:.8f} -> {meta['after_entropy']:.8f} "
        f"({meta['entropy_delta']:+.8f})"
    )
    print(
        f"Selected sink mass before/after/delta: "
        f"{meta['before_selected_sink_mass']:.8f} -> {meta['after_selected_sink_mass']:.8f} "
        f"({meta['selected_sink_mass_delta']:+.8f})"
    )
    print(
        f"Top10 mass before/after: "
        f"{meta['before_top10_mass']:.8f} -> {meta['after_top10_mass']:.8f}"
    )
    print(
        f"After new top1 excluding selected sinks: "
        f"index={meta['after_new_top1_index']}, mass={meta['after_new_top1_mass']:.8f}"
    )
    print(f"Selected tokens: {meta['generated_text'] or '[empty]'}")


if __name__ == "__main__":
    main()
