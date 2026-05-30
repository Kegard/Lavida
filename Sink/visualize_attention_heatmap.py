import argparse
import copy
import math
import sys
from contextlib import contextmanager
from pathlib import Path

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

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch


MASK_TOKEN_ID = 126336


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a standard attention heatmap for one denoising step in LaViDa "
            "discrete diffusion generation."
        )
    )
    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--image", required=True)
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
    parser.add_argument("--step-ratio", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--remasking",
        default="low_confidence",
        choices=["low_confidence", "random", "entrophy", "margin"],
    )
    parser.add_argument(
        "--schedule",
        default="none",
        choices=["shift", "cosine", "logit_normal", "none"],
    )
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--target-step", type=int, default=8)
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument(
        "--no-visual-renorm",
        action="store_true",
        help="Disable renormalization inside the visual-token region.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output", default="Sink/attention_heatmap_step8.png")
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
    return prefix_embeds, visual_positions, vis_info


def reconstruct_attention_weights(
    block_module,
    raw_q: torch.Tensor,
    raw_k: torch.Tensor,
    raw_v: torch.Tensor,
    layer_past=None,
    attention_bias=None,
    full_k: torch.Tensor = None,
    full_v: torch.Tensor = None,
) -> torch.Tensor:
    bsz, q_len, hidden = raw_q.shape
    cfg = block_module.config
    dtype = full_k.dtype if full_k is not None else raw_k.dtype

    q = raw_q
    k = full_k if full_k is not None else raw_k
    v = full_v if full_v is not None else raw_v

    if getattr(block_module, "q_norm", None) is not None and getattr(block_module, "k_norm", None) is not None:
        q = block_module.q_norm(q).to(dtype=dtype)
        k = block_module.k_norm(k).to(dtype=dtype)

    n_heads = cfg.n_heads
    n_kv_heads = cfg.effective_n_kv_heads
    head_dim = hidden // n_heads

    q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    if full_k is not None and full_v is not None:
        k = k.view(bsz, k.shape[1], n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, v.shape[1], n_kv_heads, head_dim).transpose(1, 2)
    else:
        k = k.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

    if cfg.rope and hasattr(block_module, "rotary_emb"):
        q, k = block_module.rotary_emb(q, k)

    if q.size(1) != k.size(1):
        repeat_factor = q.size(1) // k.size(1)
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
    if attention_bias is not None:
        q_now, k_now = scores.shape[-2], scores.shape[-1]
        bias = attention_bias[:, :, k_now - q_now : k_now, :k_now]
        scores = scores + bias.to(dtype=scores.dtype, device=scores.device)
    return F.softmax(scores, dim=-1).detach().cpu()


@contextmanager
def capture_attention_readonly(model, layers_to_capture):
    store = {}
    model_core = model.get_model() if hasattr(model, "get_model") else model
    blocks = model_core.transformer.blocks
    hooks = []
    layer_states = {}

    for layer_idx in layers_to_capture:
        block = blocks[layer_idx]
        layer_name = f"layer_{layer_idx}"
        store[layer_name] = []
        layer_states[layer_name] = {
            "raw_q": None,
            "raw_k": None,
            "raw_v": None,
            "attention_bias": None,
            "layer_past": None,
        }

        def make_pre_hook(current_layer_name):
            def _pre_hook(module, args, kwargs):
                state = layer_states[current_layer_name]
                state["raw_q"] = None
                state["raw_k"] = None
                state["raw_v"] = None
                state["attention_bias"] = kwargs.get("attention_bias", None)
                if "layer_past" in kwargs:
                    state["layer_past"] = kwargs["layer_past"]
                elif len(args) >= 3:
                    state["layer_past"] = args[2]
                else:
                    state["layer_past"] = None
            return _pre_hook

        def make_post_hook(current_layer_name):
            def _post_hook(module, args, output):
                state = layer_states[current_layer_name]
                if state["raw_q"] is None or state["raw_k"] is None or state["raw_v"] is None:
                    return

                full_k = None
                full_v = None
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    full_k, full_v = output[1]

                weights = reconstruct_attention_weights(
                    block_module=module,
                    raw_q=state["raw_q"],
                    raw_k=state["raw_k"],
                    raw_v=state["raw_v"],
                    layer_past=state["layer_past"],
                    attention_bias=state["attention_bias"],
                    full_k=None
                    if full_k is None
                    else full_k.transpose(1, 2).contiguous().view(full_k.shape[0], full_k.shape[2], -1),
                    full_v=None
                    if full_v is None
                    else full_v.transpose(1, 2).contiguous().view(full_v.shape[0], full_v.shape[2], -1),
                )
                store[current_layer_name].append(weights)
            return _post_hook

        hooks.append(block.register_forward_pre_hook(make_pre_hook(layer_name), with_kwargs=True))
        hooks.append(block.register_forward_hook(make_post_hook(layer_name)))

        if hasattr(block, "att_proj"):
            fused_dims = tuple(block.fused_dims)

            def make_att_proj_hook(current_layer_name, current_fused_dims):
                def _hook(module, inputs, output):
                    q, k, v = output.split(current_fused_dims, dim=-1)
                    state = layer_states[current_layer_name]
                    state["raw_q"] = q.detach()
                    state["raw_k"] = k.detach()
                    state["raw_v"] = v.detach()
                return _hook

            hooks.append(block.att_proj.register_forward_hook(make_att_proj_hook(layer_name, fused_dims)))
        else:
            def make_proj_hook(current_layer_name, key_name):
                def _hook(module, inputs, output):
                    layer_states[current_layer_name][key_name] = output.detach()
                return _hook

            hooks.append(block.q_proj.register_forward_hook(make_proj_hook(layer_name, "raw_q")))
            hooks.append(block.k_proj.register_forward_hook(make_proj_hook(layer_name, "raw_k")))
            hooks.append(block.v_proj.register_forward_hook(make_proj_hook(layer_name, "raw_v")))

    try:
        yield store
    finally:
        for hook in hooks:
            hook.remove()


def normalize_attention_map(attn_map: np.ndarray) -> np.ndarray:
    attn_map = attn_map.astype(np.float32)
    lo = float(attn_map.min())
    hi = float(attn_map.max())
    if hi - lo <= 1e-8:
        return np.zeros_like(attn_map, dtype=np.float32)
    return (attn_map - lo) / (hi - lo)


def resize_map_to_image(attn_map: np.ndarray, image: Image.Image) -> np.ndarray:
    heat = normalize_attention_map(attn_map)
    heat_img = Image.fromarray((heat * 255.0).astype(np.uint8))
    heat_img = heat_img.resize(image.size, Image.BILINEAR)
    return np.array(heat_img, dtype=np.float32) / 255.0


def maybe_visual_renorm(attn_slice: torch.Tensor, enable: bool) -> torch.Tensor:
    if not enable:
        return attn_slice
    denom = attn_slice.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return attn_slice / denom


def collect_step_heatmap(attn_store, selected_queries, visual_positions, grid_shape, visual_renorm: bool):
    if not selected_queries:
        return np.zeros(grid_shape, dtype=np.float32)

    selected_queries_tensor = torch.tensor(selected_queries, dtype=torch.long)
    visual_positions_tensor = torch.tensor(visual_positions, dtype=torch.long)

    accum = None
    used_layers = 0
    for layer_key in sorted(attn_store.keys()):
        if not attn_store[layer_key]:
            continue
        attn_probs = attn_store[layer_key][0]
        layer_values = attn_probs[0, :, selected_queries_tensor, :]
        layer_values = layer_values[:, :, visual_positions_tensor]
        layer_values = maybe_visual_renorm(layer_values, enable=visual_renorm).mean(dim=(0, 1))
        accum = layer_values.to(torch.float32) if accum is None else accum + layer_values.to(torch.float32)
        used_layers += 1

    if accum is None or used_layers == 0:
        return np.zeros(grid_shape, dtype=np.float32)

    return (accum / used_layers).view(grid_shape[0], grid_shape[1]).cpu().numpy()


def extract_target_step_heatmap(model, tokenizer, image_processor, args):
    device = args.device
    dtype = get_torch_dtype(args.torch_dtype)
    image = Image.open(args.image).convert("RGB")
    prompt = build_prompt(args.question, args.conv_template)

    prefix_embeds, visual_positions, vis_info = prepare_multimodal_prefix_from_image(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image=image,
        prompt_text=prompt,
        device=device,
        dtype=dtype,
    )

    core_model = model.get_model()
    layers_to_capture = list(range(len(core_model.transformer.blocks)))
    total_decode_steps = 0

    with torch.no_grad():
        past_key_values = core_model(None, input_embeddings=prefix_embeds, use_cache=True).attn_key_values
        x = torch.full((1, args.max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)

        if args.max_new_tokens % args.block_length != 0:
            raise ValueError("max_new_tokens must be divisible by block_length.")

        num_blocks = args.max_new_tokens // args.block_length
        steps = args.max_new_tokens // num_blocks
        if args.step_ratio is not None:
            steps = int(steps * args.step_ratio)
        if steps <= 0:
            raise ValueError("The computed number of steps per block is 0. Increase step_ratio or max_new_tokens.")

        schedule_kwargs = {"shift": args.schedule_shift} if args.schedule == "shift" else None

        for block_idx in range(num_blocks):
            block_slice = slice(block_idx * args.block_length, (block_idx + 1) * args.block_length)
            block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
            num_transfer_tokens = get_num_transfer_tokens_sch(
                block_mask_index,
                steps,
                schedule=args.schedule,
                schedule_kwargs=schedule_kwargs,
            )
            block_steps = num_transfer_tokens.shape[1]

            for step_idx in range(block_steps):
                mask_index = x == MASK_TOKEN_ID
                current_block_mask = mask_index[:, block_slice]
                if current_block_mask.sum().item() == 0:
                    continue

                current_embeds = core_model.transformer.wte(x)
                logits = core_model(
                    None,
                    input_embeddings=current_embeds,
                    past_key_values=past_key_values,
                ).logits
                logits_with_noise = add_gumbel_noise(logits, temperature=args.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if args.remasking == "low_confidence":
                    probs = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif args.remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                elif args.remasking == "entrophy":
                    epsilon = 1e-10
                    probs = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.sum(probs * torch.log(probs + epsilon), dim=-1)
                elif args.remasking == "margin":
                    probs = F.softmax(logits.to(torch.float64), dim=-1)
                    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                    x0_p = sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
                else:
                    raise NotImplementedError(args.remasking)

                x0_p[:, (block_idx + 1) * args.block_length :] = -torch.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -torch.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for batch_idx in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[batch_idx, step_idx].item())
                    _, select_index = torch.topk(confidence[batch_idx], k=k)
                    transfer_index[batch_idx, select_index] = True

                selected_queries = torch.where(transfer_index[0])[0].tolist()
                total_decode_steps += 1

                with capture_attention_readonly(model, layers_to_capture) as attn_store:
                    _ = core_model(
                        None,
                        input_embeddings=current_embeds,
                        past_key_values=past_key_values,
                    ).logits

                if total_decode_steps == args.target_step:
                    heatmap = collect_step_heatmap(
                        attn_store=attn_store,
                        selected_queries=selected_queries,
                        visual_positions=visual_positions,
                        grid_shape=(vis_info["height"], vis_info["width"]),
                        visual_renorm=not args.no_visual_renorm,
                    )
                    token_text = tokenizer.batch_decode(
                        x0[0, selected_queries].detach().cpu().unsqueeze(0),
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )[0].replace("<|endoftext|>", "").strip()
                    return image, heatmap, token_text, total_decode_steps

                x[transfer_index] = x0[transfer_index]

    raise ValueError(
        f"target step {args.target_step} is out of range. "
        f"Actual decode steps: {total_decode_steps}."
    )


def render_heatmap_figure(image, heatmap, generated_text, args):
    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    heatmap = normalize_attention_map(heatmap)
    overlay_map = resize_map_to_image(heatmap, image)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), facecolor="white")

    axes[0].imshow(image_array)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    hm = axes[1].imshow(heatmap, cmap=args.cmap, interpolation="bilinear")
    axes[1].set_title(f"Attention Heatmap\nStep {args.target_step}", fontsize=13)
    axes[1].axis("off")
    fig.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(image_array)
    axes[2].imshow(
        overlay_map,
        cmap=args.cmap,
        alpha=args.overlay_alpha,
        interpolation="bilinear",
    )
    axes[2].set_title("Overlay", fontsize=13)
    axes[2].axis("off")

    token_note = generated_text if generated_text else "[empty]"
    fig.suptitle(
        f"Attention Map at Denoising Step {args.target_step}\nSelected tokens: {token_note}",
        fontsize=13,
        y=1.03,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
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

    image, heatmap, generated_text, step = extract_target_step_heatmap(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        args=args,
    )
    _ = step
    render_heatmap_figure(image=image, heatmap=heatmap, generated_text=generated_text, args=args)

    print(f"Saved figure to {args.output}")
    print(f"Step {args.target_step} selected tokens: {generated_text}")


if __name__ == "__main__":
    main()
