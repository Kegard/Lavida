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
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch
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
DEFAULT_STEP_COLORS = [
    "#d94841",
    "#6aa84f",
    "#f1a208",
    "#7b52ab",
    "#2f6db0",
    "#b85c00",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize spatial attention snapshots for selected denoising steps and "
            "color the generated text by each token's first-unmask step."
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
    parser.add_argument("--snapshot-steps", nargs="+", type=int, default=[8, 16, 24, 32])
    parser.add_argument("--output", default="Sink/attention_step_snapshots.png")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--text-fontsize", type=float, default=15.0)
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

    return prefix_embeds, prefix_input_ids, visual_positions, vis_info


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
    dtype = (full_k.dtype if full_k is not None else raw_k.dtype)

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

    num_q_heads = q.size(1)
    num_kv_heads = k.size(1)
    if num_q_heads != num_kv_heads:
        repeat_factor = num_q_heads // num_kv_heads
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
    warned_block_mask_layers = set()

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
            "block_mask": None,
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
                if "block_mask" in kwargs:
                    state["block_mask"] = kwargs["block_mask"]
                elif len(args) >= 5:
                    state["block_mask"] = args[4]
                else:
                    state["block_mask"] = None
            return _pre_hook

        def make_post_hook(current_layer_name):
            def _post_hook(module, args, output):
                state = layer_states[current_layer_name]
                if state["block_mask"] is not None:
                    if current_layer_name not in warned_block_mask_layers:
                        print(
                            f"[Warning] {current_layer_name} uses block_mask/flex_attention; "
                            "skipping read-only capture for this call."
                        )
                        warned_block_mask_layers.add(current_layer_name)
                    return
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


def build_step_palette(snapshot_steps):
    return {
        step: DEFAULT_STEP_COLORS[idx % len(DEFAULT_STEP_COLORS)]
        for idx, step in enumerate(snapshot_steps)
    }


def normalize_attention_map(attn_map: np.ndarray) -> np.ndarray:
    attn_map = attn_map.astype(np.float32)
    attn_map = attn_map - float(attn_map.min())
    hi = float(np.quantile(attn_map, 0.99))
    if hi <= 1e-8:
        return np.zeros_like(attn_map, dtype=np.float32)
    return np.clip(attn_map / hi, 0.0, 1.0)


def collect_step_visual_map(attn_store, selected_queries, visual_positions, grid_shape):
    if not selected_queries:
        return np.zeros(grid_shape, dtype=np.float32)

    selected_queries_tensor = torch.tensor(selected_queries, dtype=torch.long)
    visual_positions_tensor = torch.tensor(visual_positions, dtype=torch.long)

    step_vector = None
    used_layers = 0
    for layer_key in sorted(attn_store.keys()):
        if not attn_store[layer_key]:
            continue
        attn_probs = attn_store[layer_key][0]
        layer_values = attn_probs[0, :, selected_queries_tensor, :]
        layer_values = layer_values[:, :, visual_positions_tensor].mean(dim=(0, 1))
        if step_vector is None:
            step_vector = layer_values.to(torch.float32)
        else:
            step_vector += layer_values.to(torch.float32)
        used_layers += 1

    if step_vector is None or used_layers == 0:
        return np.zeros(grid_shape, dtype=np.float32)

    step_vector = step_vector / used_layers
    return step_vector.view(grid_shape[0], grid_shape[1]).cpu().numpy()


def token_pieces_from_ids(tokenizer, token_ids, token_steps):
    pieces = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    all_special_ids = set(getattr(tokenizer, "all_special_ids", []))

    for idx, token_id in enumerate(token_ids):
        token_id = int(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            break

        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        text = text.replace("<|endoftext|>", "")

        if token_id in all_special_ids and not text.strip():
            continue
        if not text:
            continue
        pieces.append((text, token_steps.get(idx)))
    return pieces


def draw_colored_text(ax, pieces, step_palette, fontsize):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    x_start = 0.03
    x = x_start
    y = 0.92
    max_x = 0.97

    probe = ax.text(
        0.0,
        0.0,
        "Ag",
        fontsize=fontsize,
        transform=ax.transAxes,
        va="top",
        alpha=0.0,
    )
    line_height = probe.get_window_extent(renderer=renderer).height / ax.bbox.height * 1.35
    probe.remove()

    for text, step in pieces:
        segments = text.split("\n")
        for seg_idx, segment in enumerate(segments):
            if seg_idx > 0:
                x = x_start
                y -= line_height

            draw_text = segment
            if x == x_start:
                draw_text = draw_text.lstrip(" ")
            if not draw_text:
                continue

            probe = ax.text(
                0.0,
                0.0,
                draw_text,
                fontsize=fontsize,
                transform=ax.transAxes,
                va="top",
                alpha=0.0,
            )
            width = probe.get_window_extent(renderer=renderer).width / ax.bbox.width
            probe.remove()

            if x > x_start and x + width > max_x:
                x = x_start
                y -= line_height
                draw_text = draw_text.lstrip(" ")
                if not draw_text:
                    continue
                probe = ax.text(
                    0.0,
                    0.0,
                    draw_text,
                    fontsize=fontsize,
                    transform=ax.transAxes,
                    va="top",
                    alpha=0.0,
                )
                width = probe.get_window_extent(renderer=renderer).width / ax.bbox.width
                probe.remove()

            ax.text(
                x,
                y,
                draw_text,
                fontsize=fontsize,
                color=step_palette.get(step, "#111111"),
                transform=ax.transAxes,
                va="top",
                ha="left",
            )
            x += width

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def render_snapshot_panel(image, snapshot_maps, snapshot_steps, step_palette, pieces, output_path, dpi, overlay_alpha, text_fontsize):
    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    plt.rcParams["font.family"] = "DejaVu Serif"

    fig = plt.figure(figsize=(11.2, 6.8), facecolor="white")
    panel = FancyBboxPatch(
        (0.04, 0.08),
        0.92,
        0.84,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        linewidth=1.0,
        edgecolor="#6f6f6f",
        facecolor="white",
        transform=fig.transFigure,
        zorder=-10,
    )
    fig.add_artist(panel)

    grid = fig.add_gridspec(
        2,
        len(snapshot_steps),
        left=0.075,
        right=0.925,
        bottom=0.13,
        top=0.86,
        height_ratios=[1.0, 0.58],
        hspace=0.12,
        wspace=0.06,
    )

    height, width = image_array.shape[:2]
    for col, step in enumerate(snapshot_steps):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(image_array)

        attn_map = normalize_attention_map(snapshot_maps[step])
        overlay = np.zeros((attn_map.shape[0], attn_map.shape[1], 4), dtype=np.float32)
        overlay[..., :3] = np.asarray(to_rgb(step_palette[step]), dtype=np.float32)
        overlay[..., 3] = attn_map * overlay_alpha
        ax.imshow(overlay, extent=(0, width, height, 0), interpolation="bilinear")

        ax.set_title(
            f"Step: {step}",
            fontsize=15,
            color=step_palette[step],
            fontweight="bold",
            pad=8,
        )
        ax.axis("off")

    text_ax = fig.add_subplot(grid[1, :])
    draw_colored_text(text_ax, pieces, step_palette, fontsize=text_fontsize)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def analyze_attention_snapshots(model, tokenizer, image_processor, args):
    device = args.device
    dtype = get_torch_dtype(args.torch_dtype)
    prompt = build_prompt(args.question, args.conv_template)
    image = Image.open(args.image).convert("RGB")
    prefix_embeds, _, visual_positions, vis_info = prepare_multimodal_prefix_from_image(
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
    snapshot_steps = sorted(dict.fromkeys(args.snapshot_steps))
    snapshot_maps = {}
    token_steps = {}
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

                if total_decode_steps in snapshot_steps:
                    snapshot_maps[total_decode_steps] = collect_step_visual_map(
                        attn_store=attn_store,
                        selected_queries=selected_queries,
                        visual_positions=visual_positions,
                        grid_shape=(vis_info["height"], vis_info["width"]),
                    )

                for query_pos in selected_queries:
                    token_steps[query_pos] = total_decode_steps

                x[transfer_index] = x0[transfer_index]

    missing_snapshot_steps = [step for step in snapshot_steps if step not in snapshot_maps]
    if missing_snapshot_steps:
        raise ValueError(
            "Requested snapshot steps exceed the actual denoising process. "
            f"Missing steps: {missing_snapshot_steps}. Total decode steps: {total_decode_steps}."
        )

    token_ids = x[0].detach().cpu().tolist()
    final_text = tokenizer.batch_decode(
        x,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].replace("<|endoftext|>", "").strip()
    pieces = token_pieces_from_ids(tokenizer, token_ids, token_steps)

    return {
        "image": image,
        "snapshot_maps": snapshot_maps,
        "token_steps": token_steps,
        "pieces": pieces,
        "final_text": final_text,
        "total_decode_steps": total_decode_steps,
    }


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

    result = analyze_attention_snapshots(model, tokenizer, image_processor, args)
    snapshot_steps = sorted(dict.fromkeys(args.snapshot_steps))
    step_palette = build_step_palette(snapshot_steps)
    output_path = Path(args.output)

    render_snapshot_panel(
        image=result["image"],
        snapshot_maps=result["snapshot_maps"],
        snapshot_steps=snapshot_steps,
        step_palette=step_palette,
        pieces=result["pieces"],
        output_path=output_path,
        dpi=args.dpi,
        overlay_alpha=args.overlay_alpha,
        text_fontsize=args.text_fontsize,
    )

    print(f"Saved figure to {output_path}")
    print(f"Final text: {result['final_text']}")
    print(f"Total decode steps: {result['total_decode_steps']}")


if __name__ == "__main__":
    main()
