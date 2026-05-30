import argparse
import copy
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

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

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize query collapse / degeneracy in LaViDa by capturing decode-step "
            "queries from one layer and summarizing their cosine structure."
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
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to inspect. Negative values count from the end.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--step-ratio", type=float, default=0.25)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use-q-norm", type=int, default=1, help="Apply q_norm before capture when available.")
    parser.add_argument("--similarity-step", type=int, default=-1, help="Which decode step to use for the token-token similarity matrix.")
    parser.add_argument("--max-title-chars", type=int, default=120)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output", default="Sink/query_collapse.png")
    return parser.parse_args()


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_prompt_input_ids(tokenizer, question: str, conv_template: str, device: torch.device) -> torch.Tensor:
    prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    return tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)


def resolve_layer_index(requested_layer: int, num_layers: int) -> int:
    resolved = requested_layer if requested_layer >= 0 else num_layers + requested_layer
    if resolved < 0 or resolved >= num_layers:
        raise ValueError(f"Layer index out of range: {requested_layer} -> {resolved}, n_layers={num_layers}")
    return resolved


def resolve_step_index(requested_step: int, total_steps: int) -> int:
    if total_steps <= 0:
        return 0
    if requested_step < 0:
        requested_step = total_steps + requested_step
    return max(0, min(int(requested_step), total_steps - 1))


@contextmanager
def patch_attention_capture_queries(model, layer_idx: int, use_q_norm: bool):
    blocks = model.get_model().transformer.blocks
    block = blocks[layer_idx]
    original_attention = block.attention
    captured_queries: List[torch.Tensor] = []

    def patched_attention(q, k, v, attention_bias=None, layer_past=None, use_cache=False, block_mask=None):
        if layer_past is not None:
            q_cap = q
            if use_q_norm and getattr(block, "q_norm", None) is not None:
                q_cap = block.q_norm(q_cap).to(dtype=q_cap.dtype)

            bsz, q_len, hidden = q_cap.shape
            n_heads = block.config.n_heads
            head_dim = hidden // n_heads
            q_heads = q_cap.view(bsz, q_len, n_heads, head_dim).detach().to(torch.float32).cpu()
            captured_queries.append(q_heads[0].clone())  # [Q, H, D]

        return original_attention(
            q,
            k,
            v,
            attention_bias=attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
            block_mask=block_mask,
        )

    block.attention = patched_attention
    try:
        yield captured_queries
    finally:
        block.attention = original_attention


def compute_token_similarity_matrix(q_step: torch.Tensor) -> torch.Tensor:
    # q_step: [Q, H, D]
    q_flat = q_step.reshape(q_step.shape[0], -1)
    q_flat = F.normalize(q_flat, p=2, dim=-1, eps=1e-12)
    return q_flat @ q_flat.T


def compute_same_token_step_cosine(q_steps: torch.Tensor) -> torch.Tensor:
    # q_steps: [S, Q, H, D]
    if q_steps.shape[0] <= 1:
        return torch.empty((0, q_steps.shape[1]), dtype=torch.float32)
    prev = q_steps[:-1].reshape(q_steps.shape[0] - 1, q_steps.shape[1], -1)
    curr = q_steps[1:].reshape(q_steps.shape[0] - 1, q_steps.shape[1], -1)
    return F.cosine_similarity(prev, curr, dim=-1)  # [S-1, Q]


def compute_query_norms(q_steps: torch.Tensor) -> torch.Tensor:
    # q_steps: [S, Q, H, D]
    return torch.norm(q_steps, p=2, dim=-1).mean(dim=-1)  # [S, Q]


def compute_mean_pairwise_cosine_per_step(q_steps: torch.Tensor) -> np.ndarray:
    values = []
    for step_idx in range(q_steps.shape[0]):
        sim = compute_token_similarity_matrix(q_steps[step_idx])
        if sim.shape[0] <= 1:
            values.append(float("nan"))
            continue
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
        offdiag = sim[mask]
        values.append(float(offdiag.mean().item()))
    return np.asarray(values, dtype=np.float32)


def truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def render_overview(
    sim_mat: np.ndarray,
    step_cos: np.ndarray,
    mean_pairwise_cos: np.ndarray,
    norms: np.ndarray,
    layer_idx: int,
    step_used: int,
    generated_text: str,
    output_path: Path,
    dpi: int,
    max_title_chars: int,
):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2), facecolor="white")

    im0 = axes[0, 0].imshow(sim_mat, aspect="auto", cmap="viridis", vmin=-1.0, vmax=1.0)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[0, 0].set_title(f"Token-Token Query Cosine\nStep {step_used}")
    axes[0, 0].set_xlabel("Generated Token Index")
    axes[0, 0].set_ylabel("Generated Token Index")

    if step_cos.size > 0:
        im1 = axes[0, 1].imshow(step_cos.T, aspect="auto", cmap="viridis", vmin=-1.0, vmax=1.0)
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        axes[0, 1].set_title("Same-Token Query Cosine Across Steps")
        axes[0, 1].set_xlabel("Step Pair t->t+1")
        axes[0, 1].set_ylabel("Generated Token Index")
    else:
        axes[0, 1].text(0.5, 0.5, "Only one step captured", ha="center", va="center", fontsize=12)
        axes[0, 1].set_title("Same-Token Query Cosine Across Steps")
        axes[0, 1].axis("off")

    steps = np.arange(len(mean_pairwise_cos))
    axes[1, 0].plot(steps, mean_pairwise_cos, marker="o", linewidth=1.8, color="#4c78a8")
    axes[1, 0].set_title("Mean Off-Diagonal Query Cosine by Step")
    axes[1, 0].set_xlabel("Denoise Step")
    axes[1, 0].set_ylabel("Mean Pairwise Cosine")
    axes[1, 0].grid(alpha=0.25, linestyle="--")

    mean_norm = norms.mean(axis=1)
    axes[1, 1].plot(np.arange(len(mean_norm)), mean_norm, marker="o", linewidth=1.8, color="#dd8452")
    axes[1, 1].set_title("Mean Query L2 Norm by Step")
    axes[1, 1].set_xlabel("Denoise Step")
    axes[1, 1].set_ylabel("Mean Query L2 Norm")
    axes[1, 1].grid(alpha=0.25, linestyle="--")

    title_text = truncate_text(generated_text, max_title_chars)
    fig.suptitle(
        f"Layer {layer_idx} Query Collapse Diagnostics\nGenerated Text: {title_text}",
        fontsize=14,
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = args.device
    dtype = get_torch_dtype(args.torch_dtype)

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
        device_map=f"{device}:0" if device.startswith("cuda") else device,
        vision_kwargs=vision_kwargs,
        torch_dtype=args.torch_dtype,
    )
    model.eval()
    model.tie_weights()
    model.to(dtype)

    layer_idx = resolve_layer_index(args.layer, len(model.get_model().transformer.blocks))
    image = Image.open(args.image).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    image_tensor = image_tensor.to(dtype=dtype, device=device)
    image_sizes = [image.size]
    input_ids = build_prompt_input_ids(tokenizer, args.question, args.conv_template, torch.device(device))

    with patch_attention_capture_queries(
        model=model,
        layer_idx=layer_idx,
        use_q_norm=bool(args.use_q_norm),
    ) as captured_queries:
        with torch.no_grad():
            out = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=float(args.temperature),
                max_new_tokens=int(args.max_new_tokens),
                block_length=int(args.block_length),
                step_ratio=float(args.step_ratio),
                tokenizer=tokenizer,
                prefix_lm=True,
                verbose=False,
                schedule=args.schedule,
            )
            cont = out[0] if isinstance(out, tuple) else out

    if not captured_queries:
        raise RuntimeError(f"No decode-step queries captured for layer {layer_idx}.")

    q_steps = torch.stack(captured_queries, dim=0)  # [S, Q, H, D]
    step_used = resolve_step_index(int(args.similarity_step), int(q_steps.shape[0]))
    sim_mat = compute_token_similarity_matrix(q_steps[step_used])
    step_cos = compute_same_token_step_cosine(q_steps)
    norms = compute_query_norms(q_steps)
    mean_pairwise_cos = compute_mean_pairwise_cosine_per_step(q_steps)

    generated_text = tokenizer.batch_decode(
        cont,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].replace("<|endoftext|>", "").strip()

    output_path = Path(args.output)
    render_overview(
        sim_mat=sim_mat.cpu().numpy(),
        step_cos=step_cos.cpu().numpy(),
        mean_pairwise_cos=mean_pairwise_cos,
        norms=norms.cpu().numpy(),
        layer_idx=layer_idx,
        step_used=step_used,
        generated_text=generated_text,
        output_path=output_path,
        dpi=args.dpi,
        max_title_chars=args.max_title_chars,
    )

    summary = {
        "layer_idx": int(layer_idx),
        "num_steps": int(q_steps.shape[0]),
        "num_gen_tokens": int(q_steps.shape[1]),
        "num_heads": int(q_steps.shape[2]),
        "head_dim": int(q_steps.shape[3]),
        "similarity_step_used": int(step_used),
        "generated_text": generated_text,
        "token_similarity_mean": float(sim_mat.mean().item()),
        "token_similarity_min": float(sim_mat.min().item()),
        "token_similarity_max": float(sim_mat.max().item()),
        "mean_offdiag_query_cosine_per_step": [float(x) for x in mean_pairwise_cos.tolist()],
        "same_token_step_cosine_mean": float(step_cos.mean().item()) if step_cos.numel() > 0 else float("nan"),
        "same_token_step_cosine_min": float(step_cos.min().item()) if step_cos.numel() > 0 else float("nan"),
        "same_token_step_cosine_max": float(step_cos.max().item()) if step_cos.numel() > 0 else float("nan"),
        "mean_query_norm_per_step": [float(x) for x in norms.mean(dim=1).tolist()],
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved figure to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
