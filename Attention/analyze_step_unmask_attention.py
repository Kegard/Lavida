import argparse
import copy
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch


MASK_TOKEN_ID = 126336
CATEGORY_ORDER = ["visual", "prompt_text", "generated_text"]
CATEGORY_LABELS = {
    "visual": "Visual",
    "prompt_text": "Prompt Text",
    "generated_text": "Generated Text",
}
CATEGORY_COLORS = {
    "visual": "#ff8c8c",
    "prompt_text": "#9be7e8",
    "generated_text": "#7ae0d6",
}
GEN_TEXT_CATEGORY_ORDER = ["mask", "special", "normal"]
GEN_TEXT_CATEGORY_LABELS = {
    "mask": "Mask Token",
    "special": "Special Token",
    "normal": "Normal Token",
}
GEN_TEXT_CATEGORY_COLORS = {
    "mask": "#b7a6ff",
    "special": "#ffbf69",
    "normal": "#55c8a8",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze attention-source proportions for each generated token at the moment "
            "it is newly unmasked during LaViDa discrete diffusion generation."
        )
    )
    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--image", default="images/dog.png")
    parser.add_argument("--question", default="Describe the image in detail.")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--step-ratio", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument("--schedule", default="none")
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--output-dir", default="Attention/outputs/step_unmask_attention")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--dataset-path", default=None, help="Optional dataset path for direct loading via datasets.load_dataset, e.g. lmms-lab/LMMs-Eval-Lite.")
    parser.add_argument("--dataset-name", default=None, help="Optional dataset config name, e.g. textvqa_val.")
    parser.add_argument("--split", default="validation", help="Dataset split name.")
    parser.add_argument("--textvqa-ann", default=None, help="Path to a TextVQA annotation json for batch visualization.")
    parser.add_argument("--textvqa-image-dir", default=None, help="Root directory containing TextVQA images.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum number of TextVQA samples to process.")
    parser.add_argument("--sample-offset", type=int, default=0, help="Skip the first N TextVQA samples before visualization.")
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


def load_dataset_split(dataset_path, dataset_name, split):
    if dataset_name is None:
        return load_dataset(dataset_path, split=split)
    return load_dataset(dataset_path, dataset_name, split=split)


def construct_textvqa_question(doc):
    return f"{doc['question'].capitalize()}\nAnswer the question using a single word or phrase."


def extract_textvqa_images(doc):
    return [doc["image"].convert("RGB")]


def load_textvqa_samples(annotation_path: str):
    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "annotations", "samples", "questions"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    raise ValueError(f"Unsupported TextVQA annotation format: {annotation_path}")


def get_sample_question(sample: dict) -> str:
    for key in ("question", "text", "query", "prompt"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Cannot find question text in sample keys: {list(sample.keys())}")


def build_sample_name(sample: dict, index: int) -> str:
    qid = sample.get("question_id", sample.get("questionId", sample.get("id", index)))
    image_id = sample.get("image_id", sample.get("imageId", sample.get("image", "image")))
    return f"{index:05d}_{str(qid)}_{str(image_id)}"


def resolve_textvqa_image_path(image_root: str, sample: dict) -> str:
    candidates = []

    for key in ("image", "image_path", "img_path", "image_file"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            path = value.strip()
            candidates.append(path)
            candidates.append(os.path.join(image_root, path))

    image_id = sample.get("image_id", sample.get("imageId", sample.get("img_id")))
    if image_id is not None:
        image_id = str(image_id)
        base_candidates = [image_id]
        if not os.path.splitext(image_id)[1]:
            base_candidates.extend([f"{image_id}.jpg", f"{image_id}.png", f"{image_id}.jpeg"])
        for base in base_candidates:
            candidates.append(os.path.join(image_root, base))
            candidates.append(os.path.join(image_root, "train2014", base))
            candidates.append(os.path.join(image_root, "val2014", base))
            candidates.append(os.path.join(image_root, "images", base))

    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(norm):
            return norm

    raise FileNotFoundError(f"Could not resolve image path for sample image_id={image_id} under {image_root}")


def prepare_multimodal_prefix_from_image(model, tokenizer, image_processor, image, prompt_text, device, dtype):
    image = image.convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=dtype, device=device) for _image in image_tensor]

    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
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

    return prefix_embeds, prefix_input_ids, visual_mask


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
def capture_attention_readonly(model, layers_to_capture, capture_prefill=True, capture_decode=True):
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

        def make_pre_hook(layer_name):
            def _pre_hook(module, args, kwargs):
                state = layer_states[layer_name]
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

        def make_post_hook(layer_name):
            def _post_hook(module, args, output):
                state = layer_states[layer_name]
                if state["block_mask"] is not None:
                    if layer_name not in warned_block_mask_layers:
                        print(f"[Warning] {layer_name} uses block_mask/flex_attention; skipping read-only capture for this call.")
                        warned_block_mask_layers.add(layer_name)
                    return
                if state["raw_q"] is None or state["raw_k"] is None or state["raw_v"] is None:
                    return

                full_k = None
                full_v = None
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    full_k, full_v = output[1]

                raw_q_len = int(state["raw_q"].shape[1])
                raw_k_len = int(state["raw_k"].shape[1])
                if full_k is not None:
                    full_k_len = int(full_k.shape[-2])
                elif state["layer_past"] is not None:
                    full_k_len = int(state["layer_past"][0].shape[-2]) + raw_k_len
                else:
                    full_k_len = raw_k_len

                is_decode = full_k_len > raw_q_len
                if is_decode and not capture_decode:
                    return
                if (not is_decode) and not capture_prefill:
                    return

                weights = reconstruct_attention_weights(
                    block_module=module,
                    raw_q=state["raw_q"],
                    raw_k=state["raw_k"],
                    raw_v=state["raw_v"],
                    layer_past=state["layer_past"],
                    attention_bias=state["attention_bias"],
                    full_k=None if full_k is None else full_k.transpose(1, 2).contiguous().view(full_k.shape[0], full_k.shape[2], -1),
                    full_v=None if full_v is None else full_v.transpose(1, 2).contiguous().view(full_v.shape[0], full_v.shape[2], -1),
                )
                store[layer_name].append(weights)
            return _post_hook

        hooks.append(block.register_forward_pre_hook(make_pre_hook(layer_name), with_kwargs=True))
        hooks.append(block.register_forward_hook(make_post_hook(layer_name)))

        if hasattr(block, "att_proj"):
            fused_dims = tuple(block.fused_dims)

            def make_att_proj_hook(layer_name, fused_dims):
                def _hook(module, inputs, output):
                    q, k, v = output.split(fused_dims, dim=-1)
                    state = layer_states[layer_name]
                    state["raw_q"] = q.detach()
                    state["raw_k"] = k.detach()
                    state["raw_v"] = v.detach()
                return _hook

            hooks.append(block.att_proj.register_forward_hook(make_att_proj_hook(layer_name, fused_dims)))
        else:
            def make_proj_hook(layer_name, key_name):
                def _hook(module, inputs, output):
                    layer_states[layer_name][key_name] = output.detach()
                return _hook

            hooks.append(block.q_proj.register_forward_hook(make_proj_hook(layer_name, "raw_q")))
            hooks.append(block.k_proj.register_forward_hook(make_proj_hook(layer_name, "raw_k")))
            hooks.append(block.v_proj.register_forward_hook(make_proj_hook(layer_name, "raw_v")))

    try:
        yield store
    finally:
        for hook in hooks:
            hook.remove()


class QueryAttentionCollector:
    def __init__(
        self,
        prefix_ids: torch.Tensor,
        visual_mask: torch.Tensor,
        gen_tokens: torch.Tensor,
        special_token_ids,
        selected_queries: torch.Tensor,
    ):
        self.selected_queries = selected_queries.detach().cpu().to(dtype=torch.long)
        self.query_positions = [int(pos) for pos in self.selected_queries.tolist()]

        prefix_ids = prefix_ids.detach().cpu().to(dtype=torch.long)
        prefix_is_visual = visual_mask.detach().cpu().to(dtype=torch.bool)
        prefix_is_prompt_text = (~prefix_is_visual) & (prefix_ids != IGNORE_INDEX)
        gen_tokens = gen_tokens.detach().cpu().to(dtype=torch.long).view(-1)
        generated_region = torch.ones_like(gen_tokens, dtype=torch.bool)
        generated_is_mask = gen_tokens == MASK_TOKEN_ID
        generated_is_special = torch.zeros_like(generated_is_mask)
        special_token_ids = [token_id for token_id in special_token_ids if token_id != MASK_TOKEN_ID]
        if special_token_ids:
            special_ids = torch.tensor(special_token_ids, dtype=torch.long)
            generated_is_special = torch.isin(gen_tokens, special_ids) & (~generated_is_mask)
        generated_is_normal = (~generated_is_mask) & (~generated_is_special)

        self.key_masks = {
            "visual": prefix_is_visual,
            "prompt_text": prefix_is_prompt_text,
            "generated_text": generated_region,
        }
        self.gen_text_key_masks = {
            "mask": generated_is_mask,
            "special": generated_is_special,
            "normal": generated_is_normal,
        }
        self.query_category_sums = {
            query_pos: {name: 0.0 for name in CATEGORY_ORDER}
            for query_pos in self.query_positions
        }
        self.query_gen_text_category_sums = {
            query_pos: {name: 0.0 for name in GEN_TEXT_CATEGORY_ORDER}
            for query_pos in self.query_positions
        }

    def record(self, attn_probs: torch.Tensor):
        if self.selected_queries.numel() == 0:
            return

        probs = attn_probs[0, :, self.selected_queries, :].to(torch.float32)
        prefix_len = self.key_masks["visual"].shape[0]

        for local_idx, query_pos in enumerate(self.query_positions):
            query_probs = probs[:, local_idx, :]
            for name in CATEGORY_ORDER:
                prefix_mask = self.key_masks[name].to(device=query_probs.device)
                if name == "generated_text":
                    expanded_mask = torch.cat(
                        [
                            torch.zeros(prefix_len, dtype=torch.bool, device=query_probs.device),
                            prefix_mask,
                        ]
                    )
                else:
                    expanded_mask = torch.cat(
                        [
                            prefix_mask,
                            torch.zeros(query_probs.shape[-1] - prefix_len, dtype=torch.bool, device=query_probs.device),
                        ]
                    )
                self.query_category_sums[query_pos][name] += query_probs[:, expanded_mask].sum().item()
            for name in GEN_TEXT_CATEGORY_ORDER:
                gen_text_mask = self.gen_text_key_masks[name].to(device=query_probs.device)
                expanded_mask = torch.cat(
                    [
                        torch.zeros(prefix_len, dtype=torch.bool, device=query_probs.device),
                        gen_text_mask,
                    ]
                )
                self.query_gen_text_category_sums[query_pos][name] += query_probs[:, expanded_mask].sum().item()

    def finalize(self):
        result = {}
        for query_pos, sums in self.query_category_sums.items():
            total = sum(sums.values())
            if total <= 0:
                result[query_pos] = {name: 0.0 for name in CATEGORY_ORDER}
            else:
                result[query_pos] = {name: sums[name] / total for name in CATEGORY_ORDER}
        return result

    def finalize_gen_text(self):
        result = {}
        for query_pos, sums in self.query_gen_text_category_sums.items():
            total = sum(sums.values())
            if total <= 0:
                result[query_pos] = {name: 0.0 for name in GEN_TEXT_CATEGORY_ORDER}
            else:
                result[query_pos] = {name: sums[name] / total for name in GEN_TEXT_CATEGORY_ORDER}
        return result


def prepare_multimodal_prefix(model, tokenizer, image_processor, image_path, prompt_text, device, dtype):
    image = Image.open(image_path).convert("RGB")
    return prepare_multimodal_prefix_from_image(model, tokenizer, image_processor, image, prompt_text, device, dtype)


def save_attention_distribution_plot(token_stats, output_path: Path, dpi: int):
    x = list(range(len(token_stats)))
    bottoms = [0.0] * len(token_stats)

    plt.figure(figsize=(14, 4.8))
    for name in CATEGORY_ORDER:
        values = [token_stats[idx][name] for idx in x]
        plt.bar(
            x,
            values,
            bottom=bottoms,
            width=0.92,
            color=CATEGORY_COLORS[name],
            edgecolor="none",
            label=CATEGORY_LABELS[name],
        )
        bottoms = [bottoms[i] + values[i] for i in range(len(values))]

    plt.ylim(0.0, 1.0)
    plt.xlim(-0.5, len(token_stats) - 0.5)
    plt.xlabel("Generated Token Index")
    plt.ylabel("Attention Weight")
    plt.title("Attention Distribution")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_step_attention_distribution_plot(step_stats, output_path: Path, dpi: int):
    x = list(range(len(step_stats)))
    bottoms = [0.0] * len(step_stats)

    plt.figure(figsize=(14, 4.8))
    for name in CATEGORY_ORDER:
        values = [step_stats[idx][name] for idx in x]
        plt.bar(
            x,
            values,
            bottom=bottoms,
            width=0.92,
            color=CATEGORY_COLORS[name],
            edgecolor="none",
            label=CATEGORY_LABELS[name],
        )
        bottoms = [bottoms[i] + values[i] for i in range(len(values))]

    plt.ylim(0.0, 1.0)
    plt.xlim(-0.5, len(step_stats) - 0.5)
    plt.xlabel("Diffusion Step Index")
    plt.ylabel("Attention Weight")
    plt.title("Attention Distribution Over Steps")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_layer_step_heatmap(layer_step_stats, category: str, output_path: Path, dpi: int):
    if not layer_step_stats:
        raise ValueError("No layer-step statistics to plot.")

    data = torch.tensor(
        [[layer_stats[category] for layer_stats in step_stats] for step_stats in layer_step_stats],
        dtype=torch.float32,
    ).transpose(0, 1)

    width = max(8.0, min(28.0, 0.28 * data.shape[1] + 4.0))
    height = max(5.0, min(18.0, 0.22 * data.shape[0] + 2.5))

    plt.figure(figsize=(width, height))
    im = plt.imshow(data.numpy(), aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(im, label=f"{CATEGORY_LABELS[category]} Attention Ratio")
    plt.xlabel("Diffusion Step Index")
    plt.ylabel("Layer Index")
    plt.title(f"{CATEGORY_LABELS[category]} Attention Ratio by Layer and Step")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_gen_text_attention_distribution_plot(step_stats, output_path: Path, dpi: int):
    x = list(range(len(step_stats)))
    bottoms = [0.0] * len(step_stats)

    plt.figure(figsize=(14, 4.8))
    for name in GEN_TEXT_CATEGORY_ORDER:
        values = [step_stats[idx][name] for idx in x]
        plt.bar(
            x,
            values,
            bottom=bottoms,
            width=0.92,
            color=GEN_TEXT_CATEGORY_COLORS[name],
            edgecolor="none",
            label=GEN_TEXT_CATEGORY_LABELS[name],
        )
        bottoms = [bottoms[i] + values[i] for i in range(len(values))]

    plt.ylim(0.0, 1.0)
    plt.xlim(-0.5, len(step_stats) - 0.5)
    plt.xlabel("Diffusion Step Index")
    plt.ylabel("Attention Weight Within Generated Text")
    plt.title("Generated Text Internal Attention Distribution")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_visual_attention_curve(token_stats, output_path: Path, dpi: int):
    x = list(range(len(token_stats)))
    y = [token_stats[idx]["visual"] for idx in x]

    plt.figure(figsize=(14, 4.8))
    plt.plot(x, y, color="purple", linewidth=1.7, marker="o", markersize=2.8)
    plt.xlabel("Generated Token Index")
    plt.ylabel("Visual Attention")
    plt.title("Visual Attention Over Time")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_visual_attention_over_steps(step_stats, output_path: Path, dpi: int):
    x = list(range(len(step_stats)))
    y = [step_stats[idx]["visual"] for idx in x]

    plt.figure(figsize=(14, 4.8))
    plt.plot(x, y, color="purple", linewidth=1.7, marker="o", markersize=2.8)
    plt.xlabel("Diffusion Step Index")
    plt.ylabel("Visual Attention")
    plt.title("Visual Attention Over Steps")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def average_stat_sequences(stat_sequences):
    if not stat_sequences:
        raise ValueError("No statistic sequences to average.")

    min_len = min(len(seq) for seq in stat_sequences)
    if min_len <= 0:
        raise ValueError("Statistic sequences are empty.")

    averaged = []
    for idx in range(min_len):
        averaged_item = {}
        for name in CATEGORY_ORDER:
            averaged_item[name] = sum(seq[idx][name] for seq in stat_sequences) / len(stat_sequences)
        averaged.append(averaged_item)
    return averaged


def average_layer_step_stat_sequences(stat_sequences):
    if not stat_sequences:
        raise ValueError("No layer-step statistic sequences to average.")

    min_len = min(len(seq) for seq in stat_sequences)
    if min_len <= 0:
        raise ValueError("Layer-step statistic sequences are empty.")

    averaged = []
    for step_idx in range(min_len):
        min_layers = min(len(seq[step_idx]) for seq in stat_sequences)
        averaged_step = []
        for layer_idx in range(min_layers):
            averaged_item = {}
            for name in CATEGORY_ORDER:
                averaged_item[name] = sum(seq[step_idx][layer_idx][name] for seq in stat_sequences) / len(stat_sequences)
            averaged_step.append(averaged_item)
        averaged.append(averaged_step)
    return averaged


def average_gen_text_stat_sequences(stat_sequences):
    if not stat_sequences:
        raise ValueError("No statistic sequences to average.")

    min_len = min(len(seq) for seq in stat_sequences)
    if min_len <= 0:
        raise ValueError("Statistic sequences are empty.")

    averaged = []
    for idx in range(min_len):
        averaged_item = {}
        for name in GEN_TEXT_CATEGORY_ORDER:
            averaged_item[name] = sum(seq[idx][name] for seq in stat_sequences) / len(stat_sequences)
        averaged.append(averaged_item)
    return averaged


def get_special_token_ids(tokenizer):
    special_ids = getattr(tokenizer, "all_special_ids", None)
    if special_ids is None:
        return []
    return [int(token_id) for token_id in special_ids if token_id is not None]


def average_query_stats(finalized_stats):
    if not finalized_stats:
        return {name: 0.0 for name in CATEGORY_ORDER}
    return {
        name: sum(stats[name] for stats in finalized_stats.values()) / len(finalized_stats)
        for name in CATEGORY_ORDER
    }


def analyze_single_example(model, tokenizer, image_processor, args, image_path: str, question: str, output_dir: Path = None, save_plots: bool = True):
    device = args.device
    dtype = get_torch_dtype(args.torch_dtype)
    if save_plots:
        if output_dir is None:
            raise ValueError("output_dir is required when save_plots=True.")
        output_dir.mkdir(parents=True, exist_ok=True)
    vision_kwargs = dict(
        mm_vision_tower=args.vision_tower,
        mm_resampler_type=None,
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=1152,
        use_mm_proj=True,
    )
    _ = vision_kwargs
    prompt = build_prompt(question, args.conv_template)
    prefix_embeds, prefix_input_ids, visual_mask = prepare_multimodal_prefix(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_path=image_path,
        prompt_text=prompt,
        device=device,
        dtype=dtype,
    )

    core_model = model.get_model()
    layers_to_capture = list(range(len(core_model.transformer.blocks)))
    special_token_ids = get_special_token_ids(tokenizer)

    with torch.no_grad():
        # Build KV cache from the multimodal prefix first.
        # This prefill pass is intentionally NOT captured.
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

        token_stats = {}
        step_stats = []
        gen_text_step_stats = []
        layer_step_stats = []

        for block_idx in range(num_blocks):
            block_slice = slice(block_idx * args.block_length, (block_idx + 1) * args.block_length)
            block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
            schedule_kwargs = {"shift": args.schedule_shift} if args.schedule == "shift" else None
            num_transfer_tokens = get_num_transfer_tokens_sch(
                block_mask_index,
                steps,
                schedule=args.schedule,
                schedule_kwargs=schedule_kwargs,
            )
            block_steps = num_transfer_tokens.shape[1]

            for step_idx in range(block_steps):
                mask_index = x == MASK_TOKEN_ID
                block_mask_index = mask_index[:, block_slice]
                if block_mask_index.sum().item() == 0:
                    continue

                current_embeds = core_model.transformer.wte(x)
                logits = core_model(None, input_embeddings=current_embeds, past_key_values=past_key_values).logits
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

                selected_queries = torch.where(transfer_index[0])[0]
                collector = QueryAttentionCollector(
                    prefix_ids=prefix_input_ids,
                    visual_mask=visual_mask,
                    gen_tokens=x[0],
                    special_token_ids=special_token_ids,
                    selected_queries=selected_queries,
                )

                # Capture attention for the decode pass only.
                # Query positions are the current generated slots, while the prefix only
                # serves as cached KV context from the earlier uncaptured prefill.
                with capture_attention_readonly(model, layers_to_capture, capture_prefill=False, capture_decode=True) as attn_store:
                    _ = core_model(None, input_embeddings=current_embeds, past_key_values=past_key_values).logits

                current_layer_step_stats = []
                for layer_idx in layers_to_capture:
                    layer_key = f"layer_{layer_idx}"
                    if layer_key in attn_store and attn_store[layer_key]:
                        layer_attn = attn_store[layer_key][0]
                        collector.record(layer_attn)
                        layer_collector = QueryAttentionCollector(
                            prefix_ids=prefix_input_ids,
                            visual_mask=visual_mask,
                            gen_tokens=x[0],
                            special_token_ids=special_token_ids,
                            selected_queries=selected_queries,
                        )
                        layer_collector.record(layer_attn)
                        current_layer_step_stats.append(average_query_stats(layer_collector.finalize()))
                    else:
                        current_layer_step_stats.append({name: 0.0 for name in CATEGORY_ORDER})
                layer_step_stats.append(current_layer_step_stats)

                current_token_stats = collector.finalize()
                current_gen_text_stats = collector.finalize_gen_text()
                for query_pos, stats in current_token_stats.items():
                    token_stats[query_pos] = stats
                if current_token_stats:
                    step_mean = {}
                    for name in CATEGORY_ORDER:
                        step_mean[name] = sum(v[name] for v in current_token_stats.values()) / len(current_token_stats)
                    step_stats.append(step_mean)
                if current_gen_text_stats:
                    gen_text_step_mean = {}
                    for name in GEN_TEXT_CATEGORY_ORDER:
                        gen_text_step_mean[name] = sum(v[name] for v in current_gen_text_stats.values()) / len(current_gen_text_stats)
                    gen_text_step_stats.append(gen_text_step_mean)

                x[transfer_index] = x0[transfer_index]

        missing_positions = [idx for idx in range(args.max_new_tokens) if idx not in token_stats]
        if missing_positions:
            raise RuntimeError(f"Some generated token positions were never recorded: {missing_positions[:10]}")

        ordered_token_stats = [token_stats[idx] for idx in range(args.max_new_tokens)]
        attention_dist_path = None
        step_attention_dist_path = None
        visual_layer_step_heatmap_path = None
        prompt_layer_step_heatmap_path = None
        gen_text_layer_step_heatmap_path = None
        gen_text_attention_dist_path = None
        visual_curve_path = None
        visual_step_curve_path = None
        if save_plots:
            attention_dist_path = output_dir / "attention_distribution.png"
            step_attention_dist_path = output_dir / "step_attention_distribution.png"
            visual_layer_step_heatmap_path = output_dir / "layer_step_visual_attention_heatmap.png"
            prompt_layer_step_heatmap_path = output_dir / "layer_step_prompt_text_attention_heatmap.png"
            gen_text_layer_step_heatmap_path = output_dir / "layer_step_generated_text_attention_heatmap.png"
            gen_text_attention_dist_path = output_dir / "gen_text_attention_distribution.png"
            visual_curve_path = output_dir / "visual_attention_over_time.png"
            visual_step_curve_path = output_dir / "visual_attention_over_steps.png"
            save_attention_distribution_plot(ordered_token_stats, attention_dist_path, args.dpi)
            save_step_attention_distribution_plot(step_stats, step_attention_dist_path, args.dpi)
            save_layer_step_heatmap(layer_step_stats, "visual", visual_layer_step_heatmap_path, args.dpi)
            save_layer_step_heatmap(layer_step_stats, "prompt_text", prompt_layer_step_heatmap_path, args.dpi)
            save_layer_step_heatmap(layer_step_stats, "generated_text", gen_text_layer_step_heatmap_path, args.dpi)
            save_gen_text_attention_distribution_plot(gen_text_step_stats, gen_text_attention_dist_path, args.dpi)
            save_visual_attention_curve(ordered_token_stats, visual_curve_path, args.dpi)
            save_visual_attention_over_steps(step_stats, visual_step_curve_path, args.dpi)

        final_text = tokenizer.batch_decode(x, skip_special_tokens=False)[0].replace("<|endoftext|>", "")

    return {
        "image_path": image_path,
        "question": question,
        "attention_distribution": None if attention_dist_path is None else str(attention_dist_path),
        "step_attention_distribution": None if step_attention_dist_path is None else str(step_attention_dist_path),
        "layer_step_visual_attention_heatmap": None if visual_layer_step_heatmap_path is None else str(visual_layer_step_heatmap_path),
        "layer_step_prompt_text_attention_heatmap": None if prompt_layer_step_heatmap_path is None else str(prompt_layer_step_heatmap_path),
        "layer_step_generated_text_attention_heatmap": None if gen_text_layer_step_heatmap_path is None else str(gen_text_layer_step_heatmap_path),
        "gen_text_attention_distribution": None if gen_text_attention_dist_path is None else str(gen_text_attention_dist_path),
        "visual_attention_over_time": None if visual_curve_path is None else str(visual_curve_path),
        "visual_attention_over_steps": None if visual_step_curve_path is None else str(visual_step_curve_path),
        "final_text": final_text,
        "token_stats": ordered_token_stats,
        "step_stats": step_stats,
        "layer_step_stats": layer_step_stats,
        "gen_text_step_stats": gen_text_step_stats,
    }


def analyze_single_dataset_example(model, tokenizer, image_processor, args, image, question: str):
    device = args.device
    dtype = get_torch_dtype(args.torch_dtype)
    prompt = build_prompt(question, args.conv_template)
    prefix_embeds, prefix_input_ids, visual_mask = prepare_multimodal_prefix_from_image(
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
    special_token_ids = get_special_token_ids(tokenizer)

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

        token_stats = {}
        step_stats = []
        gen_text_step_stats = []
        layer_step_stats = []

        for block_idx in range(num_blocks):
            block_slice = slice(block_idx * args.block_length, (block_idx + 1) * args.block_length)
            block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
            schedule_kwargs = {"shift": args.schedule_shift} if args.schedule == "shift" else None
            num_transfer_tokens = get_num_transfer_tokens_sch(
                block_mask_index,
                steps,
                schedule=args.schedule,
                schedule_kwargs=schedule_kwargs,
            )
            block_steps = num_transfer_tokens.shape[1]

            for step_idx in range(block_steps):
                mask_index = x == MASK_TOKEN_ID
                block_mask_index = mask_index[:, block_slice]
                if block_mask_index.sum().item() == 0:
                    continue

                current_embeds = core_model.transformer.wte(x)
                logits = core_model(None, input_embeddings=current_embeds, past_key_values=past_key_values).logits
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

                selected_queries = torch.where(transfer_index[0])[0]
                collector = QueryAttentionCollector(
                    prefix_ids=prefix_input_ids,
                    visual_mask=visual_mask,
                    gen_tokens=x[0],
                    special_token_ids=special_token_ids,
                    selected_queries=selected_queries,
                )

                with capture_attention_readonly(model, layers_to_capture, capture_prefill=False, capture_decode=True) as attn_store:
                    _ = core_model(None, input_embeddings=current_embeds, past_key_values=past_key_values).logits

                current_layer_step_stats = []
                for layer_idx in layers_to_capture:
                    layer_key = f"layer_{layer_idx}"
                    if layer_key in attn_store and attn_store[layer_key]:
                        layer_attn = attn_store[layer_key][0]
                        collector.record(layer_attn)
                        layer_collector = QueryAttentionCollector(
                            prefix_ids=prefix_input_ids,
                            visual_mask=visual_mask,
                            gen_tokens=x[0],
                            special_token_ids=special_token_ids,
                            selected_queries=selected_queries,
                        )
                        layer_collector.record(layer_attn)
                        current_layer_step_stats.append(average_query_stats(layer_collector.finalize()))
                    else:
                        current_layer_step_stats.append({name: 0.0 for name in CATEGORY_ORDER})
                layer_step_stats.append(current_layer_step_stats)

                finalized = collector.finalize()
                finalized_gen_text = collector.finalize_gen_text()
                for query_pos, stats in finalized.items():
                    token_stats[query_pos] = stats
                if finalized:
                    step_mean = {}
                    for name in CATEGORY_ORDER:
                        step_mean[name] = sum(v[name] for v in finalized.values()) / len(finalized)
                    step_stats.append(step_mean)
                if finalized_gen_text:
                    gen_text_step_mean = {}
                    for name in GEN_TEXT_CATEGORY_ORDER:
                        gen_text_step_mean[name] = sum(v[name] for v in finalized_gen_text.values()) / len(finalized_gen_text)
                    gen_text_step_stats.append(gen_text_step_mean)

                x[transfer_index] = x0[transfer_index]

        missing_positions = [idx for idx in range(args.max_new_tokens) if idx not in token_stats]
        if missing_positions:
            raise RuntimeError(f"Some generated token positions were never recorded: {missing_positions[:10]}")

        ordered_token_stats = [token_stats[idx] for idx in range(args.max_new_tokens)]
        final_text = tokenizer.batch_decode(x, skip_special_tokens=False)[0].replace("<|endoftext|>", "")

    return {
        "question": question,
        "final_text": final_text,
        "token_stats": ordered_token_stats,
        "step_stats": step_stats,
        "layer_step_stats": layer_step_stats,
        "gen_text_step_stats": gen_text_step_stats,
    }


def run_generation_analysis(args):
    device = args.device
    dtype = get_torch_dtype(args.torch_dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    if args.dataset_path is not None:
        dataset = load_dataset_split(args.dataset_path, args.dataset_name, args.split)
        if args.sample_offset > 0:
            dataset = dataset.select(range(args.sample_offset, len(dataset)))
        if args.max_samples > 0:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        if len(dataset) == 0:
            raise ValueError("No dataset samples selected for processing.")

        print(f"Processing {len(dataset)} dataset samples from {args.dataset_path}")
        batch_token_stats = []
        batch_step_stats = []
        batch_layer_step_stats = []
        batch_gen_text_step_stats = []
        for sample_idx, sample in enumerate(dataset):
            question = construct_textvqa_question(sample)
            image = extract_textvqa_images(sample)[0]
            result = analyze_single_dataset_example(model, tokenizer, image_processor, args, image, question)
            batch_token_stats.append(result["token_stats"])
            batch_step_stats.append(result["step_stats"])
            batch_layer_step_stats.append(result["layer_step_stats"])
            batch_gen_text_step_stats.append(result["gen_text_step_stats"])
            print(f"Processed sample {sample_idx + args.sample_offset}")
            print(result["final_text"])

        avg_token_stats = average_stat_sequences(batch_token_stats)
        avg_step_stats = average_stat_sequences(batch_step_stats)
        avg_layer_step_stats = average_layer_step_stat_sequences(batch_layer_step_stats)
        avg_gen_text_step_stats = average_gen_text_stat_sequences(batch_gen_text_step_stats)

        attention_dist_path = output_dir / "attention_distribution.png"
        step_attention_dist_path = output_dir / "step_attention_distribution.png"
        visual_layer_step_heatmap_path = output_dir / "layer_step_visual_attention_heatmap.png"
        prompt_layer_step_heatmap_path = output_dir / "layer_step_prompt_text_attention_heatmap.png"
        gen_text_layer_step_heatmap_path = output_dir / "layer_step_generated_text_attention_heatmap.png"
        gen_text_attention_dist_path = output_dir / "gen_text_attention_distribution.png"
        visual_curve_path = output_dir / "visual_attention_over_time.png"
        visual_step_curve_path = output_dir / "visual_attention_over_steps.png"
        save_attention_distribution_plot(avg_token_stats, attention_dist_path, args.dpi)
        save_step_attention_distribution_plot(avg_step_stats, step_attention_dist_path, args.dpi)
        save_layer_step_heatmap(avg_layer_step_stats, "visual", visual_layer_step_heatmap_path, args.dpi)
        save_layer_step_heatmap(avg_layer_step_stats, "prompt_text", prompt_layer_step_heatmap_path, args.dpi)
        save_layer_step_heatmap(avg_layer_step_stats, "generated_text", gen_text_layer_step_heatmap_path, args.dpi)
        save_gen_text_attention_distribution_plot(avg_gen_text_step_stats, gen_text_attention_dist_path, args.dpi)
        save_visual_attention_curve(avg_token_stats, visual_curve_path, args.dpi)
        save_visual_attention_over_steps(avg_step_stats, visual_step_curve_path, args.dpi)

        print(f"Saved: {attention_dist_path}")
        print(f"Saved: {step_attention_dist_path}")
        print(f"Saved: {visual_layer_step_heatmap_path}")
        print(f"Saved: {prompt_layer_step_heatmap_path}")
        print(f"Saved: {gen_text_layer_step_heatmap_path}")
        print(f"Saved: {gen_text_attention_dist_path}")
        print(f"Saved: {visual_curve_path}")
        print(f"Saved: {visual_step_curve_path}")
        return

    result = analyze_single_example(model, tokenizer, image_processor, args, args.image, args.question, output_dir)
    print(f"Saved: {result['attention_distribution']}")
    print(f"Saved: {result['step_attention_distribution']}")
    print(f"Saved: {result['layer_step_visual_attention_heatmap']}")
    print(f"Saved: {result['layer_step_prompt_text_attention_heatmap']}")
    print(f"Saved: {result['layer_step_generated_text_attention_heatmap']}")
    print(f"Saved: {result['gen_text_attention_distribution']}")
    print(f"Saved: {result['visual_attention_over_time']}")
    print(f"Saved: {result['visual_attention_over_steps']}")
    print("Final generation:")
    print(result["final_text"])


if __name__ == "__main__":
    args = parse_args()
    run_generation_analysis(args)
