#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from llava.model.builder import load_pretrained_model
from visualize_kv_norm_vs_attention import (
    build_prompt,
    capture_attention_readonly,
    compute_attention_vector,
    get_torch_dtype,
    prepare_multimodal_prefix_from_image,
    resolve_layer_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether visual sink-token K/V and attention are sensitive to image perturbations. "
            "The script selects the original image's top visual sink token, then tracks that same visual "
            "position under perturbed images."
        )
    )
    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--image", default="", help="Single-image path. If omitted, use --dataset-path.")
    parser.add_argument("--question", default="", help="Single-image question. If omitted with --image, use a generic prompt.")
    parser.add_argument("--dataset-path", default="", help="Optional HuggingFace dataset path, e.g. lmms-lab/textvqa.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--image-root", default="", help="Optional root for datasets whose image field is a relative path.")
    parser.add_argument(
        "--dataset-question-suffix",
        default="Answer the question using a single word or phrase.",
        help="Suffix appended to dataset questions. Use an empty string to disable.",
    )
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to inspect. Negative values count from the end.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--blur-radius", type=float, default=10.0)
    parser.add_argument("--shuffle-grid", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--perturbations",
        default="original,blur,blank,noise,patch_shuffle",
        help="Comma-separated subset of: original,blur,blank,noise,patch_shuffle.",
    )
    parser.add_argument(
        "--no-visual-renorm",
        action="store_true",
        help="Disable renormalization inside visual-token attention before averaging.",
    )
    parser.add_argument("--output", default="Sink/image_perturbation_sensitivity.json")
    return parser.parse_args()


def parse_perturbations(raw: str) -> List[str]:
    allowed = {"original", "blur", "blank", "noise", "patch_shuffle"}
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(f"Unsupported perturbations: {unknown}. Allowed: {sorted(allowed)}")
    if "original" not in names:
        names.insert(0, "original")
    return names


def mean_color_image(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = tuple(int(round(x)) for x in arr.reshape(-1, 3).mean(axis=0))
    return Image.new("RGB", image.size, color)


def noise_image(image: Image.Image, std: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    noisy = arr + rng.normal(loc=0.0, scale=float(std), size=arr.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def patch_shuffle_image(image: Image.Image, grid: int, seed: int) -> Image.Image:
    if grid <= 1:
        return image.copy()

    image = image.convert("RGB")
    width, height = image.size
    xs = [round(i * width / grid) for i in range(grid + 1)]
    ys = [round(i * height / grid) for i in range(grid + 1)]
    boxes = [(xs[x], ys[y], xs[x + 1], ys[y + 1]) for y in range(grid) for x in range(grid)]
    patches = [image.crop(box) for box in boxes]

    order = list(range(len(patches)))
    random.Random(seed).shuffle(order)
    out = Image.new("RGB", image.size)
    for dst_box, src_idx in zip(boxes, order):
        out.paste(patches[src_idx].resize((dst_box[2] - dst_box[0], dst_box[3] - dst_box[1])), dst_box)
    return out


def build_perturbed_images(image: Image.Image, args: argparse.Namespace) -> Dict[str, Image.Image]:
    image = image.convert("RGB")
    return {
        "original": image,
        "blur": image.filter(ImageFilter.GaussianBlur(radius=float(args.blur_radius))),
        "blank": mean_color_image(image),
        "noise": noise_image(image, std=args.noise_std, seed=args.seed),
        "patch_shuffle": patch_shuffle_image(image, grid=args.shuffle_grid, seed=args.seed),
    }


def load_hf_dataset(path: str, name: str | None, split: str):
    from datasets import load_dataset

    if name:
        return load_dataset(path, name, split=split)
    return load_dataset(path, split=split)


def image_from_value(value, image_root: str) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute() and image_root:
            path = Path(image_root) / path
        return Image.open(path).convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(value)!r}")


def dataset_question(doc: Dict[str, object], suffix: str) -> str:
    question = str(doc.get("question", "")).strip()
    if not question:
        raise ValueError("Dataset sample does not contain a non-empty 'question' field.")
    suffix = suffix.strip()
    if suffix:
        return f"{question}\n{suffix}"
    return question


def iter_input_samples(args: argparse.Namespace):
    if args.dataset_path:
        dataset = load_hf_dataset(args.dataset_path, args.dataset_name, args.split)
        if args.limit <= 0:
            raise ValueError("--limit must be positive in dataset mode.")
        written = 0
        for dataset_index in range(args.start_index, len(dataset)):
            if written >= args.limit:
                break
            doc = dataset[dataset_index]
            if "image" not in doc or doc.get("image") is None:
                continue
            yield {
                "source": "dataset",
                "dataset_path": args.dataset_path,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "dataset_index": int(dataset_index),
                "question_id": doc.get("question_id", doc.get("id")),
                "image": image_from_value(doc["image"], args.image_root),
                "image_ref": str(doc.get("image", "")) if not isinstance(doc.get("image"), Image.Image) else "",
                "question": dataset_question(doc, args.dataset_question_suffix),
            }
            written += 1
        return

    if not args.image:
        raise ValueError("Provide either --image for single-image mode or --dataset-path for dataset mode.")
    question = args.question.strip() or "Describe the image in detail."
    yield {
        "source": "single_image",
        "image": Image.open(args.image).convert("RGB"),
        "image_ref": str(args.image),
        "question": question,
    }


def selected_text_queries(prefix_len: int, visual_positions: List[int]) -> List[int]:
    visual = set(int(pos) for pos in visual_positions)
    return [idx for idx in range(prefix_len) if idx not in visual]


def extract_visual_kv(kv_cache, layer_idx: int, visual_positions: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    past_k, past_v = kv_cache[layer_idx]
    if past_k.ndim != 4 or past_v.ndim != 4:
        raise ValueError(f"Expected KV tensors with shape [B,H,T,D], got {tuple(past_k.shape)} and {tuple(past_v.shape)}")
    positions = torch.tensor(visual_positions, dtype=torch.long, device=past_k.device)
    k_vis = past_k[0, :, positions, :].detach().float().cpu()
    v_vis = past_v[0, :, positions, :].detach().float().cpu()
    return k_vis, v_vis


def flat_token_vectors(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(0, 1).contiguous().view(x.shape[1], -1)


def cosine_same_positions(reference: torch.Tensor, current: torch.Tensor) -> np.ndarray:
    n = min(reference.shape[1], current.shape[1])
    ref_flat = flat_token_vectors(reference[:, :n, :])
    cur_flat = flat_token_vectors(current[:, :n, :])
    return F.cosine_similarity(ref_flat, cur_flat, dim=-1).numpy().astype(np.float32)


def analyze_image(
    *,
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    prompt: str,
    layer_idx: int,
    device: str,
    dtype: torch.dtype,
    visual_renorm: bool,
) -> Dict[str, object]:
    prefix_embeds, visual_positions, vis_info = prepare_multimodal_prefix_from_image(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image=image,
        prompt_text=prompt,
        device=device,
        dtype=dtype,
    )
    text_queries = selected_text_queries(prefix_embeds.shape[1], visual_positions)

    core_model = model.get_model()
    with torch.no_grad(), capture_attention_readonly(model, [layer_idx]) as attn_store:
        output = core_model(None, input_embeddings=prefix_embeds, use_cache=True)

    if output.attn_key_values is None:
        raise RuntimeError("Prefill did not return KV cache.")

    attention = compute_attention_vector(
        attn_store=attn_store,
        layer_idx=layer_idx,
        selected_queries=text_queries,
        visual_positions=visual_positions,
        visual_renorm=visual_renorm,
    )
    k_vis, v_vis = extract_visual_kv(output.attn_key_values, layer_idx, visual_positions)
    return {
        "attention": attention,
        "k_vis": k_vis,
        "v_vis": v_vis,
        "visual_positions": [int(x) for x in visual_positions],
        "num_text_queries": int(len(text_queries)),
        "vis_info": vis_info,
    }


def analyze_sample(
    *,
    sample: Dict[str, object],
    model,
    tokenizer,
    image_processor,
    args: argparse.Namespace,
    layer_idx: int,
    dtype: torch.dtype,
    perturbation_names: List[str],
) -> Dict[str, object]:
    prompt = build_prompt(str(sample["question"]), args.conv_template)
    perturbed_images = build_perturbed_images(sample["image"], args)

    all_stats = {}
    for name in perturbation_names:
        all_stats[name] = analyze_image(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image=perturbed_images[name],
            prompt=prompt,
            layer_idx=layer_idx,
            device=args.device,
            dtype=dtype,
            visual_renorm=not args.no_visual_renorm,
        )

    original = all_stats["original"]
    original_attention = original["attention"]
    sink_idx = int(np.argmax(original_attention))
    perturbation_records = [
        token_summary(name, all_stats[name], original, sink_idx=sink_idx, top_k=args.top_k)
        for name in perturbation_names
    ]

    output = {
        "source": sample.get("source"),
        "image": sample.get("image_ref", ""),
        "question": sample["question"],
        "layer_idx": int(layer_idx),
        "visual_renorm": bool(not args.no_visual_renorm),
        "original_sink_visual_index": int(sink_idx),
        "original_sink_attention": float(original_attention[sink_idx]),
        "num_text_queries": int(original["num_text_queries"]),
        "vis_info": original["vis_info"],
        "perturbations": perturbation_records,
    }
    for key in ("dataset_path", "dataset_name", "split", "dataset_index", "question_id"):
        if key in sample:
            output[key] = sample[key]
    return output


def token_summary(name: str, stats: Dict[str, object], reference: Dict[str, object], sink_idx: int, top_k: int) -> Dict[str, object]:
    attention = stats["attention"]
    k_cos = cosine_same_positions(reference["k_vis"], stats["k_vis"])
    v_cos = cosine_same_positions(reference["v_vis"], stats["v_vis"])
    top_indices = np.argsort(-attention)[: min(top_k, attention.shape[0])]
    rank_order = np.argsort(-attention)
    sink_rank = int(np.where(rank_order == sink_idx)[0][0] + 1) if sink_idx < attention.shape[0] else None

    return {
        "name": name,
        "same_sink_visual_index": int(sink_idx),
        "same_sink_attention": float(attention[sink_idx]) if sink_idx < attention.shape[0] else None,
        "same_sink_attention_rank": sink_rank,
        "same_sink_k_cosine_to_original": float(k_cos[sink_idx]) if sink_idx < k_cos.shape[0] else None,
        "same_sink_v_cosine_to_original": float(v_cos[sink_idx]) if sink_idx < v_cos.shape[0] else None,
        "mean_visual_k_cosine_to_original": float(np.mean(k_cos)),
        "mean_visual_v_cosine_to_original": float(np.mean(v_cos)),
        "top_attention_indices": [int(x) for x in top_indices.tolist()],
        "top_attention_values": [float(attention[x]) for x in top_indices.tolist()],
        "new_top_visual_index": int(top_indices[0]) if top_indices.size else None,
        "num_visual_tokens": int(attention.shape[0]),
    }


def main() -> None:
    args = parse_args()
    perturbation_names = parse_perturbations(args.perturbations)
    dtype = get_torch_dtype(args.torch_dtype)

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
    model.to(dtype)

    core_model = model.get_model()
    layer_idx = resolve_layer_index(args.layer, len(core_model.transformer.blocks))
    records = []
    for sample in iter_input_samples(args):
        record = analyze_sample(
            sample=sample,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            args=args,
            layer_idx=layer_idx,
            dtype=dtype,
            perturbation_names=perturbation_names,
        )
        records.append(record)
        print(
            f"sample={record.get('dataset_index', 0)} "
            f"sink={record['original_sink_visual_index']} "
            f"attn={record['original_sink_attention']:.6f}"
        )

    if not records:
        raise RuntimeError("No analyzable samples were found.")

    output = records[0] if len(records) == 1 and not args.dataset_path else {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "start_index": int(args.start_index),
        "num_records": int(len(records)),
        "layer_idx": int(layer_idx),
        "records": records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved perturbation sensitivity summary to {output_path}")
    first = records[0]
    print(
        "First sample original top sink visual index: "
        f"{first['original_sink_visual_index']}, attention={first['original_sink_attention']:.6f}"
    )
    for record in first["perturbations"]:
        print(
            f"{record['name']:>13s} "
            f"attn={record['same_sink_attention']:.6f} "
            f"rank={record['same_sink_attention_rank']} "
            f"k_cos={record['same_sink_k_cosine_to_original']:.4f} "
            f"v_cos={record['same_sink_v_cosine_to_original']:.4f}"
        )


if __name__ == "__main__":
    main()
