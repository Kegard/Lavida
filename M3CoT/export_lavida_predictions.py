import argparse
import copy
import json
import os
import sys
from pathlib import Path

import datasets
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_ROOT = REPO_ROOT / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from lmms_eval.models.llava_llada import Llava_Llada
from M3CoT.native_vrg_generate import generate_with_native_vrg


LETTER_MAP = "ABCDEFG"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LaViDa on M3CoT and export predictions in official custom-jsonl format."
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-path", default="/data/jindong_gu/LaViDa/weight/lavida-reason")
    parser.add_argument("--method", default="base", choices=["base", "vrg"])
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--vision-tower", default="/data/jindong_gu/LaViDa/weight/siglip")
    parser.add_argument("--vision-projector", default="mlp2x_gelu")
    parser.add_argument("--vision-hidden-size", type=int, default=1152)
    parser.add_argument("--mm-pooler-ratio", type=int, default=2)
    parser.add_argument("--prompt", default="cot", choices=["direct", "cot", "ccot", "dsp"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--step-per-block", type=int, default=None)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-mode", default="random", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vrg-alpha-start", type=float, default=0.0)
    parser.add_argument("--vrg-alpha-end", type=float, default=2.0)
    parser.add_argument("--vrg-alpha-schedule", default="linear", choices=["linear", "cosine", "power"])
    parser.add_argument("--vrg-alpha-power", type=float, default=2.0)
    parser.add_argument("--vrg-null-visual-mode", default="zeros")
    parser.add_argument("--vrg-gate", default="none", choices=["none", "entropy"])
    parser.add_argument("--vrg-entropy-threshold", type=float, default=0.0)
    parser.add_argument("--save-vrg-gate-stats", action="store_true")
    return parser.parse_args()


def get_torch_dtype(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_choice_block(choices):
    return "\n".join(f"{LETTER_MAP[i]}. {choice}" for i, choice in enumerate(choices))


def build_base_prompt(doc):
    parts = []
    context = (doc.get("context") or "").strip()
    if context:
        parts.append(context)
    parts.append(doc["question"])
    parts.append(build_choice_block(doc["choices"]))
    return "\n".join(parts)


def build_prompt(doc, prompt_style):
    base = build_base_prompt(doc)
    if prompt_style == "direct":
        return base + "\n\nAnswer with the option's letter from the given choices directly."
    if prompt_style == "cot":
        return (
            base
            + "\n\nPlease reason step by step, and answer the question with option letter "
            + "from given choices in the format of Answer: <option letter>."
        )
    if prompt_style == "dsp":
        return (
            base
            + "\n\nFirst describe the image information relevant to the question. "
            + "Then reason briefly and provide the final answer in the format [Answer] (X)."
        )
    if prompt_style == "ccot":
        return (
            base
            + "\n\nFirst identify the relevant objects, attributes, and relationships in the image as a compact scene graph. "
            + "Then solve the question and provide the final answer in the format [Answer] (X)."
        )
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def build_model(args):
    common_kwargs = dict(
        pretrained=args.model_path,
        truncation=True,
        device=args.device,
        batch_size=1,
        model_name="llava_llada",
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        conv_template=args.conv_template,
        use_cache=True,
        truncate_context=False,
        customized_config=None,
        max_frames_num=32,
        mm_spatial_pool_stride=2,
        mm_spatial_pool_mode="bilinear",
        token_strategy="single",
        video_decode_backend="decord",
        mc_num=16,
    )

    os.environ["LLADA_VISION_ENCODER"] = args.vision_tower
    os.environ["LLADA_VISION_PROJECTOR"] = args.vision_projector
    os.environ["LLADA_VISION_ENCODER_HIDDEN_SIZE"] = str(args.vision_hidden_size)
    os.environ["LLADA_MM_POOLER_RATIO"] = str(args.mm_pooler_ratio)
    return Llava_Llada(**common_kwargs)


def prepare_sample_inputs(model_wrapper, doc, prompt_text, torch_dtype):
    tokenizer = model_wrapper.tokenizer
    image = doc["image"].convert("RGB")
    image_tensor = process_images([image], model_wrapper._image_processor, model_wrapper.config)
    dtype = get_torch_dtype(torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=model_wrapper.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=model_wrapper.device)

    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    if "llama_3" in model_wrapper.conv_template or "llada" in model_wrapper.conv_template:
        conv = copy.deepcopy(conv_templates[model_wrapper.conv_template])
    else:
        conv = conv_templates[model_wrapper.conv_template].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model_wrapper.device)
    attention_masks = torch.ones_like(input_ids, dtype=torch.bool, device=model_wrapper.device)
    return image, image_tensor, input_ids, attention_masks


def generate_one(model_wrapper, doc, prompt_text, args):
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    image, image_tensor, input_ids, attention_masks = prepare_sample_inputs(
        model_wrapper,
        doc,
        prompt_text,
        args.torch_dtype,
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "image_sizes": [image.size],
    }
    effective_step_ratio = None if args.step_per_block is not None else args.step_ratio
    if args.block_length is not None:
        gen_kwargs["block_length"] = args.block_length
    if args.step_per_block is not None:
        gen_kwargs["step_per_block"] = args.step_per_block
    elif effective_step_ratio is not None:
        gen_kwargs["step_ratio"] = effective_step_ratio

    if args.method == "vrg":
        result = generate_with_native_vrg(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_masks,
            images=image_tensor,
            image_sizes=[image.size],
            max_new_tokens=int(args.max_new_tokens),
            block_length=int(args.block_length or min(128, int(args.max_new_tokens))),
            temperature=float(args.temperature),
            step_ratio=effective_step_ratio,
            step_per_block=args.step_per_block,
            remasking="low_confidence",
            alpha_start=float(args.vrg_alpha_start),
            alpha_end=float(args.vrg_alpha_end),
            alpha_schedule=args.vrg_alpha_schedule,
            alpha_power=float(args.vrg_alpha_power),
            null_visual_mode=args.vrg_null_visual_mode,
            vrg_gate=args.vrg_gate,
            vrg_entropy_threshold=float(args.vrg_entropy_threshold),
            return_gate_stats=bool(args.save_vrg_gate_stats),
        )
        if args.save_vrg_gate_stats:
            sequences, gate_stats = result
        else:
            sequences = result
            gate_stats = None
    else:
        with torch.inference_mode():
            sequences = model.generate(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_id,
                images=image_tensor,
                use_cache=True,
                **gen_kwargs,
            )
        gate_stats = None
    text = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
    return text.lstrip("!").strip(), gate_stats


def load_existing_ids(output_path):
    if not output_path.exists():
        return set()
    ids = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = obj.get("id")
            if sample_id is not None:
                ids.add(sample_id)
    return ids


def main():
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        done_ids = load_existing_ids(output_path)
        file_mode = "a"
    else:
        done_ids = set()
        file_mode = "w"

    dataset = datasets.load_dataset(args.dataset_path, split=args.split)
    if args.sample_mode == "random":
        dataset = dataset.shuffle(seed=args.sample_seed)
    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    model_wrapper = build_model(args)

    num_written = 0
    with output_path.open(file_mode, encoding="utf-8") as fout:
        for doc in tqdm(dataset, desc=f"M3CoT {args.split}"):
            if doc["id"] in done_ids:
                continue

            prompt_text = build_prompt(doc, args.prompt)
            prediction, gate_stats = generate_one(model_wrapper, doc, prompt_text, args)

            record = {
                "id": doc["id"],
                "choices": list(doc["choices"]),
                "answer": doc["answer"],
                "domain": doc["domain"],
                "topic": doc["topic"],
                "method": args.method,
                "vrg_gate": args.vrg_gate if args.method == "vrg" else "none",
                "messages": [prompt_text, prediction],
            }
            if gate_stats is not None:
                record["vrg_gate_stats"] = gate_stats
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1
            if num_written % max(1, args.save_every) == 0:
                fout.flush()

    print(f"Wrote {num_written} predictions to {output_path}")


if __name__ == "__main__":
    main()
