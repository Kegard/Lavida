import argparse
import copy
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LETTER_MAP = "ABCDEFG"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LaViDa native prefix-LM inference on M3CoT with official prompt formatting."
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output-jsonl", required=True)

    parser.add_argument("--model-path", default="/data/jindong_gu/LaViDa/weight/lavida-reason")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="/data/jindong_gu/LaViDa/weight/siglip")
    parser.add_argument("--vision-projector", default="mlp2x_gelu")
    parser.add_argument("--vision-hidden-size", type=int, default=1152)
    parser.add_argument("--mm-pooler-ratio", type=int, default=2)
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16"])

    parser.add_argument(
        "--prompt",
        default="paper_cot",
        choices=["paper_cot", "paper_cot_eval_format", "official_m3cot_cot", "official_direct"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--step-per-block", type=int, default=None)
    parser.add_argument("--schedule", default="none", choices=["none", "shift", "cosine", "logit_normal"])
    parser.add_argument("--schedule-shift", type=float, default=0.33)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.set_defaults(prefix_lm=True)
    parser.add_argument("--prefix-lm", dest="prefix_lm", action="store_true")
    parser.add_argument("--no-prefix-lm", dest="prefix_lm", action="store_false")
    parser.add_argument("--verbose-generation", action="store_true")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-mode", default="sequential", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_torch_dtype(dtype_name):
    import torch

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_official_base_prompt(doc):
    parts = []
    context = (doc.get("context") or "").strip()
    if context:
        parts.append(f"[Context]\n{context}")
    parts.append(f"[Question]\n{doc['question']}")
    choices = "\n".join(f"({LETTER_MAP[i]}) {choice}" for i, choice in enumerate(doc["choices"]))
    parts.append(f"[Choices]\n{choices}")
    return "\n".join(parts)


def build_prompt(doc, prompt_style):
    base = build_official_base_prompt(doc)
    if prompt_style == "paper_cot":
        return (
            base
            + "\n\nPlease reason step by step, and answer the question with option letter "
            + "from given choices in the format of Answer: <option letter>."
        )
    if prompt_style == "paper_cot_eval_format":
        return (
            base
            + "\n\nPlease reason step by step, and answer the question with option letter "
            + "from given choices. End your response exactly in the format Answer: (X), "
            + "where X is the option letter."
        )
    if prompt_style == "official_m3cot_cot":
        return base + "\n\nLet's think step-by-step!"
    if prompt_style == "official_direct":
        return base
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def clean_generated_text(text):
    return text.lstrip("!").strip()


def normalize_answer(answer):
    if isinstance(answer, int):
        return LETTER_MAP[answer]
    return str(answer)


def load_done_ids(output_path):
    if not output_path.exists():
        return set()
    done_ids = set()
    with output_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = obj.get("id")
            if sample_id is not None:
                done_ids.add(sample_id)
    return done_ids


def select_dataset(args):
    import datasets

    dataset = datasets.load_dataset(args.dataset_path, split=args.split)
    if args.sample_mode == "random":
        dataset = dataset.shuffle(seed=args.sample_seed)
    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    return dataset


def load_lavida_model(args):
    from llava.model.builder import load_pretrained_model

    dtype = get_torch_dtype(args.torch_dtype)
    vision_kwargs = {
        "mm_vision_tower": args.vision_tower,
        "mm_resampler_type": None,
        "mm_projector_type": args.vision_projector,
        "mm_hidden_size": args.vision_hidden_size,
        "mm_pooler_ratio": args.mm_pooler_ratio,
        "use_mm_proj": True,
        "mm_patch_merge_type": "spatial_unpad",
    }
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        args.model_name,
        device_map=args.device_map,
        vision_kwargs=vision_kwargs,
        torch_dtype=args.torch_dtype,
    )
    model.eval()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.to(dtype)
    return tokenizer, model, image_processor


def prepare_inputs(args, tokenizer, model, image_processor, doc, prompt_text):
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    image = doc["image"].convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    dtype = get_torch_dtype(args.torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=args.device)

    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    if "llama_3" in args.conv_template or "llada" in args.conv_template:
        conv = copy.deepcopy(conv_templates[args.conv_template])
    else:
        conv = conv_templates[args.conv_template].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(args.device)
    return image, image_tensor, input_ids


def build_generation_kwargs(args, tokenizer, image):
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kwargs = {
        "pad_token_id": pad_token_id,
        "image_sizes": [image.size],
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "block_length": args.block_length,
        "prefix_lm": args.prefix_lm,
        "tokenizer": tokenizer,
        "verbose": args.verbose_generation,
    }
    if args.step_per_block is not None:
        gen_kwargs["step_per_block"] = args.step_per_block
    else:
        gen_kwargs["step_ratio"] = args.step_ratio
    if args.schedule != "none":
        gen_kwargs["schedule"] = args.schedule
        if args.schedule == "shift":
            gen_kwargs["schedule_kwargs"] = {"shift": args.schedule_shift}
    return gen_kwargs


def generate_one(args, tokenizer, model, image_processor, doc, prompt_text):
    import torch

    image, image_tensor, input_ids = prepare_inputs(args, tokenizer, model, image_processor, doc, prompt_text)
    with torch.inference_mode():
        sequences = model.generate(
            input_ids,
            images=image_tensor,
            **build_generation_kwargs(args, tokenizer, image),
        )
    if args.verbose_generation:
        sequences = sequences[0]
    text = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
    return clean_generated_text(text)


def generation_metadata(args):
    return {
        "model_path": args.model_path,
        "vision_tower": args.vision_tower,
        "conv_template": args.conv_template,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "block_length": args.block_length,
        "step_ratio": None if args.step_per_block is not None else args.step_ratio,
        "step_per_block": args.step_per_block,
        "schedule": args.schedule,
        "schedule_shift": args.schedule_shift if args.schedule == "shift" else None,
        "temperature": args.temperature,
        "prefix_lm": args.prefix_lm,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
    }


def main():
    from tqdm import tqdm

    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set() if args.overwrite else load_done_ids(output_path)
    file_mode = "w" if args.overwrite or not output_path.exists() else "a"

    dataset = select_dataset(args)
    tokenizer, model, image_processor = load_lavida_model(args)
    metadata = generation_metadata(args)

    written = 0
    with output_path.open(file_mode, encoding="utf-8") as fout:
        for doc in tqdm(dataset, desc=f"M3CoT {args.split} aligned"):
            if doc["id"] in done_ids:
                continue
            prompt_text = build_prompt(doc, args.prompt)
            prediction = generate_one(args, tokenizer, model, image_processor, doc, prompt_text)
            record = {
                "id": doc["id"],
                "category": doc.get("category"),
                "context": doc.get("context", ""),
                "question": doc["question"],
                "choices": list(doc["choices"]),
                "image_id": doc.get("image_id"),
                "answer": normalize_answer(doc["answer"]),
                "domain": doc["domain"],
                "topic": doc["topic"],
                "split": args.split,
                "messages": [prompt_text, prediction],
                "generation": metadata,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % max(1, args.save_every) == 0:
                fout.flush()
    print(f"Wrote {written} predictions to {output_path}")


if __name__ == "__main__":
    main()
