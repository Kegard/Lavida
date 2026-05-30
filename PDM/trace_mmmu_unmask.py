import argparse
import ast
import copy
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token


MC_PROMPT = "Answer with the option's letter from the given choices directly."
DEFAULT_MASK_ID = 126336


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace the diffusion step where the MMMU multiple-choice answer token is unmasked."
    )
    parser.add_argument("--dataset-path", default="lmms-lab/MMMU")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", default="experiment/mmmu_unmask_trace.jsonl")

    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")

    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--step-ratio", type=float, default=1)
    parser.add_argument("--remasking", default="margin", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=0.33)
    parser.add_argument("--mask-id", type=int, default=DEFAULT_MASK_ID)
    return parser.parse_args()


def parse_options(raw_options):
    if isinstance(raw_options, str):
        return ast.literal_eval(raw_options)
    return list(raw_options)


def option_letters(num_options):
    return [chr(ord("A") + idx) for idx in range(num_options)]


def format_options(options):
    letters = option_letters(len(options))
    return "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))


def construct_prompt(doc):
    question = doc["question"]
    options = parse_options(doc["options"])
    return f"{question}\n{format_options(options)}\n\n{MC_PROMPT}"


def extract_images(doc):
    images = []
    for idx in range(1, 8):
        key = f"image_{idx}"
        image = doc.get(key)
        if image is not None:
            images.append(image.convert("RGB"))
    return images


def build_conversation_prompt(context, num_images, conv_template):
    question = context
    if num_images > 0 and DEFAULT_IMAGE_TOKEN not in question:
        image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * num_images)
        question = f"{image_tokens}\n{question}"

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def move_images_to_device(images, image_processor, model_config, device):
    if not images:
        return None
    image_tensor = process_images(images, image_processor, model_config)
    if isinstance(image_tensor, list):
        return [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
    return image_tensor.to(dtype=torch.bfloat16, device=device)


def parse_choice_from_response(response, choices):
    response = response.strip().strip(",.!?;:'\"")
    padded = f" {response} "

    candidates = []
    for choice in choices:
        if f"({choice})" in padded:
            candidates.append((choice, padded.rfind(f"({choice})")))
        elif f"{choice}." in padded:
            candidates.append((choice, padded.rfind(f"{choice}.")))
        elif f" {choice} " in padded:
            candidates.append((choice, padded.rfind(f" {choice} ")))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def normalize_choice_token(token_text, choices):
    normalized = token_text.strip().strip(",.!?;:'\"()[]{}")
    return normalized if normalized in choices else None


def find_choice_token_position(tokenizer, token_ids, answer_letter, choices, mask_id):
    if answer_letter is None:
        return None
    for pos, token_id in enumerate(token_ids):
        token_id = int(token_id)
        if token_id == mask_id:
            continue
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        if normalize_choice_token(token_text, choices) == answer_letter:
            return pos
    return None


def first_unmask_step(history, position, mask_id):
    if position is None:
        return None
    for step_idx, state in enumerate(history):
        if position < state.shape[1] and int(state[0, position].item()) != mask_id:
            return step_idx + 1
    return None


def load_mmmu_split(dataset_path, dataset_name, split):
    if dataset_name is None:
        return load_dataset(dataset_path, split=split)
    return load_dataset(dataset_path, dataset_name, split=split)


def clean_model_output(text, conv_template):
    text = text.lstrip("!")
    text = text.replace("<|im_end|>\n", "")
    text = text.replace("<|im_end|>", "")

    conv = conv_templates[conv_template]
    stop_strings = [conv.sep]
    if conv.sep2 is not None:
        stop_strings.append(conv.sep2)

    for stop_str in stop_strings:
        if stop_str and stop_str in text:
            text = text.split(stop_str, 1)[0]
    return text.strip()


def run_generation(model, input_ids, image_tensor, images, tokenizer, args, schedule, schedule_kwargs):
    image_sizes = [image.size for image in images] if images else None

    return model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=args.max_new_tokens,
        block_length=args.max_new_tokens,
        step_ratio=args.step_ratio,
        tokenizer=tokenizer,
        prefix_lm=True,
        verbose=True,
        remasking=args.remasking,
        schedule=schedule,
        schedule_kwargs=schedule_kwargs,
    )


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
        device_map=args.device_map,
        vision_kwargs=vision_kwargs,
        torch_dtype="bfloat16",
    )
    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)

    dataset = load_mmmu_split(args.dataset_path, args.dataset_name, args.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schedule = None if args.schedule == "none" else args.schedule
    schedule_kwargs = {"shift": args.schedule_shift} if schedule == "shift" else None

    num_written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for dataset_index, doc in enumerate(dataset):
            if dataset_index < args.start_index:
                continue
            if doc.get("question_type") != "multiple-choice":
                continue
            if args.limit is not None and num_written >= args.limit:
                break

            context = construct_prompt(doc)
            images = extract_images(doc)
            prompt = build_conversation_prompt(context, len(images), args.conv_template)
            image_tensor = move_images_to_device(images, image_processor, model.config, args.device)
            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(args.device)

            t0 = time.time()
            cont, history = run_generation(
                model,
                input_ids,
                image_tensor,
                images,
                tokenizer,
                args,
                schedule,
                schedule_kwargs,
            )
            elapsed = time.time() - t0

            generated_ids = cont[0].detach().cpu().tolist()
            raw_output = clean_model_output(
                tokenizer.batch_decode(cont, skip_special_tokens=True)[0],
                args.conv_template,
            )
            choices = option_letters(len(parse_options(doc["options"])))
            pred_letter = parse_choice_from_response(raw_output, choices)
            pred_position = find_choice_token_position(
                tokenizer,
                generated_ids,
                pred_letter,
                choices,
                args.mask_id,
            )

            first_token_step = first_unmask_step(history, 0, args.mask_id)
            answer_step = first_unmask_step(history, pred_position, args.mask_id)
            record = {
                "dataset_index": dataset_index,
                "id": doc.get("id"),
                "question_type": doc.get("question_type"),
                "gold": doc.get("answer"),
                "backend": "llada",
                "pred": pred_letter,
                "raw_output": raw_output,
                "answer_token_position": pred_position,
                "answer_unmask_step": answer_step,
                "first_token_unmask_step": first_token_step,
                "num_history_steps": len(history),
                "max_new_tokens": args.max_new_tokens,
                "block_length": args.max_new_tokens,
                "step_ratio": args.step_ratio,
                "schedule": schedule,
                "schedule_kwargs": schedule_kwargs,
                "elapsed_sec": elapsed,
                "remasking": args.remasking,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            num_written += 1
            print(
                f"[{num_written}] id={record['id']} gold={record['gold']} "
                f"pred={record['pred']} step={record['answer_unmask_step']} output={raw_output!r}"
            )

    print(f"Wrote {num_written} records to {output_path}")


if __name__ == "__main__":
    main()
