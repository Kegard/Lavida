import argparse
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


TEXTVQA_PROMPT = "Answer the question using a single word or phrase."
DEFAULT_MASK_ID = 126336


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace the diffusion step where each TextVQA generated token is unmasked."
    )
    parser.add_argument("--dataset-path", default="lmms-lab/textvqa")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", default="experiment/textvqa_unmask_trace.jsonl")

    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")

    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--step-ratio", type=float, default=1.0)
    parser.add_argument("--remasking", default="margin", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=0.33)
    parser.add_argument("--mask-id", type=int, default=DEFAULT_MASK_ID)
    return parser.parse_args()


def construct_prompt(doc):
    return f"{doc['question'].capitalize()}\n{TEXTVQA_PROMPT}"


def extract_images(doc):
    image = doc.get("image")
    if image is None:
        return []
    return [image.convert("RGB")]


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


def load_textvqa_split(dataset_path, dataset_name, split):
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


def first_unmask_step(history, position, mask_id):
    for step_idx, state in enumerate(history):
        if position < state.shape[1] and int(state[0, position].item()) != mask_id:
            return step_idx + 1
    return None


def is_normal_token(tokenizer, token_id, mask_id):
    token_id = int(token_id)
    if token_id == mask_id:
        return False
    token_text = tokenizer.decode([token_id], skip_special_tokens=True)
    return token_text != ""


def build_position_records(tokenizer, token_ids, history, mask_id):
    all_records = []
    token_records = []

    for position, token_id in enumerate(token_ids):
        token_id = int(token_id)
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        normal_token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        record = {
            "position": position,
            "token_id": token_id,
            "token_text": token_text,
            "normal_token_text": normal_token_text,
            "is_normal_token": is_normal_token(tokenizer, token_id, mask_id),
            "unmask_step": first_unmask_step(history, position, mask_id),
        }
        all_records.append(record)
        if record["is_normal_token"]:
            token_records.append(record)

    return token_records, all_records


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

    dataset = load_textvqa_split(args.dataset_path, args.dataset_name, args.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schedule = None if args.schedule == "none" else args.schedule
    schedule_kwargs = {"shift": args.schedule_shift} if schedule == "shift" else None

    num_written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for dataset_index, doc in enumerate(dataset):
            if dataset_index < args.start_index:
                continue
            if args.limit is not None and num_written >= args.limit:
                break

            images = extract_images(doc)
            if not images:
                continue

            context = construct_prompt(doc)
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
            token_records, all_position_records = build_position_records(
                tokenizer,
                generated_ids,
                history,
                args.mask_id,
            )

            record = {
                "dataset_index": dataset_index,
                "doc_id": str(doc.get("question_id")),
                "question_id": doc.get("question_id"),
                "question_type": "open-ended",
                "question": context,
                "answers": doc.get("answers"),
                "ocr_tokens": doc.get("ocr_tokens"),
                "backend": "llada",
                "raw_output": raw_output,
                "generated_token_ids": generated_ids,
                "token_records": token_records,
                "token_unmask_steps": [item["unmask_step"] for item in token_records],
                "all_position_records": all_position_records,
                "num_normal_tokens": len(token_records),
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
                f"[{num_written}] id={record['doc_id']} "
                f"tokens={record['num_normal_tokens']} steps={record['token_unmask_steps']} "
                f"output={raw_output!r}"
            )

    print(f"Wrote {num_written} records to {output_path}")


if __name__ == "__main__":
    main()
