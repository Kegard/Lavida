import argparse
import ast
import copy
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute token-level PDM scores for LaViDa and visualize them."
    )
    parser.add_argument("--image", default=None, help="Path to the input image.")
    parser.add_argument("--question", default=None, help="User text prompt.")
    parser.add_argument("--answer-text", default=None, help="Optional answer text. If omitted, the model will generate one.")
    parser.add_argument("--output-json", default="experiment/lavida_pdm.json")
    parser.add_argument("--output-plot", default="experiment/lavida_pdm.png")
    parser.add_argument("--dataset-path", default=None, help="Optional dataset path, e.g. lmms-lab/MMMU.")
    parser.add_argument("--dataset-type", default="mmmu", choices=["mmmu", "textvqa"])
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset-index", type=int, default=None)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-dataset-answer", action="store_true")
    parser.add_argument("--output-jsonl", default="experiment/lavida_pdm_batch.jsonl")
    parser.add_argument("--output-plot-dir", default="experiment/lavida_pdm_plots")
    parser.add_argument("--aggregate-plot", default="experiment/lavida_pdm_batch_hist.png")

    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")

    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=0.33)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--mask-id", type=int, default=126336)
    return parser.parse_args()


def validate_args(args):
    using_dataset = args.dataset_path is not None
    if using_dataset:
        if args.dataset_index is not None and args.doc_id is not None:
            raise ValueError("Pass at most one of --dataset-index and --doc-id.")
    elif args.image is None or args.question is None:
        raise ValueError("Pass --image and --question, or use --dataset-path with a dataset sample selector.")


def parse_options(raw_options):
    if isinstance(raw_options, str):
        return ast.literal_eval(raw_options)
    return list(raw_options)


def option_letters(num_options):
    return [chr(ord("A") + idx) for idx in range(num_options)]


def format_options(options):
    letters = option_letters(len(options))
    return "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))


def construct_mmmu_question(doc):
    question = doc["question"]
    if doc.get("question_type") == "multiple-choice":
        options = parse_options(doc["options"])
        return f"{question}\n{format_options(options)}\n\nAnswer with the option's letter from the given choices directly."
    return f"{question}\n\nAnswer the question using a single word or phrase."


def construct_textvqa_question(doc):
    return f"{doc['question'].capitalize()}\nAnswer the question using a single word or phrase."


def extract_mmmu_images(doc):
    images = []
    for idx in range(1, 8):
        key = f"image_{idx}"
        image = doc.get(key)
        if image is not None:
            images.append(image.convert("RGB"))
    return images


def extract_textvqa_images(doc):
    return [doc["image"].convert("RGB")]


def build_dataset_question(doc, dataset_type):
    if dataset_type == "textvqa":
        return construct_textvqa_question(doc)
    return construct_mmmu_question(doc)


def extract_dataset_images(doc, dataset_type):
    if dataset_type == "textvqa":
        return extract_textvqa_images(doc)
    return extract_mmmu_images(doc)


def dataset_doc_id(doc, dataset_type):
    if dataset_type == "textvqa":
        return str(doc.get("question_id"))
    return doc.get("id")


def dataset_question_type(doc, dataset_type):
    if dataset_type == "textvqa":
        return "open-ended"
    return doc.get("question_type")


def load_dataset_doc(dataset_path, dataset_name, split, dataset_index, doc_id):
    dataset = load_dataset_split(dataset_path, dataset_name, split)

    if doc_id is not None:
        for idx, doc in enumerate(dataset):
            if doc.get("id") == doc_id:
                return idx, doc
        raise ValueError(f"Could not find doc_id={doc_id!r} in {dataset_path}:{split}.")

    if dataset_index is None or dataset_index < 0 or dataset_index >= len(dataset):
        raise ValueError(f"--dataset-index must be within [0, {len(dataset) - 1}].")
    return dataset_index, dataset[dataset_index]


def load_dataset_split(dataset_path, dataset_name, split):
    if dataset_name is None:
        return load_dataset(dataset_path, split=split)
    return load_dataset(dataset_path, dataset_name, split=split)


def build_prompt(question, conv_template, num_images):
    text = question
    if num_images > 0 and DEFAULT_IMAGE_TOKEN not in text:
        image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * num_images)
        text = f"{image_tokens}\n{text}"

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def load_image_tensor(image_path, image_processor, model_config, device):
    image = load_image(image_path)
    image_tensor = process_images([image], image_processor, model_config)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=torch.bfloat16, device=device)
    return image, image_tensor


def load_images_tensor(images, image_processor, model_config, device):
    image_tensor = process_images(images, image_processor, model_config)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=torch.bfloat16, device=device)
    return image_tensor


def clean_output(text, conv_template):
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


def prepare_prefix_cache(model, input_ids, images=None, image_sizes=None):
    position_ids = None
    attention_mask = None
    if images is not None:
        (_, position_ids, attention_mask, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            None,
            None,
            images,
            ["image"],
            image_sizes=image_sizes,
        )
    else:
        inputs_embeds = model.get_model().embed_tokens(input_ids)

    prefill = model.get_model()(None, input_embeddings=inputs_embeds, use_cache=True)
    return prefill.attn_key_values


def get_state_logits(base_model, past_key_values, state_tokens):
    inputs_embeds = base_model.transformer.wte(state_tokens)
    return base_model(None, input_embeddings=inputs_embeds, past_key_values=past_key_values).logits


def token_labels(tokenizer, answer_ids):
    labels = []
    for token_id in answer_ids.tolist():
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        token_text = token_text.replace("\n", "\\n")
        labels.append(token_text if token_text else f"<{token_id}>")
    return labels


def generate_trace(model, tokenizer, input_ids, image_tensor, image_sizes, args):
    schedule = None if args.schedule == "none" else args.schedule
    schedule_kwargs = {"shift": args.schedule_shift} if schedule == "shift" else None
    cont, history = model.generate(
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
        remasking=args.remasking,
        verbose=True,
        schedule=schedule,
        schedule_kwargs=schedule_kwargs,
    )
    raw_output = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    return cont, history, clean_output(raw_output, args.conv_template)


def select_effective_answer_tokens(tokenizer, generated_ids, answer_text):
    if not answer_text:
        return generated_ids[:0]
    answer_token_count = tokenizer(answer_text, return_tensors="pt").input_ids.shape[1]
    return generated_ids[:answer_token_count]


def compute_pdm(model, tokenizer, cond_input_ids, uncond_input_ids, generated_ids, answer_ids, history, image_tensor, image_sizes, mask_id):
    cond_past = prepare_prefix_cache(model, cond_input_ids, images=image_tensor, image_sizes=image_sizes)
    uncond_past = prepare_prefix_cache(model, uncond_input_ids, images=None, image_sizes=None)
    base_model = model.get_model()
    answer_len = answer_ids.shape[0]
    full_len = generated_ids.shape[0]

    token_steps = [None] * answer_len
    cond_scores = [None] * answer_len
    uncond_scores = [None] * answer_len
    pdm_scores = [None] * answer_len
    step_records = []

    previous_state = torch.full((1, full_len), mask_id, dtype=torch.long, device=answer_ids.device)
    for step_idx, current_state_cpu in enumerate(history, start=1):
        current_state = current_state_cpu[:, :full_len].to(answer_ids.device)
        newly_unmasked = (previous_state == mask_id) & (current_state != mask_id)
        new_positions = torch.nonzero(newly_unmasked[0], as_tuple=False).flatten().tolist()

        cond_logits = get_state_logits(base_model, cond_past, previous_state)
        uncond_logits = get_state_logits(base_model, uncond_past, previous_state)
        cond_log_probs = F.log_softmax(cond_logits, dim=-1)
        uncond_log_probs = F.log_softmax(uncond_logits, dim=-1)

        step_record = {
            "step": step_idx,
            "new_positions": new_positions,
            "tokens": [],
            "step_mean_pdm": None,
        }
        step_pdms = []
        for pos in new_positions:
            target_id = int(current_state[0, pos].item())
            cond_score = float(cond_log_probs[0, pos, target_id].item())
            uncond_score = float(uncond_log_probs[0, pos, target_id].item())
            pdm_score = cond_score - uncond_score
            step_pdms.append(pdm_score)

            if pos < answer_len:
                token_steps[pos] = step_idx
                cond_scores[pos] = cond_score
                uncond_scores[pos] = uncond_score
                pdm_scores[pos] = pdm_score
            step_record["tokens"].append(
                {
                    "position": pos,
                    "token_id": target_id,
                    "token_text": tokenizer.decode([target_id], skip_special_tokens=False).replace("\n", "\\n"),
                    "conditional_log_prob": cond_score,
                    "unconditional_log_prob": uncond_score,
                    "pdm": pdm_score,
                }
            )
        if step_pdms:
            step_record["step_mean_pdm"] = sum(step_pdms) / len(step_pdms)
        step_records.append(step_record)
        previous_state = current_state

    return {
        "conditional_log_probs": cond_scores,
        "unconditional_log_probs": uncond_scores,
        "pdm_scores": pdm_scores,
        "token_steps": token_steps,
        "step_records": step_records,
    }


def plot_pdm(step_records, output_path, title):
    steps = [record["step"] for record in step_records]
    pdm_values = [np.nan if record["step_mean_pdm"] is None else record["step_mean_pdm"] for record in step_records]

    if not steps:
        steps = [1]
        pdm_values = [0.0]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        steps,
        pdm_values,
        color="#2E8B57",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("PDM")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks(steps)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_batch_summary(mean_pdms, output_path, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(mean_pdms, bins=min(20, max(5, len(mean_pdms))), color="#2E8B57", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Mean PDM")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_sample_result(model, tokenizer, image_processor, args, images, question, sample_info):
    image_tensor = load_images_tensor(images, image_processor, model.config, args.device)

    cond_prompt = build_prompt(question, args.conv_template, num_images=len(images))
    uncond_prompt = build_prompt(question, args.conv_template, num_images=0)
    cond_input_ids = tokenizer_image_token(
        cond_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    uncond_input_ids = tokenizer_image_token(
        uncond_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)

    cont, history, generated_answer_text = generate_trace(
        model,
        tokenizer,
        cond_input_ids,
        image_tensor,
        [image.size for image in images],
        args,
    )
    generated_ids = cont[0]
    answer_text_override = sample_info.get("answer_text_override")
    if answer_text_override is not None and answer_text_override != generated_answer_text:
        raise ValueError(
            "History-based PDM must follow the model's actual generation trajectory. "
            "The provided --answer-text does not match the generated answer."
        )
    answer_text = generated_answer_text if answer_text_override is None else answer_text_override

    answer_ids = select_effective_answer_tokens(tokenizer, generated_ids, answer_text).to(args.device)
    stats = compute_pdm(
        model,
        tokenizer,
        cond_input_ids,
        uncond_input_ids,
        generated_ids,
        answer_ids,
        history,
        image_tensor,
        [image.size for image in images],
        args.mask_id,
    )
    token_texts = token_labels(tokenizer, answer_ids.cpu())

    result = {
        "image": sample_info.get("image_path"),
        "question": question,
        "answer_text": answer_text,
        "generated_answer_text": generated_answer_text,
        "answer_token_ids": answer_ids.detach().cpu().tolist(),
        "answer_tokens": token_texts,
        "conditional_log_probs": stats["conditional_log_probs"],
        "unconditional_log_probs": stats["unconditional_log_probs"],
        "pdm_scores": stats["pdm_scores"],
        "token_steps": stats["token_steps"],
        "step_records": stats["step_records"],
        "total_pdm": float(sum(score for score in stats["pdm_scores"] if score is not None)),
        "mean_pdm": float(
            sum(score for score in stats["pdm_scores"] if score is not None) /
            max(1, sum(score is not None for score in stats["pdm_scores"]))
        ),
    }
    result.update(sample_info)
    return result, token_texts


def save_single_result(result, token_texts, output_json, output_plot):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)

    plot_pdm(result["step_records"], output_plot, title="LaViDa Step-level PDM")


def main():
    args = parse_args()
    validate_args(args)
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

    if args.dataset_path is None:
        result, token_texts = compute_sample_result(
            model,
            tokenizer,
            image_processor,
            args,
            [load_image(args.image)],
            args.question,
            {"image_path": args.image, "answer_text_override": args.answer_text},
        )
        save_single_result(result, token_texts, Path(args.output_json), Path(args.output_plot))
        print(f"Saved PDM JSON to {args.output_json}")
        print(f"Saved PDM plot to {args.output_plot}")
        return

    if args.dataset_index is not None or args.doc_id is not None:
        dataset_index, doc = load_dataset_doc(
            args.dataset_path,
            args.dataset_name,
            args.split,
            args.dataset_index,
            args.doc_id,
        )
        images = extract_dataset_images(doc, args.dataset_type)
        if not images:
            raise ValueError("The selected dataset sample has no images.")
        sample_info = {
            "dataset_path": args.dataset_path,
            "dataset_type": args.dataset_type,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "dataset_index": dataset_index,
            "doc_id": dataset_doc_id(doc, args.dataset_type),
            "question_type": dataset_question_type(doc, args.dataset_type),
            "answer_text_override": str(doc.get("answer")) if args.use_dataset_answer and doc.get("answer") is not None else args.answer_text,
        }
        result, token_texts = compute_sample_result(
            model,
            tokenizer,
            image_processor,
            args,
            images,
            build_dataset_question(doc, args.dataset_type),
            sample_info,
        )
        save_single_result(result, token_texts, Path(args.output_json), Path(args.output_plot))
        print(f"Saved PDM JSON to {args.output_json}")
        print(f"Saved PDM plot to {args.output_plot}")
        return

    dataset = load_dataset_split(args.dataset_path, args.dataset_name, args.split)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.output_plot_dir)
    aggregate_plot = Path(args.aggregate_plot)
    mean_pdms = []
    num_written = 0

    with output_jsonl.open("w", encoding="utf-8") as fout:
        for dataset_index, doc in enumerate(dataset):
            if dataset_index < args.start_index:
                continue
            if args.limit is not None and num_written >= args.limit:
                break

            images = extract_dataset_images(doc, args.dataset_type)
            if not images:
                continue

            sample_info = {
                "dataset_path": args.dataset_path,
                "dataset_type": args.dataset_type,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "dataset_index": dataset_index,
                "doc_id": dataset_doc_id(doc, args.dataset_type),
                "question_type": dataset_question_type(doc, args.dataset_type),
                "answer_text_override": str(doc.get("answer")) if args.use_dataset_answer and doc.get("answer") is not None else args.answer_text,
            }
            result, token_texts = compute_sample_result(
                model,
                tokenizer,
                image_processor,
                args,
                images,
                build_dataset_question(doc, args.dataset_type),
                sample_info,
            )
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            sample_plot = plot_dir / f"{dataset_index:06d}_{dataset_doc_id(doc, args.dataset_type) or 'sample'}.png"
            plot_pdm(
                result["step_records"],
                sample_plot,
                title=f"LaViDa PDM: {dataset_doc_id(doc, args.dataset_type) or dataset_index}",
            )
            mean_pdms.append(result["mean_pdm"])
            num_written += 1
            print(
                f"[{num_written}] id={result.get('doc_id')} "
                f"answer={result['answer_text']!r} mean_pdm={result['mean_pdm']:.4f}"
            )

    if mean_pdms:
        plot_batch_summary(mean_pdms, aggregate_plot, title="LaViDa Mean PDM Distribution")
    print(f"Saved batch PDM JSONL to {output_jsonl}")
    print(f"Saved per-sample PDM plots to {plot_dir}")
    if mean_pdms:
        print(f"Saved aggregate PDM plot to {aggregate_plot}")


if __name__ == "__main__":
    main()
