import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scale_Attention.reweight_patch import (
    build_prefix_from_multimodal_inputs,
    build_prompt,
    get_torch_dtype,
    maybe_disable_torch_compile,
)
from VRG.timestep_vrg import compute_step_vrg_alpha, generate_with_timestep_vrg
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run timestep-aware VRG on one image or a small dataset slice.")
    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--image", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--dataset-path", default=None, help="Optional dataset path, e.g. lmms-lab/textvqa.")
    parser.add_argument("--dataset-type", default="textvqa", choices=["textvqa"])
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--step-ratio", type=float, default=1.0)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument("--vrg-alpha-start", type=float, default=0.0)
    parser.add_argument("--vrg-alpha-end", type=float, default=1.0)
    parser.add_argument("--vrg-alpha-schedule", default="linear", choices=["linear", "cosine", "power"])
    parser.add_argument("--vrg-alpha-power", type=float, default=2.0)
    parser.add_argument("--vrg-null-visual-mode", default="zeros", choices=["zeros", "mask_token"])
    parser.add_argument("--output", default="VRG/timestep_vrg_output.json")
    parser.add_argument("--save-trace", action="store_true", help="Save per-step cond/uncond logits-difference traces.")
    parser.add_argument("--trace-topk", type=int, default=10, help="Top-k positive/negative delta tokens to save per representative position.")
    parser.add_argument("--trace-max-positions", type=int, default=4, help="How many representative active positions to save per step.")
    parser.add_argument("--trace-output", default=None, help="Optional path for single-sample trace JSON output.")
    parser.add_argument("--trace-pt-output", default=None, help="Optional path for single-sample trace torch.save output.")
    parser.add_argument("--batch-output-dir", default=None, help="Directory for dataset-slice outputs. Defaults to output stem + '_batch'.")
    return parser.parse_args()


def load_dataset_split(dataset_path, dataset_name, split):
    if dataset_name is None:
        return load_dataset(dataset_path, split=split)
    return load_dataset(dataset_path, dataset_name, split=split)


def get_textvqa_doc(dataset, dataset_index):
    if dataset_index is None or dataset_index < 0 or dataset_index >= len(dataset):
        raise ValueError(f"--dataset-index must be within [0, {len(dataset) - 1}].")
    return dataset_index, dataset[dataset_index]


def build_textvqa_question(doc):
    return f"{doc['question'].capitalize()}\nAnswer the question using a single word or phrase."


def build_textvqa_sample(args, doc, dataset_index):
    image = doc["image"].convert("RGB")
    question = build_textvqa_question(doc)
    return {
        "image": image,
        "image_source": f"{args.dataset_path}:{args.split}:{dataset_index}",
        "question": question,
        "dataset_meta": {
            "dataset_path": args.dataset_path,
            "dataset_name": args.dataset_name,
            "dataset_type": args.dataset_type,
            "split": args.split,
            "dataset_index": int(dataset_index),
            "question_id": str(doc.get("question_id")),
            "answers": doc.get("answers"),
            "ocr_tokens": doc.get("ocr_tokens"),
        },
    }


def load_single_debug_sample(args):
    if args.dataset_path is None:
        if args.image is None or args.question is None:
            raise ValueError("Pass --image and --question, or pass --dataset-path for a dataset sample.")
        image = Image.open(args.image).convert("RGB")
        return {
            "image": image,
            "image_source": args.image,
            "question": args.question,
            "dataset_meta": None,
        }

    if args.dataset_type != "textvqa":
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")

    dataset = load_dataset_split(args.dataset_path, args.dataset_name, args.split)
    dataset_index, doc = get_textvqa_doc(dataset, args.start_index)
    return build_textvqa_sample(args, doc, dataset_index)


def iter_debug_samples(args):
    if args.dataset_path is None:
        yield 0, load_single_debug_sample(args)
        return

    if args.dataset_type != "textvqa":
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")
    if args.limit <= 0:
        raise ValueError("--limit must be > 0 for dataset-slice debugging.")

    dataset = load_dataset_split(args.dataset_path, args.dataset_name, args.split)
    end_index = min(args.start_index + args.limit, len(dataset))
    if args.start_index < 0 or args.start_index >= len(dataset):
        raise ValueError(f"--start-index must be within [0, {len(dataset) - 1}].")

    for dataset_index in range(args.start_index, end_index):
        yield dataset_index, build_textvqa_sample(args, dataset[dataset_index], dataset_index)


def summarize_alpha_schedule(max_new_tokens, block_length, step_ratio, alpha_start, alpha_end, alpha_schedule, alpha_power):
    num_blocks = max_new_tokens // block_length
    steps = max_new_tokens // num_blocks
    if step_ratio:
        steps = int(steps * step_ratio)
    total_steps = num_blocks * steps
    alpha_trace = [
        compute_step_vrg_alpha(
            global_step_idx=step_idx,
            total_steps=total_steps,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            schedule=alpha_schedule,
            power=alpha_power,
        )
        for step_idx in range(total_steps)
    ]
    return {
        "num_blocks": int(num_blocks),
        "steps_per_block": int(steps),
        "total_denoising_steps": int(total_steps),
        "alpha_trace": alpha_trace,
    }


def run_one_sample(args, model, tokenizer, image_processor, sample, output_path, trace_output_path=None, trace_pt_output_path=None):
    image = sample["image"]
    question = sample["question"]

    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=get_torch_dtype(args.torch_dtype), device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=get_torch_dtype(args.torch_dtype), device=args.device)

    prompt = build_prompt(question, args.conv_template)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)

    prefix_embeds, prefix_input_ids_full = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        attention_mask=attention_mask,
    )

    alpha_summary = summarize_alpha_schedule(
        max_new_tokens=int(args.max_new_tokens),
        block_length=int(args.block_length),
        step_ratio=float(args.step_ratio),
        alpha_start=float(args.vrg_alpha_start),
        alpha_end=float(args.vrg_alpha_end),
        alpha_schedule=args.vrg_alpha_schedule,
        alpha_power=float(args.vrg_alpha_power),
    )

    with torch.no_grad():
        vrg_outputs = generate_with_timestep_vrg(
            core_model=model.get_model(),
            prefix_embeds=prefix_embeds,
            prefix_input_ids_full=prefix_input_ids_full,
            max_new_tokens=int(args.max_new_tokens),
            block_length=int(args.block_length),
            temperature=float(args.temperature),
            remasking=args.remasking,
            schedule=args.schedule,
            schedule_shift=float(args.schedule_shift),
            step_ratio=float(args.step_ratio),
            alpha_start=float(args.vrg_alpha_start),
            alpha_end=float(args.vrg_alpha_end),
            alpha_schedule=args.vrg_alpha_schedule,
            alpha_power=float(args.vrg_alpha_power),
            null_visual_mode=args.vrg_null_visual_mode,
            return_trace=bool(args.save_trace),
            trace_topk=int(args.trace_topk),
            trace_max_positions=int(args.trace_max_positions),
            tokenizer=tokenizer,
        )
    if args.save_trace:
        sequences, last_step_meta, final_meta, trace_records = vrg_outputs
    else:
        sequences, last_step_meta, final_meta = vrg_outputs
        trace_records = None

    final_text = tokenizer.batch_decode(
        sequences,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].replace("<|endoftext|>", "").strip()

    payload = {
        "image": sample["image_source"],
        "question": question,
        "dataset_meta": sample["dataset_meta"],
        "prompt": prompt,
        "final_text": final_text,
        "vrg_alpha_start": float(args.vrg_alpha_start),
        "vrg_alpha_end": float(args.vrg_alpha_end),
        "vrg_alpha_schedule": args.vrg_alpha_schedule,
        "vrg_alpha_power": float(args.vrg_alpha_power),
        "vrg_null_visual_mode": args.vrg_null_visual_mode,
        "schedule": args.schedule,
        "schedule_shift": float(args.schedule_shift),
        "step_ratio": float(args.step_ratio),
        "block_length": int(args.block_length),
        "max_new_tokens": int(args.max_new_tokens),
        "alpha_summary": alpha_summary,
        "last_step_meta": last_step_meta,
        "final_meta": final_meta,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_trace:
        trace_output_path = Path(trace_output_path) if trace_output_path else output_path.with_suffix(".trace.json")
        trace_output_path.parent.mkdir(parents=True, exist_ok=True)
        trace_payload = {
            "image": sample["image_source"],
            "question": question,
            "dataset_meta": sample["dataset_meta"],
            "prompt": prompt,
            "final_text": final_text,
            "max_new_tokens": int(args.max_new_tokens),
            "block_length": int(args.block_length),
            "step_ratio": float(args.step_ratio),
            "alpha_summary": alpha_summary,
            "last_step_meta": last_step_meta,
            "final_meta": final_meta,
            "trace_records": trace_records,
        }
        trace_output_path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        trace_pt_output_path = Path(trace_pt_output_path) if trace_pt_output_path else output_path.with_suffix(".trace.pt")
        trace_pt_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trace_payload, trace_pt_output_path)

    print(f"Saved summary to {output_path}")
    if args.save_trace:
        print(f"Saved trace JSON to {trace_output_path}")
        print(f"Saved trace PT to {trace_pt_output_path}")
    print(f"Alpha schedule: {args.vrg_alpha_schedule} from {args.vrg_alpha_start} to {args.vrg_alpha_end}")
    print(f"Total denoising steps: {alpha_summary['total_denoising_steps']}")
    print(f"Final text: {final_text}")
    return payload


def batch_output_dir(args):
    if args.batch_output_dir is not None:
        return Path(args.batch_output_dir)
    output_path = Path(args.output)
    return output_path.with_suffix("")


def main():
    args = parse_args()
    restore_compile = maybe_disable_torch_compile()

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

    summaries = []
    if args.dataset_path is None:
        sample = load_single_debug_sample(args)
        run_one_sample(
            args=args,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            sample=sample,
            output_path=Path(args.output),
            trace_output_path=args.trace_output,
            trace_pt_output_path=args.trace_pt_output,
        )
    else:
        out_dir = batch_output_dir(args)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_jsonl = out_dir / "batch_summary.jsonl"
        for sample_idx, sample in iter_debug_samples(args):
            stem = f"sample_{sample_idx:06d}"
            output_path = out_dir / f"{stem}.json"
            trace_output_path = out_dir / f"{stem}.trace.json" if args.save_trace else None
            trace_pt_output_path = out_dir / f"{stem}.trace.pt" if args.save_trace else None
            print(f"Running dataset sample {sample_idx}: {sample['question'][:120]}")
            payload = run_one_sample(
                args=args,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                sample=sample,
                output_path=output_path,
                trace_output_path=trace_output_path,
                trace_pt_output_path=trace_pt_output_path,
            )
            summaries.append(
                {
                    "dataset_index": sample_idx,
                    "question_id": sample["dataset_meta"]["question_id"],
                    "question": sample["question"],
                    "final_text": payload["final_text"],
                    "output_path": str(output_path),
                    "trace_output_path": str(trace_output_path) if trace_output_path else None,
                }
            )

        with summary_jsonl.open("w", encoding="utf-8") as f:
            for summary in summaries:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"Saved batch summary to {summary_jsonl}")

    restore_compile()


if __name__ == "__main__":
    main()
