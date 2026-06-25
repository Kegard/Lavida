#!/usr/bin/env python
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from M3CoT.PostMaSK.run_m3cot_postmask import generate_with_postmask
from Scale_Attention.reweight_patch import (
    build_prefix_from_multimodal_inputs,
    get_torch_dtype,
    maybe_disable_torch_compile,
)
from VRG.run_textvqa_proposal_refine import add_diffusion_noise
from VRG.run_textvqa_visual_warmup import (
    build_prompt,
    compute_textvqa_score,
    construct_textvqa_prompt,
    load_textvqa_split,
    normalize_answers,
    prepare_prefix,
)
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TextVQA with the same draft/PostMask/refill pipeline used by M3CoT PostVRG."
    )
    parser.add_argument("--dataset-path", default="lmms-lab/textvqa")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default="VRG/outputs/textvqa_postmask")

    parser.add_argument("--pretrained", default="weight/lavida-reason")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--prompt-mode", default="reasoning", choices=["auto", "short", "reasoning"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--step-per-block", type=int, default=None)
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])

    parser.add_argument("--draft-steps", type=int, default=16)
    parser.add_argument("--postmask-steps", type=int, default=16)
    parser.add_argument("--remask-per-step", type=int, default=4)
    parser.add_argument("--remask-selection", default="proposal_confidence")
    parser.add_argument("--postmask-mode", default="fixed_set", choices=["dynamic", "fixed_set"])
    parser.add_argument("--fixed-set-size", type=int, default=32)
    parser.add_argument("--fixed-refill-per-step", type=int, default=2)
    parser.add_argument("--loo-chunk-size", type=int, default=16)

    parser.add_argument("--null-visual-mode", default="zeros", choices=["zeros", "mask_token"])
    parser.add_argument("--selector-weak-visual-mode", default="null_visual", choices=["null_visual", "diffusion_noise"])
    parser.add_argument("--draft-guidance", default="none", choices=["none", "vcd"])
    parser.add_argument("--draft-weak-visual-mode", default="diffusion_noise", choices=["null_visual", "diffusion_noise"])
    parser.add_argument("--vcd-draft-alpha", type=float, default=1.0)
    parser.add_argument("--draft-guidance-ratio", type=float, default=1.0)
    parser.add_argument("--refill-guidance", default="none", choices=["none", "vcd"])
    parser.add_argument("--refill-weak-visual-mode", default="diffusion_noise", choices=["null_visual", "diffusion_noise"])
    parser.add_argument("--vcd-refill-alpha", type=float, default=0.5)
    parser.add_argument("--refill-vrg-calibration", default="none", choices=["none", "soft_confidence", "hard_confidence"])
    parser.add_argument("--refill-vrg-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--refill-guidance-steps", type=int, default=None)
    parser.add_argument("--vcd-noise-step", type=int, default=500)
    parser.add_argument("--vcd-noise-seed", type=int, default=42)

    parser.add_argument("--selector-candidate-size", type=int, default=64)
    parser.add_argument("--rank-visual-weight", type=float, default=1.0)
    parser.add_argument("--quadrant-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--quadrant-visual-threshold", type=float, default=0.2)
    parser.add_argument("--normalized-visual-alpha", type=float, default=1.0)
    parser.add_argument("--normalized-confidence-beta", type=float, default=1.0)
    parser.add_argument("--negative-gain-lambda", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def needs_diffusion_noise_prefix(args):
    visual_selector_modes = {
        "visual_gain",
        "conf_low_visual_gain",
        "proposal_then_visual_gain",
        "rank_conf_visual_gain",
        "quadrant_conf_visual_gain",
        "norm_conf_visual_gain",
        "highgain_lowconf",
        "highgain_lowconf_negboost",
    }
    return (
        (args.remask_selection in visual_selector_modes and args.selector_weak_visual_mode == "diffusion_noise")
        or (args.draft_guidance == "vcd" and args.draft_weak_visual_mode == "diffusion_noise")
        or (args.refill_guidance == "vcd" and args.refill_weak_visual_mode == "diffusion_noise")
    )


def build_textvqa_diffusion_noise_prefix(args, model, tokenizer, image_processor, doc):
    image = doc["image"].convert("RGB")
    context = construct_textvqa_prompt(
        doc,
        prompt_mode=args.prompt_mode,
        pretrained_path=args.pretrained,
    )
    prompt = build_prompt(context, args.conv_template)

    image_tensor = process_images([image], image_processor, model.config)
    dtype = get_torch_dtype(args.torch_dtype)
    if isinstance(image_tensor, list):
        image_tensor = [_image.to(dtype=dtype, device=args.device) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=dtype, device=args.device)
    noisy_image_tensor = add_diffusion_noise(
        image_tensor,
        noise_step=args.vcd_noise_step,
        seed=args.vcd_noise_seed,
    )

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=args.device)
    weak_prefix_embeds, _ = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=noisy_image_tensor,
        image_sizes=[image.size],
        attention_mask=attention_mask,
    )
    return weak_prefix_embeds


def main():
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be > 0.")

    restore_compile = maybe_disable_torch_compile()

    from llava.model.builder import load_pretrained_model

    vision_kwargs = dict(
        mm_vision_tower=args.vision_tower,
        mm_resampler_type=None,
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=1152,
        mm_pooler_ratio=2,
        mm_patch_merge_type="spatial_unpad",
        use_mm_proj=True,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        None,
        args.model_name,
        device_map=args.device_map,
        vision_kwargs=vision_kwargs,
        torch_dtype=args.torch_dtype,
    )
    model.eval()
    model.tie_weights()
    model.to(get_torch_dtype(args.torch_dtype))
    core_model = model.get_model()

    dataset = load_textvqa_split(args.dataset_path, args.dataset_name, args.split)
    answer_processor = EvalAIAnswerProcessor()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"

    total_elapsed = 0.0
    written = 0
    draft_score_total = 0.0
    final_score_total = 0.0
    improved_after_postmask = 0
    worsened_after_postmask = 0
    unchanged_after_postmask = 0
    postmask_step_totals = defaultdict(float)
    postmask_step_counts = defaultdict(int)

    with records_path.open("w", encoding="utf-8") as fout:
        for dataset_index in range(args.start_index, len(dataset)):
            if written >= args.limit:
                break
            doc = dataset[dataset_index]
            if doc.get("image") is None:
                continue

            t0 = time.time()
            context, prompt, prefix_embeds, prefix_input_ids_full = prepare_prefix(
                args,
                model,
                tokenizer,
                image_processor,
                doc,
            )

            weak_prefix_embeds = None
            if needs_diffusion_noise_prefix(args):
                weak_prefix_embeds = build_textvqa_diffusion_noise_prefix(
                    args,
                    model,
                    tokenizer,
                    image_processor,
                    doc,
                )

            selector_weak_prefix_embeds = (
                weak_prefix_embeds
                if args.selector_weak_visual_mode == "diffusion_noise"
                else None
            )
            draft_weak_prefix_embeds = (
                weak_prefix_embeds
                if args.draft_guidance == "vcd" and args.draft_weak_visual_mode == "diffusion_noise"
                else None
            )
            refill_weak_prefix_embeds = (
                weak_prefix_embeds
                if args.refill_guidance == "vcd" and args.refill_weak_visual_mode == "diffusion_noise"
                else None
            )

            run_output = generate_with_postmask(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                prefix_input_ids_full=prefix_input_ids_full,
                selector_weak_prefix_embeds=selector_weak_prefix_embeds,
                draft_weak_prefix_embeds=draft_weak_prefix_embeds,
                refill_weak_prefix_embeds=refill_weak_prefix_embeds,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                step_per_block=args.step_per_block,
                step_ratio=args.step_ratio,
                temperature=args.temperature,
                remasking=args.remasking,
                draft_steps=args.draft_steps,
                postmask_steps=args.postmask_steps,
                remask_per_step=args.remask_per_step,
                remask_selection=args.remask_selection,
                postmask_mode=args.postmask_mode,
                fixed_set_size=args.fixed_set_size,
                fixed_refill_per_step=args.fixed_refill_per_step,
                loo_chunk_size=args.loo_chunk_size,
                null_visual_mode=args.null_visual_mode,
                selector_weak_visual_mode=args.selector_weak_visual_mode,
                draft_guidance=args.draft_guidance,
                draft_weak_visual_mode=args.draft_weak_visual_mode,
                vcd_draft_alpha=args.vcd_draft_alpha,
                draft_guidance_ratio=args.draft_guidance_ratio,
                refill_guidance=args.refill_guidance,
                refill_weak_visual_mode=args.refill_weak_visual_mode,
                vcd_refill_alpha=args.vcd_refill_alpha,
                refill_vrg_calibration=args.refill_vrg_calibration,
                refill_vrg_confidence_threshold=args.refill_vrg_confidence_threshold,
                refill_guidance_steps=args.refill_guidance_steps,
                prompt_contrast_alpha=0.5,
                prompt_contrast_mode="confused_option_mapping",
                vcd_noise_step=args.vcd_noise_step,
                vcd_noise_seed=args.vcd_noise_seed,
                selector_candidate_size=args.selector_candidate_size,
                rank_visual_weight=args.rank_visual_weight,
                quadrant_confidence_threshold=args.quadrant_confidence_threshold,
                quadrant_visual_threshold=args.quadrant_visual_threshold,
                normalized_visual_alpha=args.normalized_visual_alpha,
                normalized_confidence_beta=args.normalized_confidence_beta,
                negative_gain_lambda=args.negative_gain_lambda,
            )

            normalized_answers = normalize_answers(doc, answer_processor)
            draft_score, draft_prediction = compute_textvqa_score(
                normalized_answers,
                run_output["draft_text"],
                answer_processor,
            )
            final_score, final_prediction = compute_textvqa_score(
                normalized_answers,
                run_output["final_text"],
                answer_processor,
            )

            for record in run_output["postmask_records"]:
                step_score, step_prediction = compute_textvqa_score(
                    normalized_answers,
                    record["state_text"],
                    answer_processor,
                )
                record["normalized_prediction"] = step_prediction
                record["exact_match"] = step_score
                local_step = int(record["step"]) - int(args.draft_steps)
                postmask_step_totals[local_step] += step_score
                postmask_step_counts[local_step] += 1

            elapsed = time.time() - t0
            total_elapsed += elapsed
            draft_score_total += draft_score
            final_score_total += final_score
            if final_score > draft_score:
                improved_after_postmask += 1
            elif final_score < draft_score:
                worsened_after_postmask += 1
            else:
                unchanged_after_postmask += 1

            output_record = {
                "dataset_index": int(dataset_index),
                "question_id": doc.get("question_id"),
                "question": context,
                "answers": doc.get("answers"),
                "normalized_answers": normalized_answers,
                "ocr_tokens": doc.get("ocr_tokens"),
                "prompt": prompt,
                "elapsed_sec": elapsed,
                "draft_text": run_output["draft_text"],
                "draft_prediction": draft_prediction,
                "draft_exact_match": draft_score,
                "final_text": run_output["final_text"],
                "final_prediction": final_prediction,
                "final_exact_match": final_score,
                "draft_records": run_output["draft_records"],
                "postmask_records": run_output["postmask_records"],
                "draft_answer_ids": run_output["draft_answer_ids"],
                "final_answer_ids": run_output["final_answer_ids"],
                "proposal_confidence": run_output["proposal_confidence"],
                "meta": run_output["meta"],
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                print(
                    f"[{written}] dataset_index={dataset_index} "
                    f"draft={draft_score:.3f} final={final_score:.3f} elapsed={elapsed:.2f}s"
                )

    postmask_step_summary = []
    for step in sorted(postmask_step_counts):
        count = postmask_step_counts[step]
        postmask_step_summary.append(
            {
                "postmask_step": int(step),
                "mean_exact_match": postmask_step_totals[step] / count if count else None,
                "count": int(count),
            }
        )

    summary = {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "start_index": args.start_index,
        "num_samples": written,
        "pipeline_definition": "TextVQA runner using M3CoT PostMask generate_with_postmask: draft -> fixed-set remask -> refill.",
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": total_elapsed / written if written else None,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "step_per_block": args.step_per_block,
            "step_ratio": args.step_ratio,
            "remasking": args.remasking,
            "temperature": args.temperature,
        },
        "postmask": {
            "draft_steps": args.draft_steps,
            "postmask_steps": args.postmask_steps,
            "remask_per_step": args.remask_per_step,
            "remask_selection": args.remask_selection,
            "postmask_mode": args.postmask_mode,
            "fixed_set_size": args.fixed_set_size,
            "fixed_refill_per_step": args.fixed_refill_per_step,
            "refill_guidance": args.refill_guidance,
            "refill_weak_visual_mode": args.refill_weak_visual_mode if args.refill_guidance == "vcd" else None,
            "vcd_refill_alpha": args.vcd_refill_alpha if args.refill_guidance == "vcd" else None,
            "refill_guidance_steps": args.refill_guidance_steps,
            "vcd_noise_step": (
                args.vcd_noise_step
                if args.refill_guidance == "vcd" and args.refill_weak_visual_mode == "diffusion_noise"
                else None
            ),
            "vcd_noise_seed": (
                args.vcd_noise_seed
                if args.refill_guidance == "vcd"
                and args.refill_weak_visual_mode == "diffusion_noise"
                and args.vcd_noise_seed is not None
                else None
            ),
        },
        "draft_mean_exact_match": draft_score_total / written if written else None,
        "final_mean_exact_match": final_score_total / written if written else None,
        "mean_gain_from_postmask": (final_score_total - draft_score_total) / written if written else None,
        "num_improved_after_postmask": improved_after_postmask,
        "num_worsened_after_postmask": worsened_after_postmask,
        "num_unchanged_after_postmask": unchanged_after_postmask,
        "postmask_step_summary": postmask_step_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote records to {records_path}")
    print(f"Wrote summary to {summary_path}")
    print(
        "Mean EM: "
        f"draft={summary['draft_mean_exact_match']:.4f}, "
        f"final={summary['final_mean_exact_match']:.4f}, "
        f"gain={summary['mean_gain_from_postmask']:.4f}"
    )
    restore_compile()


if __name__ == "__main__":
    main()
