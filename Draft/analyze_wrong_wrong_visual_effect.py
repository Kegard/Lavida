import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Draft.analyze_textvqa_stepwise_cases import classify_case
from Draft.analyze_wrong_wrong_changing_logits import (
    DEFAULT_MASK_ID,
    build_conversation_prompt,
    build_target_positions,
    choose_canonical_answer,
    construct_prompt,
    extract_images,
    get_num_transfer_tokens_sch,
    load_records_by_question_id,
    load_textvqa_split,
    move_images_to_device,
)
from Scale_Attention.reweight_patch import build_prefix_from_multimodal_inputs
from VRG.timestep_vrg import build_unconditional_prefix_embeds, compute_remasking_confidence
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare correct-vs-wrong logit gaps with image vs no-image on wrong-wrong TextVQA samples."
    )
    parser.add_argument("--dataset-path", default="lmms-lab/textvqa")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--input-jsonl", default="Draft/textvqa_stepwise_eval_samples.jsonl")
    parser.add_argument("--labeled-jsonl", default="Draft/textvqa_stepwise_eval_samples_labeled.jsonl")
    parser.add_argument(
        "--case-filter",
        default="wrong_to_wrong_changing",
        help="Comma-separated case names, e.g. wrong_to_wrong_changing or wrong_to_wrong_changing,wrong_to_wrong_stable",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-jsonl", default="Draft/wrong_wrong_visual_effect.jsonl")
    parser.add_argument("--summary-json", default="Draft/wrong_wrong_visual_effect_summary.json")
    parser.add_argument("--output-plot", default="Draft/wrong_wrong_visual_effect.svg")
    parser.add_argument("--pretrained", default="weight/lavida")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--step-ratio", type=float, default=1.0)
    parser.add_argument(
        "--remasking",
        default="margin",
        choices=["low_confidence", "random", "entrophy", "margin"],
    )
    parser.add_argument(
        "--schedule",
        default="none",
        choices=["shift", "cosine", "logit_normal", "none"],
    )
    parser.add_argument("--schedule-shift", type=float, default=0.33)
    parser.add_argument("--mask-id", type=int, default=DEFAULT_MASK_ID)
    parser.add_argument("--correct-threshold", type=float, default=0.0)
    parser.add_argument(
        "--null-visual-mode",
        default="zeros",
        choices=["zeros", "mask_token"],
        help="How to construct the no-image prefix while preserving multimodal token positions.",
    )
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args()


def build_case_records(sample_records, labeled_records, case_names, threshold):
    selected = []
    for record in sample_records.values():
        labeled = labeled_records.get(str(record["question_id"]))
        if labeled is not None:
            case_name = labeled.get("case_analysis", {}).get("case")
        else:
            case_name = classify_case(record["step_results"], threshold)["case"]
        if case_name in case_names:
            selected.append(record)
    selected.sort(key=lambda item: int(item["dataset_index"]))
    return selected


def rank_of_token(logits_row, token_id):
    greater = (logits_row > logits_row[token_id]).sum().item()
    return int(greater) + 1


@torch.no_grad()
def trace_visual_effect_sample(
    model,
    tokenizer,
    prefix_embeds_with_image,
    prefix_embeds_no_image,
    max_new_tokens,
    block_length,
    remasking,
    schedule,
    schedule_kwargs,
    mask_id,
    step_ratio,
    target_positions,
):
    core_model = model.get_model()
    gen_length = max_new_tokens
    if block_length is None:
        block_length = max_new_tokens
    if gen_length % block_length != 0:
        raise ValueError(
            f"max_new_tokens ({gen_length}) must be divisible by block_length ({block_length})."
        )

    x = torch.full((1, gen_length), mask_id, dtype=torch.long, device=core_model.device)
    past_with_image = core_model(
        None,
        input_embeddings=prefix_embeds_with_image,
        use_cache=True,
    ).attn_key_values
    past_no_image = core_model(
        None,
        input_embeddings=prefix_embeds_no_image,
        use_cache=True,
    ).attn_key_values

    steps = max_new_tokens
    num_blocks = gen_length // block_length
    steps = steps // num_blocks
    if step_ratio is not None:
        steps = int(steps * step_ratio)
    if steps <= 0:
        raise ValueError(f"step_ratio={step_ratio} makes steps become {steps}.")

    per_position_records = {item["position"]: [] for item in target_positions}
    global_step = 0

    for block_idx in range(num_blocks):
        block_slice = slice(block_idx * block_length, (block_idx + 1) * block_length)
        block_mask_index = x[:, block_slice] == mask_id
        num_transfer_tokens = get_num_transfer_tokens_sch(
            block_mask_index,
            steps,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs,
        )

        for step_idx in range(steps):
            mask_index = x == mask_id
            block_mask_index = mask_index[:, block_slice]
            if block_mask_index.sum() == 0:
                continue

            inputs_embeds_curr = core_model.transformer.wte(x)
            logits_with_image = core_model(
                None,
                input_embeddings=inputs_embeds_curr,
                past_key_values=past_with_image,
            ).logits
            logits_no_image = core_model(
                None,
                input_embeddings=inputs_embeds_curr,
                past_key_values=past_no_image,
            ).logits

            x0_with_image = torch.argmax(logits_with_image, dim=-1)
            confidence = compute_remasking_confidence(
                logits_with_image,
                x0_with_image,
                remasking,
            )
            confidence[:, (block_idx + 1) * block_length :] = float("-inf")
            x0_with_image = torch.where(mask_index, x0_with_image, x)
            confidence = torch.where(mask_index, confidence, float("-inf"))

            transfer_index = torch.zeros_like(x0_with_image, dtype=torch.bool, device=x0_with_image.device)
            k = int(num_transfer_tokens[0, step_idx].item())
            if k > 0:
                _, select_index = torch.topk(confidence[0], k=k)
                transfer_index[0, select_index] = True
                selected_positions = [int(item) for item in select_index.detach().cpu().tolist()]
            else:
                selected_positions = []

            global_step += 1
            for target in target_positions:
                position = target["position"]
                if position >= logits_with_image.shape[1]:
                    continue
                row_with_image = logits_with_image[0, position]
                row_no_image = logits_no_image[0, position]
                correct_id = target["correct_token_id"]
                wrong_id = target["wrong_token_id"]

                correct_with_image = float(row_with_image[correct_id].item())
                wrong_with_image = float(row_with_image[wrong_id].item())
                delta_with_image = correct_with_image - wrong_with_image

                correct_no_image = float(row_no_image[correct_id].item())
                wrong_no_image = float(row_no_image[wrong_id].item())
                delta_no_image = correct_no_image - wrong_no_image

                per_position_records[position].append(
                    {
                        "step": global_step,
                        "block_index": block_idx + 1,
                        "step_in_block": step_idx + 1,
                        "num_transferred": k,
                        "position": position,
                        "selected_positions_with_image": selected_positions,
                        "selected_this_step_with_image": bool(transfer_index[0, position].item()),
                        "position_masked_before": bool(mask_index[0, position].item()),
                        "correct_token_id": correct_id,
                        "wrong_token_id": wrong_id,
                        "correct_logit_with_image": correct_with_image,
                        "wrong_logit_with_image": wrong_with_image,
                        "delta_with_image": delta_with_image,
                        "correct_rank_with_image": rank_of_token(row_with_image, correct_id),
                        "wrong_rank_with_image": rank_of_token(row_with_image, wrong_id),
                        "top1_token_id_with_image": int(torch.argmax(row_with_image).item()),
                        "correct_logit_no_image": correct_no_image,
                        "wrong_logit_no_image": wrong_no_image,
                        "delta_no_image": delta_no_image,
                        "correct_rank_no_image": rank_of_token(row_no_image, correct_id),
                        "wrong_rank_no_image": rank_of_token(row_no_image, wrong_id),
                        "top1_token_id_no_image": int(torch.argmax(row_no_image).item()),
                        "visual_delta_gain": delta_with_image - delta_no_image,
                    }
                )

            x[transfer_index] = x0_with_image[transfer_index]

    return per_position_records


def aggregate_mean_curves(sample_records):
    with_image_values = defaultdict(list)
    no_image_values = defaultdict(list)
    gain_values = defaultdict(list)
    for sample_record in sample_records:
        for position_records in sample_record.values():
            for item in position_records:
                step = int(item["step"])
                with_image_values[step].append(float(item["delta_with_image"]))
                no_image_values[step].append(float(item["delta_no_image"]))
                gain_values[step].append(float(item["visual_delta_gain"]))

    def mean_dict(values_by_step):
        return {
            step: statistics.mean(values_by_step[step])
            for step in sorted(values_by_step)
        }

    return mean_dict(with_image_values), mean_dict(no_image_values), mean_dict(gain_values)


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_visual_effect_svg(with_image_curve, no_image_curve, gain_curve, output_path):
    width = 1400
    height = 480
    gap = 24
    panel_width = (width - gap) // 2

    def draw_curve_panel(curves, title, origin_x):
        left = origin_x + 70
        right = 20
        top = 50
        bottom = 70
        chart_width = panel_width - 70 - right
        chart_height = height - top - bottom
        first_spec = next(iter(curves.values())) if curves else None
        steps = sorted(first_spec["curve"]) if first_spec else []
        values = []
        for spec in curves.values():
            curve = spec["curve"]
            values.extend(curve[step] for step in sorted(curve))
        min_value = min(values + [0.0])
        max_value = max(values + [0.0])
        if math.isclose(min_value, max_value):
            min_value -= 1.0
            max_value += 1.0

        def x_for(step):
            if len(steps) == 1:
                return left + chart_width / 2.0
            return left + (step - steps[0]) / (steps[-1] - steps[0]) * chart_width

        def y_for(value):
            ratio = (value - min_value) / (max_value - min_value)
            return top + chart_height - ratio * chart_height

        parts = [
            f'<rect x="{origin_x}" y="0" width="{panel_width}" height="{height}" fill="white"/>',
            f'<text x="{origin_x + panel_width / 2}" y="28" text-anchor="middle" font-size="18" font-family="Arial">{svg_escape(title)}</text>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#333" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#333" stroke-width="1.5"/>',
        ]
        zero_y = y_for(0.0)
        parts.append(
            f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + chart_width}" y2="{zero_y:.2f}" stroke="#adb5bd" stroke-dasharray="6,4" stroke-width="1.2"/>'
        )

        for tick_value in [min_value, 0.0, max_value]:
            y = y_for(tick_value)
            parts.append(
                f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{tick_value:.3f}</text>'
            )

        for curve_name, curve_spec in curves.items():
            color = curve_spec["color"]
            curve = curve_spec["curve"]
            points = " ".join(f"{x_for(step):.2f},{y_for(curve[step]):.2f}" for step in sorted(curve))
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
            legend_y = top + 18 + 18 * curve_spec["legend_index"]
            parts.append(f'<line x1="{left + 10}" y1="{legend_y}" x2="{left + 34}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>')
            parts.append(
                f'<text x="{left + 42}" y="{legend_y + 4}" font-size="11" font-family="Arial">{svg_escape(curve_name)}</text>'
            )

        for step in [steps[0], steps[len(steps) // 2], steps[-1]]:
            x = x_for(step)
            parts.append(
                f'<text x="{x:.2f}" y="{top + chart_height + 18:.2f}" text-anchor="middle" font-size="11" font-family="Arial">{step}</text>'
            )
        return "\n".join(parts)

    left_panel = draw_curve_panel(
        {
            "with image Δ_t": {"curve": with_image_curve, "color": "#c2255c", "legend_index": 0},
            "no image Δ_t": {"curve": no_image_curve, "color": "#1971c2", "legend_index": 1},
        },
        "correct-vs-wrong Δ_t",
        origin_x=0,
    )
    right_panel = draw_curve_panel(
        {
            "visual gain = Δ_t(img) - Δ_t(noimg)": {"curve": gain_curve, "color": "#2f9e44", "legend_index": 0},
        },
        "Visual Effect on Δ_t",
        origin_x=panel_width + gap,
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#f8f9fa"/>
{left_panel}
{right_panel}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def main():
    args = parse_args()
    from llava.model.builder import load_pretrained_model

    case_names = {item.strip() for item in args.case_filter.split(",") if item.strip()}
    sample_records = load_records_by_question_id(args.input_jsonl)
    labeled_records = load_records_by_question_id(args.labeled_jsonl)
    selected_records = build_case_records(
        sample_records,
        labeled_records,
        case_names,
        args.correct_threshold,
    )
    selected_records = selected_records[args.start_index :]
    if args.limit is not None:
        selected_records = selected_records[: args.limit]

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
    answer_processor = EvalAIAnswerProcessor()
    schedule = None if args.schedule == "none" else args.schedule
    schedule_kwargs = {"shift": args.schedule_shift} if schedule == "shift" else None

    output_jsonl = Path(args.output_jsonl)
    summary_json = Path(args.summary_json)
    output_plot = Path(args.output_plot)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    traced_samples = []
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for sample_idx, record in enumerate(selected_records, start=1):
            dataset_index = int(record["dataset_index"])
            doc = dataset[dataset_index]
            images = extract_images(doc)
            context = construct_prompt(doc)
            prompt = build_conversation_prompt(context, len(images), args.conv_template)
            image_tensor = move_images_to_device(images, image_processor, model.config, args.device)
            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(args.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
            prefix_with_image, prefix_input_ids_full = build_prefix_from_multimodal_inputs(
                model=model,
                input_ids=input_ids,
                images=image_tensor,
                image_sizes=[image.size for image in images],
                attention_mask=attention_mask,
            )
            prefix_no_image, _ = build_unconditional_prefix_embeds(
                model.get_model(),
                prefix_with_image,
                prefix_input_ids_full,
                null_visual_mode=args.null_visual_mode,
            )

            canonical_correct_text = choose_canonical_answer(doc.get("answers", []), answer_processor)
            final_wrong_text = record["final_text"]
            target_positions = build_target_positions(
                tokenizer,
                canonical_correct_text,
                final_wrong_text,
                args.max_new_tokens,
            )
            if not target_positions:
                continue

            per_position_records = trace_visual_effect_sample(
                model=model,
                tokenizer=tokenizer,
                prefix_embeds_with_image=prefix_with_image,
                prefix_embeds_no_image=prefix_no_image,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                remasking=args.remasking,
                schedule=schedule,
                schedule_kwargs=schedule_kwargs,
                mask_id=args.mask_id,
                step_ratio=args.step_ratio,
                target_positions=target_positions,
            )

            sample_output = {
                "dataset_index": dataset_index,
                "doc_id": str(record["doc_id"]),
                "question_id": record["question_id"],
                "case_filter": sorted(case_names),
                "question": record["question"],
                "canonical_correct_text": canonical_correct_text,
                "final_wrong_text": final_wrong_text,
                "target_positions": target_positions,
                "per_position_records": per_position_records,
            }
            fout.write(json.dumps(sample_output, ensure_ascii=False) + "\n")
            traced_samples.append(sample_output)

            if args.print_every > 0 and sample_idx % args.print_every == 0:
                print(
                    f"[{sample_idx}] id={record['doc_id']} positions={len(target_positions)} "
                    f"correct={canonical_correct_text!r} wrong={final_wrong_text!r}"
                )

    position_record_groups = [sample["per_position_records"] for sample in traced_samples]
    with_image_curve, no_image_curve, gain_curve = aggregate_mean_curves(position_record_groups)

    event_counter = Counter()
    for sample in traced_samples:
        for position_records in sample["per_position_records"].values():
            for item in position_records:
                gain = float(item["visual_delta_gain"])
                if gain > 0:
                    event_counter["visual_helps"] += 1
                elif gain < 0:
                    event_counter["visual_hurts"] += 1
                else:
                    event_counter["visual_neutral"] += 1

    if gain_curve:
        max_gain_step = max(gain_curve, key=lambda step: gain_curve[step])
        min_gain_step = min(gain_curve, key=lambda step: gain_curve[step])
        num_positive_gain_steps = sum(1 for value in gain_curve.values() if value > 0)
    else:
        max_gain_step = None
        min_gain_step = None
        num_positive_gain_steps = 0

    summary = {
        "case_filter": sorted(case_names),
        "null_visual_mode": args.null_visual_mode,
        "num_samples": len(traced_samples),
        "event_counts": dict(event_counter),
        "num_positive_gain_steps": num_positive_gain_steps,
        "positive_gain_steps": [step for step, value in gain_curve.items() if value > 0],
        "max_gain_step": max_gain_step,
        "max_gain_value": gain_curve[max_gain_step] if max_gain_step is not None else None,
        "min_gain_step": min_gain_step,
        "min_gain_value": gain_curve[min_gain_step] if min_gain_step is not None else None,
        "mean_delta_with_image_curve": [
            {"step": step, "mean_delta_with_image": with_image_curve[step]}
            for step in sorted(with_image_curve)
        ],
        "mean_delta_no_image_curve": [
            {"step": step, "mean_delta_no_image": no_image_curve[step]}
            for step in sorted(no_image_curve)
        ],
        "mean_visual_gain_curve": [
            {"step": step, "mean_visual_gain": gain_curve[step]}
            for step in sorted(gain_curve)
        ],
    }
    with summary_json.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)

    if with_image_curve and no_image_curve and gain_curve:
        write_visual_effect_svg(with_image_curve, no_image_curve, gain_curve, output_plot)

    print(f"Wrote per-sample visual comparison to {output_jsonl}")
    print(f"Wrote summary to {summary_json}")
    if with_image_curve and no_image_curve and gain_curve:
        print(f"Wrote plot to {output_plot}")
        print(
            f"Visual helps events: {event_counter.get('visual_helps', 0)}, "
            f"hurts events: {event_counter.get('visual_hurts', 0)}, "
            f"neutral events: {event_counter.get('visual_neutral', 0)}"
        )


if __name__ == "__main__":
    main()
