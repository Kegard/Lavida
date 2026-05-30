import argparse
import copy
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Draft.analyze_textvqa_stepwise_cases import classify_case
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


TEXTVQA_PROMPT = "Answer the question using a single word or phrase."
DEFAULT_MASK_ID = 126336


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze wrong->wrong changing TextVQA x0 trajectories via correct-vs-wrong token logits."
    )
    parser.add_argument("--dataset-path", default="lmms-lab/textvqa")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--input-jsonl",
        default="Draft/textvqa_stepwise_eval_samples.jsonl",
        help="Step-wise sample output produced by eval_textvqa_stepwise_llada.py",
    )
    parser.add_argument(
        "--labeled-jsonl",
        default="Draft/textvqa_stepwise_eval_samples_labeled.jsonl",
        help="Optional labeled jsonl. If missing, the case will be recomputed from step_results.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start index inside the filtered wrong->wrong changing list.")
    parser.add_argument("--limit", type=int, default=None, help="How many wrong->wrong changing samples to analyze.")
    parser.add_argument(
        "--output-jsonl",
        default="Draft/wrong_wrong_changing_logit_analysis.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        default="Draft/wrong_wrong_changing_logit_summary.json",
    )
    parser.add_argument(
        "--output-plot",
        default="Draft/wrong_wrong_changing_delta_curve.svg",
    )
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
    parser.add_argument("--print-every", type=int, default=1)
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


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def cosine_schedule(x):
    x = torch.clamp(x, 0, 1)
    return 1 - 0.5 * (1 + torch.cos(torch.pi * x))


def sigmoid_normal_cdf(y):
    logit_y = torch.log(y / (1 - y))
    return 0.5 * (1 + torch.erf(logit_y / torch.sqrt(torch.tensor(2.0))))


def logit_normal_schedule(shift, sigmas):
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def get_num_transfer_tokens_sch(mask_index, steps, schedule=None, schedule_kwargs=None):
    if schedule is None:
        return get_num_transfer_tokens(mask_index, steps)
    if schedule_kwargs is None:
        schedule_kwargs = {}

    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = int(min(steps, mask_num[0]))
    t = torch.linspace(0, 1, steps + 1)
    if schedule == "logit_normal":
        sigmas = sigmoid_normal_cdf(t)
    elif schedule == "shift":
        sigmas = logit_normal_schedule(schedule_kwargs.get("shift", 3), t)
    elif schedule == "cosine":
        sigmas = cosine_schedule(t)
    else:
        sigmas = t

    sigmas = sigmas.to(mask_num.device)
    num_transfer_tokens = torch.zeros(
        mask_num.size(0),
        steps,
        device=mask_index.device,
        dtype=torch.int64,
    )

    for i in range(mask_num.size(0)):
        sigmas_sample = (sigmas * mask_num[i]).to(torch.int64)
        sigmas_sample = sigmas_sample[1:] - sigmas_sample[:-1]
        sigmas_sample = torch.clamp(sigmas_sample, 1, None)
        delta = sigmas_sample.sum() - mask_num[i]
        j = 0
        while delta > 0:
            j = j % len(sigmas_sample)
            if sigmas_sample[j] == 1:
                j += 1
                continue
            delta -= 1
            sigmas_sample[j] -= 1
            j += 1
        num_transfer_tokens[i] = sigmas_sample
    return num_transfer_tokens.flip(-1)


def prepare_inputs_embeds(model, input_ids, image_tensor, images):
    if image_tensor is None:
        return model.get_model().embed_tokens(input_ids)
    image_sizes = [image.size for image in images]
    _, _, _, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        None,
        None,
        None,
        image_tensor,
        ["image"],
        image_sizes=image_sizes,
    )
    return inputs_embeds


def load_records_by_question_id(path):
    path = Path(path)
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["question_id"])] = record
    return records


def normalize_prediction(text, answer_processor):
    return answer_processor(text)


def choose_canonical_answer(answers, answer_processor):
    normalized_answers = [answer_processor(answer) for answer in answers if answer is not None]
    if not normalized_answers:
        return ""
    counts = Counter(normalized_answers)
    best_count = max(counts.values())
    candidates = [answer for answer, count in counts.items() if count == best_count]
    candidates.sort(key=lambda item: (-len(item), item))
    return candidates[0]


def build_wrong_wrong_changing_records(sample_records, labeled_records, threshold):
    wrong_wrong_changing = []
    for record in sample_records.values():
        labeled = labeled_records.get(str(record["question_id"]))
        if labeled is not None:
            case_name = labeled.get("case_analysis", {}).get("case")
        else:
            case_name = classify_case(record["step_results"], threshold)["case"]
        if case_name == "wrong_to_wrong_changing":
            wrong_wrong_changing.append(record)
    wrong_wrong_changing.sort(key=lambda item: int(item["dataset_index"]))
    return wrong_wrong_changing


def build_target_positions(tokenizer, correct_text, wrong_text, max_new_tokens):
    correct_ids = tokenizer.encode(correct_text, add_special_tokens=False)
    wrong_ids = tokenizer.encode(wrong_text, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    max_len = min(max(len(correct_ids), len(wrong_ids)), max_new_tokens)

    positions = []
    for idx in range(max_len):
        correct_id = correct_ids[idx] if idx < len(correct_ids) else eos_id
        wrong_id = wrong_ids[idx] if idx < len(wrong_ids) else eos_id
        if correct_id == wrong_id:
            continue
        positions.append(
            {
                "position": idx,
                "correct_token_id": int(correct_id),
                "wrong_token_id": int(wrong_id),
                "correct_token_text": tokenizer.decode([correct_id], skip_special_tokens=False),
                "wrong_token_text": tokenizer.decode([wrong_id], skip_special_tokens=False),
            }
        )
    return positions


def summarize_position_records(position_records):
    steps = [item["step"] for item in position_records]
    deltas = [item["delta"] for item in position_records]
    if not deltas:
        return {
            "step_max_delta": None,
            "max_delta": None,
            "step_closest_to_zero": None,
            "delta_closest_to_zero": None,
            "num_positive_delta_steps": 0,
            "positive_delta_steps": [],
            "positive_delta_no_unmask_reason_counts": {},
        }

    max_delta_idx = max(range(len(deltas)), key=lambda idx: deltas[idx])
    closest_to_zero_idx = min(range(len(deltas)), key=lambda idx: abs(deltas[idx]))
    positive_delta_steps = [record["step"] for record in position_records if record["delta"] > 0]
    positive_delta_reason_counts = Counter(
        record["no_unmask_reason"]
        for record in position_records
        if record["delta"] > 0 and not record["position_selected_for_unmask"]
    )
    return {
        "step_max_delta": steps[max_delta_idx],
        "max_delta": deltas[max_delta_idx],
        "step_closest_to_zero": steps[closest_to_zero_idx],
        "delta_closest_to_zero": deltas[closest_to_zero_idx],
        "num_positive_delta_steps": len(positive_delta_steps),
        "positive_delta_steps": positive_delta_steps,
        "positive_delta_no_unmask_reason_counts": dict(positive_delta_reason_counts),
    }


def rank_of_token(logits_row, token_id):
    greater = (logits_row > logits_row[token_id]).sum().item()
    return int(greater) + 1


@torch.no_grad()
def trace_wrong_wrong_sample(
    model,
    tokenizer,
    inputs_embeds,
    conv_template,
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
    past_key_values = core_model(
        None,
        input_embeddings=inputs_embeds,
        use_cache=True,
    ).attn_key_values

    steps = max_new_tokens
    num_blocks = gen_length // block_length
    steps = steps // num_blocks
    if step_ratio is not None:
        steps = int(steps * step_ratio)
    if steps <= 0:
        raise ValueError(f"step_ratio={step_ratio} makes steps become {steps}.")

    target_by_position = {item["position"]: item for item in target_positions}
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
            logits = core_model(
                None,
                input_embeddings=inputs_embeds_curr,
                past_key_values=past_key_values,
            ).logits
            x0 = torch.argmax(logits, dim=-1)

            if remasking == "low_confidence":
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == "entrophy":
                epsilon = 1e-10
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                log_probs = torch.log(probs + epsilon)
                x0_p = torch.sum(probs * log_probs, dim=-1)
            elif remasking == "margin":
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                x0_p = sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
            else:
                raise NotImplementedError(remasking)

            x0_p[:, (block_idx + 1) * block_length :] = float("-inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, float("-inf"))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            selected_positions = []
            k = int(num_transfer_tokens[0, step_idx].item())
            if k > 0:
                _, select_index = torch.topk(confidence[0], k=k)
                transfer_index[0, select_index] = True
                selected_positions = [int(item) for item in select_index.detach().cpu().tolist()]

            global_step += 1

            top2_values, top2_indices = torch.topk(logits[0], k=2, dim=-1)
            selected_confidences = []
            if selected_positions:
                selected_confidences = [
                    float(confidence[0, position].item())
                    for position in selected_positions
                ]
            selected_confidence_floor = min(selected_confidences) if selected_confidences else None

            for position, target in target_by_position.items():
                if position >= logits.shape[1]:
                    continue
                logits_row = logits[0, position]
                correct_id = target["correct_token_id"]
                wrong_id = target["wrong_token_id"]
                correct_logit = float(logits_row[correct_id].item())
                wrong_logit = float(logits_row[wrong_id].item())
                delta = correct_logit - wrong_logit
                top1_id = int(torch.argmax(logits_row).item())
                top1_logit = float(logits_row[top1_id].item())
                top2_id = int(top2_indices[position, 1].item())
                top2_logit = float(top2_values[position, 1].item())
                position_masked_before = bool(mask_index[0, position].item())
                position_selected_for_unmask = bool(transfer_index[0, position].item())
                predicted_token_id = int(x0[0, position].item())
                predicted_token_text = tokenizer.decode([predicted_token_id], skip_special_tokens=False)
                position_confidence = (
                    float(confidence[0, position].item())
                    if position_masked_before
                    else None
                )

                if not position_masked_before:
                    no_unmask_reason = "already_unmasked"
                elif position_selected_for_unmask:
                    if top1_id == correct_id:
                        no_unmask_reason = "unmasked_correct"
                    elif top1_id == wrong_id:
                        no_unmask_reason = "unmasked_wrong"
                    else:
                        no_unmask_reason = "unmasked_other"
                else:
                    if top1_id == correct_id:
                        no_unmask_reason = "correct_top1_but_not_selected"
                    elif delta > 0:
                        no_unmask_reason = "correct_beats_wrong_but_third_token_top1"
                    else:
                        no_unmask_reason = "correct_below_wrong"

                per_position_records[position].append(
                    {
                        "step": global_step,
                        "block_index": block_idx + 1,
                        "step_in_block": step_idx + 1,
                        "num_transferred": k,
                        "position": position,
                        "position_masked_before": position_masked_before,
                        "position_selected_for_unmask": position_selected_for_unmask,
                        "selected_positions": selected_positions,
                        "selected_confidence_floor": selected_confidence_floor,
                        "position_confidence": position_confidence,
                        "correct_token_id": correct_id,
                        "wrong_token_id": wrong_id,
                        "correct_logit": correct_logit,
                        "wrong_logit": wrong_logit,
                        "delta": delta,
                        "correct_rank": rank_of_token(logits_row, correct_id),
                        "wrong_rank": rank_of_token(logits_row, wrong_id),
                        "top1_token_id": top1_id,
                        "top1_token_text": tokenizer.decode([top1_id], skip_special_tokens=False),
                        "top1_logit": top1_logit,
                        "top2_token_id": top2_id,
                        "top2_token_text": tokenizer.decode([top2_id], skip_special_tokens=False),
                        "top2_logit": top2_logit,
                        "predicted_token_id": predicted_token_id,
                        "predicted_token_text": predicted_token_text,
                        "no_unmask_reason": no_unmask_reason,
                    }
                )

            x[transfer_index] = x0[transfer_index]

    final_text = clean_model_output(
        tokenizer.batch_decode(x.detach().cpu(), skip_special_tokens=True)[0],
        conv_template,
    )
    return final_text, per_position_records


def build_mean_delta_curve(sample_position_summaries):
    step_to_values = defaultdict(list)
    for summary in sample_position_summaries:
        for item in summary:
            step_to_values[item["step"]].append(float(item["delta"]))
    mean_curve = {}
    for step in sorted(step_to_values):
        mean_curve[step] = statistics.mean(step_to_values[step])
    return mean_curve


def build_positive_delta_reason_summary(sample_position_summaries):
    reason_counts = Counter()
    for summary in sample_position_summaries:
        for item in summary:
            if item["delta"] > 0 and not item["position_selected_for_unmask"]:
                reason_counts[item["no_unmask_reason"]] += 1
    return dict(reason_counts)


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_delta_svg(mean_curve, output_path, title):
    width = 900
    height = 420
    left = 70
    right = 30
    top = 50
    bottom = 70
    chart_width = width - left - right
    chart_height = height - top - bottom

    steps = sorted(mean_curve)
    values = [mean_curve[step] for step in steps]
    min_value = min(values + [0.0])
    max_value = max(values + [0.0])
    if math.isclose(min_value, max_value):
        min_value -= 1.0
        max_value += 1.0

    def y_for(value):
        ratio = (value - min_value) / (max_value - min_value)
        return top + chart_height - ratio * chart_height

    def x_for(step):
        if len(steps) == 1:
            return left + chart_width / 2.0
        return left + (step - steps[0]) / (steps[-1] - steps[0]) * chart_width

    points = " ".join(f"{x_for(step):.2f},{y_for(mean_curve[step]):.2f}" for step in steps)
    zero_y = y_for(0.0)
    max_step = max(steps, key=lambda step: mean_curve[step])
    max_value_at_step = mean_curve[max_step]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-family="Arial">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + chart_width}" y2="{zero_y:.2f}" stroke="#adb5bd" stroke-dasharray="6,4" stroke-width="1.2"/>',
        f'<polyline fill="none" stroke="#c2255c" stroke-width="2.5" points="{points}"/>',
    ]

    for tick_value in [min_value, 0.0, max_value]:
        y = y_for(tick_value)
        parts.append(f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{tick_value:.3f}</text>')

    for step in [steps[0], steps[len(steps) // 2], steps[-1]]:
        x = x_for(step)
        parts.append(f'<text x="{x:.2f}" y="{top + chart_height + 18:.2f}" text-anchor="middle" font-size="11" font-family="Arial">{step}</text>')

    max_x = x_for(max_step)
    max_y = y_for(max_value_at_step)
    parts.append(f'<circle cx="{max_x:.2f}" cy="{max_y:.2f}" r="4" fill="#d9480f"/>')
    parts.append(
        f'<text x="{max_x:.2f}" y="{max_y - 10:.2f}" text-anchor="middle" font-size="11" font-family="Arial">max @ step {max_step}: {max_value_at_step:.3f}</text>'
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    args = parse_args()
    from llava.model.builder import load_pretrained_model

    sample_records = load_records_by_question_id(args.input_jsonl)
    labeled_records = load_records_by_question_id(args.labeled_jsonl)
    filtered_records = build_wrong_wrong_changing_records(
        sample_records,
        labeled_records,
        args.correct_threshold,
    )

    filtered_records = filtered_records[args.start_index :]
    if args.limit is not None:
        filtered_records = filtered_records[: args.limit]

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
    output_jsonl = Path(args.output_jsonl)
    summary_json = Path(args.summary_json)
    output_plot = Path(args.output_plot)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    schedule = None if args.schedule == "none" else args.schedule
    schedule_kwargs = {"shift": args.schedule_shift} if schedule == "shift" else None
    answer_processor = EvalAIAnswerProcessor()

    sample_position_summaries = []
    aggregate_position_summaries = []
    sample_results = []

    with output_jsonl.open("w", encoding="utf-8") as fout:
        for sample_idx, record in enumerate(filtered_records, start=1):
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
            inputs_embeds = prepare_inputs_embeds(model, input_ids, image_tensor, images)

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

            traced_final_text, per_position_records = trace_wrong_wrong_sample(
                model=model,
                tokenizer=tokenizer,
                inputs_embeds=inputs_embeds,
                conv_template=args.conv_template,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                remasking=args.remasking,
                schedule=schedule,
                schedule_kwargs=schedule_kwargs,
                mask_id=args.mask_id,
                step_ratio=args.step_ratio,
                target_positions=target_positions,
            )

            position_summaries = []
            for position_info in target_positions:
                position = position_info["position"]
                records_for_position = per_position_records[position]
                aggregate_position_summaries.extend(records_for_position)
                sample_position_summaries.append(records_for_position)
                position_summary = dict(position_info)
                position_summary.update(summarize_position_records(records_for_position))
                position_summaries.append(position_summary)

            sample_output = {
                "dataset_index": dataset_index,
                "doc_id": str(record["doc_id"]),
                "question_id": record["question_id"],
                "question": record["question"],
                "ocr_tokens": record.get("ocr_tokens"),
                "canonical_correct_text": canonical_correct_text,
                "final_wrong_text": final_wrong_text,
                "traced_final_text": traced_final_text,
                "target_positions": target_positions,
                "position_summaries": position_summaries,
                "per_position_records": per_position_records,
            }
            fout.write(json.dumps(sample_output, ensure_ascii=False) + "\n")
            sample_results.append(sample_output)

            if args.print_every > 0 and sample_idx % args.print_every == 0:
                print(
                    f"[{sample_idx}] id={record['doc_id']} positions={len(target_positions)} "
                    f"correct={canonical_correct_text!r} wrong={final_wrong_text!r}"
                )

    mean_curve = build_mean_delta_curve(sample_position_summaries)
    if mean_curve:
        step_of_max_mean_delta = max(mean_curve, key=lambda step: mean_curve[step])
        max_mean_delta = mean_curve[step_of_max_mean_delta]
        num_positive_mean_steps = sum(1 for value in mean_curve.values() if value > 0)
    else:
        step_of_max_mean_delta = None
        max_mean_delta = None
        num_positive_mean_steps = 0

    positive_delta_reason_counts = build_positive_delta_reason_summary(sample_position_summaries)
    summary = {
        "input_jsonl": args.input_jsonl,
        "labeled_jsonl": args.labeled_jsonl,
        "num_filtered_samples": len(sample_results),
        "num_traced_positions": len(sample_position_summaries),
        "canonical_answer_selection": "majority normalized answer",
        "token_alignment": "left-to-right token alignment with EOS padding for length mismatch",
        "step_of_max_mean_delta": step_of_max_mean_delta,
        "max_mean_delta": max_mean_delta,
        "num_positive_mean_steps": num_positive_mean_steps,
        "positive_mean_steps": [step for step, value in mean_curve.items() if value > 0],
        "positive_delta_no_unmask_reason_counts": positive_delta_reason_counts,
        "mean_delta_curve": [
            {"step": step, "mean_delta": mean_curve[step]}
            for step in sorted(mean_curve)
        ],
    }
    with summary_json.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)

    title = "wrong->wrong changing: mean Δ_t over step"
    if mean_curve:
        write_delta_svg(mean_curve, output_plot, title)

    print(f"Wrote per-sample analysis to {output_jsonl}")
    print(f"Wrote summary to {summary_json}")
    if mean_curve:
        print(f"Wrote plot to {output_plot}")
        print(
            f"Max mean Δ_t at step {step_of_max_mean_delta} with value {max_mean_delta:.4f}; "
            f"{num_positive_mean_steps} steps have mean Δ_t > 0."
        )
        print(f"Positive-Δ no-unmask reasons: {json.dumps(positive_delta_reason_counts, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
