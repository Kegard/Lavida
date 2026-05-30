import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


TEXTVQA_PROMPT = "Answer the question using a single word or phrase."
DEFAULT_MASK_ID = 126336


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TextVQA accuracy for each llada diffusion step independently."
    )
    parser.add_argument("--dataset-path", default="lmms-lab/textvqa")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--output-jsonl",
        default="Draft/textvqa_stepwise_eval_samples.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        default="Draft/textvqa_stepwise_eval_summary.json",
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


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


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
    x = np.clip(x, 0, 1)
    return 1 - 0.5 * (1 + np.cos(np.pi * x))


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


@torch.no_grad()
def generate_step_candidates(
    model,
    tokenizer,
    inputs_embeds,
    conv_template,
    max_new_tokens,
    block_length,
    temperature,
    remasking,
    schedule,
    schedule_kwargs,
    mask_id,
    step_ratio,
):
    core_model = model.get_model()
    bsz = inputs_embeds.shape[0]
    gen_length = max_new_tokens
    steps = max_new_tokens

    if block_length is None:
        block_length = max_new_tokens
    if gen_length % block_length != 0:
        raise ValueError(
            f"max_new_tokens ({gen_length}) must be divisible by block_length ({block_length})."
        )

    x = torch.full((bsz, gen_length), mask_id, dtype=torch.long, device=core_model.device)
    past_key_values = core_model(
        None,
        input_embeddings=inputs_embeds,
        use_cache=True,
    ).attn_key_values

    num_blocks = gen_length // block_length
    steps = steps // num_blocks
    if step_ratio is not None:
        steps = int(steps * step_ratio)
    if steps <= 0:
        raise ValueError(f"step_ratio={step_ratio} makes the number of steps become {steps}.")

    step_records = []
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
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

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

            x0_p[:, (block_idx + 1) * block_length :] = -np.inf
            x0 = torch.where(mask_index, x0, x)

            candidate_ids = x0[0].detach().cpu().tolist()
            candidate_text = clean_model_output(
                tokenizer.batch_decode(x0.detach().cpu(), skip_special_tokens=True)[0],
                conv_template,
            )

            confidence = torch.where(mask_index, x0_p, -np.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for batch_idx in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[batch_idx],
                    k=num_transfer_tokens[batch_idx, step_idx],
                )
                transfer_index[batch_idx, select_index] = True
            x[transfer_index] = x0[transfer_index]

            global_step += 1
            step_records.append(
                {
                    "step": global_step,
                    "block_index": block_idx + 1,
                    "step_in_block": step_idx + 1,
                    "num_transferred": int(num_transfer_tokens[0, step_idx].item()),
                    "num_masked_after_step": int((x == mask_id).sum().item()),
                    "candidate_token_ids": candidate_ids,
                    "candidate_text": candidate_text,
                }
            )

    final_text = clean_model_output(
        tokenizer.batch_decode(x.detach().cpu(), skip_special_tokens=True)[0],
        conv_template,
    )
    return final_text, step_records


def normalize_answers(doc, answer_processor):
    answers = doc.get("answers")
    if not answers:
        return []
    return [answer_processor(answer) for answer in answers]


def compute_textvqa_score(normalized_answers, prediction, answer_processor):
    normalized_prediction = answer_processor(prediction)
    if not normalized_answers:
        return 0.0, normalized_prediction

    gt_acc = []
    for idx in range(len(normalized_answers)):
        other_answers = [
            normalized_answers[j] for j in range(len(normalized_answers)) if j != idx
        ]
        matching = [answer for answer in other_answers if answer == normalized_prediction]
        gt_acc.append(min(1.0, float(len(matching)) / 3.0))
    return statistics.mean(gt_acc), normalized_prediction


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
    output_jsonl = Path(args.output_jsonl)
    summary_json = Path(args.summary_json)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    schedule = None if args.schedule == "none" else args.schedule
    schedule_kwargs = {"shift": args.schedule_shift} if schedule == "shift" else None
    answer_processor = EvalAIAnswerProcessor()

    step_totals = {}
    step_counts = {}
    num_written = 0
    total_final_score = 0.0
    t0_all = time.time()

    with output_jsonl.open("w", encoding="utf-8") as fout:
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
            image_tensor = move_images_to_device(
                images,
                image_processor,
                model.config,
                args.device,
            )
            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(args.device)
            inputs_embeds = prepare_inputs_embeds(model, input_ids, image_tensor, images)

            t0 = time.time()
            final_text, step_records = generate_step_candidates(
                model=model,
                tokenizer=tokenizer,
                inputs_embeds=inputs_embeds,
                conv_template=args.conv_template,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                temperature=0.0,
                remasking=args.remasking,
                schedule=schedule,
                schedule_kwargs=schedule_kwargs,
                mask_id=args.mask_id,
                step_ratio=args.step_ratio,
            )
            elapsed = time.time() - t0

            normalized_answers = normalize_answers(doc, answer_processor)
            step_results = []
            for step_record in step_records:
                exact_match, normalized_prediction = compute_textvqa_score(
                    normalized_answers,
                    step_record["candidate_text"],
                    answer_processor,
                )
                step = step_record["step"]
                step_totals[step] = step_totals.get(step, 0.0) + exact_match
                step_counts[step] = step_counts.get(step, 0) + 1

                step_results.append(
                    {
                        "step": step,
                        "block_index": step_record["block_index"],
                        "step_in_block": step_record["step_in_block"],
                        "num_transferred": step_record["num_transferred"],
                        "num_masked_after_step": step_record["num_masked_after_step"],
                        "candidate_text": step_record["candidate_text"],
                        "normalized_prediction": normalized_prediction,
                        "exact_match": exact_match,
                    }
                )

            final_score = step_results[-1]["exact_match"] if step_results else 0.0
            total_final_score += final_score
            record = {
                "dataset_index": dataset_index,
                "doc_id": str(doc.get("question_id")),
                "question_id": doc.get("question_id"),
                "question": context,
                "answers": doc.get("answers"),
                "ocr_tokens": doc.get("ocr_tokens"),
                "final_text": final_text,
                "final_exact_match": final_score,
                "num_steps": len(step_results),
                "elapsed_sec": elapsed,
                "step_results": step_results,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            num_written += 1
            if args.print_every > 0 and num_written % args.print_every == 0:
                first_score = step_results[0]["exact_match"] if step_results else 0.0
                print(
                    f"[{num_written}] id={record['doc_id']} steps={record['num_steps']} "
                    f"score@1={first_score:.3f} score@final={final_score:.3f} "
                    f"final={final_text!r}"
                )

    step_summary = []
    for step in sorted(step_totals):
        step_summary.append(
            {
                "step": step,
                "mean_exact_match": step_totals[step] / step_counts[step],
                "count": step_counts[step],
            }
        )

    summary = {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "pretrained": args.pretrained,
        "vision_tower": args.vision_tower,
        "model_name": args.model_name,
        "conv_template": args.conv_template,
        "max_new_tokens": args.max_new_tokens,
        "block_length": args.block_length if args.block_length is not None else args.max_new_tokens,
        "step_ratio": args.step_ratio,
        "remasking": args.remasking,
        "schedule": schedule,
        "schedule_kwargs": schedule_kwargs,
        "num_samples": num_written,
        "mean_final_exact_match": (total_final_score / num_written) if num_written else 0.0,
        "step_summary": step_summary,
        "elapsed_sec_total": time.time() - t0_all,
        "output_jsonl": str(output_jsonl),
    }

    with summary_json.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)

    print(f"Wrote {num_written} sample records to {output_jsonl}")
    print(f"Wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
