import argparse
import json
import math
import sys
import time
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from M3CoT.run_m3cot_stepwise_x0 import (
    MASK_TOKEN_ID,
    clean_generated_text,
    prepare_prefix,
)
from M3CoT.utils.metric import judge_answer
from Scale_Attention.reweight_patch import get_torch_dtype, maybe_disable_torch_compile
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a two-stage M3CoT experiment: read a full x0 proposal at step k, "
            "remask selected positions, then run a short refinement pass."
        )
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--sample-mode", default="sequential", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="M3CoT/outputs/proposal_refine")

    parser.add_argument("--pretrained", default="weight/lavida-reason")
    parser.add_argument("--model-name", default="llava_llada")
    parser.add_argument("--vision-tower", default="weight/siglip")
    parser.add_argument("--conv-template", default="llada")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--prompt", default="cot", choices=["direct", "cot", "ccot", "dsp"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--step-per-block", type=int, default=None)
    parser.add_argument("--step-ratio", type=float, default=None)
    parser.add_argument("--schedule", default="none", choices=["shift", "cosine", "logit_normal", "none"])
    parser.add_argument("--schedule-shift", type=float, default=1.0 / 3.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])

    parser.add_argument("--proposal-policy", default="fixed", choices=["fixed", "x0_convergence", "entropy_plateau"])
    parser.add_argument("--proposal-step", type=int, default=32)
    parser.add_argument("--x0-stable-threshold", type=float, default=0.9)
    parser.add_argument("--x0-stable-persistence", type=int, default=2)
    parser.add_argument("--entropy-plateau-threshold", type=float, default=0.01)
    parser.add_argument("--entropy-plateau-persistence", type=int, default=2)
    parser.add_argument("--proposal-remask-ratio", type=float, default=0.5)
    parser.add_argument("--late-refine-steps", type=int, default=8)
    parser.add_argument(
        "--budget-total-steps",
        default=None,
        help="Use an integer total budget, or 'native' to match the current native generation steps.",
    )
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def compute_remasking_confidence(logits, x0, remasking):
    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    if remasking == "random":
        return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    if remasking == "entrophy":
        epsilon = 1e-10
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        log_probs = torch.log(probs + epsilon)
        return torch.sum(probs * log_probs, dim=-1)
    if remasking == "margin":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        return sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
    raise NotImplementedError(remasking)


def decode_answer_tokens(tokenizer, token_ids):
    return clean_generated_text(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def decode_single_token(tokenizer, token_id):
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def compute_token_entropy(logits):
    probs = F.softmax(logits.to(torch.float64), dim=-1)
    return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)


def resolve_steps_per_block(max_new_tokens, block_length, step_per_block, step_ratio):
    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")

    num_blocks = max_new_tokens // block_length
    steps = max_new_tokens
    if steps % num_blocks != 0 and step_per_block is None:
        raise ValueError("Native generation requires steps % num_blocks == 0 unless step_per_block is set.")
    steps = steps // num_blocks

    if step_per_block is None and step_ratio is None:
        step_per_block = block_length
    if step_per_block is not None:
        if step_ratio is not None:
            raise ValueError("Do not pass both --step-per-block and --step-ratio.")
        steps = min(int(step_per_block), block_length)
    if step_ratio:
        steps = int(steps * step_ratio)
    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0. Increase step_ratio or max_new_tokens.")
    return num_blocks, steps


def build_proposal_state(proposal_answer, remasked_answer_positions, prefix_length, device):
    answer_length = proposal_answer.shape[0]
    proposal_state = proposal_answer.clone()
    if remasked_answer_positions:
        proposal_state[torch.tensor(remasked_answer_positions, device=device, dtype=torch.long)] = MASK_TOKEN_ID

    x_refine = torch.full((1, prefix_length + answer_length), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x_refine[:, :prefix_length] = 0
    x_refine[0, prefix_length:] = proposal_state
    return x_refine


def should_trigger_proposal(
    policy,
    global_step,
    proposal_step,
    proposal_answer,
    prev_proposal_answer,
    mean_entropy,
    prev_mean_entropy,
    x0_stable_threshold,
    x0_stable_persistence,
    x0_stable_persistence_needed,
    entropy_plateau_persistence,
    entropy_plateau_persistence_needed,
    entropy_plateau_threshold,
):
    if policy == "fixed":
        return global_step == proposal_step, None, x0_stable_persistence, entropy_plateau_persistence

    if policy == "x0_convergence":
        stable_frac = None
        if prev_proposal_answer is not None and proposal_answer.numel() > 0:
            stable_frac = float(proposal_answer.eq(prev_proposal_answer).to(torch.float32).mean().item())
            if stable_frac >= x0_stable_threshold:
                x0_stable_persistence += 1
            else:
                x0_stable_persistence = 0
        else:
            x0_stable_persistence = 0
        triggered = x0_stable_persistence >= x0_stable_persistence_needed
        return triggered, stable_frac, x0_stable_persistence, entropy_plateau_persistence

    if policy == "entropy_plateau":
        entropy_delta = None
        if prev_mean_entropy is not None:
            entropy_delta = float(prev_mean_entropy - mean_entropy)
            if entropy_delta <= entropy_plateau_threshold:
                entropy_plateau_persistence += 1
            else:
                entropy_plateau_persistence = 0
        else:
            entropy_plateau_persistence = 0
        triggered = entropy_plateau_persistence >= entropy_plateau_persistence_needed
        return triggered, entropy_delta, x0_stable_persistence, entropy_plateau_persistence

    raise ValueError(f"Unsupported proposal policy: {policy}")


@torch.no_grad()
def run_proposal_then_refine(
    core_model,
    tokenizer,
    prefix_embeds,
    max_new_tokens,
    block_length,
    step_per_block,
    cfg_scale,
    temperature,
    remasking,
    schedule,
    schedule_shift,
    step_ratio,
    proposal_policy,
    proposal_step,
    x0_stable_threshold,
    x0_stable_persistence_needed,
    entropy_plateau_threshold,
    entropy_plateau_persistence_needed,
    proposal_remask_ratio,
    late_refine_steps,
    budget_total_steps,
):
    if cfg_scale > 0.0:
        raise NotImplementedError("cfg_scale > 0.0 is not supported in the native path.")
    if proposal_policy == "fixed" and proposal_step <= 0:
        raise ValueError("--proposal-step must be >= 1.")
    if not 0.0 <= proposal_remask_ratio <= 1.0:
        raise ValueError("--proposal-remask-ratio must be within [0, 1].")
    if late_refine_steps < 0:
        raise ValueError("--late-refine-steps must be >= 0.")
    if budget_total_steps is not None and budget_total_steps != "native":
        try:
            budget_total_steps = int(budget_total_steps)
        except ValueError as exc:
            raise ValueError("--budget-total-steps must be an integer or 'native'.") from exc
        if budget_total_steps <= 0:
            raise ValueError("--budget-total-steps must be > 0.")

    device = prefix_embeds.device
    batch_size, prefix_length = prefix_embeds.shape[:2]
    num_blocks, steps_per_block = resolve_steps_per_block(
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        step_per_block=step_per_block,
        step_ratio=step_ratio,
    )

    x = torch.full((batch_size, prefix_length + max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    x[:, :prefix_length] = 0

    schedule_value = None if schedule == "none" else schedule
    schedule_kwargs = {"shift": schedule_shift} if schedule_value == "shift" else None

    proposal_trace = []
    global_step = 0
    proposal_payload = None
    prev_proposal_answer = None
    prev_mean_entropy = None
    x0_stable_persistence = 0
    entropy_plateau_persistence = 0
    total_native_steps = num_blocks * steps_per_block
    effective_budget_total_steps = total_native_steps if budget_total_steps == "native" else budget_total_steps

    for block_idx in range(num_blocks):
        block_start = prefix_length + block_idx * block_length
        block_end = prefix_length + (block_idx + 1) * block_length
        block_slice = slice(block_start, block_end)
        block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
        num_transfer_tokens = get_num_transfer_tokens_sch(
            block_mask_index,
            steps_per_block,
            schedule=schedule_value,
            schedule_kwargs=schedule_kwargs,
        )

        for step_idx in range(num_transfer_tokens.shape[1]):
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_slice]
            if block_mask_index.sum().item() == 0:
                continue

            current_embeds = core_model.transformer.wte(x)
            current_embeds[:, :prefix_length] = prefix_embeds
            logits = core_model(None, input_embeddings=current_embeds).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            raw_confidence = compute_remasking_confidence(logits, x0, remasking)
            x0_p = raw_confidence.clone()
            x0_p[:, prefix_length + (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)

            proposal_answer = x0[0, prefix_length:].detach().clone()
            proposal_confidence = raw_confidence[0, prefix_length:].detach().clone()
            proposal_text = decode_answer_tokens(tokenizer, proposal_answer.detach().cpu().tolist())
            token_entropy = compute_token_entropy(logits)
            mean_entropy = float(token_entropy[0, prefix_length:].mean().item())

            confidence = torch.where(mask_index, x0_p, -torch.inf)
            k = int(num_transfer_tokens[0, step_idx].item())
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            _, select_index = torch.topk(confidence[0], k=k)
            transfer_index[0, select_index] = True
            x[transfer_index] = x0[transfer_index]

            global_step += 1
            proposal_trace.append(
                {
                    "step": int(global_step),
                    "block_index": int(block_idx + 1),
                    "step_in_block": int(step_idx + 1),
                    "num_transferred": int(k),
                    "selected_positions": [int(pos) for pos in select_index.detach().cpu().tolist()],
                    "num_masked_after_step": int((x[:, prefix_length:] == MASK_TOKEN_ID).sum().item()),
                    "candidate_text": proposal_text,
                    "mean_entropy": mean_entropy,
                }
            )

            should_trigger, trigger_metric, x0_stable_persistence, entropy_plateau_persistence = should_trigger_proposal(
                policy=proposal_policy,
                global_step=global_step,
                proposal_step=proposal_step,
                proposal_answer=proposal_answer,
                prev_proposal_answer=prev_proposal_answer,
                mean_entropy=mean_entropy,
                prev_mean_entropy=prev_mean_entropy,
                x0_stable_threshold=x0_stable_threshold,
                x0_stable_persistence=x0_stable_persistence,
                x0_stable_persistence_needed=x0_stable_persistence_needed,
                entropy_plateau_persistence=entropy_plateau_persistence,
                entropy_plateau_persistence_needed=entropy_plateau_persistence_needed,
                entropy_plateau_threshold=entropy_plateau_threshold,
            )
            if should_trigger:
                proposal_payload = {
                    "proposal_answer": proposal_answer,
                    "proposal_confidence": proposal_confidence,
                    "proposal_text": proposal_text,
                    "proposal_trace": list(proposal_trace),
                    "steps_per_block": int(steps_per_block),
                    "num_blocks": int(num_blocks),
                    "total_native_steps": int(total_native_steps),
                    "proposal_metric": trigger_metric,
                    "mean_entropy": mean_entropy,
                    "x0_stable_persistence": int(x0_stable_persistence),
                    "entropy_plateau_persistence": int(entropy_plateau_persistence),
                }
                break

            prev_proposal_answer = proposal_answer
            prev_mean_entropy = mean_entropy

        if proposal_payload is not None:
            break

    if proposal_payload is None:
        if proposal_policy == "fixed":
            raise ValueError(
                f"--proposal-step={proposal_step} exceeds available native denoising steps "
                f"({num_blocks * steps_per_block})."
            )
        raise ValueError(
            f"Proposal policy {proposal_policy} did not trigger within available native denoising steps "
            f"({total_native_steps})."
        )

    proposal_answer = proposal_payload["proposal_answer"]
    proposal_confidence = proposal_payload["proposal_confidence"]
    proposal_text = proposal_payload["proposal_text"]
    proposal_trigger_step = int(len(proposal_payload["proposal_trace"]))

    effective_late_refine_steps = late_refine_steps
    budget_remask_ratio = None
    if effective_budget_total_steps is not None:
        effective_late_refine_steps = max(0, int(effective_budget_total_steps) - proposal_trigger_step)
        budget_remask_ratio = min(1.0, effective_late_refine_steps / float(proposal_payload["steps_per_block"]))

    num_answer_positions = int(proposal_answer.shape[0])
    effective_remask_ratio = proposal_remask_ratio if budget_remask_ratio is None else budget_remask_ratio
    num_to_remask = int(math.floor(num_answer_positions * effective_remask_ratio))
    if effective_remask_ratio > 0.0 and num_to_remask == 0:
        num_to_remask = 1

    remasked_answer_positions = []
    if num_to_remask > 0:
        remask_priority = 1.0 - proposal_confidence
        remask_indices = torch.topk(remask_priority, k=num_to_remask, largest=True).indices
        remasked_answer_positions = sorted(int(idx) for idx in remask_indices.detach().cpu().tolist())
    else:
        remask_priority = 1.0 - proposal_confidence

    x_refine = build_proposal_state(
        proposal_answer=proposal_answer,
        remasked_answer_positions=remasked_answer_positions,
        prefix_length=prefix_length,
        device=device,
    )

    remasked_position_details = []
    remasked_position_index = {}
    for answer_pos in remasked_answer_positions:
        token_id = int(proposal_answer[answer_pos].item())
        remasked_detail = {
            "answer_position": answer_pos,
            "sequence_position": int(prefix_length + answer_pos),
            "proposal_token_id": token_id,
            "proposal_token_text": decode_single_token(tokenizer, token_id),
            "proposal_confidence": float(proposal_confidence[answer_pos].item()),
            "remask_priority": float(remask_priority[answer_pos].item()),
            "recovered_in_refine_step": None,
            "recovered_token_id": None,
            "recovered_token_text": None,
            "recovered_confidence": None,
        }
        remasked_position_index[answer_pos] = len(remasked_position_details)
        remasked_position_details.append(remasked_detail)

    proposal_fill_steps = []
    for trace_record in proposal_payload["proposal_trace"]:
        step_selected_positions = []
        for seq_pos in trace_record["selected_positions"]:
            if seq_pos < prefix_length:
                continue
            answer_pos = int(seq_pos - prefix_length)
            if answer_pos < 0 or answer_pos >= int(proposal_answer.shape[0]):
                continue
            token_id = int(proposal_answer[answer_pos].item())
            step_selected_positions.append(
                {
                    "answer_position": answer_pos,
                    "sequence_position": int(seq_pos),
                    "proposal_token_id": token_id,
                    "proposal_token_text": decode_single_token(tokenizer, token_id),
                    "proposal_confidence": float(proposal_confidence[answer_pos].item()),
                }
            )
        proposal_fill_steps.append(
            {
                "step": int(trace_record["step"]),
                "block_index": int(trace_record["block_index"]),
                "step_in_block": int(trace_record["step_in_block"]),
                "num_transferred": int(trace_record["num_transferred"]),
                "num_masked_after_step": int(trace_record["num_masked_after_step"]),
                "mean_entropy": trace_record.get("mean_entropy"),
                "selected_positions": list(trace_record["selected_positions"]),
                "selected_tokens": step_selected_positions,
                "candidate_text": trace_record["candidate_text"],
            }
        )

    refine_records = []
    for refine_step in range(1, effective_late_refine_steps + 1):
        answer_mask = x_refine[:, prefix_length:] == MASK_TOKEN_ID
        masked_remaining = int(answer_mask.sum().item())
        if masked_remaining == 0:
            break

        current_embeds = core_model.transformer.wte(x_refine)
        current_embeds[:, :prefix_length] = prefix_embeds
        logits = core_model(None, input_embeddings=current_embeds).logits

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        x0 = torch.where(x_refine == MASK_TOKEN_ID, x0, x_refine)
        confidence = compute_remasking_confidence(logits, x0, remasking)
        confidence = torch.where(x_refine == MASK_TOKEN_ID, confidence, -torch.inf)

        remaining_steps = effective_late_refine_steps - refine_step + 1
        k = int(math.ceil(masked_remaining / remaining_steps))
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        _, select_index = torch.topk(confidence[0], k=k)
        transfer_index[0, select_index] = True

        selected_token_details = []
        for seq_pos in select_index.detach().cpu().tolist():
            if seq_pos < prefix_length:
                continue
            answer_pos = int(seq_pos - prefix_length)
            token_id = int(x0[0, seq_pos].item())
            token_conf = float(confidence[0, seq_pos].item())
            token_text = decode_single_token(tokenizer, token_id)
            selected_token_details.append(
                {
                    "answer_position": answer_pos,
                    "sequence_position": int(seq_pos),
                    "pred_token_id": token_id,
                    "pred_token_text": token_text,
                    "pred_confidence": token_conf,
                }
            )
            if answer_pos in remasked_position_index:
                detail_index = remasked_position_index[answer_pos]
                if remasked_position_details[detail_index]["recovered_in_refine_step"] is None:
                    remasked_position_details[detail_index]["recovered_in_refine_step"] = int(refine_step)
                    remasked_position_details[detail_index]["recovered_token_id"] = token_id
                    remasked_position_details[detail_index]["recovered_token_text"] = token_text
                    remasked_position_details[detail_index]["recovered_confidence"] = token_conf

        x_refine[transfer_index] = x0[transfer_index]

        candidate_text = decode_answer_tokens(
            tokenizer,
            x0[0, prefix_length:].detach().cpu().tolist(),
        )
        refine_state_text = decode_answer_tokens(
            tokenizer,
            x_refine[0, prefix_length:].detach().cpu().tolist(),
        )
        refine_records.append(
            {
                "refine_step": int(refine_step),
                "num_transferred": int(k),
                "masked_before_step": int(masked_remaining),
                "masked_after_step": int((x_refine[:, prefix_length:] == MASK_TOKEN_ID).sum().item()),
                "selected_positions": [int(pos) for pos in select_index.detach().cpu().tolist()],
                "selected_tokens": selected_token_details,
                "candidate_text": candidate_text,
                "refine_state_text": refine_state_text,
            }
        )

    final_answer_ids = x_refine[0, prefix_length:].detach().cpu().tolist()
    final_text = decode_answer_tokens(tokenizer, final_answer_ids)
    meta = {
        "prefix_length": int(prefix_length),
        "max_new_tokens": int(max_new_tokens),
        "block_length": int(block_length),
        "num_blocks": proposal_payload["num_blocks"],
        "steps_per_block": proposal_payload["steps_per_block"],
        "total_native_steps": proposal_payload["total_native_steps"],
        "proposal_policy": proposal_policy,
        "proposal_step": int(proposal_step),
        "proposal_trigger_step": proposal_trigger_step,
        "proposal_metric": proposal_payload["proposal_metric"],
        "proposal_mean_entropy": float(proposal_payload["mean_entropy"]),
        "proposal_remask_ratio": float(proposal_remask_ratio),
        "effective_proposal_remask_ratio": float(effective_remask_ratio),
        "budget_total_steps": int(effective_budget_total_steps) if effective_budget_total_steps is not None else None,
        "budget_total_steps_arg": budget_total_steps,
        "num_remasked_positions": int(len(remasked_answer_positions)),
        "late_refine_steps_requested": int(late_refine_steps),
        "effective_late_refine_steps": int(effective_late_refine_steps),
        "late_refine_steps_run": int(len(refine_records)),
    }
    return {
        "proposal_text": proposal_text,
        "proposal_trace": proposal_payload["proposal_trace"],
        "proposal_fill_steps": proposal_fill_steps,
        "proposal_confidence": proposal_confidence.detach().cpu().tolist(),
        "remask_priority": remask_priority.detach().cpu().tolist(),
        "proposal_answer_ids": proposal_answer.detach().cpu().tolist(),
        "remasked_positions": remasked_position_details,
        "refine_records": refine_records,
        "final_text": final_text,
        "meta": meta,
    }


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

    dataset = datasets.load_dataset(args.dataset_path, split=args.split)
    if args.sample_mode == "random":
        dataset = dataset.shuffle(seed=args.sample_seed)
    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"

    proposal_correct = 0
    final_correct = 0
    improved_after_refine = 0
    worsened_after_refine = 0
    unchanged_after_refine = 0
    total_elapsed = 0.0
    written = 0
    refine_step_correct = {}
    refine_step_count = {}

    with records_path.open("w", encoding="utf-8") as fout:
        for dataset_index, doc in enumerate(dataset):
            if doc.get("image") is None:
                continue

            t0 = time.time()
            context, prompt, prefix_embeds, _ = prepare_prefix(
                args,
                model,
                tokenizer,
                image_processor,
                doc,
            )
            run_output = run_proposal_then_refine(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                step_per_block=args.step_per_block,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                remasking=args.remasking,
                schedule=args.schedule,
                schedule_shift=args.schedule_shift,
                step_ratio=args.step_ratio,
                proposal_policy=args.proposal_policy,
                proposal_step=args.proposal_step,
                x0_stable_threshold=args.x0_stable_threshold,
                x0_stable_persistence_needed=args.x0_stable_persistence,
                entropy_plateau_threshold=args.entropy_plateau_threshold,
                entropy_plateau_persistence_needed=args.entropy_plateau_persistence,
                proposal_remask_ratio=args.proposal_remask_ratio,
                late_refine_steps=args.late_refine_steps,
                budget_total_steps=args.budget_total_steps,
            )
            elapsed = time.time() - t0
            total_elapsed += elapsed

            proposal_text = run_output["proposal_text"]
            final_text = run_output["final_text"]
            proposal_is_correct = bool(judge_answer(proposal_text, doc["choices"], doc["answer"]))
            final_is_correct = bool(judge_answer(final_text, doc["choices"], doc["answer"]))

            proposal_correct += int(proposal_is_correct)
            final_correct += int(final_is_correct)
            if not proposal_is_correct and final_is_correct:
                improved_after_refine += 1
            elif proposal_is_correct and not final_is_correct:
                worsened_after_refine += 1
            else:
                unchanged_after_refine += 1

            scored_refine_records = []
            for refine_record in run_output["refine_records"]:
                is_correct = bool(judge_answer(refine_record["refine_state_text"], doc["choices"], doc["answer"]))
                step = int(refine_record["refine_step"])
                refine_step_correct[step] = refine_step_correct.get(step, 0) + int(is_correct)
                refine_step_count[step] = refine_step_count.get(step, 0) + 1

                scored_refine_record = dict(refine_record)
                scored_refine_record["correct"] = is_correct
                scored_refine_records.append(scored_refine_record)

            record = {
                "dataset_index": int(dataset_index),
                "id": doc["id"],
                "question": context,
                "choices": list(doc["choices"]),
                "answer": doc["answer"],
                "domain": doc["domain"],
                "topic": doc["topic"],
                "prompt": prompt,
                "elapsed_sec": elapsed,
                "proposal_text": proposal_text,
                "proposal_correct": proposal_is_correct,
                "final_text": final_text,
                "final_correct": final_is_correct,
                "proposal_answer_ids": run_output["proposal_answer_ids"],
                "proposal_confidence": run_output["proposal_confidence"],
                "remask_priority": run_output["remask_priority"],
                "remasked_positions": run_output["remasked_positions"],
                "proposal_trace": run_output["proposal_trace"],
                "refine_records": scored_refine_records,
                "meta": run_output["meta"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                print(
                    f"[{written}] dataset_index={dataset_index} id={doc['id']} "
                    f"proposal={proposal_is_correct} final={final_is_correct} "
                    f"remasked={len(run_output['remasked_positions'])} elapsed={elapsed:.2f}s",
                    flush=True,
                )

    refine_step_summary = []
    for step in sorted(refine_step_count):
        count = refine_step_count[step]
        refine_step_summary.append(
            {
                "refine_step": int(step),
                "mean_acc": refine_step_correct[step] / count if count else None,
                "count": int(count),
            }
        )

    summary = {
        "dataset_path": args.dataset_path,
        "split": args.split,
        "start_index": args.start_index,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed if args.sample_mode == "random" else None,
        "num_samples": written,
        "prompt": args.prompt,
        "proposal_definition": (
            "Read the full answer-position x0 at proposal_step, keep higher-confidence positions, "
            "remask lower-confidence positions, then run a short refinement pass."
        ),
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": total_elapsed / written if written else None,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "step_per_block": args.step_per_block,
            "step_ratio": args.step_ratio,
            "schedule": args.schedule,
            "schedule_shift": args.schedule_shift,
            "cfg_scale": args.cfg_scale,
            "remasking": args.remasking,
            "temperature": args.temperature,
        },
        "proposal_refine": {
            "proposal_policy": args.proposal_policy,
            "proposal_step": args.proposal_step,
            "x0_stable_threshold": args.x0_stable_threshold,
            "x0_stable_persistence": args.x0_stable_persistence,
            "entropy_plateau_threshold": args.entropy_plateau_threshold,
            "entropy_plateau_persistence": args.entropy_plateau_persistence,
            "proposal_remask_ratio": args.proposal_remask_ratio,
            "late_refine_steps": args.late_refine_steps,
            "budget_total_steps": args.budget_total_steps,
        },
        "proposal_mean_acc": proposal_correct / written if written else None,
        "final_mean_acc": final_correct / written if written else None,
        "mean_gain_from_refine": ((final_correct - proposal_correct) / written) if written else None,
        "num_improved_after_refine": improved_after_refine,
        "num_worsened_after_refine": worsened_after_refine,
        "num_unchanged_after_refine": unchanged_after_refine,
        "refine_step_summary": refine_step_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote records to {records_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)
    restore_compile()


if __name__ == "__main__":
    main()
