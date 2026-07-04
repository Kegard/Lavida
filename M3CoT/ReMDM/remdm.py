import argparse
import json
import sys
import time
from pathlib import Path

import datasets
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "eval"
for path in (REPO_ROOT, EVAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from M3CoT.run_m3cot_stepwise_x0 import (
    MASK_TOKEN_ID,
    clean_generated_text,
    compute_remasking_confidence,
    prepare_prefix,
)
from M3CoT.utils.metric import judge_answer
from Scale_Attention.reweight_patch import get_torch_dtype, maybe_disable_torch_compile
from llava.model.language_model.llada.generate import add_gumbel_noise


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run ReMDM-style iterative propose-remask decoding on M3CoT: "
            "each step proposes several masked tokens, then remasks low-confidence tokens."
        )
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--domain-filter", default=None)
    parser.add_argument("--sample-mode", default="random", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="M3CoT/ReMDM/outputs/remdm_m64_s32_p4_r2_seed42_n400")

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
    parser.add_argument("--step-ratio", type=float, default=0.5)
    parser.add_argument("--proposal-per-step", type=int, default=4)
    parser.add_argument("--remask-per-step", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])
    parser.add_argument(
        "--remask-scope",
        default="filled",
        choices=["filled", "new"],
        help=(
            "filled: remask the lowest-confidence tokens among all currently filled answer tokens; "
            "new: remask only among tokens proposed in the current step."
        ),
    )
    parser.add_argument(
        "--remask-final-step",
        action="store_true",
        help="Also remask after the final step. By default the final step is kept complete.",
    )
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--no-records", action="store_true")
    return parser.parse_args()


def resolve_total_steps(max_new_tokens, block_length, step_per_block, step_ratio):
    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")
    num_blocks = max_new_tokens // block_length
    if num_blocks != 1:
        raise ValueError("remdm.py currently expects block_length == max_new_tokens.")

    steps = max_new_tokens
    if step_per_block is not None:
        if step_ratio is not None:
            raise ValueError("Do not pass both --step-per-block and --step-ratio.")
        steps = min(int(step_per_block), block_length)
    elif step_ratio is not None:
        steps = int(steps * float(step_ratio))
    if steps <= 0:
        raise ValueError("The computed total step count is 0.")
    return steps


def decode_answer(tokenizer, answer_ids):
    return clean_generated_text(
        tokenizer.decode(
            answer_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def forward_logits(core_model, x, prefix_embeds, prefix_length):
    current_embeds = core_model.transformer.wte(x)
    current_embeds[:, :prefix_length] = prefix_embeds
    return core_model(None, input_embeddings=current_embeds).logits


def answer_position_list(seq_positions, prefix_length):
    return [
        int(seq_pos - prefix_length)
        for seq_pos in seq_positions.detach().cpu().tolist()
        if int(seq_pos) >= int(prefix_length)
    ]


def build_remask_scores(
    x,
    confidence,
    answer_slice,
    filled_answer_mask,
    fill_answer_positions,
    remask_scope,
):
    answer_confidence = confidence[0, answer_slice].to(torch.float64)
    scores = torch.full_like(answer_confidence, float("inf"), dtype=torch.float64)

    if remask_scope == "new":
        if fill_answer_positions.numel() > 0:
            scores[fill_answer_positions] = answer_confidence[fill_answer_positions]
        return scores

    if remask_scope == "filled":
        scores[filled_answer_mask] = answer_confidence[filled_answer_mask]
        return scores

    raise ValueError(f"Unsupported remask_scope: {remask_scope}")


@torch.no_grad()
def generate_with_remdm(
    core_model,
    tokenizer,
    prefix_embeds,
    max_new_tokens,
    block_length,
    step_per_block,
    step_ratio,
    proposal_per_step,
    remask_per_step,
    temperature,
    remasking,
    remask_scope,
    remask_final_step,
):
    total_steps = resolve_total_steps(
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        step_per_block=step_per_block,
        step_ratio=step_ratio,
    )
    if proposal_per_step <= 0:
        raise ValueError("proposal_per_step must be > 0.")
    if remask_per_step < 0:
        raise ValueError("remask_per_step must be >= 0.")
    if proposal_per_step <= remask_per_step and not remask_final_step:
        raise ValueError("proposal_per_step should be greater than remask_per_step.")

    device = prefix_embeds.device
    batch_size, prefix_length = prefix_embeds.shape[:2]
    if batch_size != 1:
        raise ValueError("generate_with_remdm currently expects batch size 1.")

    x = torch.full(
        (batch_size, prefix_length + max_new_tokens),
        MASK_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    x[:, :prefix_length] = 0
    answer_slice = slice(prefix_length, prefix_length + max_new_tokens)
    state_confidence = torch.full(
        (max_new_tokens,),
        float("inf"),
        dtype=torch.float64,
        device=device,
    )
    step_records = []

    for step_idx in range(1, total_steps + 1):
        mask_index = x == MASK_TOKEN_ID
        answer_mask_index = mask_index[:, answer_slice]
        remaining_masked = int(answer_mask_index.sum().item())
        if remaining_masked == 0:
            break

        pre_remask_answer_positions = torch.empty(0, dtype=torch.long, device=device)
        pre_remasked_token_ids = []
        if remaining_masked < int(proposal_per_step):
            filled_answer_mask = x[0, answer_slice] != MASK_TOKEN_ID
            pre_remask_scores = torch.full_like(
                state_confidence,
                float("inf"),
                dtype=torch.float64,
            )
            pre_remask_scores[filled_answer_mask] = state_confidence[filled_answer_mask]
            pre_remask_count = min(
                int(proposal_per_step) - remaining_masked,
                int(torch.isfinite(pre_remask_scores).sum().item()),
            )
            if pre_remask_count > 0:
                pre_remask_answer_positions = torch.topk(
                    pre_remask_scores,
                    k=pre_remask_count,
                    largest=False,
                ).indices
                pre_remasked_token_ids = [
                    int(token_id)
                    for token_id in x[0, prefix_length + pre_remask_answer_positions].detach().cpu().tolist()
                ]
                x[0, prefix_length + pre_remask_answer_positions] = MASK_TOKEN_ID
                mask_index = x == MASK_TOKEN_ID
                answer_mask_index = mask_index[:, answer_slice]
                remaining_masked = int(answer_mask_index.sum().item())

        logits = forward_logits(core_model, x, prefix_embeds, prefix_length)
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        proposal_ids = torch.argmax(logits_with_noise, dim=-1)
        confidence = compute_remasking_confidence(logits, proposal_ids, remasking)
        state_confidence = confidence[0, answer_slice].detach().to(torch.float64)
        proposal_ids = torch.where(mask_index, proposal_ids, x)

        masked_confidence = torch.where(mask_index, confidence, -torch.inf)
        fill_count = min(int(proposal_per_step), remaining_masked)
        _, fill_seq_positions = torch.topk(masked_confidence[0], k=fill_count)
        x[0, fill_seq_positions] = proposal_ids[0, fill_seq_positions]
        fill_answer_positions = torch.tensor(
            answer_position_list(fill_seq_positions, prefix_length),
            dtype=torch.long,
            device=device,
        )

        remask_answer_positions = torch.empty(0, dtype=torch.long, device=device)
        remasked_token_ids = []
        should_remask = (
            remask_per_step > 0
            and (remask_final_step or step_idx < total_steps)
        )
        if should_remask:
            filled_answer_mask = x[0, answer_slice] != MASK_TOKEN_ID
            remask_scores = build_remask_scores(
                x=x,
                confidence=confidence,
                answer_slice=answer_slice,
                filled_answer_mask=filled_answer_mask,
                fill_answer_positions=fill_answer_positions,
                remask_scope=remask_scope,
            )
            available = int(torch.isfinite(remask_scores).sum().item())
            remask_count = min(int(remask_per_step), available)
            if remask_count > 0:
                remask_answer_positions = torch.topk(
                    remask_scores,
                    k=remask_count,
                    largest=False,
                ).indices
                remasked_token_ids = [
                    int(token_id)
                    for token_id in x[0, prefix_length + remask_answer_positions].detach().cpu().tolist()
                ]
                x[0, prefix_length + remask_answer_positions] = MASK_TOKEN_ID

        candidate_answer_ids = proposal_ids[0, answer_slice].detach().cpu().tolist()
        state_answer_ids = x[0, answer_slice].detach().cpu().tolist()
        step_records.append(
            {
                "step": int(step_idx),
                "num_proposed": int(fill_count),
                "pre_remasked_answer_positions": [
                    int(pos) for pos in pre_remask_answer_positions.detach().cpu().tolist()
                ],
                "pre_remasked_token_ids": pre_remasked_token_ids,
                "num_remasked": int(remask_answer_positions.numel()),
                "filled_answer_positions": [
                    int(pos) for pos in fill_answer_positions.detach().cpu().tolist()
                ],
                "remasked_answer_positions": [
                    int(pos) for pos in remask_answer_positions.detach().cpu().tolist()
                ],
                "remasked_token_ids": remasked_token_ids,
                "candidate_text": decode_answer(tokenizer, candidate_answer_ids),
                "state_text": decode_answer(tokenizer, state_answer_ids),
                "num_masked_after_step": int((x[:, answer_slice] == MASK_TOKEN_ID).sum().item()),
            }
        )

    final_answer_ids = x[0, answer_slice].detach().cpu().tolist()
    final_text = decode_answer(tokenizer, final_answer_ids)
    meta = {
        "max_new_tokens": int(max_new_tokens),
        "block_length": int(block_length),
        "total_steps": int(total_steps),
        "executed_steps": int(len(step_records)),
        "proposal_per_step": int(proposal_per_step),
        "remask_per_step": int(remask_per_step),
        "remask_scope": remask_scope,
        "remask_final_step": bool(remask_final_step),
        "temperature": float(temperature),
        "remasking": remasking,
        "num_masked_final": int((x[:, answer_slice] == MASK_TOKEN_ID).sum().item()),
    }
    return {
        "final_text": final_text,
        "final_answer_ids": final_answer_ids,
        "step_records": step_records,
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
    if args.domain_filter:
        dataset = dataset.filter(lambda row: row.get("domain") == args.domain_filter)
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
    write_records = not args.no_records

    total_elapsed = 0.0
    written = 0
    correct_total = 0

    record_file = records_path.open("w", encoding="utf-8") if write_records else None
    try:
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
            run_output = generate_with_remdm(
                core_model=core_model,
                tokenizer=tokenizer,
                prefix_embeds=prefix_embeds,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                step_per_block=args.step_per_block,
                step_ratio=args.step_ratio,
                proposal_per_step=args.proposal_per_step,
                remask_per_step=args.remask_per_step,
                temperature=args.temperature,
                remasking=args.remasking,
                remask_scope=args.remask_scope,
                remask_final_step=args.remask_final_step,
            )
            elapsed = time.time() - t0
            total_elapsed += elapsed

            final_correct = bool(judge_answer(run_output["final_text"], doc["choices"], doc["answer"]))
            correct_total += int(final_correct)

            if write_records:
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
                    "final_text": run_output["final_text"],
                    "final_answer_ids": run_output["final_answer_ids"],
                    "final_correct": final_correct,
                    "step_records": run_output["step_records"],
                    "meta": run_output["meta"],
                }
                record_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_file.flush()
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                print(
                    f"[{written}] dataset_index={dataset_index} id={doc['id']} "
                    f"final={final_correct} elapsed={elapsed:.2f}s",
                    flush=True,
                )
    finally:
        if record_file is not None:
            record_file.close()

    summary = {
        "dataset_path": args.dataset_path,
        "split": args.split,
        "start_index": args.start_index,
        "domain_filter": args.domain_filter,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed if args.sample_mode == "random" else None,
        "num_samples": written,
        "prompt": args.prompt,
        "total_elapsed_sec": total_elapsed,
        "mean_elapsed_sec": total_elapsed / written if written else None,
        "accuracy": correct_total / written if written else None,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "step_per_block": args.step_per_block,
            "step_ratio": args.step_ratio,
            "proposal_per_step": args.proposal_per_step,
            "remask_per_step": args.remask_per_step,
            "remask_scope": args.remask_scope,
            "remask_final_step": args.remask_final_step,
            "temperature": args.temperature,
            "remasking": args.remasking,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if write_records:
        print(f"Wrote records to {records_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)
    restore_compile()


if __name__ == "__main__":
    main()
