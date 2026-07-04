import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch


MASK_TOKEN_ID = 126336


def compute_remasking_confidence(logits: torch.Tensor, x0: torch.Tensor, remasking: str) -> torch.Tensor:
    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    if remasking == "random":
        return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    if remasking == "entrophy":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-12))
        return torch.sum(probs * log_probs, dim=-1)
    if remasking == "margin":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        return sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
    raise ValueError(f"Unsupported remasking strategy: {remasking}")


def decode_answer_tokens(tokenizer, token_ids: List[int]) -> str:
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    text = text.lstrip("!")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("<|eot_id|>", "")
    text = text.replace("<|im_end|>\n", "")
    text = text.replace("<|im_end|>", "")
    return text.strip()


def decode_single_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def resolve_steps_per_block(
    max_new_tokens: int,
    block_length: int,
    step_per_block: Optional[int],
    step_ratio: Optional[float],
) -> Tuple[int, int]:
    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")

    num_blocks = max_new_tokens // block_length
    steps = max_new_tokens // num_blocks
    if step_per_block is not None:
        if step_ratio is not None:
            raise ValueError("Do not pass both step_per_block and step_ratio.")
        steps = min(int(step_per_block), block_length)
    elif step_ratio is not None:
        steps = int(steps * float(step_ratio))
    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0.")
    return num_blocks, steps


def forward_logits(core_model, x: torch.Tensor, prefix_embeds: torch.Tensor, prefix_length: int) -> torch.Tensor:
    current_embeds = core_model.transformer.wte(x)
    current_embeds[:, :prefix_length] = prefix_embeds
    return core_model(None, input_embeddings=current_embeds).logits


@torch.no_grad()
def compute_leave_one_out_metrics(
    core_model,
    x: torch.Tensor,
    prefix_embeds: torch.Tensor,
    prefix_length: int,
    candidate_positions: torch.Tensor,
    chunk_size: int,
    previous_log_probs_by_pos: Dict[int, torch.Tensor],
    correction_metric: str,
    record_detailed_metrics: bool,
) -> Dict[str, torch.Tensor]:
    """Score generated tokens by masking each position and predicting its current value."""
    if candidate_positions.numel() == 0:
        empty = torch.empty(0, dtype=torch.float64, device=x.device)
        return {
            "log_likelihood": empty,
            "confidence": empty,
            "topk_margin": empty,
            "entropy": empty,
            "kl_divergence": empty,
        }

    if x.shape[0] != 1:
        raise ValueError("leave-one-out scoring currently expects batch size 1.")

    need_probs = record_detailed_metrics or correction_metric in {"topk_margin", "kl_divergence"}
    need_log_probs = need_probs or correction_metric in {"confidence", "time_aggregation"}

    log_likelihoods = []
    confidences = []
    topk_margins = []
    entropies = []
    kl_divergences = []
    positions = candidate_positions.to(device=x.device, dtype=torch.long)
    for start in range(0, int(positions.numel()), int(chunk_size)):
        pos_chunk = positions[start : start + int(chunk_size)]
        x_chunk = x.repeat(pos_chunk.numel(), 1)
        row_index = torch.arange(pos_chunk.numel(), device=x.device)
        target_ids = x_chunk[row_index, pos_chunk].clone()
        x_chunk[row_index, pos_chunk] = MASK_TOKEN_ID
        prefix_chunk = prefix_embeds.repeat(pos_chunk.numel(), 1, 1)
        logits = forward_logits(core_model, x_chunk, prefix_chunk, prefix_length)
        selected_logits = logits[row_index, pos_chunk].to(torch.float64)

        if need_log_probs:
            log_probs = F.log_softmax(selected_logits, dim=-1)
            target_log_likelihood = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        else:
            log_probs = None
            target_log_likelihood = torch.empty(0, dtype=torch.float64, device=x.device)

        if need_probs:
            probs = F.softmax(selected_logits, dim=-1) if log_probs is None else log_probs.exp()
        else:
            probs = None

        if correction_metric in {"confidence", "time_aggregation"} or record_detailed_metrics:
            log_likelihoods.append(target_log_likelihood)
            confidences.append(target_log_likelihood.exp())

        if correction_metric == "topk_margin" or record_detailed_metrics:
            top2_probs = torch.topk(probs, k=2, dim=-1).values
            topk_margins.append(top2_probs[:, 0] - top2_probs[:, 1])

        if record_detailed_metrics:
            entropies.append(-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1))

        if correction_metric == "kl_divergence" or record_detailed_metrics:
            if log_probs is None:
                log_probs = torch.log(probs.clamp_min(1e-12))
            chunk_kl = torch.zeros(pos_chunk.numel(), dtype=torch.float64, device=x.device)
            for local_idx, pos in enumerate(pos_chunk.detach().cpu().tolist()):
                previous_log_probs = previous_log_probs_by_pos.get(int(pos))
                if previous_log_probs is not None:
                    previous_log_probs = previous_log_probs.to(device=x.device, dtype=torch.float64)
                    chunk_kl[local_idx] = (probs[local_idx] * (log_probs[local_idx] - previous_log_probs)).sum()
                previous_log_probs_by_pos[int(pos)] = log_probs[local_idx].detach().to(device="cpu", dtype=torch.float32)
            kl_divergences.append(chunk_kl)

    return {
        "log_likelihood": torch.cat(log_likelihoods, dim=0) if log_likelihoods else torch.empty(0, dtype=torch.float64, device=x.device),
        "confidence": torch.cat(confidences, dim=0) if confidences else torch.empty(0, dtype=torch.float64, device=x.device),
        "topk_margin": torch.cat(topk_margins, dim=0) if topk_margins else torch.empty(0, dtype=torch.float64, device=x.device),
        "entropy": torch.cat(entropies, dim=0) if entropies else torch.empty(0, dtype=torch.float64, device=x.device),
        "kl_divergence": torch.cat(kl_divergences, dim=0) if kl_divergences else torch.empty(0, dtype=torch.float64, device=x.device),
    }


def build_metric_scores(
    metrics: Dict[str, torch.Tensor],
    cumulative_log_likelihood: torch.Tensor,
    correction_metric: str,
) -> torch.Tensor:
    if correction_metric == "confidence":
        return metrics["confidence"]
    if correction_metric == "time_aggregation":
        return cumulative_log_likelihood
    if correction_metric == "topk_margin":
        return metrics["topk_margin"]
    if correction_metric == "kl_divergence":
        return -metrics["kl_divergence"]
    raise ValueError(f"Unsupported correction_metric: {correction_metric}")


def select_remask_positions(
    positions: torch.Tensor,
    scores: torch.Tensor,
    remask_count: int,
    rule: str,
    stochastic_temperature: float,
) -> torch.Tensor:
    if remask_count <= 0 or positions.numel() == 0:
        return positions.new_empty(0)
    remask_count = min(int(remask_count), int(positions.numel()))
    if rule == "deterministic":
        selected = torch.topk(scores, k=remask_count, largest=False).indices
        return positions[selected]
    if rule == "stochastic":
        temperature = max(float(stochastic_temperature), 1e-6)
        weights = torch.softmax((-scores) / temperature, dim=0)
        selected = torch.multinomial(weights, num_samples=remask_count, replacement=False)
        return positions[selected]
    raise ValueError(f"Unsupported correction rule: {rule}")


@torch.no_grad()
def generate_with_free_correction(
    core_model,
    tokenizer,
    prefix_embeds: torch.Tensor,
    max_new_tokens: int,
    block_length: int,
    step_per_block: Optional[int],
    temperature: float,
    remasking: str,
    schedule: str,
    schedule_shift: float,
    step_ratio: Optional[float],
    correction_score: str = "cumulated",
    correction_metric: Optional[str] = None,
    correction_rule: str = "deterministic",
    transfer_per_step: Optional[int] = None,
    remask_ratio: float = 0.25,
    remask_per_step: Optional[int] = None,
    max_remask_per_step: Optional[int] = None,
    correction_scope: str = "current_block",
    loo_chunk_size: int = 16,
    stochastic_temperature: float = 1.0,
    skip_final_step_remask: bool = True,
    record_detailed_metrics: bool = False,
) -> Dict:
    if correction_metric is None:
        correction_metric = "time_aggregation" if correction_score == "cumulated" else "confidence"
    if correction_score not in {"current", "cumulated"}:
        raise ValueError("correction_score must be 'current' or 'cumulated'.")
    if correction_metric not in {"confidence", "time_aggregation", "topk_margin", "kl_divergence"}:
        raise ValueError(
            "correction_metric must be one of: confidence, time_aggregation, topk_margin, kl_divergence."
        )
    if correction_scope not in {"current_block", "generated"}:
        raise ValueError("correction_scope must be 'current_block' or 'generated'.")
    if not 0.0 <= float(remask_ratio) <= 1.0:
        raise ValueError("remask_ratio must be within [0, 1].")
    if transfer_per_step is not None and transfer_per_step <= 0:
        raise ValueError("transfer_per_step must be > 0.")
    if remask_per_step is not None and remask_per_step < 0:
        raise ValueError("remask_per_step must be >= 0.")
    if loo_chunk_size <= 0:
        raise ValueError("loo_chunk_size must be > 0.")

    device = prefix_embeds.device
    batch_size, prefix_length = prefix_embeds.shape[:2]
    if batch_size != 1:
        raise ValueError("generate_with_free_correction currently expects batch size 1.")

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
    cumulative_scores = torch.zeros_like(x, dtype=torch.float64)
    previous_log_probs_by_pos = {}
    correction_records = []
    global_step = 0

    for block_idx in range(num_blocks):
        block_start = prefix_length + block_idx * block_length
        block_end = prefix_length + (block_idx + 1) * block_length
        block_slice = slice(block_start, block_end)
        initial_block_mask = x[:, block_slice] == MASK_TOKEN_ID
        num_transfer_tokens = get_num_transfer_tokens_sch(
            initial_block_mask,
            steps_per_block,
            schedule=schedule_value,
            schedule_kwargs=schedule_kwargs,
        )

        for step_idx in range(steps_per_block):
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_slice]
            masked_remaining = int(block_mask_index.sum().item())
            if masked_remaining == 0:
                global_step += 1
                continue

            remaining_steps = steps_per_block - step_idx
            scheduled_k = int(num_transfer_tokens[0, min(step_idx, num_transfer_tokens.shape[1] - 1)].item())
            required_k = int(math.ceil(masked_remaining / remaining_steps))
            base_k = int(transfer_per_step) if transfer_per_step is not None else scheduled_k
            k = min(masked_remaining, max(base_k, required_k))

            logits = forward_logits(core_model, x, prefix_embeds, prefix_length)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0_p = compute_remasking_confidence(logits, x0, remasking)
            x0_p[:, prefix_length + (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            _, select_index = torch.topk(confidence[0], k=k)
            transfer_index[0, select_index] = True
            x[transfer_index] = x0[transfer_index]

            can_remask = not (skip_final_step_remask and step_idx == steps_per_block - 1)
            generated_mask = x[0, prefix_length : prefix_length + max_new_tokens] != MASK_TOKEN_ID
            if correction_scope == "current_block":
                scope_mask = x[0, block_slice] != MASK_TOKEN_ID
                candidate_positions = torch.nonzero(scope_mask, as_tuple=False).flatten() + block_start
            else:
                candidate_positions = torch.nonzero(generated_mask, as_tuple=False).flatten() + prefix_length

            remasked_positions = torch.empty(0, dtype=torch.long, device=device)
            loo_metrics = None
            ranked_scores = torch.empty(0, dtype=torch.float64, device=device)
            if remask_per_step is None:
                base_remask_count = int(math.floor(float(k) * float(remask_ratio)))
                if remask_ratio > 0.0 and base_remask_count == 0:
                    base_remask_count = 1
            else:
                base_remask_count = int(remask_per_step)
            if max_remask_per_step is not None:
                base_remask_count = min(base_remask_count, int(max_remask_per_step))
            remask_count = min(base_remask_count, max(0, int(candidate_positions.numel()) - 1))

            if can_remask and candidate_positions.numel() > 0 and remask_count > 0:
                loo_metrics = compute_leave_one_out_metrics(
                    core_model=core_model,
                    x=x,
                    prefix_embeds=prefix_embeds,
                    prefix_length=prefix_length,
                    candidate_positions=candidate_positions,
                    chunk_size=loo_chunk_size,
                    previous_log_probs_by_pos=previous_log_probs_by_pos,
                    correction_metric=correction_metric,
                    record_detailed_metrics=record_detailed_metrics,
                )
                if loo_metrics["log_likelihood"].numel() > 0:
                    cumulative_scores[0, candidate_positions] += loo_metrics["log_likelihood"]
                ranked_scores = build_metric_scores(
                    metrics=loo_metrics,
                    cumulative_log_likelihood=cumulative_scores[0, candidate_positions],
                    correction_metric=correction_metric,
                )

                remasked_positions = select_remask_positions(
                    positions=candidate_positions,
                    scores=ranked_scores,
                    remask_count=remask_count,
                    rule=correction_rule,
                    stochastic_temperature=stochastic_temperature,
                )
                if remasked_positions.numel() > 0:
                    x[0, remasked_positions] = MASK_TOKEN_ID

            state_ids = x[0, prefix_length:].detach().cpu().tolist()
            global_step += 1
            score_by_pos = {
                int(pos): float(score)
                for pos, score in zip(candidate_positions.detach().cpu().tolist(), ranked_scores.detach().cpu().tolist())
            }
            candidate_metric_details = []
            if loo_metrics is not None and record_detailed_metrics:
                metric_cpu = {name: value.detach().cpu().tolist() for name, value in loo_metrics.items()}
                cumulative_cpu = cumulative_scores[0, candidate_positions].detach().cpu().tolist()
                ranked_cpu = ranked_scores.detach().cpu().tolist()
                for idx, pos in enumerate(candidate_positions.detach().cpu().tolist()):
                    token_id = int(x0[0, pos].item())
                    candidate_metric_details.append(
                        {
                            "sequence_position": int(pos),
                            "answer_position": int(pos - prefix_length),
                            "token_id": token_id,
                            "token_text": decode_single_token(tokenizer, token_id),
                            "log_likelihood": float(metric_cpu["log_likelihood"][idx]),
                            "confidence": float(metric_cpu["confidence"][idx]),
                            "time_aggregation": float(cumulative_cpu[idx]),
                            "topk_margin": float(metric_cpu["topk_margin"][idx]),
                            "entropy": float(metric_cpu["entropy"][idx]),
                            "kl_divergence": float(metric_cpu["kl_divergence"][idx]),
                            "selected_metric_score": float(ranked_cpu[idx]),
                        }
                    )
            correction_records.append(
                {
                    "step": int(global_step),
                    "block_index": int(block_idx + 1),
                    "step_in_block": int(step_idx + 1),
                    "num_transferred": int(k),
                    "selected_positions": [int(pos) for pos in select_index.detach().cpu().tolist()],
                    "candidate_positions": [int(pos) for pos in candidate_positions.detach().cpu().tolist()],
                    "candidate_metrics": candidate_metric_details,
                    "remasked_positions": [int(pos) for pos in remasked_positions.detach().cpu().tolist()],
                    "remasked_tokens": [
                        {
                            "sequence_position": int(pos),
                            "answer_position": int(pos - prefix_length),
                            "token_id": int(x0[0, pos].item()),
                            "token_text": decode_single_token(tokenizer, int(x0[0, pos].item())),
                            "correction_score": score_by_pos.get(int(pos)),
                            "correction_metric": correction_metric,
                        }
                        for pos in remasked_positions.detach().cpu().tolist()
                    ],
                    "num_masked_after_step": int((x[:, prefix_length:] == MASK_TOKEN_ID).sum().item()),
                    "state_text": decode_answer_tokens(tokenizer, state_ids),
                }
            )

    final_ids = x[0, prefix_length:].detach().cpu().tolist()
    final_text = decode_answer_tokens(tokenizer, final_ids)
    return {
        "final_text": final_text,
        "final_answer_ids": final_ids,
        "correction_records": correction_records,
        "meta": {
            "prefix_length": int(prefix_length),
            "max_new_tokens": int(max_new_tokens),
            "block_length": int(block_length),
            "num_blocks": int(num_blocks),
            "steps_per_block": int(steps_per_block),
            "total_steps": int(num_blocks * steps_per_block),
            "correction_score": correction_score,
            "correction_metric": correction_metric,
            "correction_rule": correction_rule,
            "transfer_per_step": int(transfer_per_step) if transfer_per_step is not None else None,
            "remask_ratio": float(remask_ratio),
            "remask_per_step": int(remask_per_step) if remask_per_step is not None else None,
            "max_remask_per_step": int(max_remask_per_step) if max_remask_per_step is not None else None,
            "correction_scope": correction_scope,
            "loo_chunk_size": int(loo_chunk_size),
            "skip_final_step_remask": bool(skip_final_step_remask),
            "record_detailed_metrics": bool(record_detailed_metrics),
        },
    }
