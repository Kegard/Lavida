from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch


MASK_TOKEN_ID = 126336


def _decode_token(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    except TypeError:
        token_text = tokenizer.decode([token_id])
    except Exception:
        return str(token_id)
    token_text = token_text.replace("\n", "\\n")
    return token_text if token_text else str(token_id)


def _topk_delta_entries(delta_row: torch.Tensor, tokenizer, topk: int, largest: bool) -> List[Dict]:
    if topk <= 0:
        return []
    k = min(int(topk), int(delta_row.shape[-1]))
    values, indices = torch.topk(delta_row, k=k, largest=largest)
    entries = []
    for value, index in zip(values.tolist(), indices.tolist()):
        token_id = int(index)
        entries.append(
            {
                "token_id": token_id,
                "token_text": _decode_token(tokenizer, token_id),
                "delta_logit": float(value),
            }
        )
    return entries


def _compute_step_trace(
    logits_cond: torch.Tensor,
    logits_uncond: torch.Tensor,
    logits_guided: torch.Tensor,
    active_positions: torch.Tensor,
    alpha_t: float,
    block_idx: int,
    step_idx: int,
    global_step_idx: int,
    total_denoising_steps: int,
    transfer_index: torch.Tensor,
    tokenizer=None,
    trace_topk: int = 10,
    trace_max_positions: int = 4,
) -> Dict:
    if active_positions.numel() == 0:
        return {
            "block_idx": int(block_idx),
            "step_idx": int(step_idx),
            "global_step_idx": int(global_step_idx),
            "total_denoising_steps": int(total_denoising_steps),
            "alpha_t": float(alpha_t),
            "num_active_positions": 0,
            "active_positions": [],
            "selected_positions": [],
            "per_position_mean_abs_delta": [],
            "per_position_max_abs_delta": [],
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0.0,
            "rms_delta": 0.0,
            "mean_kl_cond_uncond": 0.0,
            "mean_kl_uncond_cond": 0.0,
            "representative_positions": [],
        }

    cond_active = logits_cond[0, active_positions, :].to(torch.float32)
    uncond_active = logits_uncond[0, active_positions, :].to(torch.float32)
    guided_active = logits_guided[0, active_positions, :].to(torch.float32)
    delta_active = cond_active - uncond_active
    abs_delta = delta_active.abs()

    per_position_mean_abs = abs_delta.mean(dim=-1)
    per_position_max_abs = abs_delta.max(dim=-1).values
    rms_delta = torch.sqrt((delta_active.pow(2)).mean())

    cond_log_probs = F.log_softmax(cond_active, dim=-1)
    uncond_log_probs = F.log_softmax(uncond_active, dim=-1)
    cond_probs = cond_log_probs.exp()
    uncond_probs = uncond_log_probs.exp()
    kl_cond_uncond = (cond_probs * (cond_log_probs - uncond_log_probs)).sum(dim=-1)
    kl_uncond_cond = (uncond_probs * (uncond_log_probs - cond_log_probs)).sum(dim=-1)

    selected_positions = torch.nonzero(transfer_index[0], as_tuple=False).squeeze(-1)
    selected_position_set = {int(pos) for pos in selected_positions.tolist()}

    representative_positions = []
    num_repr = min(int(trace_max_positions), int(active_positions.numel()))
    repr_indices = torch.topk(per_position_mean_abs, k=num_repr, largest=True).indices
    for repr_idx in repr_indices.tolist():
        pos = int(active_positions[repr_idx].item())
        cond_row = cond_active[repr_idx]
        uncond_row = uncond_active[repr_idx]
        guided_row = guided_active[repr_idx]
        delta_row = delta_active[repr_idx]

        cond_token_id = int(torch.argmax(cond_row).item())
        uncond_token_id = int(torch.argmax(uncond_row).item())
        guided_token_id = int(torch.argmax(guided_row).item())

        representative_positions.append(
            {
                "position": pos,
                "selected_this_step": pos in selected_position_set,
                "mean_abs_delta": float(per_position_mean_abs[repr_idx].item()),
                "max_abs_delta": float(per_position_max_abs[repr_idx].item()),
                "cond_token_id": cond_token_id,
                "cond_token_text": _decode_token(tokenizer, cond_token_id),
                "cond_token_logit": float(cond_row[cond_token_id].item()),
                "uncond_token_id": uncond_token_id,
                "uncond_token_text": _decode_token(tokenizer, uncond_token_id),
                "uncond_token_logit": float(uncond_row[uncond_token_id].item()),
                "guided_token_id": guided_token_id,
                "guided_token_text": _decode_token(tokenizer, guided_token_id),
                "guided_token_logit": float(guided_row[guided_token_id].item()),
                "top_positive_deltas": _topk_delta_entries(delta_row, tokenizer, trace_topk, largest=True),
                "top_negative_deltas": _topk_delta_entries(delta_row, tokenizer, trace_topk, largest=False),
            }
        )

    return {
        "block_idx": int(block_idx),
        "step_idx": int(step_idx),
        "global_step_idx": int(global_step_idx),
        "total_denoising_steps": int(total_denoising_steps),
        "alpha_t": float(alpha_t),
        "num_active_positions": int(active_positions.numel()),
        "active_positions": [int(pos) for pos in active_positions.tolist()],
        "selected_positions": sorted(selected_position_set),
        "per_position_mean_abs_delta": [float(value) for value in per_position_mean_abs.tolist()],
        "per_position_max_abs_delta": [float(value) for value in per_position_max_abs.tolist()],
        "mean_abs_delta": float(abs_delta.mean().item()),
        "max_abs_delta": float(abs_delta.max().item()),
        "rms_delta": float(rms_delta.item()),
        "mean_kl_cond_uncond": float(kl_cond_uncond.mean().item()),
        "mean_kl_uncond_cond": float(kl_uncond_cond.mean().item()),
        "representative_positions": representative_positions,
    }


def compute_remasking_confidence(logits: torch.Tensor, x0: torch.Tensor, remasking: str) -> torch.Tensor:
    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    if remasking == "random":
        return torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    if remasking == "entrophy":
        epsilon = 1e-10
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        return torch.sum(probs * torch.log(probs + epsilon), dim=-1)
    if remasking == "margin":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        return sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
    raise NotImplementedError(remasking)


def build_unconditional_prefix_embeds(
    core_model,
    prefix_embeds: torch.Tensor,
    prefix_input_ids_full: torch.Tensor,
    null_visual_mode: str = "zeros",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prefix_input_ids_full.dim() != 1:
        raise ValueError("prefix_input_ids_full must be a 1D tensor for timestep VRG.")

    visual_mask = prefix_input_ids_full.eq(IMAGE_TOKEN_INDEX)
    uncond_prefix_embeds = prefix_embeds.clone()
    if not visual_mask.any():
        return uncond_prefix_embeds, visual_mask

    if null_visual_mode == "zeros":
        null_visual = torch.zeros(
            (1, 1, prefix_embeds.shape[-1]),
            dtype=prefix_embeds.dtype,
            device=prefix_embeds.device,
        )
    elif null_visual_mode == "mask_token":
        null_visual = core_model.transformer.wte(
            torch.tensor([MASK_TOKEN_ID], dtype=torch.long, device=prefix_embeds.device)
        ).view(1, 1, -1).to(dtype=prefix_embeds.dtype)
    else:
        raise ValueError(f"Unsupported null_visual_mode: {null_visual_mode}")

    uncond_prefix_embeds[:, visual_mask, :] = null_visual
    return uncond_prefix_embeds, visual_mask


def compute_step_vrg_alpha(
    global_step_idx: int,
    total_steps: int,
    alpha_start: float,
    alpha_end: float,
    schedule: str,
    power: float,
) -> float:
    if total_steps <= 1:
        progress = 1.0
    else:
        progress = float(global_step_idx) / float(total_steps - 1)

    if schedule == "linear":
        shaped = progress
    elif schedule == "cosine":
        shaped = 1.0 - torch.cos(torch.tensor(progress) * torch.pi / 2).item()
    elif schedule == "power":
        shaped = progress ** power
    else:
        raise ValueError(f"Unsupported vrg schedule: {schedule}")

    return float(alpha_start + (alpha_end - alpha_start) * shaped)


def _cat_past_key_values(cond_past_key_values, uncond_past_key_values):
    merged = []
    for cond_layer, uncond_layer in zip(cond_past_key_values, uncond_past_key_values):
        cond_key, cond_value = cond_layer
        uncond_key, uncond_value = uncond_layer
        merged.append(
            (
                torch.cat([cond_key, uncond_key], dim=0),
                torch.cat([cond_value, uncond_value], dim=0),
            )
        )
    return tuple(merged)


@torch.no_grad()
def generate_with_timestep_vrg(
    core_model,
    prefix_embeds: torch.Tensor,
    prefix_input_ids_full: torch.Tensor,
    max_new_tokens: int,
    block_length: int,
    temperature: float,
    remasking: str,
    schedule: str,
    schedule_shift: float,
    step_ratio: float,
    alpha_start: float = 0.0,
    alpha_end: float = 1.0,
    alpha_schedule: str = "linear",
    alpha_power: float = 2.0,
    null_visual_mode: str = "zeros",
    return_trace: bool = False,
    trace_topk: int = 10,
    trace_max_positions: int = 4,
    tokenizer=None,
):
    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")
    if alpha_power <= 0:
        raise ValueError("alpha_power must be > 0.")

    device = prefix_embeds.device
    uncond_prefix_embeds, visual_mask = build_unconditional_prefix_embeds(
        core_model=core_model,
        prefix_embeds=prefix_embeds,
        prefix_input_ids_full=prefix_input_ids_full,
        null_visual_mode=null_visual_mode,
    )

    cond_past = core_model(None, input_embeddings=prefix_embeds, use_cache=True).attn_key_values
    uncond_past = core_model(None, input_embeddings=uncond_prefix_embeds, use_cache=True).attn_key_values
    paired_past = _cat_past_key_values(cond_past, uncond_past)

    x = torch.full((prefix_embeds.shape[0], max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=device)
    num_blocks = max_new_tokens // block_length
    steps = max_new_tokens // num_blocks
    if step_ratio:
        steps = int(steps * step_ratio)
    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0. Increase step_ratio or max_new_tokens.")

    schedule_kwargs = {"shift": schedule_shift} if schedule == "shift" else None
    total_denoising_steps = num_blocks * steps
    global_step_idx = 0
    last_step_meta: Dict = {}
    trace_records: List[Dict] = []

    for block_idx in range(num_blocks):
        block_slice = slice(block_idx * block_length, (block_idx + 1) * block_length)
        block_mask_index = x[:, block_slice] == MASK_TOKEN_ID
        num_transfer_tokens = get_num_transfer_tokens_sch(
            block_mask_index,
            steps,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs,
        )

        for step_idx in range(num_transfer_tokens.shape[1]):
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_slice]
            if block_mask_index.sum().item() == 0:
                global_step_idx += 1
                continue

            current_embeds = core_model.transformer.wte(x)
            paired_embeds = torch.cat([current_embeds, current_embeds], dim=0)
            paired_logits = core_model(
                None,
                input_embeddings=paired_embeds,
                past_key_values=paired_past,
            ).logits
            logits_cond, logits_uncond = torch.chunk(paired_logits, 2, dim=0)

            alpha_t = compute_step_vrg_alpha(
                global_step_idx=global_step_idx,
                total_steps=total_denoising_steps,
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                schedule=alpha_schedule,
                power=alpha_power,
            )
            logits = logits_uncond + (alpha_t + 1.0) * (logits_cond - logits_uncond)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0_p = compute_remasking_confidence(logits, x0, remasking)

            x0_p[:, (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for batch_idx in range(confidence.shape[0]):
                k = int(num_transfer_tokens[batch_idx, step_idx].item())
                _, select_index = torch.topk(confidence[batch_idx], k=k)
                transfer_index[batch_idx, select_index] = True
            x[transfer_index] = x0[transfer_index]

            if return_trace:
                active_positions = torch.nonzero(block_mask_index[0], as_tuple=False).squeeze(-1) + int(block_slice.start)
                trace_records.append(
                    _compute_step_trace(
                        logits_cond=logits_cond,
                        logits_uncond=logits_uncond,
                        logits_guided=logits,
                        active_positions=active_positions,
                        alpha_t=alpha_t,
                        block_idx=block_idx,
                        step_idx=step_idx,
                        global_step_idx=global_step_idx,
                        total_denoising_steps=total_denoising_steps,
                        transfer_index=transfer_index,
                        tokenizer=tokenizer,
                        trace_topk=trace_topk,
                        trace_max_positions=trace_max_positions,
                    )
                )

            last_step_meta = {
                "block_idx": int(block_idx),
                "step_idx": int(step_idx),
                "global_step_idx": int(global_step_idx),
                "total_denoising_steps": int(total_denoising_steps),
                "alpha_t": float(alpha_t),
                "num_visual_tokens": int(visual_mask.sum().item()),
            }
            global_step_idx += 1

    final_meta = {
        "alpha_start": float(alpha_start),
        "alpha_end": float(alpha_end),
        "alpha_schedule": alpha_schedule,
        "alpha_power": float(alpha_power),
        "null_visual_mode": null_visual_mode,
        "num_visual_tokens": int(visual_mask.sum().item()),
        "total_denoising_steps": int(total_denoising_steps),
    }
    if return_trace:
        trace_meta = {
            "trace_topk": int(trace_topk),
            "trace_max_positions": int(trace_max_positions),
            "num_trace_steps": int(len(trace_records)),
        }
        final_meta["trace_meta"] = trace_meta
        return x, last_step_meta, final_meta, trace_records
    return x, last_step_meta, final_meta
