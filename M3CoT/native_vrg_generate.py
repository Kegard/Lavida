import torch

from llava.model.language_model.llada.generate import add_gumbel_noise, get_num_transfer_tokens_sch
from Scale_Attention.reweight_patch import build_prefix_from_multimodal_inputs
from VRG.timestep_vrg import (
    build_unconditional_prefix_embeds,
    compute_remasking_confidence,
    compute_step_vrg_alpha,
)


MASK_TOKEN_ID = 126336


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.to(torch.float64), dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    return -(probs * log_probs).sum(dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.mean().item())


@torch.no_grad()
def generate_with_native_vrg(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    images,
    image_sizes,
    max_new_tokens: int,
    block_length: int,
    temperature: float,
    step_ratio: float = None,
    step_per_block: int = None,
    remasking: str = "low_confidence",
    alpha_start: float = 0.0,
    alpha_end: float = 1.0,
    alpha_schedule: str = "linear",
    alpha_power: float = 2.0,
    null_visual_mode: str = "zeros",
    vrg_gate: str = "none",
    vrg_entropy_threshold: float = 0.0,
    return_gate_stats: bool = False,
):
    if vrg_gate not in {"none", "entropy"}:
        raise ValueError(f"Unsupported vrg_gate: {vrg_gate}")

    core_model = model.get_model()
    cond_inputs_embeds, prefix_input_ids_full = build_prefix_from_multimodal_inputs(
        model=model,
        input_ids=input_ids,
        images=images,
        image_sizes=image_sizes,
        attention_mask=attention_mask,
    )
    uncond_inputs_embeds, _ = build_unconditional_prefix_embeds(
        core_model=core_model,
        prefix_embeds=cond_inputs_embeds,
        prefix_input_ids_full=prefix_input_ids_full,
        null_visual_mode=null_visual_mode,
    )

    bsz, prompt_len = cond_inputs_embeds.shape[:2]
    x = torch.full((bsz, prompt_len + max_new_tokens), MASK_TOKEN_ID, dtype=torch.long, device=cond_inputs_embeds.device)
    x[:, :prompt_len] = 0

    if max_new_tokens % block_length != 0:
        raise ValueError("max_new_tokens must be divisible by block_length.")

    num_blocks = max_new_tokens // block_length
    steps = max_new_tokens // num_blocks
    if step_per_block is not None:
        steps = min(int(step_per_block), block_length)
    elif step_ratio is not None:
        steps = int(steps * float(step_ratio))
    if steps <= 0:
        raise ValueError("The computed number of steps per block is 0. Increase step_ratio or max_new_tokens.")

    total_denoising_steps = num_blocks * steps
    global_step_idx = 0
    gate_records = []

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = prompt_len + (block_idx + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == MASK_TOKEN_ID
        num_transfer_tokens = get_num_transfer_tokens_sch(
            block_mask_index,
            steps,
            schedule=None,
            schedule_kwargs=None,
        )

        for step_idx in range(num_transfer_tokens.shape[1]):
            mask_index = x == MASK_TOKEN_ID
            block_mask_index = mask_index[:, block_start:block_end]
            if block_mask_index.sum().item() == 0:
                global_step_idx += 1
                continue

            current_embeds = core_model.transformer.wte(x)
            cond_current_embeds = current_embeds.clone()
            uncond_current_embeds = current_embeds.clone()
            cond_current_embeds[:, :prompt_len] = cond_inputs_embeds
            uncond_current_embeds[:, :prompt_len] = uncond_inputs_embeds

            logits_cond = core_model(
                None,
                input_embeddings=cond_current_embeds,
            ).logits
            logits_uncond = core_model(
                None,
                input_embeddings=uncond_current_embeds,
            ).logits

            alpha_t = compute_step_vrg_alpha(
                global_step_idx=global_step_idx,
                total_steps=total_denoising_steps,
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                schedule=alpha_schedule,
                power=alpha_power,
            )
            logits = logits_cond + alpha_t * (logits_cond - logits_uncond)
            use_vrg = True
            entropy_base = None
            entropy_vrg = None
            delta_entropy = None
            if vrg_gate == "entropy":
                token_entropy_base = compute_token_entropy(logits_cond)
                token_entropy_vrg = compute_token_entropy(logits)
                entropy_base = masked_mean(token_entropy_base[:, block_start:block_end], block_mask_index)
                entropy_vrg = masked_mean(token_entropy_vrg[:, block_start:block_end], block_mask_index)
                delta_entropy = entropy_base - entropy_vrg
                use_vrg = delta_entropy > float(vrg_entropy_threshold)
                if not use_vrg:
                    logits = logits_cond

            if return_gate_stats:
                gate_records.append(
                    {
                        "step": int(global_step_idx + 1),
                        "block_index": int(block_idx + 1),
                        "step_in_block": int(step_idx + 1),
                        "alpha_t": float(alpha_t),
                        "use_vrg": bool(use_vrg),
                        "entropy_base": entropy_base,
                        "entropy_vrg": entropy_vrg,
                        "delta_entropy": delta_entropy,
                    }
                )

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            x0_p = compute_remasking_confidence(logits, x0, remasking)

            x0_p[:, prompt_len + (block_idx + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for batch_idx in range(confidence.shape[0]):
                k = int(num_transfer_tokens[batch_idx, step_idx].item())
                _, select_index = torch.topk(confidence[batch_idx], k=k)
                transfer_index[batch_idx, select_index] = True
            x[transfer_index] = x0[transfer_index]
            global_step_idx += 1

    if return_gate_stats:
        return x, gate_records
    return x
