import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


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
from M3CoT.PostVRG.dataset_adapters import add_dataset_adapter_args, load_postvrg_dataset
from M3CoT.PostVRG.postvrg_final import get_torch_dtype, maybe_disable_torch_compile
from M3CoT.utils.metric import judge_answer


def cli_value(flag, default):
    if flag not in sys.argv:
        return default
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


def add_default(flag, value):
    if value is None:
        return
    if flag not in sys.argv:
        sys.argv.extend([flag, str(value)])


def safe_name(value):
    return str(value).replace(".", "p").replace("-", "m")


def apply_remdm_defaults():
    sampler = cli_value("--sampler", "remdm-rescale")
    eta = cli_value("--eta", "1.0")
    sample_seed = cli_value("--sample-seed", "42")
    limit = cli_value("--limit", "400")
    default_output = (
        "M3CoT/ReMDM/outputs/"
        f"{sampler}_eta{safe_name(eta)}_seed{sample_seed}_n{limit}"
    )
    defaults = {
        "--prompt": "cot",
        "--max-new-tokens": 64,
        "--block-length": 64,
        "--step-ratio": 0.5,
        "--limit": limit,
        "--sample-mode": "random",
        "--sample-seed": sample_seed,
        "--sampler": sampler,
        "--eta": eta,
        "--output-dir": default_output,
    }
    for flag, value in defaults.items():
        add_default(flag, value)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run ReMDM / MDLM masked-diffusion decoding on M3CoT-style "
            "multimodal multiple-choice benchmarks."
        )
    )
    parser.add_argument("--dataset-path", default="LightChen2333/M3CoT")
    parser.add_argument("--split", default="test")
    add_dataset_adapter_args(parser)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--domain-filter", default=None)
    parser.add_argument("--sample-mode", default="sequential", choices=["sequential", "random"])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="M3CoT/ReMDM/outputs/remdm")

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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random", "entrophy", "margin"])

    parser.add_argument(
        "--sampler",
        default="remdm-rescale",
        choices=["mdlm", "remdm-cap", "remdm-rescale", "remdm-conf", "remdm-loop"],
        help="Reverse sampler. ReMDM variants mirror the public ReMDM implementation.",
    )
    parser.add_argument("--eta", type=float, default=1.0, help="ReMDM remasking strength.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Final reverse-time epsilon.")
    parser.add_argument(
        "--nucleus-p",
        type=float,
        default=1.0,
        help="Top-p truncation applied to p(x0|xt) before the reverse update.",
    )
    parser.add_argument(
        "--noise-removal",
        action="store_true",
        default=True,
        help="After the last reverse step, replace remaining masks with argmax denoiser predictions.",
    )
    parser.add_argument(
        "--no-noise-removal",
        action="store_false",
        dest="noise_removal",
        help="Leave any final masks untouched for ablation.",
    )
    parser.add_argument("--loop-t-on", type=float, default=0.8)
    parser.add_argument("--loop-t-off", type=float, default=0.2)
    parser.add_argument("--loop-alpha-on", type=float, default=0.5)
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


def compute_prefix_kv_cache(core_model, prefix_embeds):
    output = core_model(None, input_embeddings=prefix_embeds, use_cache=True)
    return output.attn_key_values


def forward_answer_logits(core_model, x, prefix_embeds, prefix_length, prefix_kv_cache=None):
    if prefix_kv_cache is not None:
        answer_embeds = core_model.transformer.wte(x[:, prefix_length:])
        output = core_model(None, input_embeddings=answer_embeds, past_key_values=prefix_kv_cache)
        return output.logits.to(x.device)
    current_embeds = core_model.transformer.wte(x)
    current_embeds[:, :prefix_length] = prefix_embeds
    return core_model(None, input_embeddings=current_embeds).logits[:, prefix_length:].to(x.device)


def apply_nucleus(probs, nucleus_p):
    if nucleus_p >= 1.0:
        return probs
    if nucleus_p <= 0.0:
        raise ValueError("--nucleus-p must be in (0, 1].")
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    top_p_mask = cumulative_probs <= nucleus_p
    top_p_mask[..., 0] = True
    nucleus_probs = sorted_probs * top_p_mask
    nucleus_probs = nucleus_probs / nucleus_probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return torch.zeros_like(probs).scatter_(-1, sorted_indices, nucleus_probs)


def sample_categorical(categorical_probs, temperature):
    categorical_probs = categorical_probs.to(torch.float64).clamp_min(0.0)
    if temperature == 0:
        return categorical_probs.argmax(dim=-1)
    noise = torch.rand_like(categorical_probs, dtype=torch.float64).clamp_min(1e-30)
    gumbel_norm = (-torch.log(noise)) ** float(temperature)
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def masked_softmax(values, mask):
    scores = values.to(torch.float64).masked_fill(~mask, -torch.inf)
    if not torch.isfinite(scores).any():
        return torch.zeros_like(scores)
    return torch.softmax(scores, dim=-1)


def update_confidence(confidence, x_prev, x_next, p_x0):
    unmask_mask = (x_prev == MASK_TOKEN_ID) & (x_next != MASK_TOKEN_ID)
    if unmask_mask.any():
        selected_probs = torch.gather(
            p_x0,
            dim=-1,
            index=x_next.clamp_min(0).unsqueeze(-1),
        ).squeeze(-1)
        confidence[unmask_mask] = -selected_probs[unmask_mask].to(confidence.dtype)

    remask_mask = (x_prev != MASK_TOKEN_ID) & (x_next == MASK_TOKEN_ID)
    confidence[remask_mask] = -torch.inf
    return confidence


def remdm_q_from_sigma(p_x0, x, alpha_t, alpha_s, sigma):
    if not torch.is_tensor(sigma):
        sigma = torch.full(x.shape, float(sigma), dtype=p_x0.dtype, device=x.device)
    else:
        sigma = sigma.to(dtype=p_x0.dtype, device=x.device)
    sigma = sigma.clamp(0.0, 1.0)

    q_unmasked = p_x0 * (1.0 - sigma.unsqueeze(-1))
    q_unmasked[..., MASK_TOKEN_ID] = sigma

    denom = max(1e-12, 1.0 - float(alpha_t))
    coef = (float(alpha_s) - (1.0 - sigma.unsqueeze(-1)) * float(alpha_t)) / denom
    mask_prob = (1.0 - float(alpha_s) - sigma * float(alpha_t)) / denom
    q_masked = p_x0 * coef
    q_masked[..., MASK_TOKEN_ID] = mask_prob

    copy_flag = x != MASK_TOKEN_ID
    q = torch.where(copy_flag.unsqueeze(-1), q_unmasked, q_masked)
    return q.clamp_min(0.0)


def mdlm_update(x, p_x0, t, dt, temperature):
    move_chance_t = float(t)
    move_chance_s = max(float(t) - float(dt), 0.0)
    q_xs = p_x0 * max(move_chance_t - move_chance_s, 0.0)
    q_xs[..., MASK_TOKEN_ID] = move_chance_s
    sampled = sample_categorical(q_xs, temperature=temperature)
    copy_flag = x != MASK_TOKEN_ID
    return torch.where(copy_flag, x, sampled), {
        "sigma": None,
        "sigma_max": None,
        "mode": "mdlm",
    }


def loop_move_chances(t, dt, loop_t_on, loop_t_off, loop_alpha_on):
    if t > loop_t_on:
        move_t = 1.0 - (1.0 - t) * loop_alpha_on / max(1e-12, 1.0 - loop_t_on)
        move_s = 1.0 - (1.0 - t + dt) * loop_alpha_on / max(1e-12, 1.0 - loop_t_on)
    elif t <= loop_t_off:
        move_t = t * (1.0 - loop_alpha_on) / max(1e-12, loop_t_off)
        move_s = (t - dt) * (1.0 - loop_alpha_on) / max(1e-12, loop_t_off)
    else:
        return None, None
    return max(move_t, 0.0), max(move_s, 0.0)


def remdm_update(
    x,
    p_x0,
    confidence,
    sampler,
    t,
    dt,
    eta,
    temperature,
    loop_t_on,
    loop_t_off,
    loop_alpha_on,
):
    if sampler == "mdlm":
        x_next, meta = mdlm_update(x, p_x0, t, dt, temperature)
        return x_next, confidence, meta

    if sampler == "remdm-loop":
        move_t, move_s = loop_move_chances(t, dt, loop_t_on, loop_t_off, loop_alpha_on)
        if move_t is not None:
            q_xs = p_x0 * max(move_t - move_s, 0.0)
            q_xs[..., MASK_TOKEN_ID] = move_s
            sampled = sample_categorical(q_xs, temperature=temperature)
            copy_flag = x != MASK_TOKEN_ID
            x_next = torch.where(copy_flag, x, sampled)
            confidence = update_confidence(confidence, x, x_next, p_x0)
            return x_next, confidence, {
                "sigma": None,
                "sigma_max": None,
                "mode": "loop-mdlm",
                "move_chance_t": move_t,
                "move_chance_s": move_s,
            }

    alpha_t = max(0.0, 1.0 - float(t))
    alpha_s = max(0.0, 1.0 - (float(t) - float(dt)))
    if alpha_t > 0.0:
        sigma_max = min(1.0, max(0.0, (1.0 - alpha_s) / alpha_t))
    else:
        sigma_max = 1.0

    if sampler == "remdm-cap":
        sigma = min(float(eta), sigma_max)
        q_xs = remdm_q_from_sigma(p_x0, x, alpha_t, alpha_s, sigma)
    elif sampler == "remdm-rescale":
        sigma = float(eta) * sigma_max
        q_xs = remdm_q_from_sigma(p_x0, x, alpha_t, alpha_s, sigma)
    elif sampler == "remdm-conf":
        eligible = x != MASK_TOKEN_ID
        eta_by_pos = masked_softmax(confidence, eligible)
        sigma = eta_by_pos * sigma_max
        q_xs = remdm_q_from_sigma(p_x0, x, alpha_t, alpha_s, sigma)
    elif sampler == "remdm-loop":
        sigma = float(eta)
        q_xs = remdm_q_from_sigma(
            p_x0,
            x,
            alpha_t=float(loop_alpha_on),
            alpha_s=float(loop_alpha_on),
            sigma=sigma,
        )
    else:
        raise ValueError(f"Unsupported sampler: {sampler}")

    x_next = sample_categorical(q_xs, temperature=temperature)
    confidence = update_confidence(confidence, x, x_next, p_x0)
    sigma_value = float(sigma.mean().item()) if torch.is_tensor(sigma) else float(sigma)
    return x_next, confidence, {
        "sigma": sigma_value,
        "sigma_max": float(sigma_max),
        "mode": sampler,
    }


@torch.no_grad()
def generate_with_remdm(
    core_model,
    tokenizer,
    prefix_embeds,
    max_new_tokens,
    block_length,
    step_per_block,
    step_ratio,
    temperature,
    remasking,
    sampler,
    eta,
    eps,
    nucleus_p,
    noise_removal,
    loop_t_on,
    loop_t_off,
    loop_alpha_on,
):
    total_steps = resolve_total_steps(
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        step_per_block=step_per_block,
        step_ratio=step_ratio,
    )
    if not (0.0 < eps < 1.0):
        raise ValueError("--eps must be in (0, 1).")
    if eta < 0.0:
        raise ValueError("--eta must be >= 0.")
    if not (0.0 < nucleus_p <= 1.0):
        raise ValueError("--nucleus-p must be in (0, 1].")
    if sampler == "remdm-loop" and not (0.0 < loop_t_off < loop_t_on < 1.0):
        raise ValueError("--loop-t-off < --loop-t-on must both lie in (0, 1).")

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
    answer_x = x[:, answer_slice]
    confidence = torch.full(
        answer_x.shape,
        -torch.inf,
        dtype=torch.float64,
        device=device,
    )

    timesteps = torch.linspace(1.0, float(eps), total_steps + 1, device=device)
    dt = (1.0 - float(eps)) / float(total_steps)
    prefix_kv_cache = compute_prefix_kv_cache(core_model, prefix_embeds)
    step_records = []

    for step_idx in range(1, total_steps + 1):
        t = float(timesteps[step_idx - 1].item())
        answer_logits = forward_answer_logits(
            core_model,
            x,
            prefix_embeds,
            prefix_length,
            prefix_kv_cache=prefix_kv_cache,
        )
        p_x0 = F.softmax(answer_logits.to(torch.float64), dim=-1)
        p_x0 = apply_nucleus(p_x0, nucleus_p)
        candidate_ids = p_x0.argmax(dim=-1)
        proposal_confidence = compute_remasking_confidence(
            answer_logits,
            candidate_ids,
            remasking,
        )[0].detach().to(torch.float64)

        prev_answer_x = answer_x.clone()
        next_answer_x, confidence, update_meta = remdm_update(
            x=answer_x,
            p_x0=p_x0,
            confidence=confidence,
            sampler=sampler,
            t=t,
            dt=dt,
            eta=eta,
            temperature=temperature,
            loop_t_on=loop_t_on,
            loop_t_off=loop_t_off,
            loop_alpha_on=loop_alpha_on,
        )
        x[:, answer_slice] = next_answer_x
        answer_x = x[:, answer_slice]

        changed_mask = prev_answer_x != next_answer_x
        filled_mask = (prev_answer_x == MASK_TOKEN_ID) & (next_answer_x != MASK_TOKEN_ID)
        remasked_mask = (prev_answer_x != MASK_TOKEN_ID) & (next_answer_x == MASK_TOKEN_ID)
        step_records.append(
            {
                "step": int(step_idx),
                "phase": "remdm",
                "t": t,
                "dt": dt,
                "mode": update_meta["mode"],
                "sigma": update_meta["sigma"],
                "sigma_max": update_meta["sigma_max"],
                "num_changed": int(changed_mask.sum().item()),
                "num_filled": int(filled_mask.sum().item()),
                "num_remasked": int(remasked_mask.sum().item()),
                "filled_answer_positions": [
                    int(pos)
                    for pos in filled_mask[0].nonzero(as_tuple=False).flatten().detach().cpu().tolist()
                ],
                "remasked_answer_positions": [
                    int(pos)
                    for pos in remasked_mask[0].nonzero(as_tuple=False).flatten().detach().cpu().tolist()
                ],
                "candidate_text": decode_answer(tokenizer, candidate_ids[0].detach().cpu().tolist()),
                "state_text": decode_answer(tokenizer, answer_x[0].detach().cpu().tolist()),
                "num_masked_after_step": int((answer_x == MASK_TOKEN_ID).sum().item()),
                "proposal_confidence_mean": float(proposal_confidence.mean().item()),
            }
        )

    if noise_removal and (answer_x == MASK_TOKEN_ID).any():
        answer_logits = forward_answer_logits(
            core_model,
            x,
            prefix_embeds,
            prefix_length,
            prefix_kv_cache=prefix_kv_cache,
        )
        denoised = answer_logits.argmax(dim=-1)
        answer_mask = answer_x == MASK_TOKEN_ID
        answer_x = torch.where(answer_mask, denoised, answer_x)
        x[:, answer_slice] = answer_x

    final_answer_ids = answer_x[0].detach().cpu().tolist()
    final_text = decode_answer(tokenizer, final_answer_ids)
    meta = {
        "max_new_tokens": int(max_new_tokens),
        "block_length": int(block_length),
        "total_steps": int(total_steps),
        "executed_steps": int(len(step_records)),
        "sampler": sampler,
        "eta": float(eta),
        "eps": float(eps),
        "nucleus_p": float(nucleus_p),
        "noise_removal": bool(noise_removal),
        "temperature": float(temperature),
        "remasking": remasking,
        "loop_t_on": float(loop_t_on) if sampler == "remdm-loop" else None,
        "loop_t_off": float(loop_t_off) if sampler == "remdm-loop" else None,
        "loop_alpha_on": float(loop_alpha_on) if sampler == "remdm-loop" else None,
        "num_masked_final": int((answer_x == MASK_TOKEN_ID).sum().item()),
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

    from llava.conversation import conv_templates
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
    if args.conv_template in conv_templates:
        conv_templates[args.conv_template].tokenizer = tokenizer
    model.eval()
    model.tie_weights()
    model.to(get_torch_dtype(args.torch_dtype))
    core_model = model.get_model()
    if hasattr(core_model, "set_activation_checkpointing"):
        core_model.set_activation_checkpointing(None)

    dataset = load_postvrg_dataset(args)
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

    with (records_path.open("w", encoding="utf-8") if write_records else contextlib.nullcontext()) as fout:
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
                temperature=args.temperature,
                remasking=args.remasking,
                sampler=args.sampler,
                eta=args.eta,
                eps=args.eps,
                nucleus_p=args.nucleus_p,
                noise_removal=args.noise_removal,
                loop_t_on=args.loop_t_on,
                loop_t_off=args.loop_t_off,
                loop_alpha_on=args.loop_alpha_on,
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
                    "benchmark": doc.get("benchmark"),
                    "raw_index": doc.get("raw_index"),
                    "prompt": prompt,
                    "elapsed_sec": elapsed,
                    "final_text": run_output["final_text"],
                    "final_answer_ids": run_output["final_answer_ids"],
                    "final_correct": final_correct,
                    "step_records": run_output["step_records"],
                    "meta": run_output["meta"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
            written += 1

            if args.print_every > 0 and written % args.print_every == 0:
                print(
                    f"[{written}] dataset_index={dataset_index} id={doc['id']} "
                    f"final={final_correct} elapsed={elapsed:.2f}s",
                    flush=True,
                )

    summary = {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "benchmark": args.benchmark,
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
            "temperature": args.temperature,
            "remasking": args.remasking,
            "sampler": args.sampler,
            "eta": args.eta,
            "eps": args.eps,
            "nucleus_p": args.nucleus_p,
            "noise_removal": args.noise_removal,
            "loop_t_on": args.loop_t_on if args.sampler == "remdm-loop" else None,
            "loop_t_off": args.loop_t_off if args.sampler == "remdm-loop" else None,
            "loop_alpha_on": args.loop_alpha_on if args.sampler == "remdm-loop" else None,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if write_records:
        print(f"Wrote records to {records_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)
    restore_compile()


if __name__ == "__main__":
    apply_remdm_defaults()
    main()
